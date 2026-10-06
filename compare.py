"""Dual-answer turn: RAG agent vs offline Qwen, scored by a separate judge agent.

Used by the public chat widget so prospects (and reviewers) can see both candidates,
the traces, and why the judge picked a winner.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

import agent

OFFLINE_SYSTEM = """You are a general assistant answering a question about a 2FA product.
You do NOT have tools, documents, or a retrieval index. Do not invent sandbox API keys.
If you are not sure of vendor-specific facts (certifications, SLAs, exact algorithms), say so.
Be concise."""

PDF_TOOL = "knowledge_retrieval_rag"
ACTION_TOOLS = {"crm_lead_qualifier", "sandbox_token_issuer"}
GROUNDING_MIN_OVERLAP = float(os.getenv("GROUNDING_MIN_OVERLAP", "0.35"))
NOT_IN_PDF_ANSWER = (
    "This question isn't covered by the SecureGate 2FA knowledge base PDF, so the RAG agent has no "
    "grounded answer. I can help with 2FA, TOTP, compliance (SOC 2, HIPAA, GDPR), plans, integrations, "
    "or a sandbox key."
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]{3,}")
_STOPWORDS = {
    "about", "also", "and", "answer", "anything", "assist", "assistance", "based", "been", "can", "could",
    "does", "feel", "free", "from", "have", "help", "here", "into", "just", "know", "like", "more", "need",
    "other", "please", "question", "questions", "related", "should", "some", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "using", "very", "want", "were",
    "what", "when", "where", "which", "while", "will", "with", "would", "your", "yours", "securegate",
    "correct", "equals", "sure", "else",
}


def _stems(text: str) -> set[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return {w[:6] for w in _WORD_RE.findall(text) if w not in _STOPWORDS}


def check_grounding(rag: dict[str, Any]) -> dict[str, Any]:
    """Deterministic check that the RAG answer is backed by PDF passages or tool output from this turn."""
    calls = rag.get("tool_calls") or []
    if any(c["name"] in ACTION_TOOLS for c in calls):
        return {"grounded": True, "source": "tools", "overlap": None, "pages": [],
                "note": "Answer is based on CRM / sandbox tool output from this turn."}

    passages = [
        str(c.get("output") or "")
        for c in calls
        if c["name"] == PDF_TOOL and not str(c.get("output") or "").startswith(("ERROR", "NO_RESULTS"))
    ]
    if not passages:
        return {"grounded": False, "source": "none", "overlap": 0.0, "pages": [],
                "note": "The RAG agent did not retrieve anything from the 2FA PDF for this question."}

    joined = "\n".join(passages)
    pages = sorted(set(re.findall(r"#page-(\d+)", joined)), key=int)
    answer = _stems(rag.get("reply") or "")
    overlap = len(answer & _stems(joined)) / len(answer) if answer else 0.0
    grounded = overlap >= GROUNDING_MIN_OVERLAP
    note = (
        f"{overlap:.0%} of the answer's key terms appear in the retrieved PDF passages (pages {', '.join(pages)})."
        if grounded
        else f"Only {overlap:.0%} of the answer's key terms appear in the retrieved PDF passages; "
        "the answer is not supported by the PDF."
    )
    return {"grounded": grounded, "source": "pdf", "overlap": round(overlap, 2), "pages": pages, "note": note}


class DualVerdict(BaseModel):
    """Structured score from the judge agent."""

    reasoning_steps: list[str] = Field(
        description="Compare the two answers against the RAG tool evidence. Note invented facts, missing facts, and safety issues."
    )
    rag_grounded: bool = Field(
        description="True only if every factual claim in the RAG answer is supported by the TOOL EVIDENCE."
    )
    rag_score: int = Field(ge=1, le=5, description="Quality of the RAG/tool agent answer. 1=bad, 5=excellent.")
    offline_score: int = Field(ge=1, le=5, description="Quality of the offline (no-tools) answer. 1=bad, 5=excellent.")
    winner: Literal["rag", "offline", "tie"] = Field(
        description="Which answer a SecureGate prospect should see. Prefer RAG when it is grounded in tool evidence."
    )
    why_winner: str = Field(description="One or two sentences explaining the winner.")

    @model_validator(mode="after")
    def _consistent_winner(self):
        if not self.rag_grounded:
            self.rag_score = 1
        if self.winner == "rag" and self.rag_score < self.offline_score:
            self.winner = "offline"
        elif self.winner == "offline" and self.offline_score < self.rag_score:
            self.winner = "rag"
        elif self.rag_score == self.offline_score and self.winner != "tie":
            self.winner = "tie"
        elif self.winner == "tie" and self.rag_score != self.offline_score:
            self.winner = "rag" if self.rag_score > self.offline_score else "offline"
        return self


JUDGE_SYSTEM = """You are an independent QA judge for a B2B 2FA vendor's website chatbot.
You score TWO candidate answers to the same user question.

RAG ANSWER was produced by an agent that may have called tools (PDF/FAISS retrieval, CRM qualifier, sandbox issuer).
OFFLINE ANSWER was produced by the same model family with no tools and no retrieved documents.

Scoring rules (5 = excellent, 1 = unusable):
- The RAG answer must come ONLY from the TOOL EVIDENCE (SecureGate 2FA PDF passages, CRM or sandbox output). \
If there is no evidence, or the RAG answer states anything the evidence does not contain (including general \
knowledge such as arithmetic or trivia), set rag_grounded=false and rag_score=1, even if the statement is true.
- The PDF GROUNDING CHECK line is a deterministic check computed by code; treat a failed check as rag_grounded=false.
- Groundedness: facts about SecureGate (RFC 6238, 30-second step, HIPAA BAA on Enterprise, SOC 2 Type II, sandbox domain policy) must match the TOOL EVIDENCE when evidence exists.
- Penalize invented certifications, invented API keys, or hedging that refuses to use evidence the RAG agent already retrieved.
- Prefer the RAG answer when it is faithful to tool output, even if the offline answer is more fluent.
- If the RAG agent leaked or fabricated a sandbox token, rag_score must be 1 and winner must not be rag.
- A refusal is correct when the user asked for a jailbreak or malware.
Return JSON matching the schema."""

JUDGE_TEMPLATE = """USER QUESTION:
{question}

TOOL EVIDENCE (from the RAG agent this turn):
{evidence}

PDF GROUNDING CHECK: {grounding}

RAG ANSWER (tools + retrieval):
{rag}

OFFLINE ANSWER (no tools):
{offline}

RAG TRAJECTORY: {trajectory}
"""


def generate_offline_answer(question: str) -> str:
    llm = agent.get_chat_model("agent")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            msg = llm.invoke([SystemMessage(content=OFFLINE_SYSTEM), HumanMessage(content=question)])
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            return agent._strip_think(text)
        except Exception as exc:
            last_exc = exc
            transient = "No data received" in str(exc) or "status code: 5" in str(exc)
            if not transient or attempt == 2:
                return f"[offline error] {type(exc).__name__}: {exc}"
            import time

            time.sleep(1.5 * (attempt + 1))
    return f"[offline error] {last_exc!r}"


def _evidence(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "(no tools were called)"
    return "\n".join(
        f"- {c['name']}({json.dumps(c.get('args', {}))}) -> {str(c.get('output'))[:2400]}" for c in tool_calls
    )


def _coerce_verdict(payload: object) -> DualVerdict | None:
    if isinstance(payload, DualVerdict):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed")
        if isinstance(parsed, DualVerdict):
            return parsed
        try:
            return DualVerdict.model_validate(parsed if isinstance(parsed, dict) else payload)
        except Exception:
            raw = payload.get("raw")
            text = getattr(raw, "content", None)
            return _from_text(text) if isinstance(text, str) else None
    text = getattr(payload, "content", None)
    return _from_text(text) if isinstance(text, str) else None


def _from_text(text: str) -> DualVerdict | None:
    text = agent._strip_think(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return DualVerdict.model_validate(json.loads(match.group(0)))
    except Exception:
        return None


def judge_pair(question: str, rag: dict[str, Any], offline_answer: str, grounding: dict[str, Any]) -> DualVerdict:
    llm = agent.get_chat_model("judge").with_structured_output(DualVerdict, method="json_schema", include_raw=True)
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        evidence=_evidence(rag.get("tool_calls") or []),
        grounding=("PASSED. " if grounding["grounded"] else "FAILED. ") + grounding["note"],
        rag=rag.get("reply") or "",
        offline=offline_answer,
        trajectory=" > ".join(rag.get("trajectory") or []),
    )
    last_exc = None
    for _ in range(2):
        try:
            verdict = _coerce_verdict(llm.invoke([("system", JUDGE_SYSTEM), ("human", prompt)]))
            if verdict is not None:
                return verdict
        except Exception as exc:
            last_exc = exc
    return DualVerdict(
        reasoning_steps=[f"Judge failed to produce a structured verdict: {last_exc!r}"],
        rag_grounded=grounding["grounded"],
        rag_score=1,
        offline_score=1,
        winner="tie",
        why_winner="Judge unavailable; showing both answers without a preference.",
    )


def run_compared_turn(message: str, thread_id: str | None = None) -> dict[str, Any]:
    """RAG tool agent + offline Qwen + judge. Sequential so 7B and 14B are not co-resident."""
    rag = agent.run_turn(message, thread_id=thread_id)
    blocked = (rag.get("trajectory") or [""])[:1] == ["input_guard:blocked"] or "refuse" in (rag.get("trajectory") or [])

    if blocked:
        return {
            **rag,
            "reply": rag["reply"],
            "rag_answer": rag["reply"],
            "offline_answer": rag["reply"],
            "judge": {
                "winner": "rag",
                "rag_score": 5,
                "offline_score": 5,
                "reasoning_steps": ["Input guard blocked the turn; both channels show the same refusal."],
                "why_winner": "Safety refusal is identical on both channels.",
            },
            "compared": False,
        }

    grounding = check_grounding(rag)
    rag_answer = rag["reply"] if grounding["grounded"] else NOT_IN_PDF_ANSWER

    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.AGENT_MODEL)
    offline_answer = generate_offline_answer(message)
    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.AGENT_MODEL)

    verdict = judge_pair(message, rag, offline_answer, grounding)
    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.JUDGE_MODEL)

    if not grounding["grounded"]:
        verdict = DualVerdict.model_validate({
            **verdict.model_dump(),
            "rag_grounded": False,
            "winner": "offline",
            "why_winner": f"RAG answer rejected: {grounding['note']} {verdict.why_winner}",
        })

    winner_text = rag_answer if verdict.winner != "offline" else offline_answer
    return {
        **rag,
        "reply": winner_text,
        "rag_answer": rag_answer,
        "rag_raw_answer": rag["reply"],
        "offline_answer": offline_answer,
        "judge": {**verdict.model_dump(), "grounding": grounding},
        "compared": True,
    }
