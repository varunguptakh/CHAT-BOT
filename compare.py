"""Dual-answer turn: RAG agent vs offline Qwen, scored by a separate judge agent.

Used by the public chat widget so prospects (and reviewers) can see both candidates,
the traces, and why the judge picked a winner.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

import agent

OFFLINE_SYSTEM = """You are a general assistant answering a question about a 2FA product.
You do NOT have tools, documents, or a retrieval index. Do not invent sandbox API keys.
If you are not sure of vendor-specific facts (certifications, SLAs, exact algorithms), say so.
Be concise."""


class DualVerdict(BaseModel):
    """Structured score from the judge agent."""

    reasoning_steps: list[str] = Field(
        description="Compare the two answers against the RAG tool evidence. Note invented facts, missing facts, and safety issues."
    )
    rag_score: int = Field(ge=1, le=5, description="Quality of the RAG/tool agent answer. 1=bad, 5=excellent.")
    offline_score: int = Field(ge=1, le=5, description="Quality of the offline (no-tools) answer. 1=bad, 5=excellent.")
    winner: Literal["rag", "offline", "tie"] = Field(
        description="Which answer a SecureGate prospect should see. Prefer RAG when it is grounded in tool evidence."
    )
    why_winner: str = Field(description="One or two sentences explaining the winner.")

    @model_validator(mode="after")
    def _consistent_winner(self):
        if self.winner == "rag" and self.rag_score < self.offline_score:
            self.winner = "offline"
        elif self.winner == "offline" and self.offline_score < self.rag_score:
            self.winner = "rag"
        elif self.rag_score == self.offline_score and self.winner != "tie":
            self.winner = "tie"
        return self


JUDGE_SYSTEM = """You are an independent QA judge for a B2B 2FA vendor's website chatbot.
You score TWO candidate answers to the same user question.

RAG ANSWER was produced by an agent that may have called tools (PDF/FAISS retrieval, CRM qualifier, sandbox issuer).
OFFLINE ANSWER was produced by the same model family with no tools and no retrieved documents.

Scoring rules (5 = excellent, 1 = unusable):
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
        f"- {c['name']}({json.dumps(c.get('args', {}))}) -> {str(c.get('output'))[:900]}" for c in tool_calls
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


def judge_pair(question: str, rag: dict[str, Any], offline_answer: str) -> DualVerdict:
    llm = agent.get_chat_model("judge").with_structured_output(DualVerdict, method="json_schema", include_raw=True)
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        evidence=_evidence(rag.get("tool_calls") or []),
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

    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.AGENT_MODEL)
    offline_answer = generate_offline_answer(message)
    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.AGENT_MODEL)

    verdict = judge_pair(message, rag, offline_answer)
    if agent.LLM_PROVIDER == "ollama":
        agent.release_ollama_model(agent.JUDGE_MODEL)

    winner_text = rag["reply"] if verdict.winner != "offline" else offline_answer
    return {
        **rag,
        "reply": winner_text,
        "rag_answer": rag["reply"],
        "offline_answer": offline_answer,
        "judge": verdict.model_dump(),
        "compared": True,
    }
