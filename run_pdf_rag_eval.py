"""PDF RAG evaluation: retrieve from the 2FA knowledge PDF, answer with Qwen, score vs gold.

For every evaluation question we produce two candidate answers:

  * rag_qwen        - Qwen 2.5 instructed to use only the retrieved PDF chunks
  * closed_book_qwen - Qwen 2.5 with no PDF context (baseline)

A structured LLM judge (qwen3:14b) scores faithfulness + completeness against the gold
answer. A deterministic keyword hit-rate is mixed in. The higher combined score wins.

Usage:
    python build_pdf.py && python run_pdf_rag_eval.py
    python run_pdf_rag_eval.py --limit 4
    python run_pdf_rag_eval.py --id hipaa --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import agent
from rag_pdf import PDF_PATH, QA_PAIRS_PATH

console = Console()
REPORT_PATH = Path("pdf_rag_report.json")

RAG_SYSTEM = (
    "You are Aegis, SecureGate's technical sales assistant. Answer the user using ONLY "
    "the PDF excerpts. Cite the source in square brackets (for example [securegate_2fa_knowledge.pdf#page-2]). "
    "If the excerpts do not contain the answer, say you do not know. Do not invent certifications, SLAs, or keys."
)

CLOSED_SYSTEM = (
    "You are a general assistant. Answer the two-factor authentication product question "
    "from your own knowledge. Do not claim you retrieved a company document. Be concise."
)


class AnswerScore(BaseModel):
    reasoning_steps: list[str] = Field(description="Step-by-step comparison of the candidate against the gold answer.")
    faithfulness: int = Field(ge=1, le=5, description="1=contradicts gold/PDF, 5=fully consistent.")
    completeness: int = Field(ge=1, le=5, description="1=misses the required facts, 5=covers them.")
    overall: int = Field(ge=1, le=5, description="1=unusable, 5=best possible grounded answer.")
    passed: bool = Field(description="True if overall >= 4 and faithfulness >= 4.")

    @model_validator(mode="after")
    def _align(self):
        if self.passed:
            self.overall = max(self.overall, 4)
        return self


def keyword_score(answer: str, must_contain: list[str]) -> float:
    if not must_contain:
        return 1.0
    blob = answer.lower()
    hits = sum(1 for k in must_contain if k.lower() in blob)
    return hits / len(must_contain)


def combined_score(judge: AnswerScore | None, lexical: float) -> float:
    overall = (judge.overall / 5.0) if judge else 0.0
    return round(0.6 * overall + 0.4 * lexical, 4)


def retrieve(question: str, k: int) -> list[tuple[object, float]]:
    store = agent.get_vector_store()
    try:
        pairs = store.similarity_search_with_relevance_scores(question, k=k)
        return [(doc, float(score)) for doc, score in pairs]
    except Exception:
        docs = store.similarity_search(question, k=k)
        return [(doc, 0.0) for doc in docs]


def format_context(hits: list[tuple[object, float]]) -> str:
    blocks = []
    for doc, score in hits:
        src = doc.metadata.get("source", "unknown")
        blocks.append(f"[{src}] (relevance={score:.3f})\n{doc.page_content}")
    return "\n\n".join(blocks) or "(no excerpts retrieved)"


def generate(system: str, user: str) -> str:
    llm = agent.get_chat_model("agent")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            msg = llm.invoke([("system", system), ("human", user)])
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            return agent._strip_think(text)
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                return f"[generation error] {type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return f"[generation error] {last_exc!r}"


def build_judge():
    return agent.get_chat_model("judge").with_structured_output(
        AnswerScore, method="json_schema", include_raw=True
    )


def _coerce(payload: object) -> AnswerScore | None:
    if isinstance(payload, AnswerScore):
        return payload
    if isinstance(payload, dict):
        parsed = payload.get("parsed")
        if isinstance(parsed, AnswerScore):
            return parsed
        try:
            return AnswerScore.model_validate(parsed if isinstance(parsed, dict) else payload)
        except Exception:
            raw = payload.get("raw")
            text = getattr(raw, "content", None)
            return _from_text(text) if isinstance(text, str) else None
    text = getattr(payload, "content", None)
    return _from_text(text) if isinstance(text, str) else None


def _from_text(text: str) -> AnswerScore | None:
    text = agent._strip_think(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return AnswerScore.model_validate(json.loads(match.group(0)))
    except Exception:
        return None


JUDGE_SYSTEM = """You are scoring a 2FA product chatbot answer against a gold answer from the official PDF.
Be strict about invented certifications, wrong time-steps, or sandbox policy that contradicts gold.
Do not reward fluent writing if the facts are wrong."""

JUDGE_TEMPLATE = """QUESTION:
{question}

GOLD ANSWER (from the official 2FA PDF / FAQ):
{gold}

CANDIDATE NAME: {name}

CANDIDATE ANSWER:
{answer}

PDF EXCERPTS THAT WERE RETRIEVED (may be empty for closed-book):
{context}
"""


def score_answer(judge, question: str, gold: str, name: str, answer: str, context: str) -> AnswerScore:
    prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, name=name, answer=answer, context=context[:4000])
    last_exc = None
    for _ in range(2):
        try:
            verdict = _coerce(judge.invoke([("system", JUDGE_SYSTEM), ("human", prompt)]))
            if verdict is not None:
                return verdict
        except Exception as exc:
            last_exc = exc
    return AnswerScore(
        reasoning_steps=[f"Judge failed: {last_exc!r}"],
        faithfulness=1,
        completeness=1,
        overall=1,
        passed=False,
    )


def load_pairs() -> list[dict]:
    return json.loads(QA_PAIRS_PATH.read_text())["pairs"]


def run(pairs: list[dict], top_k: int, verbose: bool) -> int:
    if not PDF_PATH.exists():
        console.print("[red]PDF missing. Run:[/red] python build_pdf.py")
        return 2

    console.print(
        Panel.fit(
            f"[bold]PDF RAG eval[/bold]\n"
            f"pdf=[cyan]{PDF_PATH.name}[/cyan]  agent=[cyan]{agent.AGENT_MODEL}[/cyan]  "
            f"judge=[cyan]{agent.JUDGE_MODEL}[/cyan]\n"
            f"questions={len(pairs)}  top_k={top_k}",
            border_style="blue",
        )
    )

    with console.status("Indexing PDF into FAISS..."):
        agent.get_vector_store()
        if agent.LLM_PROVIDER == "ollama":
            agent.release_ollama_model(agent.EMBED_MODEL)

    judge = None
    rows = []
    rag_wins = 0
    closed_wins = 0
    ties = 0

    for i, pair in enumerate(pairs, 1):
        question = pair["eval_question"]
        gold = pair["gold_answer"]
        must = pair.get("must_contain") or []

        with console.status(f"[{i}/{len(pairs)}] {pair['id']}: retrieve + generate..."):
            hits = retrieve(question, top_k)
            context = format_context(hits)
            rag_user = f"PDF EXCERPTS:\n{context}\n\nQUESTION:\n{question}"
            t0 = time.perf_counter()
            rag_answer = generate(RAG_SYSTEM, rag_user)
            if agent.LLM_PROVIDER == "ollama":
                agent.release_ollama_model(agent.AGENT_MODEL)
            closed_answer = generate(CLOSED_SYSTEM, question)
            gen_s = time.perf_counter() - t0
            if agent.LLM_PROVIDER == "ollama":
                agent.release_ollama_model(agent.AGENT_MODEL)

        if judge is None:
            judge = build_judge()

        with console.status(f"[{i}/{len(pairs)}] {pair['id']}: judging..."):
            rag_verdict = score_answer(judge, question, gold, "rag_qwen", rag_answer, context)
            closed_verdict = score_answer(judge, question, gold, "closed_book_qwen", closed_answer, "(none)")
        if agent.LLM_PROVIDER == "ollama":
            agent.release_ollama_model(agent.JUDGE_MODEL)

        rag_lex = keyword_score(rag_answer, must)
        closed_lex = keyword_score(closed_answer, must)
        rag_combo = combined_score(rag_verdict, rag_lex)
        closed_combo = combined_score(closed_verdict, closed_lex)
        if rag_combo > closed_combo:
            winner, rag_wins = "rag_qwen", rag_wins + 1
        elif closed_combo > rag_combo:
            winner, closed_wins = "closed_book_qwen", closed_wins + 1
        else:
            winner, ties = "tie", ties + 1

        row = {
            "id": pair["id"],
            "topic": pair["topic"],
            "question": question,
            "gold_answer": gold,
            "must_contain": must,
            "retrieval": [
                {"source": d.metadata.get("source"), "score": round(s, 4), "excerpt": d.page_content[:400]}
                for d, s in hits
            ],
            "candidates": {
                "rag_qwen": {
                    "answer": rag_answer,
                    "lexical": rag_lex,
                    "judge": rag_verdict.model_dump(),
                    "combined": rag_combo,
                },
                "closed_book_qwen": {
                    "answer": closed_answer,
                    "lexical": closed_lex,
                    "judge": closed_verdict.model_dump(),
                    "combined": closed_combo,
                },
            },
            "winner": winner,
            "generation_s": round(gen_s, 2),
        }
        rows.append(row)
        mark = "RAG" if winner == "rag_qwen" else ("CLOSED" if winner == "closed_book_qwen" else "TIE")
        console.print(
            f"  {pair['id']:<16} winner=[bold]{mark}[/bold]  "
            f"rag={rag_combo:.2f} (lex {rag_lex:.0%})  closed={closed_combo:.2f} (lex {closed_lex:.0%})"
        )
        if verbose:
            console.print(Panel(f"[bold]Q[/bold] {question}\n\n[bold]RAG[/bold]\n{rag_answer}\n\n"
                                f"[bold]Closed-book[/bold]\n{closed_answer}\n\n[bold]Gold[/bold]\n{gold}",
                                title=pair["id"], border_style="cyan"))

    _print_table(rows)
    REPORT_PATH.write_text(json.dumps(rows, indent=2))
    console.print(
        f"\n[bold]Scoreboard[/bold]  RAG wins {rag_wins}  |  closed-book wins {closed_wins}  |  ties {ties}"
        f"  (report: {REPORT_PATH})\n"
    )
    best = "rag_qwen" if rag_wins >= closed_wins else "closed_book_qwen"
    console.print(f"Best overall method: [bold green]{best}[/bold green]")
    return 0 if rag_wins >= closed_wins else 1


def _print_table(rows: list[dict]) -> None:
    table = Table(title="PDF RAG vs closed-book Qwen", header_style="bold magenta", show_lines=True)
    table.add_column("ID")
    table.add_column("RAG combined", justify="right")
    table.add_column("RAG judge")
    table.add_column("Closed combined", justify="right")
    table.add_column("Closed judge")
    table.add_column("Winner", justify="center")
    for r in rows:
        rag = r["candidates"]["rag_qwen"]
        clo = r["candidates"]["closed_book_qwen"]
        table.add_row(
            r["id"],
            f"{rag['combined']:.2f}",
            f"{rag['judge']['overall']}/5 f={rag['judge']['faithfulness']}",
            f"{clo['combined']:.2f}",
            f"{clo['judge']['overall']}/5 f={clo['judge']['faithfulness']}",
            r["winner"],
        )
    console.print()
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Qwen answers retrieved from the 2FA PDF.")
    parser.add_argument("--id", action="append", help="Run only this pair id (repeatable).")
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of questions.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    pairs = load_pairs()
    if args.id:
        want = set(args.id)
        pairs = [p for p in pairs if p["id"] in want]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        console.print("[red]No questions matched.[/red]")
        sys.exit(2)

    try:
        agent.ensure_credentials()
    except RuntimeError as exc:
        console.print(f"[bold red]Setup error:[/bold red] {exc}")
        sys.exit(2)

    sys.exit(run(pairs, args.top_k, args.verbose))


if __name__ == "__main__":
    main()
