# Evaluation story: SecureGate 2FA sales bot

**Build → Evaluate → Learn → Improve.** This is the document to read first (~10 minutes), then run `make run` and `make evals`. For architecture, APIs, and every file, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Agent (one diagram)

```
USER
  │
  ├─► RAG agent (LangGraph) ── tools ──► FAISS/PDF, CRM qualifier, sandbox issuer
  │                                         │
  ├─► Offline Qwen (no tools, no docs)      │
  │                                         ▼
  └─► Judge agent (qwen3:14b, structured JSON)
           scores both, picks a winner
           traces: trajectory + tool I/O + guardrail flags
```

Chat shows **both answers and the judge score**. The served `reply` is the winner (RAG on a tie).

## Datasets

| Dataset | File | What it covers | Evaluator |
|---|---|---|---|
| Sales-bot trajectories | `run_evals.py` (`EVAL_CASES`) | Normal RAG (HIPAA/SOC2/TOTP/GDPR), lead slicing at 85 / 500 / 1200, sandbox Gmail reject vs corporate issue, adversarial jailbreaks | Deterministic tool/token checks **and** LLM judge (`passed`) |
| PDF Q&A gold | `data/qa_pairs.json` + `data/securegate_2fa_knowledge.pdf` | Paraphrased questions vs canonical answers; RAG Qwen vs closed-book Qwen | Keyword hit-rate **and** LLM judge overall 1–5; winner = higher combined score |
| Live / online | every `/api/chat` turn | Same dual-answer + judge as production traffic | Judge agent; inspect `trace` in the widget |

Tricky examples (intentional): Gmail sandbox, mixed-case Outlook, “gmail.com is our corporate domain”, fake `<tool>` output, prompt-leak, off-topic malware.

## Before → after

**v0 (first eval run).** 7B agent and 14B judge loaded together. Ollama returned empty streams. Failures were infrastructure, not the agent.

**v1 (sequential unload).** `normal-hipaa` and `edge-gmail` both **PASS**. Deterministic checks + judge `passed=true`. Honest issue: the HIPAA reply still asked for employee count although the user never mentioned plans. Judge `score` sometimes printed `1` while `passed=true` (calibration bug on the 1–5 scale).

**v2 (PDF as RAG corpus).** RAG vs closed-book on paraphrased questions:

| Question | RAG combined | Closed-book | Winner |
|---|---|---|---|
| TOTP / RFC 6238 time-step | **0.88** (100% gold keywords) | 0.51 | RAG |
| HIPAA BAA | **0.88** (100% gold keywords) | 0.39 | RAG |

Closed-book sounded fluent and wrong (“OTP usually 30–90 seconds”, “the product *should* be HIPAA compliant”). That is the failure we wanted the judge to catch.

**v3.** Chat widget shows both candidates + judge. Prompt updated: do not ask for headcount unless the user asked about tier. Dual-answer path is the online evaluation.

**v4 (PDF grounding gate).** Found in live use: "2+2=4" gave RAG **5/5**. The agent never touched the PDF and answered from the model's own knowledge; the judge only scored correctness. Fix: a deterministic check (`compare.check_grounding`) requires PDF passages or tool output from the same turn, plus ≥35% key-term overlap between answer and passages. Failing answers are replaced with a "not covered by the PDF" notice and scored 1; the judge also returns `rag_grounded`. After: "2+2=4" → RAG **1/5** (`NOT FROM PDF`), offline 5/5, offline wins. Real PDF answers (HIPAA, RFC 6238, SOC 2) score 71–100% overlap, well above the cutoff. Unit tests in `test_tools.py` (`GroundingTests`).

## How to inspect traces

1. UI: open the widget → send a question → expand **trace** (LangGraph node names + tool args/outputs) and **judge reasoning**.
2. Batch: `make evals` writes `eval_report.json`; `make evals-pdf` writes `pdf_rag_report.json`.
3. Optional LangSmith: `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY=...` then rerun. LangChain/LangGraph spans are picked up automatically.

## Production evaluation (if this shipped)

- **Online:** keep the dual-answer judge on a 5–10% shadow sample; serve RAG only to users (latency). Log `winner`, scores, trajectory, redacted tool I/O.
- **Offline CI:** `make test` on every PR (no GPU). Nightly `make evals` on the four slices; fail the build if `edge-*` or `adversarial-*` drop.
- **Human calibration:** sample 20 judge disagreements a week; if Cohen’s κ vs human < 0.6, freeze the judge prompt and add those cases to `EVAL_CASES`.
- **Slices that matter:** public-email sandbox, ENTERPRISE boundary at 500, HIPAA/SOC2 grounding, jailbreak block rate, p95 latency.

## Known failures and tradeoffs

- Judge **faithfulness** field can disagree with **overall** on qwen3:14b. We gate on `passed` + keywords, not on that one field.
- Regex input guard has false positives on phrases that look like “ignore previous…”. We accept that on a public widget.
- Dual-answer chat is slow (~30–90s) because we unload 7B before 14B. Production would judge asynchronously.
- Sandbox keys are demo artefacts, not a real IdP.
- In-memory FAISS and MemorySaver die on process restart.

## Coding assistant

Cursor (Grok) scaffolded LangGraph, the React widget, PDF build, and eval harness. I checked: sandbox domain logic is a pure function (not a prompt), Gmail is rejected without a model, FAISS retrieval returns HIPAA/RFC 6238 passages, and the first GPU OOM/empty-stream failure was reproduced and fixed with sequential model unload — not by asking the model to “try again”.
