# SecureGate 2FA — Agentic Growth & Technical Sales Bot

A landing-page sales assistant that answers technical and compliance questions, qualifies enterprise leads, and issues sandbox credentials — with a **safety-first** credentialing guardrail that refuses public webmail.

The default runtime is **fully offline** (Ollama + `qwen2.5:7b-instruct` + `qwen3:14b` + `nomic-embed-text`). Flip `LLM_PROVIDER=openai` to use `gpt-4o-mini` / `gpt-4o` if you have an API key.

**Documentation**

- **[DOCUMENTATION.md](DOCUMENTATION.md)** — full project guide (architecture, every file, chat lifecycle, API, evals, how to extend)
- **[EVAL.md](EVAL.md)** — evaluation story: datasets, traces, before/after, failures, production plan
- **[README.md](README.md)** — this page: quickstart and topology

---

## System topology

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  React landing page  (:5173)                                            │
  │    ChatWidget  ──POST /api/chat──►  FastAPI  (:8000)                    │
  └──────────────────────────────────────────────┬──────────────────────────┘
                                                 │
                    ┌────────────────────────────▼─────────────────────────┐
                    │              LangGraph StateGraph                    │
                    │                                                      │
                    │         START                                        │    
                    │           │                                          │
                    │           ▼                                          │
                    │     input_guard                                      │
                    │      /         \                                     │
                    │  blocked      clean                                  │
                    │     │            │                                   │
                    │     ▼            ▼                                   │
                    │  refuse        agent ◄───────────────┐               │
                    │     │         /     \                │               │
                    │     │   tool_calls   final answer    │               │
                    │     │       │            │           │               │
                    │     │       ▼            ▼           │               │
                    │     │     tools ──►  output_guard    │               │
                    │     │                    │           │               │
                    │     └────────► END ◄─────┘           │               │
                    │                                      │               │
                    │  Tools bound to the agent            │               │
                    │    • knowledge_retrieval_rag  (FAISS │ RFC 6238,     │
                    │                                  SOC2, HIPAA, GDPR)  │
                    │    • crm_lead_qualifier       (ENTERPRISE ≥ 500)     │
                    │    • sandbox_token_issuer     (corporate domain only)│
                    │  Also: offline Qwen (no tools) scored by a judge     │
                    └──────────────────────────────────────────────────────┘
```

State is a `TypedDict` with three channels:

| Channel | Reducer | Purpose |
|---|---|---|
| `messages` | `add_messages` | Conversation history for the LLM |
| `trajectory` | `operator.add` | Ordered node / tool names for this session (eval traces) |
| `guardrail_flags` | `operator.add` | Injection / leak / fabricated-token detections |

---

## Quickstart

Requires Python 3.10+, Node 18+, and (for the default offline path) [Ollama](https://ollama.com).

```bash
git clone <this-repo> && cd chatbot
make setup          # venv + pinned deps + Ollama models + npm install
make run            # FastAPI :8000  +  React :5173
```

Open http://localhost:5173 and click **Talk to Aegis**.

| Command | What it does |
|---|---|
| `make run` | API + React UI together |
| `make api` / `make ui` | Run each side independently |
| `make chat` | Terminal REPL against the same graph |
| `make evals` | Full evaluation suite + LLM judge |
| `make evals-adversarial` | Only the prompt-injection cases |
| `make test` | Deterministic unit tests (no LLM) |
| `make pdf` | Build `data/securegate_2fa_knowledge.pdf` from the Q&A JSON |
| `make evals-pdf` | RAG from that PDF, answer with Qwen, score vs gold, pick RAG vs closed-book |

Copy `.env.example` to `.env` to change models. If `LLM_PROVIDER=openai` and `OPENAI_API_KEY` is unset, the process **prompts securely via `getpass.getpass`** — no `export` required.

---

## The three tools

### `knowledge_retrieval_rag(query)`
In-memory FAISS over `data/securegate_2fa_knowledge.pdf` only. Rebuild the PDF with `make pdf` after editing `data/qa_pairs.json`. The PDF is the canonical 2FA / compliance corpus: RFC 6238 TOTP, clock drift, HSM secrets, SOC 2 Type II, HIPAA BAA, GDPR, plans, sandbox policy, integrations. Retrieval is the only source the agent is allowed to treat as fact.

### `crm_lead_qualifier(company_name, company_size, auth_system)`
Integer headcount is the only sizing signal. `company_size >= 500` → **ENTERPRISE** (AE + solutions engineer). Otherwise **MID-MARKET** (automated self-serve). The 500-employee boundary is inclusive.

### `sandbox_token_issuer(corporate_email)`
A **deterministic, non-LLM** guardrail. The domain is extracted, lowercased, and matched against a frozen set of public providers (`gmail.com`, `yahoo.com`, `outlook.com`, plus common aliases and disposable mail). Public, reserved (`example.com`, `.test`, …) and malformed addresses return `ERROR: SANDBOX_REQUEST_REJECTED`. Only then is a `sbx_test_…` key minted. The output guard additionally redacts any `sbx_test_` string that was never issued by this process — so the model cannot invent credentials.

---

## Evaluation suite

`run_evals.py` ships with a hardcoded dataset covering four buckets:

| Category | What it proves |
|---|---|
| `normal` | RAG grounding on HIPAA / SOC 2 Type II / RFC 6238 / GDPR |
| `slicing` | Headcount parsing → ENTERPRISE vs MID-MARKET, including the 500-employee boundary |
| `edge` | Gmail/Outlook rejection; successful issue to a corporate domain |
| `adversarial` | Prompt-injection, system-prompt leak, fake tool-output, social-engineered "gmail is our domain", off-topic malware |

Each case is scored **twice**. It only passes if both layers pass:

1. **Deterministic checks** — expected/forbidden tools, token presence, canary leak, tool-output substrings.
2. **LLM judge** — `with_structured_output(JudgeVerdict)` producing `reasoning_steps[]`, `safety_violation`, `score`, and a boolean `passed`.

A Rich terminal table is printed; a machine-readable `eval_report.json` is written next to the script. Non-zero exit code on any failure, so this can gate CI.

```bash
make evals
python run_evals.py --case edge-gmail --verbose
python run_evals.py --category adversarial
```

---

## PDF RAG + scored Qwen answers

The knowledge PDF is generated from `data/qa_pairs.json` (canonical FAQ + gold answers). Evaluation questions are paraphrases of those FAQs so retrieval has to work, not just string-match.

```bash
make pdf            # writes data/securegate_2fa_knowledge.pdf
make evals-pdf      # retrieve → Qwen RAG vs closed-book → judge score → winner
python run_pdf_rag_eval.py --id hipaa --id totp-rfc6238 --verbose
```

Each question gets two candidate answers. Combined score is `0.6 * (judge overall / 5) + 0.4 * keyword hit-rate` against the gold answer. The higher score wins that question; the method that wins more questions is reported as best. Results are written to `pdf_rag_report.json`.

Do not run `make evals-pdf` at the same time as `make run` on a 16 GB GPU (the judge unloads the 7B agent first).

---

## Trade-offs

**Safety-first credentialing vs. friction-free growth.** A growth bot's conversion job is to get a prospect into a sandbox *now*. Public-webmail signup would maximize that. We deliberately do the opposite: the domain check is a pure function, not a prompt instruction, and the output guard will redact a fabricated key even if the model "plays along" with an injection. The cost is real — a founder on Gmail has to come back with a work address. The benefit is that a public landing-page bot cannot become a free credential printer for attackers, and sales still captures a verified corporate identity (which is the actual ICP signal).

**Regex input guard vs. an LLM moderator.** Cheap, deterministic, and it short-circuits the model on the most common jailbreaks. It will also refuse some legitimate phrasing that happens to look like an injection. We accept the false-positive rate on a public widget; a production deployment should log those flags and periodically tune the patterns against real traffic.

**In-memory FAISS vs. a hosted vector store.** Instant to stand up, no network, no PII leaving the box, trivial to snapshot in evals. It does not survive process restart, does not shard, and is a poor fit once the corpus is no longer a handful of compliance pages. Swap `get_vector_store()` for PGVector / Pinecone without touching the graph.

**Offline small models vs. frontier APIs.** Defaulting to Ollama means the demo, the evals, and the landing page all run with no cloud key. Tool-calling quality on 7B is good enough for this schema but will miss more edge phrasings than `gpt-4o-mini`. The provider is a single env var. If `nomic-embed-text` is not pulled, RAG falls back to deterministic hashed n-gram embeddings — good enough for this nine-document corpus, not a substitute for a real embedding model in production.

---

## Known limits

- Conversation state lives in a process-local `MemorySaver`. Restarting the API drops sessions; scale-out needs a Postgres / Redis checkpointer.
- The knowledge corpus is compiled into `agent.py`. There is no document-admin UI and no freshness pipeline.
- Injection detection is pattern-based. Novel jailbreaks that do not match `INJECTION_PATTERNS` still hit the model (and then the output canary / token redaction).
- Sandbox keys are random hex in a process-local set. They are **not** provisioned against a real IdP. Treat them as demo artefacts.
- The judge is another LLM. It can be wrong; that is why the deterministic layer is the one that can fail a release on its own.
- `qwen3:14b` as judge is slower than `gpt-4o`. Budget ~30–90s per case on Apple Silicon, longer on first load. Evals **unload the agent before loading the judge** so both 7B and 14B models are not resident in VRAM at once — do not run `make run` and `make evals` at the same time on a 16 GB GPU.

---

## Production tracing plan

1. **LangSmith (or OpenTelemetry + Langfuse).** Wrap `get_app()` with `langsmith.traceable` / set `LANGCHAIN_TRACING_V2=true`. Every node already writes `trajectory` and `guardrail_flags`; export those as span attributes (`sg.trajectory`, `sg.guardrail_flags`, `sg.tool.name`).
2. **Redact before export.** Strip `sbx_test_*` and email local-parts from traces. Keep domain + lead tier; those are the useful funnels.
3. **Eval as CI.** Run `make test` on every PR (no GPU). Nightly, run `make evals` against pinned Ollama tags (or OpenAI) and fail the build if `passed` drops on `edge-*` or `adversarial-*`. Store `eval_report.json` as an artefact and chart pass-rate over time.
4. **Online monitoring.** Log `input_guard:blocked` rate, `SANDBOX_REQUEST_REJECTED` rate, ENTERPRISE-qualification count, and p95 `/api/chat` latency. Alert on a spike in fabricated-token redactions — that is the model trying to bypass the issuer.
5. **Human review sample.** 1% of production threads (PII-redacted) into a queue labelled by trajectory prefix. Use disagreements with the judge to grow the hardcoded eval set.

---

## Project layout

```
DOCUMENTATION.md    Full project documentation
EVAL.md             Evaluation narrative (datasets, before/after)
agent.py            LangGraph topology, tools, RAG, guardrails
compare.py          RAG vs offline Qwen + judge (used by /api/chat)
server.py           FastAPI /api/chat + /api/health
run_evals.py        Dataset + deterministic checks + structured LLM judge
run_pdf_rag_eval.py RAG vs closed-book scored against gold answers
build_pdf.py / rag_pdf.py   Knowledge PDF generate + chunk
test_tools.py       Fast unit tests for the three tools (no LLM)
data/               qa_pairs.json + securegate_2fa_knowledge.pdf
requirements.txt    Pinned Python deps
Makefile            setup / run / evals / test
frontend/           React 18 + Vite landing page + chat widget
```

## License

Demo / sample code. Not an official SecureGate product.
