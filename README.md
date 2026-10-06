# SecureGate 2FA — Agentic Growth & Technical Sales Bot

A landing-page sales assistant that answers technical and compliance questions, qualifies enterprise leads, and issues sandbox credentials — with a **safety-first** credentialing guardrail that refuses public webmail.

Each chat turn shows **two answers**:

1. **RAG + tools** — LangGraph agent searches [`data/securegate_2fa_knowledge.pdf`](data/securegate_2fa_knowledge.pdf), then Qwen 7B writes from those passages (and can call CRM / sandbox tools).
2. **Offline Qwen 7B** — same model, no PDF, no tools.
3. A **judge (Qwen 14B)** scores both and marks a winner.

Default runtime is **fully offline** via [Ollama](https://ollama.com). You can switch to OpenAI with one env var.

**Docs**

| File | What it is |
|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | Full architecture, API, files, troubleshooting |
| [EVAL.md](EVAL.md) | Datasets, traces, before/after scores, known failures |

---

## Installation (GitHub)

### Prerequisites

Install these first:

| Tool | Version | Why |
|---|---|---|
| [Git](https://git-scm.com/downloads) | any recent | clone this repo |
| [Python](https://www.python.org/downloads/) | 3.10+ (3.12/3.14 work) | API, LangGraph, evals |
| [Node.js](https://nodejs.org/) | 18+ (includes `npm`) | React chat widget |
| [Ollama](https://ollama.com/download) | latest | local Qwen 7B / 14B (skip if you only use OpenAI) |
| `make` | macOS/Linux built-in; Windows: Git Bash or WSL | `make setup` / `make run` |

On macOS, `make` comes with Xcode Command Line Tools (`xcode-select --install`).

### 1. Clone the repository

```bash
git clone https://github.com/varunguptakh/CHAT-BOT.git
cd CHAT-BOT
```

### 2. Create your env file

```bash
cp .env.example .env
```

Leave `LLM_PROVIDER=ollama` for the local path. You do **not** need an OpenAI key.

### 3. Install Python deps, models, and the UI

From the repo root:

```bash
make setup
```

That command:

1. Creates `.venv` and installs [requirements.txt](requirements.txt)
2. Pulls Ollama models: `qwen2.5:7b-instruct` (agent), `qwen3:14b` (judge), `nomic-embed-text` (PDF embeddings)
3. Runs `npm install` in `frontend/`

Disk for models is roughly **7B ~5 GB + 14B ~9 GB + embed ~0.3 GB**. First pull can take several minutes.

**If you do not have `make`**, run the same steps by hand:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull qwen3:14b
ollama pull nomic-embed-text
cd frontend && npm install && cd ..
cp -n .env.example .env
```

### 4. Start Ollama

Keep this running in another terminal (or as a background service):

```bash
ollama serve
```

Confirm models:

```bash
ollama list
```

### 5. Run the app

```bash
make run
```

- React UI: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- API: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health) (if 8000 is busy, the API binds **8001** and Vite still proxies `/api`)

Open the UI → **Talk to Aegis**. A compare turn (RAG + offline + judge) often takes **30–90 seconds** on a laptop GPU.

Without `make`:

```bash
source .venv/bin/activate
python server.py &
cd frontend && npm run dev
```

### 6. Optional: OpenAI instead of Ollama

In `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

If the key is omitted, the process prompts with a hidden `getpass` input. Then `make run` again. Agent = `gpt-4o-mini`, judge = `gpt-4o`.

---

## Verify the install

```bash
make test              # tool + guardrail unit tests (no 14B judge)
make pdf               # rebuild the knowledge PDF from data/qa_pairs.json
```

Ask in the widget:

- `Are you HIPAA compliant? Do you sign a BAA?`
- `Which TOTP algorithm and time-step do you use?`
- `Can you send sandbox credentials to me@gmail.com?` (must be rejected)

---

## Useful commands

| Command | What it does |
|---|---|
| `make run` | API + React UI together |
| `make api` / `make ui` | Backend or frontend only |
| `make chat` | Terminal REPL (RAG graph only, no dual judge) |
| `make test` | Deterministic unit tests |
| `make pdf` | Build `data/securegate_2fa_knowledge.pdf` |
| `make evals` | Full evaluation suite + LLM judge |
| `make evals-pdf` | RAG vs closed-book scored against gold answers |
| `make evals-adversarial` | Jailbreak cases only |

Do **not** run `make evals` at the same time as `make run` on a 16 GB GPU (7B and 14B will fight for VRAM).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Start `ollama serve`; check `OLLAMA_BASE_URL` in `.env` |
| Missing models | `ollama pull qwen2.5:7b-instruct && ollama pull qwen3:14b && ollama pull nomic-embed-text` |
| UI says backend offline | API is not up; wait for `Uvicorn running` or check `.api_url` |
| Port 8000 in use | Normal — API moves to 8001 automatically |
| Empty / hung replies | Unload extra models: `ollama stop qwen3:14b`; don’t run evals + chat together |
| `make: command not found` | Use the manual commands in step 3, or install make / use WSL on Windows |

More detail: [DOCUMENTATION.md](DOCUMENTATION.md).

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
- The knowledge corpus is `data/securegate_2fa_knowledge.pdf` (built from `data/qa_pairs.json`). There is no document-admin UI.
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
## Image Reference
<img width="688" height="1022" alt="image" src="https://github.com/user-attachments/assets/add3b378-b683-4e8a-8f73-76ff0376d37e" />


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
