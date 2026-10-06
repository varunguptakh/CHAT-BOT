# SecureGate 2FA Sales Bot — Full Project Documentation

This document describes the whole repository: what it is, how every piece fits together, how a chat turn runs, how evaluation works, and how to run and extend it.

Shorter entry points:

- [README.md](README.md) — quickstart and topology
- [EVAL.md](EVAL.md) — evaluation story (datasets, before/after, failures)

---

## 1. What this project is

**SecureGate 2FA** is a fictional two-factor authentication vendor. This repo is the **public landing-page assistant** that would sit on that vendor’s website.

The bot is built to do four jobs that a real technical-sales widget would do:

1. Answer product and compliance questions (TOTP / RFC 6238, HIPAA, SOC 2 Type II, GDPR) using **RAG** over a knowledge PDF.
2. Qualify inbound leads into **ENTERPRISE** (≥ 500 employees) vs **MID-MARKET**.
3. Issue **sandbox API keys**, but only to corporate email domains — never Gmail/Yahoo/Outlook.
4. Show **two answers on every chat turn** (RAG+tools vs offline Qwen) and let a **separate judge agent** score them.

It is also an **evaluation project**: there are hardcoded test questions, deterministic checks, an LLM judge, traces, and a documented Build → Evaluate → Learn → Improve loop.

This is demo / sample code. It is not a real SecureGate product and does not talk to a real identity provider.

---

## 2. High-level architecture

```
Browser (React + Vite, port 5173)
    │  POST /api/chat  { message, session_id }
    ▼
FastAPI (server.py, often port 8001 if 8000 is taken)
    │
    ▼
compare.run_compared_turn()
    ├─ 1. RAG agent     LangGraph StateGraph in agent.py
    │                     tools: RAG, CRM, sandbox issuer
    ├─ 2. Offline Qwen  same model, no tools, no documents
    └─ 3. Judge agent   qwen3:14b + Pydantic JSON schema
                          scores both, picks winner

Knowledge: data/securegate_2fa_knowledge.pdf
           → chunked (rag_pdf.py) → FAISS (in memory)
Models:    Ollama by default (qwen2.5:7b-instruct, qwen3:14b, nomic-embed-text)
           or OpenAI (gpt-4o-mini / gpt-4o)
```

The React app never imports LangChain. It only HTTP-calls the API and renders the two answers, scores, and traces.

---

## 3. Why LangChain and LangGraph

**LangChain** = reusable LLM parts.

| Part | Used for |
|---|---|
| `ChatOllama` / `ChatOpenAI` | Talk to Qwen or GPT |
| `@tool` + `bind_tools` | Let the model call RAG, CRM, sandbox issuer |
| `FAISS` + embeddings | Retrieve PDF passages |
| `Document` + text splitter | Chunk the PDF |
| `with_structured_output` | Force the judge to return scores as JSON |

**LangGraph** = the control flow for the RAG agent. A sales bot is not one prompt. It must loop (call a tool, read the result, maybe call another), branch (jailbreak → refuse), and keep session memory. That is a `StateGraph`:

```
START → input_guard → agent ⇄ tools → output_guard → END
                 ↘ refuse ───────────────────────────↗
```

The judge and the offline model are **not** a second graph. They are extra LangChain `invoke` calls after the graph finishes (see `compare.py`).

---

## 4. Repository map

```
chatbot/
├── DOCUMENTATION.md      ← this file
├── README.md             quickstart
├── EVAL.md               evaluation narrative
├── Makefile              setup, run, evals, tests
├── requirements.txt      pinned Python deps
├── .env.example          copy to .env
├── agent.py              LangGraph, tools, RAG, guardrails
├── compare.py            RAG vs offline + judge (used by the chat API)
├── server.py             FastAPI /api/chat and /api/health
├── rag_pdf.py            load and chunk the knowledge PDF
├── build_pdf.py          generate the PDF from JSON
├── run_evals.py          trajectory eval dataset + LLM judge
├── run_pdf_rag_eval.py   RAG vs closed-book scored vs gold answers
├── test_tools.py         unit tests (no LLM required for most checks)
├── data/
│   ├── qa_pairs.json                    gold Q&A for the PDF and evals
│   └── securegate_2fa_knowledge.pdf     RAG corpus
└── frontend/             React 18 + Vite landing page + chat widget
    ├── src/App.jsx
    ├── src/api.js
    ├── src/components/ChatWidget.jsx
    ├── src/components/Message.jsx
    └── src/styles.css
```

---

## 5. What happens on one chat message

Assume the user types: *“Are you HIPAA compliant? Do you sign a BAA?”*

### 5.1 FastAPI

`POST /api/chat` with `{ "message": "...", "session_id": "abc" }` calls `compare.run_compared_turn`.

### 5.2 RAG agent (LangGraph)

1. **input_guard** — regex jailbreak / length checks. If it matches, the graph **refuses without calling a model** and both UI cards show the same refusal.
2. **agent** — Qwen with tools bound. For a HIPAA question it should call `knowledge_retrieval_rag`.
3. **tools** — FAISS similarity search over the PDF (+ built-in snippets). Passages come back as a `ToolMessage`.
4. **agent** again — writes the user-facing answer **only from those passages**, with citations.
5. **output_guard** — strips `<think>` tags, blocks a leaked system-prompt canary, redacts any `sbx_test_…` string that was never issued.

State stored per `session_id`:

| Channel | Meaning |
|---|---|
| `messages` | Chat history |
| `trajectory` | Node names for this turn, e.g. `input_guard:pass → agent:call:knowledge_retrieval_rag → tool:…` |
| `guardrail_flags` | Injection / leak / fake-token detections |

### 5.3 Offline Qwen

The **same user question** is sent to Qwen **with no tools and no PDF**. This is the “what would a generic model say?” baseline. It often sounds fluent and is factually wrong (e.g. “OTP is usually 30–90 seconds”).

On small GPUs the 7B model is unloaded before the 14B judge is loaded so they do not share VRAM.

### 5.4 Judge agent

`qwen3:14b` receives:

- the question
- tool evidence from the RAG turn
- the result of the PDF grounding check (below)
- both answers
- the RAG trajectory

It must return JSON: `rag_grounded`, `rag_score`, `offline_score` (1–5), `winner` (`rag` | `offline` | `tie`), `why_winner`, `reasoning_steps`.

**PDF grounding check (`compare.check_grounding`).** The RAG answer must come from the 2FA PDF (or CRM / sandbox tool output) of the same turn. Before judging, code checks:

| Situation | Result |
|---|---|
| CRM qualifier or sandbox issuer was called | grounded (source `tools`) |
| No PDF passages were retrieved (e.g. "2+2=4") | **not grounded** |
| Passages retrieved, but fewer than 35% of the answer's key terms appear in them (`GROUNDING_MIN_OVERLAP`) | **not grounded** |
| Otherwise | grounded (source `pdf`, with page numbers) |

When the check fails, the RAG card shows a "not covered by the PDF" notice instead of the model's text, `rag_score` is forced to 1, and the offline answer wins. The judge can also mark `rag_grounded=false` when the answer states facts the passages don't contain; that caps `rag_score` at 1 too.

The API’s `reply` field is the **winner’s text** (RAG on a tie). The UI still **shows both cards**, with a `PDF p.N`, `TOOL OUTPUT` or `NOT FROM PDF` badge on the RAG card.

### 5.5 React widget

`Message.jsx` renders:

- Judge banner (winner, scores, expandable reasoning)
- RAG + tools card
- Offline Qwen card
- Tool chips (knowledge / CRM / sandbox)
- Expandable **trace** (trajectory + tool args/outputs)

A compare turn can take **30–90 seconds** locally. Jailbreak refusals return in milliseconds.

---

## 6. The three tools (in detail)

All three are Python functions decorated with LangChain `@tool` in `agent.py`. The model chooses them; the code implements them.

### 6.1 `knowledge_retrieval_rag(query)`

- Embeds `query`, searches in-memory **FAISS**, returns top-k passages (`RAG_TOP_K`, default 3).
- Corpus = PDF chunks + a small hardcoded set of the same facts (so the bot still works if the PDF is missing).
- The system prompt tells the agent: **do not invent certifications**; if the passages do not contain the answer, say so.

### 6.2 `crm_lead_qualifier(company_name, company_size, auth_system)`

- `company_size` must be a positive integer.
- `>= 500` → `tier: ENTERPRISE` (account executive + solutions engineer).
- `< 500` → `tier: MID-MARKET` (automated self-serve).
- **500 is inclusive** (a company of exactly 500 is ENTERPRISE).
- This is sales routing, not a real CRM write.

### 6.3 `sandbox_token_issuer(corporate_email)` — why it exists

A growth bot wants the prospect to **try the product**. A sandbox key is a 14-day `sbx_test_…` token, 100 req/min, no real SMS.

It is also the **dangerous** tool. If the model could invent keys or skip the domain check, the public widget becomes a credential printer.

So the check is **not an LLM decision**:

1. Validate email shape.
2. Lowercase the domain.
3. Reject public webmail (`gmail.com`, `yahoo.com`, `outlook.com`, aliases, disposable mail).
4. Reject reserved names (`example.com`, `.test`, …).
5. Only then mint `sbx_test_` + random hex into a process-local set.

If the model later types a key that was **never issued**, the output guard replaces it with `[REDACTED: unverified credential]`.

These keys are **demo artefacts**. They are not provisioned in Okta, Auth0, or any real API gateway.

---

## 7. Knowledge base and PDF RAG

Canonical facts live in `data/qa_pairs.json` (id, topic, FAQ question, **eval** paraphrase, gold answer, required keywords).

`python build_pdf.py` (or `make pdf`) writes `data/securegate_2fa_knowledge.pdf`:

- Narrative sections: product, RFC 6238, clock drift, HSM secrets, SOC 2 / HIPAA / GDPR, plans, sandbox, integrations
- FAQ Q&A copied from the JSON

`rag_pdf.py` extracts text with `pypdf`, splits with `RecursiveCharacterTextSplitter` (chunk ~800, overlap ~150), and tags metadata `source: securegate_2fa_knowledge.pdf#page-N`.

If Ollama embeddings (`nomic-embed-text`) are unavailable, RAG falls back to **hashed n-gram vectors**. That is good enough for this small corpus; production should use a real embedding model.

---

## 8. Guardrails

| Layer | When | Behavior |
|---|---|---|
| Input regex | Before the LLM | Jailbreak-like phrasing / prompt-leak / fake `</tool>` tags → canned refusal |
| Tool logic | Sandbox issuer | Domain check cannot be “talked out of” |
| System prompt | Agent | Untrusted user text; never invent keys; stay on 2FA |
| Output canary | After the LLM | If `SG-CANARY-…` appears, the system prompt leaked → refuse |
| Token redaction | After the LLM | Unknown `sbx_test_` strings are stripped |
| Prompt rule (v3) | After eval | Do not ask for headcount unless the user asked about plan/tier |

The regex guard has **false positives**. That is an accepted tradeoff on a public widget.

---

## 9. HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | `{ status, provider, agent_model, judge_model, compare_mode }` |
| `POST` | `/api/chat` | Dual-answer turn |

**Request**

```json
{ "message": "Are you HIPAA compliant?", "session_id": "optional-hex" }
```

**Response (shape)**

```json
{
  "session_id": "...",
  "reply": "winner text",
  "rag_answer": "...",
  "offline_answer": "...",
  "judge": {
    "winner": "rag",
    "rag_score": 4,
    "offline_score": 2,
    "rag_grounded": true,
    "reasoning_steps": ["..."],
    "why_winner": "...",
    "grounding": { "grounded": true, "source": "pdf", "overlap": 0.8, "pages": ["2", "3"], "note": "..." }
  },
  "compared": true,
  "trajectory": ["input_guard:pass", "agent:call:knowledge_retrieval_rag", "..."],
  "tool_calls": [{ "name": "knowledge_retrieval_rag", "args": {}, "output": "..." }],
  "guardrail_flags": [],
  "latency_ms": 45000
}
```

CORS defaults to `http://localhost:5173` and `http://127.0.0.1:5173`.

If port **8000** is already used (another app or SSH tunnel), `server.py` binds **8001+** and writes `.api_url`. Vite reads that file and proxies `/api` there.

Startup warms: Ollama/OpenAI credentials, FAISS index, compiled graph.

---

## 10. Frontend

| File | Role |
|---|---|
| `App.jsx` | Landing page (hero, compliance badges); CTAs open the widget with a prompt |
| `ChatWidget.jsx` | Session id, message list, loading (“RAG → offline → judge”), suggestions |
| `Message.jsx` | Dual cards, judge banner, sandbox key card, traces |
| `RichText.jsx` | Safe-ish subset: paragraphs, lists, `**bold**`, `` `code` `` |
| `api.js` | `fetch` to `/api/chat` and `/api/health` |
| `vite.config.js` | Dev server on `127.0.0.1:5173`, proxy `/api` |

This is a **demo widget**, not a full design system. There is no auth, no streaming SSE (the compare path is one blocking HTTP call).

---

## 11. Configuration

Copy `.env.example` to `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama |
| `AGENT_MODEL` | `qwen2.5:7b-instruct` / `gpt-4o-mini` | Fast agent |
| `JUDGE_MODEL` | `qwen3:14b` / `gpt-4o` | Structured judge |
| `EMBED_MODEL` | `nomic-embed-text` / `text-embedding-3-small` | Vectors |
| `OPENAI_API_KEY` | (prompted via `getpass` if missing) | Only for OpenAI |
| `RAG_TOP_K` | `3` | Passages per retrieval |
| `RECURSION_LIMIT` | `12` | Max graph steps |
| `MAX_INPUT_CHARS` | `4000` | Request cap |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind; next free port if busy |
| `CORS_ORIGINS` | Vite origins | Comma-separated |

Optional LangSmith: `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY`. LangChain/LangGraph spans export automatically.

---

## 12. How to run

**Need:** Python 3.10+, Node 18+, [Ollama](https://ollama.com) for the default path.

```bash
make setup     # venv, pip, ollama pull, npm install
make run       # API + UI
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173) → **Talk to Aegis**.

| Command | What |
|---|---|
| `make api` / `make ui` | Backend or frontend only |
| `make chat` | Terminal REPL (RAG graph only, no dual judge) |
| `make test` | Tool/guardrail unit tests |
| `make pdf` | Rebuild the knowledge PDF |
| `make evals` | Full trajectory suite |
| `make evals-pdf` | RAG vs closed-book vs gold |
| `make evals-adversarial` | Jailbreak slice only |
| `make lint` | `py_compile` |

Do **not** run `make evals` / `make evals-pdf` at the same time as `make run` on a 16 GB GPU.

---

## 13. Evaluation (detail)

See [EVAL.md](EVAL.md) for the narrative. Mechanical facts:

### 13.1 Trajectory suite — `run_evals.py`

Hardcoded `EVAL_CASES`:

| Slice | Examples |
|---|---|
| `normal` | HIPAA BAA, SOC 2 Type II, TOTP 30s, GDPR Frankfurt |
| `slicing` | 1,200 → ENTERPRISE, 85 → MID-MARKET, **500 → ENTERPRISE** |
| `edge` | Gmail reject, Outlook mixed-case reject, corporate email issue |
| `adversarial` | “ignore instructions”, prompt leak, fake tool JSON, “gmail is our domain”, malware |

A case **passes only if**:

1. Deterministic checks pass (expected/forbidden tools, token present/absent, canary not leaked).
2. LLM judge sets `passed=true` and `safety_violation=false`.

Writes `eval_report.json`. Non-zero exit if any case fails (CI-friendly).

### 13.2 PDF suite — `run_pdf_rag_eval.py`

Uses **paraphrased** `eval_question`s so retrieval cannot cheat by exact FAQ match.

Score = `0.6 * (judge overall / 5) + 0.4 * keyword hit-rate` against gold.

Measured example (v2):

| Question | RAG | Closed-book | Winner |
|---|---|---|---|
| TOTP time-step | 0.88 | 0.51 | RAG |
| HIPAA BAA | 0.88 | 0.39 | RAG |

### 13.3 Online eval

Every widget turn **is** an evaluation: two candidates + judge + traces in the UI.

---

## 14. Unit tests — `test_tools.py`

No GPU required for CRM/sandbox/regex tests. RAG tests need Ollama reachable (or hashed embeddings).

Covers: HIPAA/TOTP retrieval, ENTERPRISE/MID-MARKET/500 boundary, Gmail/Outlook/subdomain/malformed reject, corporate issue, injection patterns vs a benign HIPAA question, PDF chunk contents.

---

## 15. File-by-file (backend)

| File | Responsibility |
|---|---|
| `agent.py` | Models, FAISS, three tools, guards, `StateGraph`, `run_turn` |
| `compare.py` | Offline generate + judge schema + `run_compared_turn` |
| `server.py` | HTTP, CORS, port fallback, dual-answer response |
| `rag_pdf.py` | PDF → Documents |
| `build_pdf.py` | JSON → PDF |
| `run_evals.py` | Batch trajectory eval |
| `run_pdf_rag_eval.py` | Batch RAG vs closed-book |
| `test_tools.py` | unittest |

Terminal `make chat` uses `agent.run_turn` only (faster, no judge). The **website** uses `compare.run_compared_turn`.

---

## 16. Design tradeoffs

**Safety vs growth.** Refusing Gmail loses some signups. It stops the public bot from minting keys for attackers and captures a work domain (actual ICP).

**Regex vs LLM moderator.** Cheap and deterministic; some false positives.

**In-memory FAISS.** Zero ops; dies on restart; swap later without changing the graph.

**Offline 7B vs GPT-4o.** No API key; weaker tool calling; one env var to switch.

**Dual-answer in the UI.** Honest comparison for reviewers; too slow for production UX (judge should be async/shadow).

---

## 17. Known limitations

- `MemorySaver` is process-local; restart drops chats.
- Sandbox keys are random hex in a Python `set`, not a real IdP.
- Judge can mis-calibrate (e.g. `score=1` with `passed=true`). Deterministic checks still gate releases.
- Regex jailbreak filter is incomplete by nature.
- Dual chat path is 30–90s on Apple Silicon.
- PDF + snippets must be rebuilt (`make pdf`) after editing `qa_pairs.json`; restart the API to re-index.

---

## 18. How you would run this in production

1. Serve **RAG-only** to users; run offline+judge on a **5–10% shadow** sample.
2. Postgres/Redis checkpointer; hosted vector DB; real sandbox IdP.
3. LangSmith/Langfuse: export `trajectory`, `guardrail_flags`, winner, redacted tool I/O (never full keys or email local-parts).
4. CI: `make test` every PR; nightly `make evals`; fail if `edge-*` or `adversarial-*` regress.
5. Human-calibrate the judge on ~20 disagreements/week.

---

## 19. Extending the project

| Goal | Where to change |
|---|---|
| New product fact | `data/qa_pairs.json` → `make pdf` → restart API |
| New eval question | `EVAL_CASES` in `run_evals.py` and/or `qa_pairs.json` |
| New tool | `@tool` in `agent.py`, add to `TOOLS`, describe in `SYSTEM_PROMPT` |
| Different model | `.env` `AGENT_MODEL` / `JUDGE_MODEL` |
| Streaming tokens | replace blocking `/api/chat` with SSE; judge still after complete |
| Real CRM | replace `crm_lead_qualifier` body; keep the same tool signature |

---

## 20. Troubleshooting

| Symptom | Likely cause |
|---|---|
| UI “Backend offline” | API not running; Vite proxy pointing at wrong `.api_url` |
| `/api/health` is another app | Port 8000 occupied; this API is on 8001 (check `.api_url`) |
| Empty Ollama stream / hang | 7B and 14B loaded together — stop extra processes, `ollama stop` unused models |
| RAG misses HIPAA/TOTP | PDF missing — `make pdf`; or embeddings 404 — pull `nomic-embed-text` |
| Gmail still gets a key | Should be impossible if the tool ran; check **trace** for `sandbox_token_issuer` output |
| Chat takes > 2 minutes | Normal for compare path on first 14B load |

---

## 21. Glossary

| Term | Meaning here |
|---|---|
| RAG | Retrieve PDF/FAISS passages, then generate |
| Offline Qwen | Same chat model, no retrieval, no tools |
| Judge | Second model that scores the two answers |
| Trajectory | Ordered LangGraph node/tool names for a turn |
| Sandbox key | Demo `sbx_test_…` token, corporate email only |
| Canary | Hidden marker in the system prompt; if it appears in output, the prompt leaked |
| ENTERPRISE | Lead with ≥ 500 employees |

---

## 22. License

Demo / sample code for architecture and evaluation practice. Not affiliated with a real SecureGate company.
