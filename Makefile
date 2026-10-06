SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON_BIN ?= python3
VENV       := .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
LLM_PROVIDER ?= $(shell grep -E '^LLM_PROVIDER=' .env 2>/dev/null | cut -d= -f2 || echo ollama)

.PHONY: help setup setup-py setup-models setup-ui run api ui chat evals evals-adversarial evals-pdf pdf test lint build-ui clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: setup-py setup-models setup-ui ## Full setup: venv + deps, offline models, React UI
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Try: make evals  |  make run"

setup-py: ## Create virtualenv and install pinned Python deps
	@test -d $(VENV) || $(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install -r requirements.txt

setup-models: ## Pull offline Ollama models (skipped for LLM_PROVIDER=openai)
	@if [ "$(LLM_PROVIDER)" = "ollama" ]; then \
		command -v ollama >/dev/null || { echo "Install Ollama: https://ollama.com/download"; exit 1; }; \
		ollama pull qwen2.5:7b-instruct && ollama pull qwen3:14b && ollama pull nomic-embed-text; \
	else echo "LLM_PROVIDER=$(LLM_PROVIDER): skipping Ollama model pulls"; fi

setup-ui: ## Install React frontend deps
	cd frontend && npm install

run: ## Start API (free port from :8000) and React UI (:5173) together
	@trap 'kill 0' EXIT INT TERM; \
	rm -f .api_url; \
	$(PY) server.py & \
	for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do test -f .api_url && break; sleep 0.5; done; \
	(cd frontend && npm run dev) & \
	wait

api: ## Start only the FastAPI backend on :8000
	$(PY) server.py

ui: ## Start only the React dev server on :5173
	cd frontend && npm run dev

chat: ## Interactive terminal chat with the agent
	$(PY) agent.py

evals: ## Run the full evaluation suite with LLM judge
	$(PY) run_evals.py

evals-adversarial: ## Run only adversarial security cases (verbose)
	$(PY) run_evals.py --category adversarial --verbose

pdf: ## Generate data/securegate_2fa_knowledge.pdf from the Q&A JSON
	$(PY) build_pdf.py

evals-pdf: pdf ## RAG from the 2FA PDF, answer with Qwen, score vs gold, pick the winner
	$(PY) run_pdf_rag_eval.py

test: ## Deterministic unit tests for tools + guardrails (no LLM)
	$(PY) test_tools.py

lint: ## Byte-compile Python sources as a quick syntax check
	$(PY) -m py_compile agent.py compare.py run_evals.py run_pdf_rag_eval.py build_pdf.py rag_pdf.py server.py

build-ui: ## Production build of the React UI into frontend/dist
	cd frontend && npm run build

clean: ## Remove caches, reports and build output
	rm -rf __pycache__ eval_report.json pdf_rag_report.json frontend/dist
