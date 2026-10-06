"""
HTTP API for the React chat widget.

    POST /api/chat    {"message": "...", "session_id": "optional"} -> reply + per-turn trace
    GET  /api/health  provider / model metadata

Run: python server.py   (or `make api`)
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent
import compare


@asynccontextmanager
async def lifespan(_: FastAPI):
    agent.ensure_credentials()
    agent.get_vector_store()
    agent.get_app()
    yield


app = FastAPI(title="SecureGate 2FA Sales Bot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=agent.MAX_INPUT_CHARS)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


class ToolCall(BaseModel):
    name: str
    args: dict
    output: str | None


class GroundingView(BaseModel):
    grounded: bool
    source: str
    overlap: float | None = None
    pages: list[str] = []
    note: str = ""


class JudgeView(BaseModel):
    winner: str
    rag_score: int
    offline_score: int
    rag_grounded: bool = True
    reasoning_steps: list[str]
    why_winner: str
    grounding: GroundingView | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    rag_answer: str
    offline_answer: str
    judge: JudgeView | None = None
    compared: bool = False
    trajectory: list[str]
    tool_calls: list[ToolCall]
    guardrail_flags: list[str]
    latency_ms: int


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": agent.LLM_PROVIDER,
        "agent_model": agent.AGENT_MODEL,
        "judge_model": agent.JUDGE_MODEL,
        "compare_mode": True,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        result = compare.run_compared_turn(req.message.strip(), thread_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model backend error: {type(exc).__name__}") from exc
    judge = result.get("judge")
    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        rag_answer=result.get("rag_answer") or result["reply"],
        offline_answer=result.get("offline_answer") or result["reply"],
        judge=JudgeView(**judge) if judge else None,
        compared=bool(result.get("compared")),
        trajectory=result["trajectory"],
        tool_calls=[ToolCall(**c) for c in result["tool_calls"]],
        guardrail_flags=result["guardrail_flags"],
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    preferred = int(os.getenv("API_PORT", "8000"))
    port = preferred
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                port = candidate
                break
            except OSError:
                continue
    else:
        raise SystemExit(f"No free TCP port in {preferred}–{preferred + 19} on {host}")
    if port != preferred:
        print(f"[warn] {host}:{preferred} is in use; serving on {host}:{port}")
    Path(".api_url").write_text(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
