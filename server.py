%%writefile /content/CHAT-BOT/server.py
"""
HTTP API for the React chat widget (Fast Single-Model Mode).
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    agent.ensure_credentials()
    agent.get_vector_store()
    agent.get_app()
    yield


app = FastAPI(title="SecureGate 2FA Sales Bot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=agent.MAX_INPUT_CHARS)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


class ToolCall(BaseModel):
    name: str
    args: dict
    output: str | None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    rag_answer: str
    offline_answer: str
    judge: dict | None = None
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
        "compare_mode": False,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        # Run single agent turn directly (Fast)
        result = agent.run_turn(req.message.strip(), thread_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model backend error: {type(exc).__name__}") from exc
    
    reply_text = result.get("reply", "")
    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        rag_answer=reply_text,
        offline_answer="",
        judge=None,
        compared=False,
        trajectory=result.get("trajectory", []),
        tool_calls=[ToolCall(**c) for c in result.get("tool_calls", [])],
        guardrail_flags=result.get("guardrail_flags", []),
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
    Path(".api_url").write_text(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
