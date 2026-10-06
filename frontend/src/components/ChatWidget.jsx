import { useEffect, useRef, useState } from "react";
import { fetchHealth, sendMessage } from "../api.js";
import Message from "./Message.jsx";

const SUGGESTIONS = [
  "Are you HIPAA compliant? Do you sign a BAA?",
  "Which TOTP algorithm and time-step do you use?",
  "We're Acme Corp, 1,200 employees on Okta. Which tier fits us?",
  "Can I get a sandbox API key?",
];

const WELCOME = {
  role: "bot",
  text:
    "Hi, I'm **Aegis** 👋 Each answer is produced twice: a **RAG + tools** agent (PDF / FAISS, CRM, sandbox guardrail) " +
    "and an **offline Qwen** with no documents. A separate **judge agent** scores both and marks a winner.",
  trajectory: [],
};

const newSessionId = () => crypto.randomUUID().replaceAll("-", "");

export default function ChatWidget({ open, onOpenChange, pendingPrompt, onPromptConsumed }) {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [health, setHealth] = useState(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "down" }));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (pendingPrompt && !loading) {
      send(pendingPrompt);
      onPromptConsumed();
    }
  }, [pendingPrompt, loading]);

  async function send(text) {
    const message = text.trim();
    if (!message || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: message }]);
    setLoading(true);
    try {
      const res = await sendMessage(message, sessionId);
      setMessages((m) => [
        ...m,
        {
          role: "bot",
          text: res.reply,
          ragAnswer: res.rag_answer,
          offlineAnswer: res.offline_answer,
          judge: res.judge,
          compared: res.compared,
          trajectory: res.trajectory,
          toolCalls: res.tool_calls,
          guardrailFlags: res.guardrail_flags,
          latencyMs: res.latency_ms,
        },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: `Couldn't reach the assistant: ${err.message}` }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function reset() {
    setSessionId(newSessionId());
    setMessages([WELCOME]);
  }

  const online = health?.status === "ok";

  return (
    <>
      <div className={`widget ${open ? "widget-open" : ""}`} role="dialog" aria-label="SecureGate assistant">
        <header className="widget-header">
          <div className="widget-title">
            <div className="avatar avatar-lg">A</div>
            <div>
              <div className="widget-name">Aegis · SecureGate</div>
              <div className="widget-status">
                <span className={`dot ${online ? "dot-on" : health ? "dot-off" : ""}`} />
                {online ? `Online · RAG vs offline + judge` : health ? "Backend offline" : "Connecting..."}
              </div>
            </div>
          </div>
          <div className="widget-actions">
            <button type="button" title="New conversation" onClick={reset}>↺</button>
            <button type="button" title="Close" onClick={() => onOpenChange(false)}>✕</button>
          </div>
        </header>

        <div className="widget-body" ref={scrollRef}>
          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}
          {loading && (
            <div className="msg-bot-wrap">
              <div className="avatar">A</div>
              <div className="msg msg-bot typing-wrap">
                <div className="typing">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="typing-caption">RAG agent → offline Qwen → judge (this can take a minute)</div>
              </div>
            </div>
          )}
          {messages.length === 1 && !loading && (
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button type="button" key={s} onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        <form
          className="widget-input"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            ref={inputRef}
            value={input}
            maxLength={4000}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about compliance, TOTP, plans, or a sandbox key..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()} aria-label="Send">
            ➤
          </button>
        </form>
        <div className="widget-footer">Each turn shows RAG vs offline Qwen. Sandbox keys need a corporate email.</div>
      </div>

      {!open && (
        <button type="button" className="launcher" onClick={() => onOpenChange(true)} aria-label="Open chat">
          💬<span>Ask Aegis</span>
        </button>
      )}
    </>
  );
}
