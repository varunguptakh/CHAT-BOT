import { useState } from "react";
import RichText from "./RichText.jsx";

const TOOL_LABELS = {
  knowledge_retrieval_rag: { icon: "📚", label: "Knowledge base" },
  crm_lead_qualifier: { icon: "🏢", label: "Lead qualifier" },
  sandbox_token_issuer: { icon: "🔑", label: "Sandbox issuer" },
};

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function CredentialCard({ data }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(data.sandbox_api_key);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="credential-card">
      <div className="credential-head">
        <span>Sandbox API key</span>
        <span className="pill pill-green">ISSUED</span>
      </div>
      <div className="credential-key">
        <code>{data.sandbox_api_key}</code>
        <button type="button" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
      </div>
      <div className="credential-meta">
        {data.email} · {data.rate_limit} · expires {new Date(data.expires_at).toLocaleDateString()}
      </div>
    </div>
  );
}

function ScorePill({ score, max = 5 }) {
  if (score == null) return null;
  const pct = Math.round((score / max) * 100);
  return (
    <span className={`score-pill ${pct >= 80 ? "score-good" : pct >= 50 ? "score-ok" : "score-bad"}`}>
      {score}/{max}
    </span>
  );
}

function AnswerCard({ label, hint, text, score, winner }) {
  return (
    <div className={`answer-card ${winner ? "answer-winner" : ""}`}>
      <div className="answer-card-head">
        <div>
          <div className="answer-label">{label}</div>
          <div className="answer-hint">{hint}</div>
        </div>
        <div className="answer-card-meta">
          <ScorePill score={score} />
          {winner && <span className="pill pill-green">JUDGE PICK</span>}
        </div>
      </div>
      <RichText text={text} />
    </div>
  );
}

export default function Message({ msg }) {
  const [showTrace, setShowTrace] = useState(false);
  const [showJudge, setShowJudge] = useState(false);

  if (msg.role === "user") {
    return <div className="msg msg-user">{msg.text}</div>;
  }
  if (msg.role === "error") {
    return <div className="msg msg-error">⚠️ {msg.text}</div>;
  }

  const toolCalls = msg.toolCalls || [];
  const flags = msg.guardrailFlags || [];
  const judge = msg.judge;
  const dual = Boolean(msg.ragAnswer && msg.offlineAnswer && msg.compared);
  const lead = toolCalls
    .filter((c) => c.name === "crm_lead_qualifier")
    .map((c) => parseJson(c.output))
    .find((d) => d?.tier);
  const credential = toolCalls
    .filter((c) => c.name === "sandbox_token_issuer")
    .map((c) => parseJson(c.output))
    .find((d) => d?.status === "ISSUED");
  const rejected = toolCalls.some(
    (c) => c.name === "sandbox_token_issuer" && String(c.output).startsWith("ERROR"),
  );

  return (
    <div className="msg-bot-wrap">
      <div className="avatar">A</div>
      <div className="msg msg-bot">
        {dual ? (
          <>
            {judge && (
              <div className="judge-banner">
                <div>
                  <strong>Judge agent</strong> picked{" "}
                  <em>{judge.winner === "offline" ? "offline Qwen" : judge.winner === "tie" ? "a tie" : "RAG + tools"}</em>
                </div>
                <div className="judge-scores">
                  RAG <ScorePill score={judge.rag_score} />
                  <span className="judge-vs">vs</span>
                  Offline <ScorePill score={judge.offline_score} />
                </div>
                <p className="judge-why">{judge.why_winner}</p>
                {judge.reasoning_steps?.length > 0 && (
                  <button type="button" className="trace-toggle" onClick={() => setShowJudge((s) => !s)}>
                    {showJudge ? "hide reasoning" : "judge reasoning"}
                  </button>
                )}
                {showJudge && (
                  <ol className="judge-steps">
                    {judge.reasoning_steps.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ol>
                )}
              </div>
            )}
            <div className="dual-grid">
              <AnswerCard
                label="RAG + tools"
                hint="FAISS / PDF retrieval, CRM, sandbox guardrail"
                text={msg.ragAnswer}
                score={judge?.rag_score}
                winner={judge?.winner === "rag" || judge?.winner === "tie"}
              />
              <AnswerCard
                label="Offline Qwen"
                hint="Same model, no tools, no documents"
                text={msg.offlineAnswer}
                score={judge?.offline_score}
                winner={judge?.winner === "offline"}
              />
            </div>
          </>
        ) : (
          <RichText text={msg.text} />
        )}
        {credential && <CredentialCard data={credential} />}

        {msg.trajectory?.length > 0 && (
          <div className="chips">
            {toolCalls.map((c, i) => {
              const meta = TOOL_LABELS[c.name] || { icon: "🛠️", label: c.name };
              return (
                <span key={i} className="chip">
                  {meta.icon} {meta.label}
                </span>
              );
            })}
            {lead && <span className={`chip ${lead.tier === "ENTERPRISE" ? "chip-purple" : "chip-blue"}`}>{lead.tier}</span>}
            {rejected && <span className="chip chip-red">Domain rejected</span>}
            {flags.length > 0 && <span className="chip chip-red">🛡️ Guardrail</span>}
            {dual && <span className="chip">⚖️ Judge</span>}
            <button type="button" className="trace-toggle" onClick={() => setShowTrace((s) => !s)}>
              {showTrace ? "hide trace" : "trace"}
            </button>
          </div>
        )}

        {showTrace && (
          <div className="trace">
            <div className="trace-path">{msg.trajectory.join(" → ")}</div>
            {toolCalls.map((c, i) => (
              <details key={i}>
                <summary>
                  {c.name}({Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
                </summary>
                <pre>{c.output}</pre>
              </details>
            ))}
            {flags.length > 0 && <div className="trace-flags">flags: {flags.join(", ")}</div>}
            {msg.latencyMs != null && <div className="trace-latency">{(msg.latencyMs / 1000).toFixed(1)}s</div>}
          </div>
        )}
      </div>
    </div>
  );
}
