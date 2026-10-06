import { useState } from "react";
import ChatWidget from "./components/ChatWidget.jsx";

const FEATURES = [
  {
    icon: "⏱️",
    title: "RFC 6238 TOTP",
    body: "Standards-based time-based one-time passwords with 30-second steps, drift tolerance and replay protection.",
    prompt: "How do you handle TOTP clock drift and replay attacks?",
  },
  {
    icon: "🛡️",
    title: "Audit-ready compliance",
    body: "SOC 2 Type II attested, HIPAA BAA on Enterprise, GDPR DPA with EU data residency.",
    prompt: "Can you walk me through your SOC 2 Type II and GDPR posture?",
  },
  {
    icon: "🔌",
    title: "Drop-in integrations",
    body: "Okta, Entra ID, Auth0, Ping, Keycloak, plus SAML/OIDC and SDKs for Python, Node, Go and Java.",
    prompt: "Do you integrate with Microsoft Entra ID and Keycloak?",
  },
];

export default function App() {
  const [chatOpen, setChatOpen] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState(null);

  const ask = (prompt) => {
    setChatOpen(true);
    setPendingPrompt(prompt);
  };

  return (
    <div className="page">
      <nav className="nav">
        <div className="logo">🔐 SecureGate</div>
        <div className="nav-links">
          <a href="#features">Product</a>
          <a href="#compliance">Compliance</a>
          <button type="button" className="btn btn-ghost" onClick={() => ask("Can I get a sandbox API key?")}>
            Get sandbox key
          </button>
        </div>
      </nav>

      <header className="hero">
        <span className="eyebrow">Two-factor authentication for regulated teams</span>
        <h1>
          Authentication your <span className="grad">auditors</span> will love.
        </h1>
        <p>
          Drop-in TOTP 2FA built on RFC 6238, backed by SOC 2 Type II, HIPAA and GDPR controls. Ask our assistant
          anything, from algorithms to BAAs, and get sandbox access in seconds.
        </p>
        <div className="hero-cta">
          <button type="button" className="btn btn-primary" onClick={() => ask("Are you HIPAA compliant? Do you sign a BAA?")}>
            Ask about compliance
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => setChatOpen(true)}>
            Talk to Aegis →
          </button>
        </div>
      </header>

      <section id="features" className="features">
        {FEATURES.map((f) => (
          <article key={f.title} className="card">
            <div className="card-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
            <button type="button" className="link" onClick={() => ask(f.prompt)}>
              Ask about this →
            </button>
          </article>
        ))}
      </section>

      <section id="compliance" className="badges">
        {["SOC 2 Type II", "HIPAA", "GDPR", "RFC 6238", "FIPS 140-2 HSM"].map((b) => (
          <span key={b} className="badge">{b}</span>
        ))}
      </section>

      <footer className="footer">© {new Date().getFullYear()} SecureGate, Inc. Demo landing page.</footer>

      <ChatWidget
        open={chatOpen}
        onOpenChange={setChatOpen}
        pendingPrompt={pendingPrompt}
        onPromptConsumed={() => setPendingPrompt(null)}
      />
    </div>
  );
}
