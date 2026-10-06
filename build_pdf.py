"""Build data/securegate_2fa_knowledge.pdf from data/qa_pairs.json.

    python build_pdf.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAIRS_PATH = DATA / "qa_pairs.json"
PDF_PATH = DATA / "securegate_2fa_knowledge.pdf"

NARRATIVE = [
    (
        "1. Product overview",
        "SecureGate 2FA is a two-factor authentication platform for regulated teams. "
        "After a password or SSO assertion succeeds, the user must prove possession of a "
        "registered authenticator. The default factor is a time-based one-time password "
        "(TOTP). Enterprise customers may also register WebAuthn / FIDO2 security keys "
        "and passkeys. The product is designed so auditors can map controls to RFC 6238, "
        "SOC 2 Type II, HIPAA, and GDPR without a custom questionnaire for every deal.",
    ),
    (
        "2. TOTP algorithm (RFC 6238)",
        "One-time passwords implement TOTP as defined in RFC 6238, which extends HOTP "
        "from RFC 4226. The moving factor is time, not a counter. Codes are derived as "
        "HMAC(K, T) where T = floor((UnixTime - T0) / X), T0 = 0, and the time-step X is "
        "30 seconds. Default display is 6 digits; 8 digits are configurable. HMAC-SHA1 is "
        "the default so Google Authenticator, Authy, and 1Password work out of the box. "
        "HMAC-SHA256 and HMAC-SHA512 are supported on the Enterprise plan.",
    ),
    (
        "3. Clock drift, replay, and lockout",
        "Phones drift. The verifier therefore accepts a validation window of +/-1 "
        "time-step: the previous, current, and next 30-second step. Each accepted code is "
        "recorded and cannot be replayed within its validity window. Repeated failures "
        "trigger exponential back-off. After 5 consecutive invalid codes the authenticator "
        "is locked until an administrator or a recovery flow resets it.",
    ),
    (
        "4. Secret generation and storage",
        "Shared secrets are generated with a CSPRNG (160-bit minimum) and provisioned to "
        "the authenticator as an otpauth:// URI encoded in a QR code. After enrollment the "
        "secret is never returned by any API. Secrets are encrypted at rest with AES-256-GCM "
        "using per-tenant keys stored in a FIPS 140-2 Level 3 HSM. All traffic uses TLS 1.2+.",
    ),
    (
        "5. Compliance: SOC 2 Type II, HIPAA, GDPR",
        "SOC 2 Type II covers Security, Availability, and Confidentiality. The audit is "
        "annual, run by an independent AICPA-accredited CPA firm over a 12-month observation "
        "period. The report is available under NDA via the Trust Center. HIPAA: SecureGate "
        "supports covered entities and business associates and signs a Business Associate "
        "Agreement (BAA) on the Enterprise plan. Logs that may contain ePHI are encrypted "
        "at rest and in transit, retained per customer policy, and access is role-restricted "
        "with audit trails. GDPR: SecureGate is a data processor, offers a DPA with EU "
        "Standard Contractual Clauses, and can pin residency to Frankfurt (eu-central). "
        "Access and erasure requests are fulfilled within 30 days.",
    ),
    (
        "6. Plans and sandbox credentials",
        "MID-MARKET (under 500 employees) is self-serve: automated onboarding, SAML SSO, "
        "standard support, 99.9% uptime SLA. ENTERPRISE (500 or more employees) adds a "
        "dedicated account executive and solutions engineer, HIPAA BAA, custom data "
        "residency, SCIM, 24/7 premium support, and a 99.99% uptime SLA. Sandbox API keys "
        "are issued only to verified corporate email domains. Gmail, Yahoo, Outlook, and "
        "other public webmail are rejected. Keys are prefixed sbx_test_, expire after 14 "
        "days, are limited to 100 requests per minute, and cannot send real SMS or push.",
    ),
    (
        "7. Integrations and factors",
        "Identity providers: Okta, Microsoft Entra ID (Azure AD), Auth0, Ping Identity, "
        "Keycloak, and any SAML 2.0 or OIDC IdP. APIs: REST and gRPC, with SDKs for Python, "
        "Node.js, Go, and Java. Webhooks report enrollment and verification. TOTP apps are "
        "the primary factor. SMS OTP is not used in sandbox and is discouraged in production "
        "because of SIM-swap risk. WebAuthn / FIDO2 keys and passkeys are available on "
        "Enterprise as a phishing-resistant factor.",
    ),
]


class KnowledgePDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 110)
        self.cell(0, 8, "SecureGate 2FA Knowledge Base  |  Confidential - prospects under NDA", align="L")
        self.ln(12)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _wrap_write(pdf: KnowledgePDF, text: str, size: int = 11) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(30, 30, 40)
    pdf.multi_cell(0, 6, text)
    pdf.ln(2)


def build() -> Path:
    payload = json.loads(PAIRS_PATH.read_text())
    DATA.mkdir(exist_ok=True)

    pdf = KnowledgePDF(format="Letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(40, 30, 110)
    pdf.multi_cell(0, 10, payload["title"])
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(70, 70, 90)
    pdf.multi_cell(0, 7, payload["subtitle"])
    pdf.ln(4)
    pdf.set_draw_color(109, 94, 252)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 206, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 40)
    pdf.multi_cell(
        0,
        6,
        "This document is the retrieval corpus for the SecureGate public-site assistant. "
        "Answers to prospect questions must be grounded in the sections below. Do not invent "
        "certifications, algorithms, SLAs, or sandbox policy that are not written here.",
    )
    pdf.ln(6)

    for heading, body in NARRATIVE:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(45, 40, 120)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(1)
        _wrap_write(pdf, body)
        pdf.ln(3)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(45, 40, 120)
    pdf.cell(0, 10, "8. Frequently asked questions", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for i, pair in enumerate(payload["pairs"], 1):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 30)
        pdf.multi_cell(0, 7, f"Q{i}. {pair['pdf_question']}")
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 90, 140)
        pdf.cell(0, 5, f"Topic: {pair['topic']}  |  id: {pair['id']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 40)
        pdf.multi_cell(0, 6, f"A{i}. {pair['gold_answer']}")
        pdf.ln(5)

    pdf.output(str(PDF_PATH))
    return PDF_PATH


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
