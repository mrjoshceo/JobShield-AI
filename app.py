import re
import time
import textwrap
from pathlib import Path

from PIL import Image

import joblib
import streamlit as st


# =========================================================
# Product configuration
# Stable public release: Version 1.0.0
# =========================================================
PRODUCT_NAME = "JobShield AI"
APP_VERSION = "1.0.0"
MODEL_PATH = Path("svm_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
FAVICON_PATH = Path("jobshield_favicon.png")

MIN_WORDS = 35
MIN_UNIQUE_WORDS = 18
MAX_CHARACTERS = 20_000

SAMPLE_JOB = """Customer Support Specialist

We are hiring a Customer Support Specialist to join our London team. The successful candidate will respond to customer enquiries by email and telephone, maintain accurate support records, and work with colleagues to resolve service issues.

Key responsibilities include managing support tickets, escalating technical issues, updating customer information, and helping improve the customer experience.

Applicants should have strong written and verbal communication skills, good attention to detail, and confidence using common office software. Previous customer service experience is desirable. The successful applicant will receive a formal employment contract and complete the organisation's standard recruitment process."""


# =========================================================
# Page setup
# =========================================================
if not FAVICON_PATH.exists():
    raise FileNotFoundError(
        f"Missing favicon file: {FAVICON_PATH.name}. "
        "Keep it in the same folder as app.py."
    )

PAGE_ICON = Image.open(FAVICON_PATH)
st.set_page_config(
    page_title=f"{PRODUCT_NAME} | AI Job Fraud Detection",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html_block(markup: str) -> None:
    """Render dedented HTML safely without Markdown treating it as code."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


# =========================================================
# Styling
# =========================================================
html_block(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --primary: #1D4ED8;
            --primary-hover: #1E40AF;
            --navy: #0A1628;
            --white: #FFFFFF;
            --background: #F8FAFC;
            --surface-soft: #FBFCFE;
            --border: #E2E8F0;
            --muted: #475569;
            --muted-light: #64748B;
            --success: #16A34A;
            --success-soft: #F0FDF4;
            --success-border: #BBF7D0;
            --warning: #D97706;
            --warning-soft: #FFFBEB;
            --warning-border: #FDE68A;
            --danger: #DC2626;
            --danger-soft: #FEF2F2;
            --danger-border: #FECACA;
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background: var(--background);
        }

        #MainMenu,
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1520px;
            padding-top: 1rem;
            padding-bottom: 2.2rem;
            animation: pageIn 0.45s cubic-bezier(.2,.8,.2,1) both;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 52px;
            margin-bottom: 1.55rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            color: var(--navy);
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .shield-logo {
            width: 50px;
            height: 50px;
            border-radius: 15px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--primary);
            color: var(--white);
            box-shadow: 0 14px 30px rgba(29, 78, 216, 0.22);
        }

        .shield-logo svg {
            width: 30px;
            height: 30px;
            display: block;
        }

        .online-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.48rem;
            padding: 0.5rem 0.78rem;
            border-radius: 999px;
            background: var(--white);
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(10, 22, 40, 0.04);
        }

        .online-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.12);
            animation: onlinePulse 2.2s ease-in-out infinite;
        }

        .product-heading {
            max-width: 820px;
            margin: 0 auto 1.55rem;
            text-align: center;
        }

        .product-kicker {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
            background: #EFF6FF;
            border: 1px solid #DBEAFE;
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.075em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .product-heading h1 {
            color: var(--navy);
            font-size: clamp(2.15rem, 4.3vw, 3.5rem);
            line-height: 1.04;
            letter-spacing: -0.06em;
            margin: 0 0 0.7rem;
            font-weight: 900;
        }

        .product-heading h1 span {
            color: var(--primary);
        }

        .product-heading p {
            color: var(--muted);
            max-width: 680px;
            margin: 0 auto;
            font-size: 0.94rem;
            line-height: 1.65;
        }


        /* Keep the two main workspace panels visually balanced */
        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
            flex: 1 1 auto;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--white);
            height: 100%;
            border: 1px solid var(--border) !important;
            border-radius: 20px !important;
            box-shadow: 0 16px 42px rgba(10, 22, 40, 0.07);
            transition: transform 220ms cubic-bezier(.2,.8,.2,1),
                        box-shadow 220ms cubic-bezier(.2,.8,.2,1);
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 56px rgba(10, 22, 40, 0.095);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.18rem;
        }

        .panel-label {
            color: var(--primary);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.075em;
            text-transform: uppercase;
            margin-bottom: 0.42rem;
        }

        .panel-title {
            color: var(--navy);
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin-bottom: 0.22rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.5;
            margin-bottom: 0.72rem;
        }

        [data-testid="stTextArea"] label {
            display: none;
        }

        [data-testid="stTextArea"] textarea {
            min-height: 350px !important;
            padding: 1rem !important;
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
            background: var(--surface-soft) !important;
            color: var(--navy) !important;
            font-size: 0.91rem !important;
            line-height: 1.62 !important;
            box-shadow: none !important;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: #93C5FD !important;
            box-shadow: 0 0 0 4px rgba(29, 78, 216, 0.10) !important;
        }

        .input-meta {
            color: var(--muted-light);
            font-size: 0.71rem;
            margin-top: -0.35rem;
        }

        .stButton > button {
            min-height: 56px;
            border-radius: 12px;
            font-weight: 700;
            transition: transform 180ms cubic-bezier(.2,.8,.2,1),
                        box-shadow 180ms cubic-bezier(.2,.8,.2,1),
                        background 180ms ease,
                        border-color 180ms ease;
        }

        .stButton > button[kind="primary"] {
            background: var(--primary);
            border: 1px solid var(--primary);
            color: var(--white);
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.2);
        }

        .stButton > button[kind="primary"]:hover {
            background: var(--primary-hover);
            border-color: var(--primary-hover);
            color: var(--white);
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(29, 78, 216, 0.24);
        }

        .stButton > button[kind="secondary"] {
            background: var(--white);
            border: 1px solid var(--border);
            color: var(--navy);
        }

        .stButton > button[kind="secondary"]:hover {
            border-color: #CBD5E1;
            color: var(--navy);
            transform: translateY(-1px);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        .empty-state {
            min-height: 350px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--surface-soft);
            padding: 2rem;
        }

        .empty-shield {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
            background: #EFF6FF;
            border: 1px solid #DBEAFE;
            margin-bottom: 0.95rem;
        }

        .empty-shield svg {
            width: 27px;
            height: 27px;
        }

        .empty-title {
            color: var(--navy);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.34rem;
        }

        .empty-copy {
            color: var(--muted);
            max-width: 350px;
            font-size: 0.81rem;
            line-height: 1.55;
        }

        .result-safe,
        .result-warning,
        .result-risk {
            border-radius: 15px;
            padding: 1.02rem 1.08rem;
            margin-bottom: 0.75rem;
            animation: resultIn 0.35s ease both;
        }

        .result-safe {
            background: var(--success-soft);
            border: 1px solid var(--success-border);
            border-left: 6px solid var(--success);
        }

        .result-warning {
            background: var(--warning-soft);
            border: 1px solid var(--warning-border);
            border-left: 6px solid var(--warning);
        }

        .result-risk {
            background: var(--danger-soft);
            border: 1px solid var(--danger-border);
            border-left: 6px solid var(--danger);
        }

        .result-badge-safe,
        .result-badge-warning,
        .result-badge-risk {
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.44rem;
        }

        .result-badge-safe { color: var(--success); }
        .result-badge-warning { color: var(--warning); }
        .result-badge-risk { color: var(--danger); }

        .result-title {
            color: var(--navy);
            font-size: 1.28rem;
            line-height: 1.2;
            font-weight: 800;
            letter-spacing: -0.034em;
            margin-bottom: 0.3rem;
        }

        .result-summary {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.56;
        }

        .gauge-wrap {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0.8rem 0 0.68rem;
            animation: gaugeIn 0.45s ease both;
        }

        .risk-gauge {
            --score: 0;
            --gauge-color: var(--primary);
            width: 164px;
            height: 164px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            position: relative;
            background: conic-gradient(
                var(--gauge-color) calc(var(--score) * 1%),
                #EAF0F7 0
            );
            animation: gaugeFill 0.85s cubic-bezier(.2,.8,.2,1) both;
            box-shadow:
                0 12px 30px color-mix(in srgb, var(--gauge-color) 18%, transparent),
                inset 0 0 0 1px rgba(255,255,255,0.65);
        }

        .risk-gauge::after {
            content: "";
            position: absolute;
            width: 62%;
            height: 26%;
            top: 10%;
            left: 19%;
            border-radius: 50%;
            background: rgba(255,255,255,0.28);
            filter: blur(5px);
            pointer-events: none;
        }

        .risk-gauge::before {
            content: "";
            position: absolute;
            inset: 12px;
            border-radius: 50%;
            background: var(--white);
            box-shadow:
                0 8px 22px rgba(10, 22, 40, 0.07),
                inset 0 2px 5px rgba(10, 22, 40, 0.035);
        }

        .gauge-content {
            position: relative;
            z-index: 1;
            text-align: center;
        }

        .gauge-number {
            display: block;
            color: var(--navy);
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.055em;
            line-height: 1;
        }

        .gauge-label {
            display: block;
            color: var(--muted-light);
            font-size: 0.68rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .confidence-line {
            display: flex;
            justify-content: center;
            margin: 0.05rem 0 0.75rem;
        }

        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            color: var(--navy);
            background: #EFF6FF;
            border: 1px solid #DBEAFE;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .confidence-badge span {
            color: var(--primary);
            font-weight: 800;
        }

        [data-testid="stMetric"] {
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: 13px;
            padding: 0.82rem 0.88rem;
            transition: transform 0.16s ease, box-shadow 0.16s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(10, 22, 40, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--navy) !important;
            font-size: 1.18rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }

        .recommendation {
            margin-top: 0.75rem;
            border-radius: 13px;
            border: 1px solid var(--border);
            border-left: 4px solid var(--recommendation-accent, var(--primary));
            background: var(--surface-soft);
            padding: 0.78rem 0.88rem;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.5;
            min-height: 0;
        }

        .recommendation strong {
            display: flex;
            align-items: center;
            gap: 0.42rem;
            color: var(--navy);
            margin-bottom: 0.3rem;
            font-size: 0.84rem;
            font-weight: 800;
        }

        .recommendation strong::before {
            content: "✓";
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            color: var(--white);
            background: var(--recommendation-accent, var(--primary));
            font-size: 0.65rem;
            font-weight: 900;
            flex: 0 0 18px;
        }


        .loading-panel {
            margin-top: 0.6rem;
            padding: 0.85rem;
            border-radius: 13px;
            background: #EFF6FF;
            border: 1px solid #DBEAFE;
            color: var(--primary);
            text-align: center;
            font-size: 0.75rem;
            font-weight: 700;
            animation: pulsePanel 1.1s ease-in-out infinite alternate;
        }

        .loading-steps {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 0.55rem;
            flex-wrap: wrap;
        }

        .loading-step {
            padding: 0.32rem 0.5rem;
            border-radius: 999px;
            background: var(--white);
            border: 1px solid #DBEAFE;
            color: var(--primary);
            font-size: 0.66rem;
            font-weight: 700;
        }

        @keyframes pulsePanel {
            from { opacity: 0.72; transform: translateY(1px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .trust-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.45rem auto 0;
            max-width: 850px;
        }

        .trust-item {
            text-align: center;
            color: var(--muted-light);
            font-size: 0.7rem;
            line-height: 1.5;
        }

        .trust-item strong {
            display: block;
            color: var(--navy);
            font-size: 0.73rem;
            margin-bottom: 0.13rem;
        }

        .footer-note {
            max-width: 760px;
            margin: 1.05rem auto 0;
            text-align: center;
            color: rgba(100, 116, 139, 0.72);
            font-size: 0.63rem;
            line-height: 1.5;
        }


        @keyframes pageIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes onlinePulse {
            0%, 100% {
                transform: scale(1);
                box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.12);
            }
            50% {
                transform: scale(1.12);
                box-shadow: 0 0 0 7px rgba(22, 163, 74, 0.06);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }

        @keyframes resultIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes gaugeIn {
            from { opacity: 0; transform: scale(0.94); }
            to { opacity: 1; transform: scale(1); }
        }

        @keyframes gaugeFill {
            from {
                opacity: 0;
                transform: scale(0.94) rotate(-8deg);
                filter: saturate(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1) rotate(0);
                filter: saturate(1);
            }
        }

        @media (max-width: 850px) {
            .block-container {
                padding: 0.85rem 1rem 2rem;
            }

            .online-pill {
                display: none;
            }

            .product-heading h1 {
                font-size: 2.4rem;
            }

            [data-testid="stTextArea"] textarea,
            .empty-state {
                min-height: 285px !important;
            }

            .trust-strip {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """
)


# =========================================================
# Load assets
# =========================================================
@st.cache_resource(show_spinner=False)
def load_assets():
    missing = [
        path.name for path in (MODEL_PATH, VECTORIZER_PATH) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if not hasattr(classifier, "predict"):
        raise TypeError("The classifier does not support prediction.")
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("The classifier does not support probability scores.")
    if not hasattr(vectorizer, "transform"):
        raise TypeError("The vectorizer cannot transform text.")

    return classifier, vectorizer


try:
    model, vectorizer = load_assets()
except Exception:
    st.error(
        "The screening service is temporarily unavailable. Please try again later.",
        icon="⚠️",
    )
    st.stop()


# =========================================================
# Session state
# =========================================================
if "job_text" not in st.session_state:
    st.session_state.job_text = ""

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def use_sample():
    st.session_state.job_text = SAMPLE_JOB
    st.session_state.analysis_result = None


def clear_input():
    st.session_state.job_text = ""
    st.session_state.analysis_result = None


# =========================================================
# Validation and analysis
# =========================================================
def validate_text(text: str) -> tuple[bool, str]:
    cleaned = text.strip()

    if not cleaned:
        return False, "Paste a job advertisement before starting the analysis."

    if len(cleaned) > MAX_CHARACTERS:
        return (
            False,
            f"The advertisement is too long. Keep it below {MAX_CHARACTERS:,} characters.",
        )

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    unique_words = {word.lower() for word in words}

    if len(words) < MIN_WORDS:
        return (
            False,
            f"Please provide a more complete advertisement containing at least {MIN_WORDS} words.",
        )

    if len(unique_words) < MIN_UNIQUE_WORDS:
        return (
            False,
            "The text does not contain enough meaningful information for a reliable assessment.",
        )

    alphabetic_count = sum(character.isalpha() for character in cleaned)
    if alphabetic_count / max(len(cleaned), 1) < 0.55:
        return (
            False,
            "The text appears incomplete or contains too many non-word characters.",
        )

    return True, ""


def analyse_text(text: str) -> dict:
    started_at = time.perf_counter()

    features = vectorizer.transform([text])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    class_positions = {
        int(class_label): index
        for index, class_label in enumerate(model.classes_)
    }

    fraud_probability = float(probabilities[class_positions[1]] * 100)
    confidence = float(max(probabilities) * 100)
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
    }


# =========================================================
# Brand and hero
# =========================================================
shield_svg = """
<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M12 3L19 6V11.3C19 15.7 16.2 19.7 12 21C7.8 19.7 5 15.7 5 11.3V6L12 3Z"
          stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
    <path d="M9.2 12.1L11.2 14.1L15.3 9.9"
          stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

html_block(
    f"""
    <div class="topbar">
        <div class="brand">
            <span class="shield-logo">{shield_svg}</span>
            <span>{PRODUCT_NAME}</span>
        </div>
        <div class="online-pill">
            <span class="online-dot"></span>
            Screening service online
        </div>
    </div>
    """
)

html_block(
    """
    <section class="product-heading">
        <div class="product-kicker">AI job fraud detection</div>
        <h1>Assess suspicious job adverts <span>before you apply.</span></h1>
        <p>
            Review a complete job advertisement and receive an instant AI-powered
            fraud risk assessment designed to support safer application decisions.
        </p>
    </section>
    """
)


# =========================================================
# Two-panel workspace
# =========================================================
input_column, result_column = st.columns([1.08, 0.92], gap="large")

with input_column:
    with st.container(border=True):
        html_block(
            """
            <div class="panel-label">Advertisement input</div>
            <div class="panel-title">Paste the complete job listing</div>
            <div class="panel-copy">
                Include the title, company information, responsibilities,
                requirements and benefits where available.
            </div>
            """
        )

        job_text = st.text_area(
            "Job advertisement",
            key="job_text",
            height=350,
            max_chars=MAX_CHARACTERS,
            placeholder=(
                "Paste the complete job advertisement here...\n\n"
                "The fuller the listing, the more meaningful the assessment."
            ),
        )

        word_count = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", job_text))
        html_block(
            f'<div class="input-meta">{word_count:,} words · {len(job_text):,} characters</div>'
        )

        sample_col, clear_col = st.columns(2)

        with sample_col:
            st.button(
                "Try an example",
                on_click=use_sample,
                use_container_width=True,
                type="secondary",
            )

        with clear_col:
            st.button(
                "Clear",
                on_click=clear_input,
                use_container_width=True,
                type="secondary",
            )

        analyse_clicked = st.button(
            "Analyse advertisement",
            use_container_width=True,
            type="primary",
        )


validation_message = None

if analyse_clicked:
    valid, validation_message = validate_text(job_text)

    if not valid:
        st.session_state.analysis_result = None
    else:
        try:
            with st.spinner("Analysing advertisement..."):
                html_block(
                    '''
                    <div class="loading-panel">
                        Analysing job advertisement
                        <div class="loading-steps">
                            <span class="loading-step">Reading content</span>
                            <span class="loading-step">Evaluating patterns</span>
                            <span class="loading-step">Preparing result</span>
                        </div>
                    </div>
                    '''
                )
                time.sleep(0.45)
                st.session_state.analysis_result = analyse_text(job_text)
        except Exception:
            st.session_state.analysis_result = None
            validation_message = (
                "The advertisement could not be analysed. Please try again."
            )


with result_column:
    with st.container(border=True):
        html_block(
            """
            <div class="panel-label">Risk assessment</div>
            <div class="panel-title">Screening result</div>
            <div class="panel-copy">
                Review the estimated fraud risk and the recommended next step.
            </div>
            """
        )

        if validation_message:
            st.warning(validation_message, icon="⚠️")

        result = st.session_state.analysis_result

        if result is None:
            html_block(
                f"""
                <div class="empty-state">
                    <div class="empty-shield">{shield_svg}</div>
                    <div class="empty-title">Ready to analyse</div>
                    <div class="empty-copy">
                        Paste a complete job advertisement to receive an
                        AI-powered fraud risk assessment.
                    </div>
                </div>
                """
            )
        else:
            prediction = result["prediction"]
            fraud_probability = result["fraud_probability"]
            confidence = result["confidence"]

            if fraud_probability >= 75:
                risk_level = "High"
                result_class = "result-risk"
                badge_class = "result-badge-risk"
                badge_text = "High risk"
                gauge_color = "#DC2626"
            elif fraud_probability >= 45:
                risk_level = "Moderate"
                result_class = "result-warning"
                badge_class = "result-badge-warning"
                badge_text = "Moderate risk"
                gauge_color = "#D97706"
            else:
                risk_level = "Low"
                result_class = "result-safe"
                badge_class = "result-badge-safe"
                badge_text = "Lower risk"
                gauge_color = "#16A34A"

            if prediction == 1:
                title = "This advertisement may be fraudulent"
                summary = (
                    "The text contains patterns associated with potentially fraudulent "
                    "job listings. Treat the opportunity cautiously and verify the employer independently."
                )
                assessment = "Potential fraud"
                recommendation = (
                    "Do not send money, identity documents, banking information or account credentials. "
                    "Confirm the vacancy using the employer's official website and independently verified contact details."
                )
            else:
                title = "This advertisement is likely genuine"
                summary = (
                    "The text does not strongly match the fraudulent patterns recognised by the screening system. "
                    "This does not guarantee that the employer or vacancy is legitimate."
                )
                assessment = "Likely genuine"
                recommendation = (
                    "You may continue with appropriate caution. Verify the employer and vacancy before sharing "
                    "identity documents, banking information or other sensitive personal data."
                )

            html_block(
                f"""
                <div class="{result_class}">
                    <div class="{badge_class}">● {badge_text}</div>
                    <div class="result-title">{title}</div>
                    <div class="result-summary">{summary}</div>
                </div>
                """
            )

            html_block(
                f"""
                <div class="gauge-wrap">
                    <div class="risk-gauge"
                         style="--score:{fraud_probability:.1f}; --gauge-color:{gauge_color};">
                        <div class="gauge-content">
                            <span class="gauge-number">{fraud_probability:.0f}%</span>
                            <span class="gauge-label">fraud risk</span>
                        </div>
                    </div>
                </div>
                <div class="confidence-line">
                    <div class="confidence-badge">
                        Model confidence <span>{confidence:.1f}%</span>
                    </div>
                </div>
                """
            )

            metric_left, metric_right = st.columns(2)

            with metric_left:
                st.metric("Assessment", assessment)

            with metric_right:
                st.metric("Risk level", risk_level)

            html_block(
                f"""
                <div class="recommendation" style="--recommendation-accent:{gauge_color};">
                    <strong>Recommended next step</strong>
                    {recommendation}
                </div>
                """
            )


# =========================================================
# Trust cues and footer
# =========================================================
html_block(
    """
    <div class="trust-strip">
        <div class="trust-item">
            <strong>Fast screening</strong>
            Receive an assessment in seconds.
        </div>
        <div class="trust-item">
            <strong>Privacy focused</strong>
            Avoid entering passwords or sensitive personal data.
        </div>
        <div class="trust-item">
            <strong>Human verification matters</strong>
            Confirm every opportunity independently.
        </div>
    </div>

    <div class="footer-note">
        JobShield AI provides an automated text-based risk assessment and may make mistakes.
        It does not verify an employer's identity and should not replace independent checks or professional advice.
    </div>
    """
)
