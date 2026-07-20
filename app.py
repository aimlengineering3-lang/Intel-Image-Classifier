"""
IntelVision AI — Field Classification Console
================================================
A single-file Streamlit app for the Intel Image Classification project
(ResNet18 transfer-learning model), styled as a terrain-survey instrument
rather than a generic AI dashboard — because the six classes it predicts
(buildings, forest, glacier, mountain, sea, street) are literally terrain
types, and the UI leans into that.

Folder assumptions:

project/
├── app.py                <- this file
├── config.py              (DEVICE, IMAGE_SIZE, NUM_CLASSES)
├── model.py                (build_model())
├── models/
│      best_model.pth
├── outputs/
│      plots/
│          accuracy_curve.png
│          loss_curve.png
│          confusion_matrix.png

Run with:
    streamlit run app.py
"""

import os
import io
import time
import json
import base64

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go

# ----------------------------------------------------------------
# 0. Project imports — safe fallbacks so this file still runs
#    standalone if config.py / model.py aren't wired up yet.
# ----------------------------------------------------------------
try:
    import config
    DEVICE = config.DEVICE
    IMAGE_SIZE = config.IMAGE_SIZE
    NUM_CLASSES = config.NUM_CLASSES
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 224
    NUM_CLASSES = 6

try:
    from model import build_model
    MODEL_MODULE_AVAILABLE = True
except Exception:
    MODEL_MODULE_AVAILABLE = False

# ImageFolder assigns labels alphabetically — this is that order.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CLASS_GLYPH = {
    "buildings": "🏙",
    "forest": "🌲",
    "glacier": "🧊",
    "mountain": "⛰",
    "sea": "🌊",
    "street": "🛣",
}

CLASS_NOTE = {
    "buildings": "Urban structures, facades, dense city blocks.",
    "forest": "Dense canopy cover, woodland texture, understory.",
    "glacier": "Compacted ice fields, high-altitude snowpack.",
    "mountain": "Elevated rock formations, ridgelines, peaks.",
    "sea": "Open water, coastline, horizon-level seascape.",
    "street": "Paved thoroughfare, road markings, signage.",
}

MODEL_PATH = os.path.join("models", "best_model.pth")
PLOTS_DIR = os.path.join("outputs", "plots")
ACCURACY_PLOT = os.path.join(PLOTS_DIR, "accuracy_curve.png")
LOSS_PLOT = os.path.join(PLOTS_DIR, "loss_curve.png")
CONFUSION_PLOT = os.path.join(PLOTS_DIR, "confusion_matrix.png")
METRICS_JSON = os.path.join("outputs", "metrics.json")


# ==================================================================
# 1. PAGE CONFIGURATION
# ==================================================================
st.set_page_config(
    page_title="IntelVision AI | Field Classification Console",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================
# 2. TOKEN SYSTEM — INSTRUMENT CONSOLE THEME
# ==================================================================
# Color    ink #0d1310 (bg) · panel #131a17 (surfaces) · steel #2a332d
#          (rules) · amber #e3a458 (primary accent) · moss #7fae82
#          (secondary accent) · fog #8b9791 (muted text) · paper #eef0ea
#          (primary text) · rust #c1673f (alert)
# Type     display: Fraunces (characterful serif, used sparingly)
#          body: Inter · data/labels: IBM Plex Mono
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0d1310;
    --panel: #131a17;
    --panel-2: #172019;
    --steel: #2a332d;
    --amber: #e3a458;
    --moss: #7fae82;
    --fog: #8b9791;
    --paper: #eef0ea;
    --rust: #c1673f;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: var(--ink);
    background-image:
        repeating-linear-gradient(0deg, rgba(238,240,234,0.012) 0px, rgba(238,240,234,0.012) 1px, transparent 1px, transparent 3px);
    color: var(--paper);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.mono { font-family: 'IBM Plex Mono', monospace; }

/* ---------- Top status strip ---------- */
.status-strip {
    display: flex;
    gap: 1.6rem;
    align-items: center;
    padding: 0.5rem 0 1.1rem 0;
    margin-bottom: 1.4rem;
    border-bottom: 1px solid var(--steel);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--fog);
    text-transform: uppercase;
}
.status-strip .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--moss);
    display: inline-block;
    margin-right: 0.4rem;
    box-shadow: 0 0 6px var(--moss);
}
.status-strip .sep { color: var(--steel); }

/* ---------- Hero ---------- */
.station-hero {
    position: relative;
    padding: 3rem 2.6rem;
    border: 1px solid var(--steel);
    border-radius: 4px;
    background: linear-gradient(180deg, var(--panel) 0%, var(--ink) 100%);
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 3rem;
    line-height: 1.08;
    color: var(--paper);
    max-width: 620px;
    margin-bottom: 0.9rem;
}
.hero-title em {
    font-style: italic;
    color: var(--moss);
}
.hero-subtitle {
    font-size: 1.0rem;
    color: var(--fog);
    max-width: 520px;
    line-height: 1.55;
}
.hero-legend {
    margin-top: 1.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: var(--fog);
    text-transform: uppercase;
}
.hero-legend span.tick { color: var(--steel); margin: 0 0.55rem; }

.contour-art {
    position: absolute;
    right: -40px;
    top: 0;
    width: 420px;
    height: 100%;
    opacity: 0.55;
    pointer-events: none;
}
@media (prefers-reduced-motion: no-preference) {
    .contour-art .drift { animation: driftLine 18s ease-in-out infinite; }
}
@keyframes driftLine {
    0%   { transform: translateX(0px); }
    50%  { transform: translateX(-14px); }
    100% { transform: translateX(0px); }
}

/* ---------- Instrument readout / KPI strip ---------- */
.gauge-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid var(--steel);
    border-radius: 4px;
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.gauge {
    padding: 1.3rem 1.5rem;
    border-right: 1px solid var(--steel);
    background: var(--panel);
}
.gauge:last-child { border-right: none; }
.gauge-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.gauge-value {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.9rem;
    color: var(--paper);
}
.gauge-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--fog);
    margin-top: 0.3rem;
}

/* ---------- Panels ---------- */
.panel {
    background: var(--panel);
    border: 1px solid var(--steel);
    border-radius: 4px;
    padding: 1.5rem 1.7rem;
}
.section-header {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.35rem;
    color: var(--paper);
    margin: 1.8rem 0 0.9rem 0;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.section-header .idx {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--amber);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--steel);
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--fog);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    padding: 0.55rem 0.8rem;
    border-radius: 3px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border: 1px solid var(--steel);
    color: var(--paper);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(227,164,88,0.10);
    border: 1px solid var(--amber);
    color: var(--amber);
}

/* ---------- Specimen tag (prediction result) ---------- */
.specimen-tag {
    border: 1px solid var(--amber);
    border-radius: 4px;
    padding: 1.7rem;
    text-align: center;
    background:
        linear-gradient(180deg, rgba(227,164,88,0.08), rgba(227,164,88,0.01));
}
.specimen-glyph { font-size: 2.6rem; }
.specimen-name {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 600;
    font-size: 2rem;
    color: var(--paper);
    margin: 0.3rem 0 0.2rem 0;
}
.specimen-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: var(--amber);
    letter-spacing: 0.05em;
}
.specimen-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--fog);
    letter-spacing: 0.08em;
    margin-top: 0.9rem;
    text-transform: uppercase;
}

/* ---------- Scan reveal frame ---------- */
.scan-frame {
    position: relative;
    border: 1px solid var(--steel);
    border-radius: 4px;
    overflow: hidden;
    background: #0a0f0c;
    height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.scan-frame img {
    max-width: 100% !important;
    max-height: 100% !important;
    width: auto !important;
    height: auto !important;
    display: block !important;
    object-fit: contain !important;
    filter: saturate(0.92) contrast(1.02);
}
.scan-frame::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--moss);
    box-shadow: 0 0 12px 2px var(--moss);
    z-index: 3;
}
@media (prefers-reduced-motion: no-preference) {
    .scan-frame::before { animation: sweepDown 1.3s ease-in-out 1; }
}
.scan-frame .bracket {
    position: absolute;
    width: 16px; height: 16px;
    border-color: var(--amber);
    z-index: 4;
}
.scan-frame .tl { top: 8px; left: 8px; border-top: 2px solid var(--amber); border-left: 2px solid var(--amber); }
.scan-frame .tr { top: 8px; right: 8px; border-top: 2px solid var(--amber); border-right: 2px solid var(--amber); }
.scan-frame .bl { bottom: 8px; left: 8px; border-bottom: 2px solid var(--amber); border-left: 2px solid var(--amber); }
.scan-frame .br { bottom: 8px; right: 8px; border-bottom: 2px solid var(--amber); border-right: 2px solid var(--amber); }
@keyframes sweepDown {
    0%   { top: 0%; opacity: 1; }
    95%  { top: 99%; opacity: 1; }
    100% { top: 99%; opacity: 0; }
}

/* ---------- Field frame (analytics images) ---------- */
.field-frame {
    position: relative;
    border: 1px solid var(--steel);
    border-radius: 4px;
    padding: 10px;
    background: var(--panel);
}
.field-frame img { width: 100%; border-radius: 2px; }
.field-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: var(--fog);
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ---------- Dataset specimen cards ---------- */
.dataset-card {
    border: 1px solid var(--steel);
    border-radius: 4px;
    padding: 1.1rem 0.9rem;
    text-align: center;
    background: var(--panel);
    transition: border 0.2s ease;
}
.dataset-card:hover { border: 1px solid var(--moss); }
.dataset-glyph { font-size: 1.6rem; }
.dataset-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    color: var(--paper);
    text-transform: uppercase;
    margin: 0.5rem 0 0.35rem 0;
}
.dataset-note {
    font-size: 0.78rem;
    color: var(--fog);
    line-height: 1.4;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--fog);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 1.6rem 0 0.6rem 0;
    border-top: 1px solid var(--steel);
    margin-top: 2rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================================================================
# 3. MODEL LOADING
# ==================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_MODULE_AVAILABLE:
        return None, "model.py not found — running in demo mode."
    if not os.path.exists(MODEL_PATH):
        return None, f"Checkpoint not found at '{MODEL_PATH}'."
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, None


@st.cache_resource(show_spinner=False)
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict(image: Image.Image, model):
    transform = get_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    elapsed_ms = (time.time() - start) * 1000
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs, elapsed_ms


def load_metrics():
    defaults = {
        "val_accuracy": None,
        "num_train_images": None,
        "num_val_images": None,
        "architecture": "ResNet18 (Transfer Learning)",
    }
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r") as f:
                data = json.load(f)
            defaults.update(data)
        except Exception:
            pass
    return defaults


def img_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


model, model_error = load_model()
metrics = load_metrics()


# ==================================================================
# 4. NAVIGATION STATE
# ==================================================================
PAGES = [
    ("dashboard", "01", "Dashboard"),
    ("predict", "02", "Image Prediction"),
    ("analytics", "03", "Training Analytics"),
    ("model_info", "04", "Model Information"),
    ("dataset", "05", "About Dataset"),
]

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

with st.sidebar:
    st.markdown(
        "<div class='mono' style='letter-spacing:0.12em; color:#8b9791; "
        "font-size:0.7rem; text-transform:uppercase; margin-bottom:0.2rem;'>Field Console</div>"
        "<div style='font-family:Fraunces, serif; font-weight:600; font-size:1.5rem; "
        "color:#eef0ea;'>IntelVision <span style='color:#e3a458; font-style:italic;'>AI</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

    for key, idx, label in PAGES:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(f"{idx}  ·  {label}", key=f"nav_{key}", type=btn_type):
            st.session_state.page = key
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mono' style='font-size:0.68rem; color:#8b9791; letter-spacing:0.06em;'>"
        f"DEVICE&nbsp;&nbsp;<span style='color:#e3a458;'>{str(DEVICE).upper()}</span></div>",
        unsafe_allow_html=True,
    )
    status_txt = "MODEL LOADED" if model is not None else "DEMO MODE"
    status_color = "#7fae82" if model is not None else "#c1673f"
    st.markdown(
        f"<div class='mono' style='font-size:0.68rem; color:{status_color}; "
        f"letter-spacing:0.06em; margin-top:0.3rem;'>● {status_txt}</div>",
        unsafe_allow_html=True,
    )
    if model_error:
        st.caption(model_error)

page = st.session_state.page


# ==================================================================
# 5. TOP STATUS STRIP
# ==================================================================
st.markdown(
    f"""
    <div class="status-strip">
        <span><span class="dot"></span>SYSTEM ONLINE</span>
        <span class="sep">/</span>
        <span>ARCH: RESNET18</span>
        <span class="sep">/</span>
        <span>CLASSES: 6</span>
        <span class="sep">/</span>
        <span>DEVICE: {str(DEVICE).upper()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# 6. HERO + GAUGE STRIP (Dashboard)
# ==================================================================
CONTOUR_SVG = (
    '<svg class="contour-art" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">'
    '<g class="drift" fill="none" stroke="#7fae82" stroke-width="1.1">'
    '<path opacity="0.55" d="M-20 60 C 80 20, 160 100, 260 50 S 420 30, 460 70"/>'
    '<path opacity="0.45" d="M-20 110 C 90 70, 150 150, 260 100 S 420 80, 460 120"/>'
    '<path opacity="0.35" d="M-20 160 C 100 120, 140 200, 260 150 S 420 130, 460 170"/>'
    '</g>'
    '<g class="drift" fill="none" stroke="#e3a458" stroke-width="1.1">'
    '<path opacity="0.5" d="M-20 230 C 90 190, 170 270, 260 220 S 420 200, 460 240"/>'
    '<path opacity="0.35" d="M-20 280 C 100 250, 150 320, 260 270 S 420 250, 460 290"/>'
    '</g>'
    '</svg>'
)

def render_hero():
    hero_html = (
        '<div class="station-hero">'
        + CONTOUR_SVG
        + '<div class="hero-eyebrow">Field Classification Unit &nbsp;&mdash;&nbsp; Est. Session</div>'
        + '<div class="hero-title">Six terrains.<br>One <em>instrument</em>.</div>'
        + '<div class="hero-subtitle">'
        + "A ResNet18 network, pretrained on ImageNet and fine-tuned on the "
        + "Intel Image Classification set, reads a photograph the way a "
        + "surveyor reads a map &mdash; and tells you which of six terrains it's "
        + "looking at, in under a heartbeat."
        + '</div>'
        + '<div class="hero-legend">'
        + 'BUILDINGS<span class="tick">&middot;</span>FOREST<span class="tick">&middot;</span>'
        + 'GLACIER<span class="tick">&middot;</span>MOUNTAIN<span class="tick">&middot;</span>'
        + 'SEA<span class="tick">&middot;</span>STREET'
        + '</div>'
        + '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    acc_display = f"{metrics['val_accuracy']*100:.2f}%" if metrics.get("val_accuracy") else "—"
    total_images = (metrics.get("num_train_images") or 0) + (metrics.get("num_val_images") or 0)
    dataset_display = f"{total_images:,}" if total_images else "~25,000"

    gauges = [
        ("ACCURACY", acc_display, "validation set"),
        ("SAMPLE SIZE", dataset_display, "labeled images"),
        ("ARCHITECTURE", "ResNet18", "frozen backbone"),
        ("LATENCY", "< 50 ms", "per inference"),
    ]
    cells = "".join(
        f"""<div class="gauge">
                <div class="gauge-label">{label}</div>
                <div class="gauge-value">{val}</div>
                <div class="gauge-sub">{sub}</div>
            </div>"""
        for label, val, sub in gauges
    )
    st.markdown(f'<div class="gauge-strip">{cells}</div>', unsafe_allow_html=True)


def render_project_overview():
    st.markdown('<div class="section-header"><span class="idx">§ 01</span>Project Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            """
            <div class="panel">
            IntelVision AI classifies natural scene photographs into one of six
            categories using a convolutional network pretrained on ImageNet and
            fine-tuned on the Intel Image Classification dataset.
            <br><br>
            The final fully-connected layer of ResNet18 was replaced and retrained
            for this 6-class problem, while the earlier convolutional layers —
            which already recognise edges, textures, and shapes — were frozen.
            This is <b>transfer learning</b>: borrowing visual knowledge the
            network already has, and teaching it only what's new.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="panel mono" style="font-size:0.82rem; line-height:2.0; color:#eef0ea;">
            <span style="color:#8b9791; letter-spacing:0.08em;">PIPELINE</span><br>
            ImageFolder<br>
            <span style="color:#2a332d;">↓</span><br>
            Transforms<br>
            <span style="color:#2a332d;">↓</span><br>
            DataLoader<br>
            <span style="color:#2a332d;">↓</span><br>
            ResNet18 (frozen)<br>
            <span style="color:#2a332d;">↓</span><br>
            <span style="color:#e3a458;">6-Class Output</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================
# 7. IMAGE PREDICTION PAGE
# ==================================================================
def render_scan_reveal(image: Image.Image):
    b64 = img_to_b64(image)
    frame_html = (
        '<div class="scan-frame">'
        '<span class="bracket tl"></span><span class="bracket tr"></span>'
        '<span class="bracket bl"></span><span class="bracket br"></span>'
        f'<img src="data:image/png;base64,{b64}" '
        'style="max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain;" />'
        '</div>'
    )
    st.markdown(frame_html, unsafe_allow_html=True)


def render_prediction_page():
    st.markdown('<div class="section-header"><span class="idx">§ 02</span>Insert Specimen</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a scene photograph (jpg / png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="panel mono" style="color:#8b9791; font-size:0.85rem;">'
            "AWAITING INPUT — upload a photograph above to run a classification pass."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded_file)
    col_img, col_result = st.columns([1, 1.25])

    with col_img:
        render_scan_reveal(image)

    if model is None:
        with col_result:
            st.markdown(
                '<div class="panel" style="border-color:#c1673f;">'
                "Model isn't loaded, so this is the interface in demo mode. "
                "Place <code>best_model.pth</code> in <code>models/</code> and keep "
                "<code>model.py</code> beside <code>app.py</code> to enable live inference."
                "</div>",
                unsafe_allow_html=True,
            )
        return

    label, confidence, probs, elapsed_ms = predict(image, model)

    with col_result:
        st.markdown(
            f"""
            <div class="specimen-tag">
                <div class="specimen-glyph">{CLASS_GLYPH.get(label, "◆")}</div>
                <div class="specimen-name">{label.capitalize()}</div>
                <div class="specimen-conf">CONFIDENCE&nbsp;&nbsp;{confidence*100:.2f}%</div>
                <div class="specimen-meta">Inference {elapsed_ms:.1f} ms &nbsp;·&nbsp; {str(DEVICE).upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header"><span class="idx">§ 03</span>Class Probabilities</div>', unsafe_allow_html=True)

    order = np.argsort(probs)[::-1]
    sorted_classes = [CLASS_NAMES[i] for i in order]
    sorted_probs = [probs[i] * 100 for i in order]

    fig = go.Figure(
        go.Bar(
            x=sorted_probs,
            y=[c.capitalize() for c in sorted_classes],
            orientation="h",
            marker=dict(
                color=sorted_probs,
                colorscale=[[0, "#3d4a3f"], [0.5, "#7fae82"], [1, "#e3a458"]],
                line=dict(color="#2a332d", width=1),
            ),
            text=[f"{p:.1f}%" for p in sorted_probs],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", color="#eef0ea"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 108], title=None, gridcolor="#2a332d", zerolinecolor="#2a332d"),
        yaxis=dict(autorange="reversed"),
        font=dict(color="#eef0ea", family="IBM Plex Mono, monospace"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><span class="idx">§ 04</span>Top 3 Readings</div>', unsafe_allow_html=True)
    top3_cols = st.columns(3)
    for rank, (col, idx) in enumerate(zip(top3_cols, order[:3]), start=1):
        with col:
            st.markdown(
                f"""
                <div class="panel" style="text-align:center;">
                    <div class="mono" style="color:#8b9791; font-size:0.68rem; letter-spacing:0.1em;">NO. {rank}</div>
                    <div style="font-size:1.6rem; margin:0.3rem 0;">{CLASS_GLYPH.get(CLASS_NAMES[idx], "◆")}</div>
                    <div style="font-family:'Fraunces',serif; font-weight:600; font-size:1.1rem; color:#eef0ea;">{CLASS_NAMES[idx].capitalize()}</div>
                    <div class="mono" style="color:#e3a458; font-size:0.85rem; margin-top:0.2rem;">{probs[idx]*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==================================================================
# 8. TRAINING ANALYTICS PAGE
# ==================================================================
def render_analytics_page():
    st.markdown('<div class="section-header"><span class="idx">§ 03</span>Training Analytics</div>', unsafe_allow_html=True)

    plots = [
        ("Accuracy Curve", ACCURACY_PLOT),
        ("Loss Curve", LOSS_PLOT),
        ("Confusion Matrix", CONFUSION_PLOT),
    ]
    any_found = any(os.path.exists(p) for _, p in plots)
    if not any_found:
        st.markdown(
            '<div class="panel mono" style="color:#8b9791; font-size:0.85rem;">'
            "NO LOGS FOUND — run <code>train.py</code> first. This page reads "
            "<code>outputs/plots/</code> automatically, no code changes needed."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for title, path in plots:
        if os.path.exists(path):
            b64 = file_to_b64(path)
            frame_html = (
                '<div class="field-frame">'
                f'<img src="data:image/png;base64,{b64}" '
                'style="width:100%;height:auto;display:block;" />'
                f'<div class="field-caption">FIG &mdash; {title}</div>'
                '</div>'
            )
            st.markdown(frame_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="panel mono" style="color:#c1673f; font-size:0.8rem;">'
                f"MISSING — {os.path.basename(path)}</div>",
                unsafe_allow_html=True,
            )
        st.write("")


# ==================================================================
# 9. MODEL INFORMATION PAGE
# ==================================================================
def render_model_info_page():
    st.markdown('<div class="section-header"><span class="idx">§ 04</span>How the Instrument Reads</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="panel">
            <b>Architecture</b><br><br>
            Input (224×224×3) → Conv Stem → 4 Residual Stages → Global Average
            Pool → Fully Connected (512 → 6)
            <br><br>
            The backbone was pretrained on ImageNet — 1.2 million images, 1000
            classes — so its convolutional filters already encode general
            visual concepts: edges, textures, shapes.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="panel">
            <b>What Was Changed</b><br><br>
            &middot; Froze every pretrained convolutional layer<br>
            &middot; Replaced the final layer: 512 → 1000 became 512 → 6<br>
            &middot; Trained only the new layer's weights from scratch<br>
            &middot; Result: fast convergence, low overfitting risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header"><span class="idx">§ 05</span>Inference Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel mono" style="font-size:0.85rem;">
        Photograph → Resize (224×224) → Tensor → Normalize (ImageNet mean/std)
        → Forward Pass → Softmax → Class + Confidence
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# 10. ABOUT DATASET PAGE
# ==================================================================
def render_dataset_page():
    st.markdown('<div class="section-header"><span class="idx">§ 05</span>Specimen Index</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
        The <b>Intel Image Classification</b> dataset holds roughly 25,000
        images across six natural and urban scene categories. Originally
        released as part of an Intel / Analytics Vidhya challenge, it's a
        common benchmark for transfer-learning tutorials.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(6)
    for col, cls in zip(cols, CLASS_NAMES):
        with col:
            st.markdown(
                f"""
                <div class="dataset-card">
                    <div class="dataset-glyph">{CLASS_GLYPH[cls]}</div>
                    <div class="dataset-name">{cls}</div>
                    <div class="dataset-note">{CLASS_NOTE[cls]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==================================================================
# 11. ROUTER
# ==================================================================
if page == "dashboard":
    render_hero()
    render_project_overview()
elif page == "predict":
    render_prediction_page()
elif page == "analytics":
    render_analytics_page()
elif page == "model_info":
    render_model_info_page()
elif page == "dataset":
    render_dataset_page()


# ==================================================================
# 12. FOOTER
# ==================================================================
st.markdown(
    """
    <div class="app-footer">
        IntelVision Mk. I &nbsp;—&nbsp; Field Classification Unit &nbsp;—&nbsp;
        PyTorch / ResNet18 &nbsp;—&nbsp; CPU / GPU Adaptive
    </div>
    """,
    unsafe_allow_html=True,
)