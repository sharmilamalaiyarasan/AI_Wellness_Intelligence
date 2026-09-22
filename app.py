"""
AI Fitness Coach — Streamlit App
=================================
Entry point: streamlit run app.py

Features:
  - Powered by TRAINED Random Forest Classifier (models/yoga_pose_classifier.pkl)
  - Trained on user's 47+ category Kaggle Yoga Posture Dataset
  - Real-time webcam pose correction + 30 FPS stream
  - Photo Upload Mode with auto exercise classification + posture correction
  - SQLite Workout History & Analytics
  - Complete Light Theme & Widget Styling Overrides:
      * Secondary buttons (e.g. 'Back to Dashboard') -> White BG + Bold Indigo Text
      * File Uploader section -> White card + Indigo border + Crisp dark text
      * Browse Files button -> Light Indigo BG + Bold Indigo Text
      * Selectbox Popover -> White BG + Dark Slate Text
"""

import os
import sys
import time
import json
import base64
import warnings
from typing import Optional, Tuple, List, Dict
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import pandas as pd
    import joblib
    _ANALYTICS_OK = True
except ImportError:
    _ANALYTICS_OK = False

# ── Make sure our package is importable ────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from core.pose_detection   import (get_detector, detect_pose, extract_keypoints,
                                    draw_landmarks_on_frame, overlay_angle_text,
                                    get_webcam)
from core.exercise_analysis import (calculate_angles, provide_specific_feedback,
                                     get_reference_angles, get_primary_angle,
                                     predict_exercise_from_angles,
                                     SUPPORTED_EXERCISES, EXERCISE_INFO,
                                     POSE_DISPLAY_NAMES)
from core.rep_counter       import RepCounter
from core.feedback          import (get_feedback_display, get_primary_tip,
                                    EXERCISE_TIPS, format_duration)
from core.voice_coach       import (VoiceCoach, get_voice_audio_html, get_voice_audio_bytes,
                                    translate_text, get_ui_text, UI_STRINGS, BODY_PARTS_TA)
from database.database      import init_db, save_workout, get_all_workouts, get_aggregate_stats


def play_audio(placeholder, a_bytes: Optional[bytes], autoplay: bool = True):
    """Safely play audio in Streamlit with zero duplicate element ID collisions and full version compatibility."""
    if not a_bytes:
        return
    # Appending timestamp comment ensures unique element ID across identical phrases
    unique_data = a_bytes + f"<!--{time.time_ns()}-->".encode("utf-8")
    try:
        placeholder.audio(unique_data, format="audio/mp3", autoplay=autoplay)
    except Exception as e:
        print(f"[VoiceCoach] Audio playback notice: {e}")


# ── App config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Wellness Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(APP_DIR, "assets")

@st.cache_data(show_spinner=False)
def _get_asset_base64(filename: str) -> str:
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        ext = os.path.splitext(filename)[1].lower().replace(".", "")
        mime = "image/gif" if ext == "gif" else f"image/{ext}"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    return ""

# ── Initialize DB ──────────────────────────────────────────────────────────────
init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS — Universal High-Contrast Light Design System
# ═══════════════════════════════════════════════════════════════════════════════
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    *, *::before, *::after {
        font-family: 'Inter', sans-serif !important;
        box-sizing: border-box;
    }
    body, .stApp {
        background-color: #F4F6FC !important;
        color: #0F172A !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }
    .main .block-container {
        padding: 1.5rem 2rem 2rem !important;
        max-width: 1400px !important;
    }

    /* Global Typography Force Dark Slate */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #0F172A;
    }

    /* Streamlit Captions */
    [data-testid="stCaptionContainer"] p, .stCaption, caption {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* ── BUTTON STYLING (Primary & Secondary) ────────────────────────────────── */
    /* Secondary & Default Buttons (e.g. Back to Dashboard) */
    .stButton > button,
    button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #4338CA !important;
        border: 2px solid #6366F1 !important;
        border-radius: 14px !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        background-color: #EEF2FF !important;
        color: #312E81 !important;
        border-color: #4338CA !important;
        transform: translateY(-1px) !important;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA, #4F46E5) !important;
        color: #FFFFFF !important;
    }

    /* ── FILE UPLOADER STYLING ────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 12px !important;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #6366F1 !important;
        border-radius: 14px !important;
        padding: 24px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
        border: 1.5px solid #A5B4FC !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 6px 18px !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #E0E7FF !important;
        color: #312E81 !important;
    }

    /* ── BASEWEB SELECTBOX & DROPDOWN POPOVER OVERRIDES ──────────────────────── */
    .stSelectbox label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        border: 2px solid #6366F1 !important;
    }

    div[data-baseweb="select"] * {
        color: #0F172A !important;
        background-color: transparent !important;
    }

    /* Selectbox Dropdown Menu Popup Overlay */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] * {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
    }

    /* Checkbox & Radio Labels */
    .stCheckbox label span, .stRadio label span {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* ── Number Input & Slider High-Contrast Modern Styling ── */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] input {
        color: #0F172A !important;
        font-weight: 700 !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #334155 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
    }

    /* Form Container Polish */
    div[data-testid="stForm"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 24px 28px !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.03) !important;
    }

    /* ── SIDEBAR STYLING & INTERACTION ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #4F46E5 0%, #4338CA 40%, #312E81 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.15) !important;
    }

    /* General typography in sidebar: do NOT target div or span inside inputs */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #FFFFFF !important;
    }

    /* Interactive Navigation Buttons */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 14px !important;
        width: 100% !important;
        padding: 11px 16px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        text-align: left !important;
        margin-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Inactive Nav Buttons (secondary/default) */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.10) !important;
        color: rgba(255, 255, 255, 0.90) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
        background: rgba(255, 255, 255, 0.24) !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.50) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.20) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:active,
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):active {
        transform: scale(0.97) translateX(3px) !important;
    }

    /* ACTIVE Nav Button (primary) - High Contrast & Elevated */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #FFFFFF !important;
        color: #312E81 !important;
        font-weight: 800 !important;
        border: 1px solid #FFFFFF !important;
        border-left: 5px solid #4F46E5 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28), 0 0 0 2px rgba(255, 255, 255, 0.5) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #FFFFFF !important;
        color: #312E81 !important;
        transform: translateX(6px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), 0 0 0 2px rgba(255, 255, 255, 0.8) !important;
    }

    /* ── SIDEBAR SELECTBOX (LANGUAGE SELECTION FIX) ─────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 4px !important;
    }

    /* The select container itself */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"]:hover {
        border-color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    /* ALL text, labels, and spans inside sidebar selectbox MUST be dark & bold */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] p,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] div,
    [data-testid="stSidebar"] div[data-baseweb="select"] input {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        background-color: transparent !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* Popover menu dropdown items attached to document body */
    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] div,
    div[data-baseweb="menu"] span,
    div[data-baseweb="menu"] li {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Dropdown arrow icon */
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #4338CA !important;
        color: #4338CA !important;
    }

    /* Checkbox inside sidebar */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        transition: transform 0.15s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"]:hover {
        transform: translateX(2px) !important;
    }

    /* Interactive Stats Card Hover Lift */
    .sidebar-stat-card {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 12px;
        padding: 12px 10px;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: default;
    }
    .sidebar-stat-card:hover {
        background: rgba(255, 255, 255, 0.22);
        border-color: rgba(255, 255, 255, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.20);
    }

    /* Card Containers */
    .fitness-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid #E2E8F0;
    }

    .metric-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 16px rgba(15, 23, 42, 0.05);
        border: 1px solid #E2E8F0;
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-top: 6px;
    }

    /* Live Header Badges */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FEF2F2;
        color: #DC2626 !important;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .5px;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #DC2626;
        animation: pulse-dot 1.2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: .5; transform: scale(1.4); }
    }

    .stat-pill {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .stat-pill .val { font-size: 1.6rem; font-weight: 800; color: #0F172A !important; }
    .stat-pill .lbl { font-size: 11px; color: #64748B !important; font-weight: 700; text-transform: uppercase; }

    /* High-Contrast Feedback Cards */
    .feedback-good {
        background: #F0FDF4 !important;
        border-left: 6px solid #16A34A !important;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #BBF7D0;
    }
    .feedback-warning {
        background: #FFFBEB !important;
        border-left: 6px solid #D97706 !important;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #FDE68A;
    }
    .feedback-bad {
        background: #FEF2F2 !important;
        border-left: 6px solid #DC2626 !important;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #FECACA;
    }

    .fb-title {
        font-size: 17px !important;
        font-weight: 800 !important;
        margin: 0 0 6px !important;
    }
    .fb-sub {
        font-size: 13px !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        margin: 0 0 10px !important;
    }
    .fb-item {
        font-size: 13px !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        margin: 5px 0 !important;
        display: flex;
        gap: 8px;
    }

    .trophy-banner {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        border-radius: 24px;
        padding: 28px;
        text-align: center;
        color: #FFFFFF !important;
        margin-bottom: 20px;
    }
    .trophy-banner h1, .trophy-banner p { color: #FFFFFF !important; }

    .page-title { font-size: 2rem; font-weight: 900; color: #0F172A !important; margin: 0 0 4px; }
    .page-sub   { font-size: 14px; color: #475569 !important; margin: 0 0 24px; font-weight: 500; }

    .no-person {
        background: #FEF2F2;
        border: 2px dashed #FECACA;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: #DC2626 !important;
        font-weight: 700;
        font-size: 14px;
    }

    /* ════════════════════════════════════════
       PHASE 5 — EXTENDED DESIGN SYSTEM
    ════════════════════════════════════════ */

    /* ── KPI Cards ───────────────────────── */
    .kpi-card {
        background:#FFFFFF;border-radius:16px;padding:18px 16px;
        border:1px solid #E2E8F0;
        box-shadow:0 2px 8px rgba(15,23,42,0.04);
        text-align:center;transition:box-shadow 0.2s ease;
    }
    .kpi-card:hover { box-shadow:0 4px 20px rgba(79,70,229,0.10); }
    .kpi-value { font-size:1.9rem;font-weight:900;color:#0F172A;line-height:1;margin:6px 0; }
    .kpi-label { font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:.5px; }

    /* ── Hero Banner ─────────────────────── */
    .hero-banner {
        background:linear-gradient(135deg,#EEF2FF 0%,#F0F9FF 45%,#F4F6FC 100%);
        border:1px solid #C7D2FE;border-radius:24px;
        padding:36px 40px 32px;margin-bottom:28px;
    }
    .hero-title { font-size:1.85rem;font-weight:900;color:#0F172A;line-height:1.25;margin:0 0 10px; }
    .hero-title span { color:#4F46E5; }
    .hero-sub { font-size:14px;color:#475569;font-weight:500;line-height:1.7;max-width:580px;margin-bottom:20px; }

    /* ── Insight Box ─────────────────────── */
    .insight-box {
        background:linear-gradient(135deg,#F5F3FF 0%,#FFFFFF 100%);
        border:1px solid #DDD6FE;border-radius:16px;padding:20px 24px;
    }
    .insight-title { font-size:14px;font-weight:800;color:#4F46E5;margin-bottom:10px;
                     display:flex;align-items:center;gap:8px; }

    /* ── Recommendation Cards ────────────── */
    .rec-card {
        background:#FFFFFF;border-radius:16px;padding:20px 24px;
        border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(15,23,42,0.04);
        margin-bottom:14px;
    }
    .rec-icon-box {
        width:44px;height:44px;border-radius:12px;
        display:flex;align-items:center;justify-content:center;
        font-size:1.4rem;flex-shrink:0;
    }
    .rec-action {
        background:#F0FDF4;border-left:3px solid #16A34A;
        border-radius:8px;padding:10px 14px;margin-top:10px;
    }

    /* ── Chart Container ─────────────────── */
    .chart-wrap {
        background:#FFFFFF;border-radius:16px;padding:4px;
        border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(15,23,42,0.04);
    }

    /* ── Section Headers ─────────────────── */
    .sec-head {
        font-size:1rem;font-weight:800;color:#0F172A;
        margin:24px 0 14px;display:flex;align-items:center;gap:8px;
    }
    .sec-sub { font-size:13px;color:#64748B;font-weight:500;margin:-8px 0 14px; }

    /* ── Lifestyle Snapshot Pills ─────────── */
    .snap-pill {
        background:#FFFFFF;border-radius:14px;padding:14px 16px;
        text-align:center;border:1px solid #E2E8F0;
        box-shadow:0 2px 6px rgba(15,23,42,0.04);
    }
    .snap-val { font-size:1.5rem;font-weight:900;color:#0F172A; }
    .snap-lbl { font-size:10px;color:#64748B;font-weight:700;
                text-transform:uppercase;letter-spacing:.5px;margin-top:4px; }

    /* ── Dimension Bars ──────────────────── */
    .dim-row { padding:10px 0;border-bottom:1px solid #F1F5F9; }
    .dim-bar { height:6px;background:#E2E8F0;border-radius:3px;overflow:hidden;margin-top:5px; }
    .dim-fill { height:100%;border-radius:3px; }

    /* ── Tab Styling ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background:#F8FAFC !important;border-radius:12px !important;
        padding:4px !important;gap:2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:10px !important;font-weight:600 !important;
        color:#475569 !important;padding:8px 16px !important;font-size:13px !important;
    }
    .stTabs [aria-selected="true"] {
        background:#4F46E5 !important;color:#FFFFFF !important;
    }

    /* ── Metric Widget Polish ────────────── */
    [data-testid="stMetricValue"] { color:#0F172A !important;font-weight:800 !important; }
    [data-testid="stMetricLabel"] { color:#64748B !important;font-weight:600 !important; }

    /* ── Wellness score big display ──────── */
    .ws-score-big {
        font-size:5rem;font-weight:900;line-height:1;
    }

    /* ── Sidebar group label ─────────────── */
    .sidebar-group-label {
        font-size:10px;font-weight:800;color:rgba(255,255,255,0.55);
        text-transform:uppercase;letter-spacing:.8px;padding:14px 0 4px;
    }

    /* ── Predict Score Page Specific Styling ─────────── */
    .nova-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 24px;
    }
    .nova-title-area h1 {
        font-size: 2.1rem;
        font-weight: 900;
        color: #0F172A !important;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nova-title-area p {
        font-size: 0.95rem;
        color: #64748B !important;
        margin: 6px 0 0;
        max-width: 580px;
        line-height: 1.5;
        font-weight: 500;
    }
    .nova-greeting-card {
        background: #F0F7FF;
        border: 1.5px solid #D0E5FF;
        border-radius: 20px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.08);
    }
    .nova-bubble-text {
        font-size: 12.5px;
        color: #1E293B !important;
        line-height: 1.45;
        max-width: 270px;
    }
    .nova-bubble-text strong {
        color: #0F172A !important;
        display: block;
        font-size: 13.5px;
        margin-bottom: 2px;
    }
    .nova-avatar-circle {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        border: 2.5px solid #38BDF8;
        object-fit: cover;
        box-shadow: 0 2px 12px rgba(56, 189, 248, 0.35);
    }
    .user-profile-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
        color: #0F172A;
        background: #FFFFFF;
        padding: 5px 12px;
        border-radius: 9999px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(15,23,42,0.04);
        margin-bottom: 6px;
    }
    
    /* ── Metric Inputs Card ── */
    .metric-inputs-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 24px 28px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.03);
        margin-bottom: 24px;
    }
    .card-top-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A !important;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }
    .card-top-subtitle {
        font-size: 0.88rem;
        color: #64748B !important;
        margin-bottom: 18px;
    }
    .pillar-pill {
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 700;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .pillar-pill.personal { background: #6366F1; }
    .pillar-pill.activity { background: #8B5CF6; }
    .pillar-pill.recovery { background: #3B82F6; }
    .pillar-pill.wellness { background: #A855F7; }

    /* ── Results Cards ── */
    .wellness-result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card-kicker {
        font-size: 11px;
        font-weight: 800;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .score-nova-row {
        display: flex;
        align-items: center;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 16px;
        flex: 1;
    }
    .nova-speech-bubble-dyn {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        font-size: 12.5px;
        color: #334155 !important;
        line-height: 1.45;
        position: relative;
        max-width: 220px;
        margin-bottom: 10px;
        text-align: left;
    }
    .nova-speech-bubble-dyn::after {
        content: '';
        position: absolute;
        bottom: -7px;
        left: 30px;
        width: 0;
        height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-top: 7px solid #FFFFFF;
    }
    .nova-speech-bubble-dyn strong {
        display: block;
        color: #0F172A !important;
        font-size: 13px;
        margin-bottom: 2px;
    }
    .nova-gif-wrap img {
        width: 195px;
        height: 195px;
        object-fit: contain;
        filter: drop-shadow(0 8px 18px rgba(99,102,241,0.15));
        mix-blend-mode: multiply;
        transition: transform 0.3s ease;
    }
    .nova-gif-wrap img:hover {
        transform: scale(1.03);
    }

    /* ── Metric Strip Card ── */
    .metric-strip-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.03);
    }
    .metric-strip-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .metric-strip-left {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
        color: #334155 !important;
    }
    .metric-strip-score {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A !important;
    }
    .metric-strip-score span {
        font-size: 0.78rem;
        color: #64748B !important;
        font-weight: 500;
    }
    .metric-strip-bar-bg {
        width: 100%;
        height: 6px;
        background: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }
    .metric-strip-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* ── Bottom Insight Cards ── */
    .bottom-insight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(15,23,42,0.03);
        height: 100%;
    }
    .bottom-card-title {
        font-size: 14px;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .driver-bar-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
        font-size: 12px;
    }
    .driver-label {
        width: 90px;
        color: #334155 !important;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .driver-val {
        width: 32px;
        text-align: right;
        font-weight: 700;
        color: #0F172A !important;
    }
    .driver-track {
        flex: 1;
        height: 7px;
        background: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }
    .driver-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* ── Analyze button gradient polish ── */
    div[data-testid="stForm"] button[kind="primary"] {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%) !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        border-radius: 12px !important;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        transform: translateY(-1px) !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Session State
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "page":              "home",
        "selected_exercise": "Utkatasana" if "Utkatasana" in SUPPORTED_EXERCISES else SUPPORTED_EXERCISES[0],
        "workout_active":    False,
        "cap":               None,
        "rep_counter":       None,
        "live_score":        0.0,
        "live_feedback":     [],
        "live_angles":       [],
        "workout_saved":     False,
        "workout_summary":   {},
        "uploaded_image":    None,
        "voice_lang":        "en",
        "voice_enabled":     True,
        "voice_coach":       VoiceCoach(language="en", is_enabled=True),
        "test_speech_html":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def nav_to(page: str):
    st.session_state.page = page


def _safe_close_cam():
    if st.session_state.get("cap") is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None
    st.session_state.workout_active = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar Navigation & Voice Coach Controls
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # ── Brand header ─────────────────────────
        current_page = st.session_state.get("page", "home")
        st.markdown("""
        <div style='padding:20px 16px 16px;border-bottom:1px solid rgba(255,255,255,0.15);margin-bottom:4px;'>
          <div style='display:flex;align-items:center;gap:10px;'>
            <div style='background:rgba(255,255,255,0.2);border-radius:10px;
                        width:38px;height:38px;display:flex;align-items:center;
                        justify-content:center;font-size:18px;'>🧠</div>
            <div>
              <div style='font-size:13px;font-weight:800;color:#fff;'>AI Wellness Intelligence</div>
              <div style='font-size:11px;color:rgba(255,255,255,0.65);'>Fitness + Analytics Platform</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Main navigation ───────────────────────
        st.markdown("<div style='padding:8px 0 0;'>", unsafe_allow_html=True)

        nav_items = [
            ("🏠",  "Dashboard",              "home"),
            ("📊",  "Wellness Intelligence",  "wellness_dashboard"),
            ("🏃",  "AI Fitness Coach",       "fitness_coach"),
            ("🧘",  "Yoga & Pose Analysis",   "yoga"),
            ("📈",  "Progress & History",     "history"),
            ("💡",  "Recommendations",        "recommendations"),
            ("🔮",  "Predict My Score",       "predict_score"),
        ]

        for icon, label, page_key in nav_items:
            is_active = (current_page == page_key)
            prefix    = "● " if is_active else "  "
            btn_lbl   = f"{prefix}{icon}  {label}"
            if st.button(
                btn_lbl,
                use_container_width=True,
                key=f"nav_{page_key}",
                type="primary" if is_active else "secondary"
            ):
                _safe_close_cam()
                nav_to(page_key)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 8px 14px;border-color:rgba(255,255,255,0.15);'>",
                    unsafe_allow_html=True)

        # ── Voice Coach controls ──────────────────
        st.markdown("""
        <div style='font-size:10px;font-weight:800;color:rgba(255,255,255,0.65);
                    text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;'>
          🔊 Voice Coach
        </div>
        """, unsafe_allow_html=True)

        lang_options = {
            "🇬🇧 English": "en",
            "🇮🇳 Tamil":   "ta",
        }
        current_lang = st.session_state.get("voice_lang", "en")
        default_idx  = 1 if current_lang == "ta" else 0

        selected_label = st.selectbox(
            "Language:", options=list(lang_options.keys()),
            index=default_idx, key="voice_lang_select",
        )
        new_lang = lang_options[selected_label]
        if new_lang != st.session_state.voice_lang:
            st.session_state.voice_lang = new_lang
            if "voice_coach" in st.session_state and st.session_state.voice_coach:
                st.session_state.voice_coach.set_language(new_lang)

        voice_on = st.checkbox(
            "Enable Voice", value=st.session_state.get("voice_enabled", True),
            key="voice_enabled_chk",
        )
        st.session_state.voice_enabled = voice_on
        if "voice_coach" in st.session_state and st.session_state.voice_coach:
            st.session_state.voice_coach.set_enabled(voice_on)

        if st.button("🔊 Test Voice", use_container_width=True, key="btn_test_voice"):
            phrase = "Lift your arm higher." if new_lang == "en" else "உங்கள் கையை மேலே உயர்த்தவும்."
            audio_bytes = get_voice_audio_bytes(phrase, new_lang)
            st.session_state.test_speech_data = (audio_bytes, phrase, new_lang)

        if st.session_state.get("test_speech_data"):
            a_bytes, phrase_text, p_lang = st.session_state.test_speech_data
            if a_bytes:
                st.markdown(
                    f"<div style='font-size:11px;color:#FFFFFF;margin:4px 0 2px;font-weight:600;'>"
                    f"🔊 {phrase_text}</div>",
                    unsafe_allow_html=True
                )
                play_audio(st, a_bytes, autoplay=True)
                # Web Speech API fallback in background to ensure sound is heard out loud
                st.components.v1.html(get_voice_audio_html(phrase_text, p_lang, show_controls=False), height=0)

        st.markdown("<hr style='margin:14px 8px;border-color:rgba(255,255,255,0.15);'>",
                    unsafe_allow_html=True)

        # ── Platform stats ────────────────────────
        stats = get_aggregate_stats()
        st.markdown(f"""
        <div style='padding:0 4px 16px;'>
          <div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.55);
                      text-transform:uppercase;margin-bottom:10px;letter-spacing:.5px;'>
            Platform Stats
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
            <div class='sidebar-stat-card'>
              <div style='font-size:1.4rem;font-weight:900;color:#fff;'>{len(SUPPORTED_EXERCISES)}</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.75);font-weight:600;'>Pose Classes</div>
            </div>
            <div class='sidebar-stat-card'>
              <div style='font-size:1.4rem;font-weight:900;color:#fff;'>{stats.get("total_reps", 0)}</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.75);font-weight:600;'>Total Reps</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def _score_color(score: float) -> str:
    if score >= 75:   return "#16A34A"
    if score >= 50:   return "#D97706"
    return "#DC2626"


def _donut_chart(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[value, 100 - value], hole=0.72,
        marker_colors=[color, "#E2E8F0"], textinfo="none", hoverinfo="skip",
    ))
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{value:.0f}%</b>", x=0.5, y=0.5, font_size=22, showarrow=False,
            font_color="#0F172A",
        )],
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        height=150, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1a — FITNESS COACH (exercise selector, the original home page)
# ═══════════════════════════════════════════════════════════════════════════════
def render_fitness_coach():
    """Exercise selector & AI model info — original home page content."""
    col_title, col_start = st.columns([3, 1])
    with col_title:
        st.markdown("<div class='page-title'>🏃 AI Fitness Coach</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-sub'>Select an exercise and start the live webcam session with real-time AI pose coaching.</div>",
                    unsafe_allow_html=True)
    with col_start:
        if st.button("▶  Start Workout", type="primary", use_container_width=True):
            nav_to("workout"); st.rerun()

    stats = get_aggregate_stats()
    total_min = (stats.get("total_seconds", 0) or 0) // 60
    avg_acc   = stats.get("avg_accuracy", 0) or 0

    s1, s2, s3, s4 = st.columns(4)
    for col, icon, val, lbl in [
        (s1, "🧘", len(SUPPORTED_EXERCISES),   "Pose Classes"),
        (s2, "🔁", stats.get("total_reps", 0), "Total Reps"),
        (s3, "⏱️", f"{total_min}m",             "Workout Time"),
        (s4, "✅", f"{avg_acc:.0f}%",            "Avg Accuracy"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style='font-size:1.5rem;margin-bottom:4px;'>{icon}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown(f"<div class='sec-head'>🎯 Select Pose / Exercise ({len(SUPPORTED_EXERCISES)} Available)</div>",
                    unsafe_allow_html=True)

        disp_options = {POSE_DISPLAY_NAMES.get(ex, ex): ex for ex in SUPPORTED_EXERCISES}
        selected_disp = st.selectbox(
            "Search or select a pose:",
            options=list(disp_options.keys()),
            index=0,
        )
        st.session_state.selected_exercise = disp_options[selected_disp]

        ex   = st.session_state.selected_exercise
        info = EXERCISE_INFO[ex]

        asset_img_name = ex.replace(" ", "_").lower() + ".jpg"
        asset_img_path = os.path.join(ASSETS_DIR, asset_img_name)

        st.markdown("<br>", unsafe_allow_html=True)
        c_img, c_meta = st.columns([1, 2])
        with c_img:
            if os.path.exists(asset_img_path):
                st.image(asset_img_path, caption=info["display_name"], use_container_width=True)
            else:
                st.markdown("<div style='font-size:5rem;text-align:center;'>🧘</div>", unsafe_allow_html=True)
        with c_meta:
            st.markdown(f"""
            <div class='fitness-card' style='border-top:4px solid #4F46E5;'>
              <div style='font-size:20px;font-weight:900;color:#0F172A;'>{info["display_name"]}</div>
              <div style='font-size:12px;color:#4338CA;font-weight:700;margin:4px 0 10px;'>{info["category"]}</div>
              <div style='font-size:13px;color:#334155;line-height:1.5;'>{info["description"]}</div>
              <div style='margin-top:12px;font-size:13px;font-weight:700;color:#3730A3;background:#EEF2FF;
                          padding:6px 12px;border-radius:10px;display:inline-block;'>
                {info["sets"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button(f"▶  Start {info['display_name']} Workout", type="primary", use_container_width=True):
            nav_to("workout"); st.rerun()

    with right_col:
        st.markdown(f"""
        <div class="fitness-card">
          <div style='font-size:1rem;font-weight:800;color:#0F172A;margin-bottom:10px;'>🤖 AI Model</div>
          <div style='font-size:13px;color:#334155;line-height:1.6;'>
            <strong style='color:#0F172A;'>Classifier:</strong> Random Forest<br>
            <strong style='color:#0F172A;'>Dataset:</strong> Kaggle Yoga Postures<br>
            <strong style='color:#0F172A;'>Pose Classes:</strong> {len(SUPPORTED_EXERCISES)}<br>
            <strong style='color:#0F172A;'>Features:</strong> 8 3D Joint Angles<br>
            <strong style='color:#0F172A;'>Landmarks:</strong> MediaPipe 33 KP
          </div>
        </div>
        """, unsafe_allow_html=True)

        recent = get_all_workouts()[:3]
        if recent:
            st.markdown("<div class='sec-head' style='margin-top:16px;'>🕐 Recent Sessions</div>",
                        unsafe_allow_html=True)
            for w in recent:
                acc   = w.get("form_accuracy", 0) or 0
                color = _score_color(acc)
                st.markdown(f"""
                <div style='background:#fff;border-radius:12px;padding:10px 14px;
                            margin-bottom:8px;border:1px solid #E2E8F0;
                            display:flex;justify-content:space-between;align-items:center;'>
                  <div>
                    <div style='font-size:13px;font-weight:700;color:#0F172A;'>🧘 {w["exercise"]}</div>
                    <div style='font-size:11px;color:#64748B;'>{w["date"]}</div>
                  </div>
                  <div style='text-align:right;'>
                    <div style='font-size:14px;font-weight:800;color:{color};'>{acc:.0f}%</div>
                    <div style='font-size:11px;color:#64748B;'>{w["total_reps"]} reps</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1b — PLATFORM DASHBOARD (new home page)
# ═══════════════════════════════════════════════════════════════════════════════
def render_home():
    """Platform landing dashboard — wellness KPIs, trend chart, AI insight."""
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">
        AI-Powered Fitness &amp; Lifestyle<br>
        <span>Wellness Intelligence</span>
      </div>
      <div class="hero-sub">
        Real-time AI fitness coaching &middot; Long-term lifestyle analytics &middot;
        Personalised wellness insights — all in one platform.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-start row ──────────────────────────────────────────────────────
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("🏃  Start Fitness Session", use_container_width=True, type="primary"):
            nav_to("fitness_coach"); st.rerun()
    with qc2:
        if st.button("📊  Wellness Analytics", use_container_width=True):
            nav_to("wellness_dashboard"); st.rerun()
    with qc3:
        if st.button("🔮  Predict My Score", use_container_width=True):
            nav_to("predict_score"); st.rerun()
    with qc4:
        if st.button("💡  Recommendations", use_container_width=True):
            nav_to("recommendations"); st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load analytics data ──────────────────────────────────────────────────
    ws = None
    if _ANALYTICS_OK:
        try:
            ws, _ = _load_wellness_data()
        except Exception:
            ws = None

    stats      = get_aggregate_stats()
    total_min  = (stats.get("total_seconds", 0) or 0) // 60
    avg_acc    = stats.get("avg_accuracy",   0) or 0

    # ── KPI cards ────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-head'>📊 Platform Overview</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    if ws is not None:
        avg_ws   = ws["Wellness_Score"].mean()
        avg_slp  = ws["Hours_Slept"].mean()
        avg_step = ws["Steps_Taken"].mean()
        avg_str  = ws["Stress_Level (1-10)"].mean()
        avg_h2o  = ws["Water_Intake (Liters)"].mean()
        kpi_data = [
            (k1, "🧠", f"{avg_ws:.0f}",        "Wellness Score", "#4F46E5"),
            (k2, "🏃", f"{avg_step:,.0f}",      "Daily Steps",    "#0891B2"),
            (k3, "😴", f"{avg_slp:.1f}h",       "Avg Sleep",      "#6366F1"),
            (k4, "😤", f"{avg_str:.1f}/10",      "Stress Level",   "#E11D48"),
            (k5, "💧", f"{avg_h2o:.1f}L",        "Hydration",      "#059669"),
        ]
    else:
        kpi_data = [
            (k1, "🧘", len(SUPPORTED_EXERCISES),    "Pose Classes",  "#4F46E5"),
            (k2, "🔁", stats.get("total_reps", 0),  "Total Reps",    "#0891B2"),
            (k3, "⏱️", f"{total_min}m",              "Workout Time",  "#6366F1"),
            (k4, "✅", f"{avg_acc:.0f}%",            "Avg Accuracy",  "#16A34A"),
            (k5, "📊", "—",                          "Wellness Score","#E11D48"),
        ]

    for col, icon, val, lbl, clr in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div style="font-size:1.6rem;">{icon}</div>
              <div class="kpi-value" style="color:{clr};">{val}</div>
              <div class="kpi-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Wellness trend + Today's summary ─────────────────────────────────────
    trend_col, summary_col = st.columns([3, 2])

    with trend_col:
        st.markdown("<div class='sec-head'>📈 Wellness Score Trend (Monthly Average)</div>",
                    unsafe_allow_html=True)
        if ws is not None and _ANALYTICS_OK:
            trend = (ws.groupby(ws["Date"].dt.month)["Wellness_Score"]
                       .mean().reset_index())
            trend.columns = ["Month", "Avg Wellness"]
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            trend["Month"] = trend["Month"].map(month_names)
            fig = px.line(trend, x="Month", y="Avg Wellness",
                          markers=True, line_shape="spline",
                          color_discrete_sequence=["#4F46E5"])
            fig.update_traces(line_width=3, marker_size=8,
                              hovertemplate="%{x}: %{y:.1f}<extra></extra>")
            fig.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                font=dict(color="#0F172A", family="Inter"),
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                           tickfont=dict(color="#475569")),
                yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                           tickfont=dict(color="#475569"), range=[50, 90],
                           title="Wellness Score"),
                showlegend=False,
            )
            st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key="home_trend")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chart-wrap" style="height:280px;display:flex;align-items:center;justify-content:center;">
              <div style="text-align:center;color:#94A3B8;">
                <div style="font-size:2.5rem;">📊</div>
                <div style="font-size:13px;font-weight:600;margin-top:8px;">
                  Run analytics pipeline to see trend
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with summary_col:
        st.markdown("<div class='sec-head'>🎯 Wellness Summary</div>", unsafe_allow_html=True)
        if ws is not None:
            avg_ws   = ws["Wellness_Score"].mean()
            avg_slp  = ws["Hours_Slept"].mean()
            avg_step = ws["Steps_Taken"].mean()
            avg_str  = ws["Stress_Level (1-10)"].mean()
            avg_h2o  = ws["Water_Intake (Liters)"].mean()

            ws_clr    = "#16A34A" if avg_ws >= 70 else "#D97706" if avg_ws >= 55 else "#DC2626"
            ws_status = "Good" if avg_ws >= 70 else "Fair" if avg_ws >= 55 else "Needs Attention"

            strong, weak = [], []
            if avg_slp  >= 7:    strong.append("Sleep")
            else:                weak.append("Sleep")
            if avg_step >= 7000: strong.append("Activity")
            else:                weak.append("Activity")
            if avg_str  <= 5:    strong.append("Stress Mgmt")
            else:                weak.append("Stress")
            if avg_h2o  >= 2.0:  strong.append("Hydration")
            else:                weak.append("Hydration")

            s_html = "".join([
                f"<div style='font-size:12px;color:#16A34A;font-weight:600;margin:2px 0;'>✓ {s}</div>"
                for s in strong[:2]
            ])
            w_html = "".join([
                f"<div style='font-size:12px;color:#D97706;font-weight:600;margin:2px 0;'>• {w}</div>"
                for w in weak[:2]
            ])
            st.markdown(f"""
            <div class="insight-box">
              <div style="text-align:center;padding:12px 0 16px;">
                <div style="font-size:3.5rem;font-weight:900;color:{ws_clr};line-height:1;">{avg_ws:.0f}</div>
                <div style="font-size:13px;color:#475569;font-weight:600;margin-top:4px;">
                  Population Average · 36,500 records
                </div>
                <div style="display:inline-block;background:#EEF2FF;color:{ws_clr};
                            font-size:11px;font-weight:700;padding:3px 12px;border-radius:20px;margin-top:6px;">
                  {ws_status}
                </div>
              </div>
              <div style="border-top:1px solid #E2E8F0;padding-top:12px;">
                <div style="font-size:12px;font-weight:700;color:#0F172A;margin-bottom:6px;">Strong Areas</div>
                {s_html}
                <div style="font-size:12px;font-weight:700;color:#0F172A;margin:10px 0 6px;">Opportunities</div>
                {w_html}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box" style="text-align:center;padding:40px 24px;">
              <div style="font-size:2.5rem;">🧠</div>
              <div style="font-size:14px;font-weight:600;color:#0F172A;margin-top:8px;">No wellness data</div>
              <div style="font-size:12px;color:#64748B;margin-top:4px;">
                Run the analytics pipeline first.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Lifestyle snapshot ────────────────────────────────────────────────────
    if ws is not None:
        st.markdown("<div class='sec-head'>🏃 Lifestyle Snapshot (Dataset Averages)</div>",
                    unsafe_allow_html=True)
        ls1, ls2, ls3, ls4 = st.columns(4)
        for col, icon, val, lbl in [
            (ls1, "👣", f"{ws['Steps_Taken'].mean():,.0f}",       "Steps / Day"),
            (ls2, "⏱️", f"{ws['Active_Minutes'].mean():.0f}m",    "Active / Day"),
            (ls3, "🔥", f"{ws['Calories_Burned'].mean():.0f}",    "Calories / Day"),
            (ls4, "💓", f"{ws['Heart_Rate (bpm)'].mean():.0f}",   "Avg Heart Rate"),
        ]:
            with col:
                st.markdown(f"""
                <div class="snap-pill">
                  <div style="font-size:1.4rem;">{icon}</div>
                  <div class="snap-val">{val}</div>
                  <div class="snap-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── AI Insight ────────────────────────────────────────────────────────────
    if ws is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        avg_ws   = ws["Wellness_Score"].mean()
        avg_slp  = ws["Hours_Slept"].mean()
        avg_step = ws["Steps_Taken"].mean()
        avg_str  = ws["Stress_Level (1-10)"].mean()
        avg_h2o  = ws["Water_Intake (Liters)"].mean()

        strong_list, improve_list = [], []
        if avg_slp  >= 7:    strong_list.append("Sleep duration is adequate")
        else:                improve_list.append(f"Average sleep ({avg_slp:.1f}h) is below the 7h target")
        if avg_step >= 7000: strong_list.append("Daily activity level is healthy")
        else:                improve_list.append(f"Average steps ({avg_step:,.0f}) below 7,000 target")
        if avg_h2o  >= 2.0:  strong_list.append("Hydration is on track")
        else:                improve_list.append(f"Water intake ({avg_h2o:.1f}L) below 2.0L target")
        if avg_str  <= 5:    strong_list.append("Stress levels are manageable")
        else:                improve_list.append(f"Stress level ({avg_str:.1f}/10) is elevated")

        s_items = "".join([
            f"<div style='margin:3px 0;font-size:13px;color:#16A34A;'>✓ {s}</div>"
            for s in strong_list
        ])
        i_items = "".join([
            f"<div style='margin:3px 0;font-size:13px;color:#D97706;'>• {s}</div>"
            for s in improve_list
        ])

        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-title">🧠 Today's Wellness Insight</div>
          <div style="font-size:13px;color:#475569;margin-bottom:14px;">
            Population average wellness score is
            <strong style="color:#4F46E5;">{avg_ws:.0f}/100</strong> —
            based on {len(ws):,} daily records across {ws["User_ID"].nunique()} users.
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div>
              <div style="font-size:12px;font-weight:700;color:#0F172A;margin-bottom:6px;">Strong Areas</div>
              {s_items}
            </div>
            <div>
              <div style="font-size:12px;font-weight:700;color:#0F172A;margin-bottom:6px;">Opportunities</div>
              {i_items}
            </div>
          </div>
          <div style="margin-top:14px;font-size:11px;color:#94A3B8;border-top:1px solid #E2E8F0;padding-top:10px;">
            📊 Kaggle Comprehensive Fitness &amp; Health Tracking Dataset · 36,500 records · 100 users
          </div>
        </div>
        """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — LIVE WORKOUT
# ═══════════════════════════════════════════════════════════════════════════════
def render_workout():
    ex_name = st.session_state.selected_exercise
    info    = EXERCISE_INFO.get(ex_name, {"display_name": ex_name, "rep_type": "reps"})
    disp_title = info.get("display_name", ex_name)
    lang    = st.session_state.get("voice_lang", "en")
    v_on    = st.session_state.get("voice_enabled", True)

    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div style='font-size:1.2rem;font-weight:800;color:#0F172A;'>🏋️ Live Workout — {disp_title}</div>",
                    unsafe_allow_html=True)
    with col_badge:
        voice_badge_lbl = "🔊 Voice: ON" if v_on else "🔇 Voice: MUTED"
        st.markdown(f"<span class='live-badge'><span class='live-dot'></span>LIVE</span> &nbsp; <span style='font-size:12px;font-weight:700;color:#4338CA;background:#EEF2FF;padding:4px 8px;border-radius:8px;'>{voice_badge_lbl} ({'தமிழ்' if lang=='ta' else 'English'})</span>",
                    unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        run_cam = st.checkbox("🔴 Start Camera Feed", value=True, key="cam_active_chk")
    with ctrl2:
        if st.button("⏹ End Workout", type="primary", use_container_width=True):
            _end_workout()
            return

    p1, p2, p3, p4 = st.columns(4)
    rep_p   = p2.empty()
    time_p  = p3.empty()
    stage_p = p4.empty()

    p1.markdown(f"<div class='stat-pill'><div class='val'>{ex_name}</div><div class='lbl'>EXERCISE</div></div>",
                unsafe_allow_html=True)

    vid_col, info_col = st.columns([3, 2])
    frame_placeholder    = vid_col.empty()
    analysis_placeholder = info_col.empty()
    voice_placeholder    = st.empty()

    if not run_cam:
        frame_placeholder.markdown("<div class='no-person'>Camera paused. Check checkbox to resume.</div>",
                                   unsafe_allow_html=True)
        return

    if st.session_state.cap is None:
        try:
            cap = get_webcam(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Grant camera permission.")
                return
            st.session_state.cap = cap
        except Exception as e:
            st.error(f"Camera error: {e}")
            return

    if st.session_state.rep_counter is None:
        with st.spinner("⏳ Loading PoseLandmarker model…"):
            get_detector()
        st.session_state.rep_counter    = RepCounter(ex_name, info)
        st.session_state.workout_active = True

    cap = st.session_state.cap
    rc: RepCounter = st.session_state.rep_counter
    vc: VoiceCoach = st.session_state.get("voice_coach", VoiceCoach(language=lang, is_enabled=v_on))
    vc.set_language(lang)
    vc.set_enabled(v_on)

    prev_reps = rc.reps

    while st.session_state.workout_active and run_cam:
        ret, frame_bgr = cap.read()
        if not ret:
            frame_placeholder.markdown("<div class='no-person'>Frame capture failed.</div>", unsafe_allow_html=True)
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            results   = detect_pose(frame_rgb)
            landmarks = extract_keypoints(results)
        except Exception:
            landmarks = []

        if landmarks:
            user_angles = calculate_angles(landmarks)
            ref_angles  = get_reference_angles(ex_name)
            feedback, score = provide_specific_feedback(ref_angles, user_angles, None)

            st.session_state.live_score    = score
            st.session_state.live_feedback = feedback
            rc.update(user_angles, score)

            # Check if rep count incremented
            if rc.reps > prev_reps:
                prev_reps = rc.reps
                if vc.can_speak():
                    rep_msg = f"Rep {rc.reps} completed!" if lang == "en" else f"சுற்று {rc.reps} முடிந்தது!"
                    a_bytes = vc.trigger_speech_bytes(rep_msg)
                    if a_bytes:
                        play_audio(voice_placeholder, a_bytes, autoplay=True)
            elif vc.can_speak():
                en_cmd, loc_cmd = vc.extract_voice_command(feedback, score)
                if loc_cmd:
                    a_bytes = vc.trigger_speech_bytes(loc_cmd)
                    if a_bytes:
                        play_audio(voice_placeholder, a_bytes, autoplay=True)

            annotated = draw_landmarks_on_frame(frame_rgb, landmarks, form_score=score)
            primary_idx   = info.get("primary_angle_idx", 7)
            primary_angle = get_primary_angle(user_angles, ex_name)
            angle_label   = info.get("primary_angle_label", "Angle")

            if primary_angle is not None:
                annotated = overlay_angle_text(annotated, landmarks, primary_angle, primary_idx, angle_label)

            sc_color = (22, 163, 74) if score >= 75 else (217, 119, 6) if score >= 50 else (220, 38, 38)
            cv2.putText(annotated, f"Form: {score:.0f}%", (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc_color, 2, cv2.LINE_AA)

            frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

            reps_val = rc.reps if info.get("rep_type") == "reps" else "—"
            rep_p.markdown(f"<div class='stat-pill'><div class='val'>{reps_val}</div><div class='lbl'>{get_ui_text('reps', lang)}</div></div>", unsafe_allow_html=True)
            time_p.markdown(f"<div class='stat-pill'><div class='val'>{rc.elapsed_str}</div><div class='lbl'>{get_ui_text('time', lang)}</div></div>", unsafe_allow_html=True)
            stage_p.markdown(f"<div class='stat-pill'><div class='val'>{rc.stage.upper()}</div><div class='lbl'>{get_ui_text('stage', lang)}</div></div>", unsafe_allow_html=True)

            disp  = get_feedback_display(feedback, score)
            tip_en = get_primary_tip(ex_name, feedback)
            tip_ta = translate_text(tip_en, "ta")
            tip    = tip_ta if lang == "ta" else tip_en

            p_ang = primary_angle or 0

            fb_cls = {"good": "feedback-good", "warning": "feedback-warning", "incorrect": "feedback-bad"}.get(disp["status"], "feedback-warning")

            # Format bilingual improve messages
            imp_items = []
            for m in disp["improve_messages"]:
                m_ta = translate_text(m, "ta")
                disp_m = f"⚠️ {m_ta}<br><span style='font-size:11px;color:#64748B;'>({m})</span>" if lang == "ta" else f"⚠️ {m}<br><span style='font-size:11px;color:#64748B;'>({m_ta})</span>"
                imp_items.append(f"<div class='fb-item'>{disp_m}</div>")
            imp_html = "".join(imp_items)

            good_items = []
            for m in disp["good_messages"][:2]:
                m_ta = translate_text(m, "ta")
                disp_g = f"✅ {m_ta}" if lang == "ta" else f"✅ {m}"
                good_items.append(f"<div class='fb-item'>{disp_g}</div>")
            imp_html += "".join(good_items)

            active_voice_text = vc.last_spoken_phrase or (tip_ta if lang == "ta" else tip_en)
            voice_banner = f"""
            <div style='background:#EEF2FF;border-left:4px solid #4F46E5;border-radius:10px;padding:8px 12px;margin-bottom:10px;display:flex;align-items:center;gap:10px;'>
              <span style='font-size:20px;'>🔊</span>
              <div>
                <div style='font-size:10px;font-weight:800;color:#4338CA;text-transform:uppercase;'>
                  {get_ui_text("current_voice_cue", lang)} ({'தமிழ்' if lang=='ta' else 'English'})
                </div>
                <div style='font-size:13px;font-weight:700;color:#1E1B4B;'>
                  "{active_voice_text}"
                </div>
              </div>
            </div>
            """

            analysis_placeholder.markdown(f"""
            <div style='padding:4px 0;'>
              {voice_banner}
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;'>
                <div class='metric-card'>
                  <div class='metric-value' style='color:{disp["color"]};'>{score:.0f}%</div>
                  <div class='metric-label'>{get_ui_text('form_score', lang)}</div>
                </div>
                <div class='metric-card'>
                  <div class='metric-value'>{p_ang:.0f}°</div>
                  <div class='metric-label'>{angle_label}</div>
                </div>
                <div class='metric-card'>
                  <div class='metric-value' style='color:#16A34A;'>{rc.correct_reps}</div>
                  <div class='metric-label'>{get_ui_text('correct_reps', lang)}</div>
                </div>
                <div class='metric-card'>
                  <div class='metric-value' style='color:#DC2626;'>{rc.incorrect_reps}</div>
                  <div class='metric-label'>{get_ui_text('need_work', lang)}</div>
                </div>
              </div>
              <div class='{fb_cls}'>
                <div class='fb-title' style='color:{disp["color"]};'>{disp["emoji"]} {disp["title"]}</div>
                <div class='fb-sub' style='color:#0F172A !important;'>{tip}</div>
                {imp_html}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            no_person_msg = "🚶 No person detected. Move into camera frame." if lang == "en" else "🚶 கேமராவில் ஆள் தெரியவில்லை. சட்டகத்திற்குள் வரவும்."
            analysis_placeholder.markdown(f"<div class='no-person'>{no_person_msg}</div>",
                                           unsafe_allow_html=True)

        time.sleep(0.01)


def _end_workout():
    rc: RepCounter = st.session_state.rep_counter
    summary = rc.get_summary() if rc else {}
    summary["last_score"]    = st.session_state.live_score
    summary["last_feedback"] = st.session_state.live_feedback

    st.session_state.workout_summary = summary
    st.session_state.workout_active  = False
    st.session_state.workout_saved   = False
    _safe_close_cam()
    nav_to("summary"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — IMAGE UPLOAD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def render_yoga():
    lang = st.session_state.get("voice_lang", "en")
    st.markdown("<div class='page-title'>🧘 Yoga &amp; Pose Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Upload any yoga/fitness photo for AI pose detection, form analysis and multilingual voice coaching.</div>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr_img    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_img is None:
            st.error("Failed to read image file.")
            return

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        with st.spinner("🔍 Running Trained Random Forest Classifier on dataset features..."):
            results   = detect_pose(rgb_img)
            landmarks = extract_keypoints(results)

        if not landmarks:
            st.warning("⚠️ No person detected in the uploaded image. Please upload a clear full-body photo.")
            st.image(rgb_img, caption="Uploaded Image", use_container_width=True)
            return

        user_angles = calculate_angles(landmarks)
        predicted_ex, confidence = predict_exercise_from_angles(user_angles)
        disp_name   = POSE_DISPLAY_NAMES.get(predicted_ex, predicted_ex)
        ref_angles  = get_reference_angles(predicted_ex)
        feedback, score = provide_specific_feedback(ref_angles, user_angles, None)

        annotated = draw_landmarks_on_frame(rgb_img, landmarks, form_score=score)

        left, right = st.columns([3, 2])

        with left:
            st.image(annotated, caption=f"Analyzed Pose Skeleton — Detected: {disp_name}", use_container_width=True)

        with right:
            disp = get_feedback_display(feedback, score)

            # Compute primary correction for audio
            improve_msgs = [msg for _, typ, msg in feedback if typ == "improve"]
            if improve_msgs:
                top_raw = improve_msgs[0]
                parts = top_raw.split(". ")
                top_cue_en = parts[-1].strip() if len(parts) >= 2 else top_raw
                top_cue_ta = translate_text(top_cue_en, "ta")
            else:
                top_cue_en = "Great alignment and posture!"
                top_cue_ta = "அருமையான தோரணை மற்றும் நிலை!"

            st.markdown(f"""
            <div class="fitness-card" style='border-top:4px solid #4F46E5;'>
              <div style='display:flex;align-items:center;gap:12px;margin-bottom:12px;'>
                <div style='font-size:2.5rem;'>🧘</div>
                <div>
                  <div style='font-size:11px;font-weight:800;color:#4338CA;text-transform:uppercase;'>Trained RF Model Prediction</div>
                  <div style='font-size:18px;font-weight:900;color:#0F172A;'>{disp_name}</div>
                  <div style='font-size:13px;color:#334155;'>Dataset Class Match: <strong style='color:#0F172A;'>{confidence}%</strong></div>
                </div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0;'>
                <div class='metric-card'>
                  <div class='metric-value' style='color:{disp["color"]};'>{score:.0f}%</div>
                  <div class='metric-label'>Form Accuracy</div>
                </div>
                <div class='metric-card'>
                  <div class='metric-value'>{score_status_str(score)}</div>
                  <div class='metric-label'>Posture Grade</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Voice Coach Audio Player buttons for Photo Upload
            st.markdown("<div style='font-size:13px;font-weight:800;color:#0F172A;margin:12px 0 6px;'>🔊 Spoken Voice Guidance (English & Tamil):</div>", unsafe_allow_html=True)
            v_btn1, v_btn2 = st.columns(2)
            with v_btn1:
                if st.button("🇬🇧 Listen in English", use_container_width=True, key="yoga_listen_en"):
                    a_bytes = get_voice_audio_bytes(top_cue_en, "en")
                    st.session_state.yoga_audio = (a_bytes, top_cue_en)
            with v_btn2:
                if st.button("🇮🇳 தமிழில் கேட்கவும்", use_container_width=True, key="yoga_listen_ta"):
                    a_bytes = get_voice_audio_bytes(top_cue_ta, "ta")
                    st.session_state.yoga_audio = (a_bytes, top_cue_ta)

            if st.session_state.get("yoga_audio"):
                y_bytes, y_text = st.session_state.yoga_audio
                if y_bytes:
                    st.markdown(f"<div style='font-size:12px;font-weight:700;color:#4F46E5;margin-top:6px;'>🔊 {y_text}</div>", unsafe_allow_html=True)
                    play_audio(st, y_bytes, autoplay=True)

            fb_cls = {"good": "feedback-good", "warning": "feedback-warning", "incorrect": "feedback-bad"}.get(disp["status"], "feedback-warning")

            imp_items = []
            for m in disp["improve_messages"]:
                m_ta = translate_text(m, "ta")
                imp_items.append(f"<div class='fb-item'>⚠️ <strong>{m}</strong><br><span style='color:#4338CA;font-weight:600;font-size:12px;'>🇮🇳 தமிழ்: {m_ta}</span></div>")
            imp_html = "".join(imp_items)

            good_items = []
            for m in disp["good_messages"][:3]:
                m_ta = translate_text(m, "ta")
                good_items.append(f"<div class='fb-item'>✅ {m} <span style='color:#16A34A;font-weight:600;font-size:12px;'>({m_ta})</span></div>")
            imp_html += "".join(good_items)

            st.markdown(f"""
            <div class='{fb_cls}'>
              <div class='fb-title' style='color:{disp["color"]};'>{disp["emoji"]} Form Analysis & Multilingual Corrections</div>
              <div class='fb-sub' style='color:#0F172A !important; font-weight:800 !important; font-size:13px !important;'>
                Corrections vs Dataset Reference Pose ({predicted_ex}):
              </div>
              {imp_html}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='font-size:14px;font-weight:800;color:#0F172A;margin:16px 0 8px;'>📐 Joint Angles Measured</div>",
                        unsafe_allow_html=True)
            body_parts = ["L Elbow", "R Elbow", "L Shoulder", "R Shoulder", "L Hip", "R Hip", "L Knee", "R Knee"]
            rows_html = "".join([
                f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px solid #E2E8F0;'>"
                f"<span style='color:#475569;font-weight:600;'>{bp} ({BODY_PARTS_TA.get(bp, '')})</span>"
                f"<span style='font-weight:800;color:#0F172A;'>{ang:.1f}°</span></div>"
                for bp, ang in zip(body_parts, user_angles)
            ])
            st.markdown(f"<div style='background:#FFFFFF;border-radius:12px;padding:12px;border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(15,23,42,0.04);'>{rows_html}</div>",
                        unsafe_allow_html=True)


def score_status_str(score: float) -> str:
    if score >= 75: return "Grade A"
    if score >= 50: return "Grade B"
    return "Grade C"


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — WORKOUT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def render_summary():
    s = st.session_state.workout_summary
    if not s:
        st.warning("No workout data. Start a workout first.")
        if st.button("← Go Home"):
            nav_to("home"); st.rerun()
        return

    lang         = st.session_state.get("voice_lang", "en")
    ex_name      = s.get("exercise", "Unknown")
    disp_name    = POSE_DISPLAY_NAMES.get(ex_name, ex_name)
    total_reps   = s.get("total_reps", 0)
    correct_reps = s.get("correct_reps", 0)
    bad_reps     = s.get("incorrect_reps", 0)
    duration_str = s.get("duration_str", "00:00")
    rep_type     = s.get("rep_type", "reps")
    last_score   = s.get("last_score", 0.0)

    form_acc = (correct_reps / total_reps * 100) if total_reps > 0 else last_score
    form_acc = round(form_acc, 1)

    title_txt = "Great Job! 🏆" if lang == "en" else "அருமை! நன்று! 🏆"
    sub_txt   = f"You completed your <strong>{disp_name}</strong> session!" if lang == "en" else f"உங்கள் <strong>{disp_name}</strong> பயிற்சி வெற்றிகரமாக முடிந்தது!"

    st.markdown(f"""
    <div class='trophy-banner'>
      <div style='font-size:3.5rem;'>🏆</div>
      <h1 style='margin:8px 0 4px;'>{title_txt}</h1>
      <p>{sub_txt}</p>
    </div>
    """, unsafe_allow_html=True)

    # Voice Summary Cue Buttons
    summary_cue_en = f"Great job! You completed your {disp_name} session with {form_acc:.0f} percent form accuracy!"
    summary_cue_ta = f"அருமையான முயற்சி! உங்கள் {disp_name} பயிற்சி முடிந்தது. தோரணை தரம் {form_acc:.0f} சதவீதம்!"

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("🇬🇧 Listen in English", use_container_width=True, key="sum_listen_en"):
            st.session_state.summary_audio = (get_voice_audio_bytes(summary_cue_en, "en"), summary_cue_en)
    with s_col2:
        if st.button("🇮🇳 தமிழில் கேட்கவும்", use_container_width=True, key="sum_listen_ta"):
            st.session_state.summary_audio = (get_voice_audio_bytes(summary_cue_ta, "ta"), summary_cue_ta)

    if st.session_state.get("summary_audio"):
        s_bytes, s_text = st.session_state.summary_audio
        if s_bytes:
            st.markdown(f"<div style='font-size:12px;font-weight:700;color:#4F46E5;margin:8px 0 4px;'>🔊 Summary: {s_text}</div>", unsafe_allow_html=True)
            play_audio(st, s_bytes, autoplay=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, icon, val, lbl in [
        (m1, "🔁", total_reps,   get_ui_text("reps", lang)),
        (m2, "✅", correct_reps, get_ui_text("correct_reps", lang)),
        (m3, "❌", bad_reps,    get_ui_text("need_work", lang)),
        (m4, "⏱️", duration_str, get_ui_text("duration", lang)),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='font-size:1.5rem;margin-bottom:4px;'>{icon}</div>
              <div class='metric-value'>{val}</div>
              <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"""
        <div class='fitness-card'>
          <div style='display:flex;align-items:center;gap:14px;'>
            <div style='font-size:2.5rem;'>🧘</div>
            <div>
              <div style='font-size:18px;font-weight:800;color:#0F172A;'>{disp_name}</div>
              <div style='font-size:12px;color:#4338CA;font-weight:600;'>Session Completed</div>
            </div>
          </div>
          <div style='margin-top:14px;background:#E2E8F0;border-radius:10px;height:8px;overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#4F46E5,#7C3AED);
                        height:100%;width:{min(form_acc,100):.0f}%;border-radius:10px;'></div>
          </div>
          <div style='display:flex;justify-content:space-between;margin-top:4px;'>
            <div style='font-size:11px;color:#64748B;'>Form Accuracy</div>
            <div style='font-size:11px;font-weight:700;color:#4F46E5;'>{form_acc:.0f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        acc_color = _score_color(form_acc)
        donut = _donut_chart(form_acc, acc_color)
        st.plotly_chart(donut, use_container_width=True, key="summary_donut")

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    with b1:
        saved = st.session_state.workout_saved
        label = "✓ Saved!" if saved else "💾 Save Session"
        if st.button(label, type="primary", use_container_width=True, disabled=saved):
            save_workout(
                exercise=disp_name,
                total_reps=total_reps,
                correct_reps=correct_reps,
                incorrect_reps=bad_reps,
                duration_seconds=s.get("duration_seconds", 0),
                form_accuracy=form_acc,
                rep_type=rep_type,
            )
            st.session_state.workout_saved = True
            st.success("✅ Session saved to SQLite database!")
            st.rerun()

    with b2:
        if st.button("📊 View History", use_container_width=True):
            nav_to("history"); st.rerun()

    with b3:
        if st.button("▶ Another Session", use_container_width=True):
            nav_to("home"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PROGRESS & HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
def render_history():
    st.markdown("<div class='page-title'>📈 Progress &amp; History</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Track your fitness journey and improvement over time.</div>",
                unsafe_allow_html=True)

    workouts = get_all_workouts()

    if not workouts:
        st.markdown("""
        <div style="text-align:center;padding:60px;background:#FFFFFF;border-radius:20px;
                    border:2px dashed #E2E8F0;margin:20px 0;">
          <div style="font-size:3rem;">📊</div>
          <div style="font-size:16px;font-weight:700;color:#0F172A;margin:12px 0 6px;">No workout history yet</div>
          <div style="font-size:13px;color:#64748B;">Complete your first workout session to start tracking progress.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Start First Workout", type="primary"):
            nav_to("fitness_coach"); st.rerun()
        return

    # ── Summary KPIs ────────────────────────────────────────────────────────
    total_sec = sum(w.get("duration_seconds", 0) or 0 for w in workouts)
    total_min = total_sec // 60
    avg_acc   = sum(w.get("form_accuracy", 0) or 0 for w in workouts) / len(workouts)
    total_reps = sum(w.get("total_reps", 0) or 0 for w in workouts)

    s1, s2, s3, s4 = st.columns(4)
    for col, icon, val, lbl in [
        (s1, "💪", len(workouts),     "Sessions"),
        (s2, "🔁", total_reps,        "Total Reps"),
        (s3, "⏱️", f"{total_min}m",   "Total Time"),
        (s4, "✅", f"{avg_acc:.0f}%",  "Avg Accuracy"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style='font-size:1.5rem;margin-bottom:4px;'>{icon}</div>
              <div class='metric-value'>{val}</div>
              <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Accuracy trend ───────────────────────────────────────────────────────
    if len(workouts) >= 2 and _ANALYTICS_OK:
        st.markdown("<div class='sec-head'>📊 Form Accuracy Trend</div>", unsafe_allow_html=True)
        accs  = [w.get("form_accuracy", 0) or 0 for w in reversed(workouts)]
        dates = [w.get("date", "") for w in reversed(workouts)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(accs)+1)), y=accs,
            mode="lines+markers",
            line=dict(color="#4F46E5", width=3),
            marker=dict(size=8, color="#4F46E5"),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
            hovertemplate="Session %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=75, line_dash="dash", line_color="#16A34A",
                      annotation_text="Target 75%",
                      annotation_font_color="#16A34A", annotation_font_size=11)
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            font=dict(color="#0F172A", family="Inter"),
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                       title="Session #", tickfont=dict(color="#475569")),
            yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                       title="Accuracy (%)", range=[0, 105],
                       tickfont=dict(color="#475569")),
            showlegend=False,
        )
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="history_trend")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Session history table ────────────────────────────────────────────────
    st.markdown("<div class='sec-head'>📋 Session History</div>", unsafe_allow_html=True)

    if _ANALYTICS_OK:
        import pandas as _pd
        df_display = _pd.DataFrame([{
            "Date":          w.get("date", ""),
            "Exercise":      w.get("exercise", ""),
            "Total Reps":    w.get("total_reps", 0),
            "✅ Correct":    w.get("correct_reps", 0),
            "❌ Incorrect":  w.get("incorrect_reps", 0),
            "Accuracy %":    round(w.get("form_accuracy", 0) or 0, 1),
            "Duration":      format_duration(w.get("duration_seconds", 0) or 0),
        } for w in workouts])
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        for w in workouts:
            acc   = w.get("form_accuracy", 0) or 0
            color = _score_color(acc)
            dur   = format_duration(w.get("duration_seconds", 0) or 0)
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:2fr 2fr 1fr 1fr 1fr 1fr 1fr;
                        gap:8px;background:#fff;border-radius:12px;
                        padding:10px 16px;margin-bottom:6px;
                        border:1px solid #E2E8F0;align-items:center;'>
              <div style='font-size:12px;color:#475569;'>{w["date"]}</div>
              <div style='font-size:13px;font-weight:700;color:#0F172A;'>🧘 {w["exercise"]}</div>
              <div style='font-weight:700;'>{w["total_reps"]}</div>
              <div style='color:#16A34A;font-weight:700;'>{w["correct_reps"]}</div>
              <div style='color:#DC2626;font-weight:700;'>{w["incorrect_reps"]}</div>
              <div style='font-weight:800;color:{color};'>{acc:.0f}%</div>
              <div style='font-size:13px;color:#475569;'>{dur}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("← Back to Dashboard"):
        nav_to("home"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def render_recommendations():
    st.markdown("<div class='page-title'>💡 Wellness Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Data-driven insights derived from lifestyle analytics — for educational purposes only.</div>",
                unsafe_allow_html=True)

    if not _ANALYTICS_OK:
        st.error("Analytics library (pandas/joblib) not available.")
        return
    try:
        ws, _ = _load_wellness_data()
    except Exception as e:
        st.error(f"Could not load analytics data: {e}")
        return

    avg_sleep  = ws["Hours_Slept"].mean()
    avg_steps  = ws["Steps_Taken"].mean()
    avg_stress = ws["Stress_Level (1-10)"].mean()
    avg_water  = ws["Water_Intake (Liters)"].mean()
    avg_hr     = ws["Heart_Rate (bpm)"].mean()

    recs = []

    # Sleep
    if avg_sleep < 7:
        p = "⚠️ Below Target"
        i = f"Population average sleep is {avg_sleep:.1f}h per night — below the recommended 7–9 hours. Sleep duration is the strongest predictor of wellness score in the ML model (36% feature importance)."
        a = "Aim for 7–9 hours. Set a fixed bedtime and limit screens 1 hour before sleep."
    else:
        p = "✅ On Track"
        i = f"Sleep duration ({avg_sleep:.1f}h) is within the healthy range of 7–9 hours."
        a = "Maintain your current sleep schedule and protect sleep quality."
    recs.append(("😴", "Sleep Quality", p, i, a, "#6366F1"))

    # Activity
    if avg_steps < 6000:
        p = "⚠️ Below Target"
        i = f"Average daily steps ({avg_steps:,.0f}) are below the recommended 7,000–10,000. Low activity strongly correlates with reduced wellness scores."
        a = "Add a 20–30 minute daily walk. Take stairs when possible and set hourly movement reminders."
    elif avg_steps >= 10000:
        p = "🌟 Excellent"
        i = f"Step count ({avg_steps:,.0f}/day) exceeds the 10,000 target. High activity is strongly linked to better wellness outcomes."
        a = "Maintain this level. Ensure adequate recovery, sleep, and hydration."
    else:
        p = "✅ Good"
        i = f"Steps ({avg_steps:,.0f}/day) are within the healthy range."
        a = "Try to maintain or increase daily steps towards 10,000."
    recs.append(("🏃", "Physical Activity", p, i, a, "#0891B2"))

    # Hydration
    if avg_water < 2.0:
        p = "⚠️ Below Target"
        i = f"Average water intake ({avg_water:.1f}L/day) is below 2.0–2.5L. Hydration is the 3rd strongest predictor of wellness score in the Gradient Boosting model."
        a = "Keep a 2L water bottle visible. Track intake with timed reminders and aim for 8 glasses daily."
    else:
        p = "✅ Adequate"
        i = f"Hydration ({avg_water:.1f}L/day) meets the minimum recommendation."
        a = "Continue consistent hydration. Increase intake during exercise or hot weather."
    recs.append(("💧", "Hydration", p, i, a, "#059669"))

    # Stress
    if avg_stress > 6:
        p = "⚠️ Elevated"
        i = f"Average stress ({avg_stress:.1f}/10) is elevated. Stress is the 2nd strongest predictor of poor wellness scores (24% feature importance in the ML model)."
        a = "Practice 10 minutes of daily breathing exercises or mindfulness. Reduce high-stress commitments where possible."
    elif avg_stress > 4:
        p = "📊 Moderate"
        i = f"Stress levels ({avg_stress:.1f}/10) are moderate. Sustained moderate stress can reduce wellness scores over time."
        a = "Consider light mindfulness, yoga, or short daily walks to manage stress."
    else:
        p = "✅ Well Managed"
        i = f"Stress is well-managed ({avg_stress:.1f}/10). This is a strong contributor to high wellness scores."
        a = "Maintain current stress management practices."
    recs.append(("🧘", "Stress Management", p, i, a, "#E11D48"))

    # Heart Rate
    if avg_hr > 90:
        p = "⚠️ Elevated"
        i = f"Average resting heart rate ({avg_hr:.0f} bpm) is on the higher side. Elevated resting HR can reflect cardiovascular stress."
        a = "Regular aerobic exercise lowers resting heart rate over time. Consult a healthcare professional if concerned."
    else:
        p = "✅ Normal Range"
        i = f"Average heart rate ({avg_hr:.0f} bpm) is within a normal range."
        a = "Continue regular physical activity to maintain cardiovascular health."
    recs.append(("❤️", "Heart Health", p, i, a, "#DC2626"))

    # Render
    for icon, category, problem, insight, action, color in recs:
        sev_clr = (
            "#DC2626" if "⚠️" in problem else
            "#16A34A" if any(x in problem for x in ("✅","🌟")) else
            "#D97706"
        )
        st.markdown(f"""
        <div class="rec-card">
          <div style="display:flex;align-items:flex-start;gap:14px;">
            <div class="rec-icon-box" style="background:#EEF2FF;">{icon}</div>
            <div style="flex:1;">
              <div style="font-size:16px;font-weight:700;color:#0F172A;">{category}</div>
              <div style="font-size:12px;font-weight:600;color:{sev_clr};margin-top:2px;">{problem}</div>
              <div style="font-size:13px;color:#475569;margin-top:10px;line-height:1.6;">{insight}</div>
              <div class="rec-action">
                <span style="font-size:10px;font-weight:700;color:#16A34A;text-transform:uppercase;letter-spacing:.5px;">Recommended Action</span>
                <div style="font-size:13px;color:#0F172A;font-weight:500;margin-top:3px;">{action}</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#F8FAFC;border-radius:12px;padding:14px 18px;margin-top:8px;
                border:1px solid #E2E8F0;font-size:12px;color:#64748B;">
      ℹ️ <strong style="color:#0F172A;">Data Note:</strong> These recommendations are based on
      population-level patterns from the Kaggle Fitness &amp; Health Tracking Dataset
      (36,500 records, 100 users). They are for educational purposes and do not constitute
      medical advice. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — PREDICT MY SCORE (standalone)
# ═══════════════════════════════════════════════════════════════════════════════
def render_predict_score():
    # ── Header with NOVA Greeting & User Profile ──
    avatar_b64 = _get_asset_base64("nova_avatar.png")
    
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-bottom:4px;">
      <div class="user-profile-badge">
        <img src="{avatar_b64}" style="width:22px;height:22px;border-radius:50%;object-fit:cover;" />
        <span>Hello, User</span>
      </div>
    </div>
    <div class="nova-header-container">
      <div class="nova-title-area">
        <h1><span>✨</span> Predict My Wellness Score</h1>
        <p>Tell us about your daily lifestyle and let NOVA calculate your personalized wellness score using AI-powered analytics.</p>
      </div>
      <div class="nova-greeting-card">
        <div class="nova-bubble-text">
          <strong>Hi! I'm NOVA ✨</strong>
          Your AI wellness companion. I'll help you understand your lifestyle and suggest simple ways to improve your wellbeing!
        </div>
        <img src="{avatar_b64}" class="nova-avatar-circle" alt="NOVA Avatar" />
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not _ANALYTICS_OK:
        st.error("Analytics library not available. Please check environment packages.")
        return

    ml_payload = _load_ml_model()

    # ── Initialize session state for persistent interactive inputs ──
    if "pred_inputs" not in st.session_state:
        st.session_state["pred_inputs"] = {
            "age": 30,
            "gender": "Male",
            "bmi": 22.50,
            "steps": 8000,
            "active_min": 45,
            "calories": 2500,
            "workout_type": "Strength",
            "sleep": 7.00,
            "water": 2.50,
            "hr": 72,
            "weekend": False,
            "stress": 4,
            "mood": "😊 Neutral",
        }

    inputs = st.session_state["pred_inputs"]

    # ── 1. Enter Your Daily Metrics Card ──
    st.markdown("""
    <div class="card-top-title">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
        <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
      </svg>
      1. Enter Your Daily Metrics
    </div>
    <div class="card-top-subtitle">Provide your latest lifestyle data for an accurate wellness prediction.</div>
    """, unsafe_allow_html=True)

    with st.form("wellness_metric_inputs_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown('<div class="pillar-pill personal"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg> Personal</div>', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=10, max_value=100, value=inputs["age"], step=1, key="ps_in_age")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if inputs["gender"] == "Male" else 1, key="ps_in_gender")
            bmi = st.number_input("BMI", min_value=12.0, max_value=50.0, value=float(inputs["bmi"]), step=0.1, format="%.2f", key="ps_in_bmi")

        with c2:
            st.markdown('<div class="pillar-pill activity"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Activity</div>', unsafe_allow_html=True)
            steps = st.number_input("Steps Taken", min_value=0, max_value=50000, value=inputs["steps"], step=500, key="ps_in_steps")
            active_min = st.number_input("Active Minutes", min_value=0, max_value=480, value=inputs["active_min"], step=5, key="ps_in_active")
            calories = st.number_input("Calories Burned", min_value=0, max_value=6000, value=inputs["calories"], step=50, key="ps_in_cal")
            workout_type = st.selectbox("Workout Type", ["Strength", "Cardio", "Yoga", "HIIT", "Walking", "None"], index=["Strength", "Cardio", "Yoga", "HIIT", "Walking", "None"].index(inputs.get("workout_type", "Strength")), key="ps_in_workout")

        with c3:
            st.markdown('<div class="pillar-pill recovery"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg> Recovery</div>', unsafe_allow_html=True)
            sleep = st.number_input("Hours Slept", min_value=0.0, max_value=16.0, value=float(inputs["sleep"]), step=0.5, format="%.2f", key="ps_in_sleep")
            water = st.number_input("Water Intake (L)", min_value=0.0, max_value=8.0, value=float(inputs["water"]), step=0.25, format="%.2f", key="ps_in_water")
            hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=inputs["hr"], step=1, key="ps_in_hr")
            weekend = st.checkbox("Is Weekend?", value=inputs["weekend"], key="ps_in_weekend")

        with c4:
            st.markdown('<div class="pillar-pill wellness"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg> Wellness</div>', unsafe_allow_html=True)
            stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=inputs["stress"], key="ps_in_stress")
            mood_options = ["😊 Neutral", "🤩 Energized", "😊 Happy", "🥱 Tired", "😣 Stressed"]
            cur_mood = inputs.get("mood", "😊 Neutral")
            mood_idx = mood_options.index(cur_mood) if cur_mood in mood_options else 0
            mood = st.selectbox("Mood", mood_options, index=mood_idx, key="ps_in_mood")

        submitted = st.form_submit_button("✨ Analyze My Wellness", use_container_width=True, type="primary")

    if submitted:
        st.session_state["pred_inputs"] = {
            "age": age, "gender": gender, "bmi": bmi,
            "steps": steps, "active_min": active_min, "calories": calories,
            "workout_type": workout_type, "sleep": sleep, "water": water,
            "hr": hr, "weekend": weekend, "stress": stress, "mood": mood
        }

    cur_data = st.session_state["pred_inputs"]

    # ── Score & Dimensions Calculation ──
    try:
        gender_enc = 1 if cur_data["gender"] == "Male" else 0
        input_row = [
            cur_data["age"], gender_enc, cur_data["bmi"], cur_data["steps"],
            cur_data["active_min"], cur_data["calories"], cur_data["sleep"],
            cur_data["water"], cur_data["hr"], cur_data["stress"], int(cur_data["weekend"])
        ]
        if ml_payload and "pipeline" in ml_payload:
            feature_names = ml_payload.get("feature_names", [
                'Age', 'Gender', 'BMI', 'Daily_Steps', 'Active_Minutes',
                'Calories_Burned', 'Sleep_Hours', 'Water_Intake_L',
                'Heart_Rate_Avg', 'Stress_Level', 'Is_Weekend'
            ])
            features_df = pd.DataFrame([input_row], columns=feature_names)
            raw_prediction = float(ml_payload["pipeline"].predict(features_df)[0])
        else:
            # Fallback estimation
            raw_prediction = 75.0
    except Exception:
        raw_prediction = 75.0

    prediction = max(0.0, min(100.0, raw_prediction))

    # Normalized dimension subscores (0-100)
    sleep_score = int(np.clip((cur_data["sleep"] / 8.5) * 100, 15, 100))
    act_steps = (cur_data["steps"] / 10000.0) * 50
    act_mins = (cur_data["active_min"] / 60.0) * 50
    activity_score = int(np.clip(act_steps + act_mins, 15, 100))
    hydration_score = int(np.clip((cur_data["water"] / 3.7) * 100, 15, 100))
    hr_score = int(np.clip(100 - abs(cur_data["hr"] - 60) * 0.9, 15, 100))
    stress_score = int(np.clip(100 - (cur_data["stress"] - 1) * 8.0, 15, 100))

    # ── Dynamic NOVA Companion & Tiers Selection ──
    if prediction >= 80:
        gif_file = "nova_companion_high.gif"
        score_color = "#10B981"
        pill_label = "Good Balance" if prediction < 85 else "Peak Wellness"
        pill_icon = "🌿" if prediction < 85 else "🌟"
        pill_bg = "#ECFDF5"
        pill_border = "#10B981"
        pill_text_color = "#059669"
        nova_title = "Outstanding!"
        nova_dialogue = "Your lifestyle is in peak condition today! Radiating positive energy and wellness."
    elif prediction >= 60:
        gif_file = "nova_companion_good.gif"
        score_color = "#10B981"
        pill_label = "Good Balance"
        pill_icon = "🌿"
        pill_bg = "#ECFDF5"
        pill_border = "#10B981"
        pill_text_color = "#059669"
        nova_title = "Great job!"
        nova_dialogue = "Your lifestyle is looking well balanced today. Keep it up!"
    elif prediction >= 40:
        gif_file = "nova_companion_fair.gif"
        score_color = "#F59E0B"
        pill_label = "Moderate Focus"
        pill_icon = "⚡"
        pill_bg = "#FFFBEB"
        pill_border = "#F59E0B"
        pill_text_color = "#B45309"
        nova_title = "Steady progress!"
        nova_dialogue = "You're making steady progress! Let's focus on sleep and hydration to power up."
    else:
        gif_file = "nova_companion_rest.gif"
        score_color = "#EF4444"
        pill_label = "Rest & Recovery"
        pill_icon = "🌱"
        pill_bg = "#FEF2F2"
        pill_border = "#EF4444"
        pill_text_color = "#B91C1C"
        nova_title = "Time to recharge!"
        nova_dialogue = "Take it easy today. Prioritize sound sleep, gentle hydration, and recharge your body!"

    gif_b64 = _get_asset_base64(gif_file)

    # Calculate SVG Gauge Arc
    circumference = 389.56
    dashoffset = circumference * (1.0 - (prediction / 100.0))

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 1 — PREDICTED WELLNESS (GAUGE + NOVA) & RADAR BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.markdown(f"""
        <div class="wellness-result-card">
          <div class="card-kicker">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            YOUR PREDICTED WELLNESS
          </div>
          <div class="score-nova-row">
            <!-- Gauge Column -->
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:180px;">
              <div style="position:relative;width:170px;height:170px;">
                <svg viewBox="0 0 160 160" width="170" height="170" style="transform:rotate(-90deg);">
                  <circle cx="80" cy="80" r="62" fill="none" stroke="#E2E8F0" stroke-width="13" stroke-linecap="round" />
                  <circle cx="80" cy="80" r="62" fill="none" stroke="{score_color}" stroke-width="13" stroke-linecap="round"
                          stroke-dasharray="389.56" stroke-dashoffset="{dashoffset:.2f}"
                          style="transition: stroke-dashoffset 0.8s ease;" />
                </svg>
                <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
                  <div style="font-size:3.2rem;font-weight:900;color:#0F172A;line-height:1;">{prediction:.0f}</div>
                  <div style="font-size:0.85rem;font-weight:600;color:#64748B;margin-top:2px;">/100</div>
                  <div style="font-size:1.2rem;margin-top:4px;">🍃</div>
                </div>
              </div>
              <div style="background:{pill_bg};border:1.5px solid {pill_border};border-radius:9999px;padding:6px 16px;margin-top:14px;font-size:13px;font-weight:700;color:{pill_text_color};display:inline-flex;align-items:center;gap:6px;">
                <span>{pill_icon}</span> {pill_label}
              </div>
            </div>
            <!-- NOVA Looping Companion Column -->
            <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
              <div class="nova-speech-bubble-dyn">
                <strong>{nova_title}</strong>
                {nova_dialogue}
              </div>
              <div class="nova-gif-wrap">
                <img src="{gif_b64}" alt="NOVA AI Companion" />
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Plotly Polar Radar Chart
        categories = ['Sleep', 'Activity', 'Hydration', 'Stress', 'Heart Rate']
        values = [sleep_score, activity_score, hydration_score, stress_score, hr_score]
        cat_closed = categories + [categories[0]]
        val_closed = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=val_closed,
            theta=cat_closed,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.22)',
            line=dict(color='#0284C7', width=2.5),
            marker=dict(size=8, color='#0284C7', symbol='circle'),
            name='Components'
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='#FFFFFF',
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=False,
                    gridcolor='#F1F5F9',
                    linecolor='#E2E8F0',
                ),
                angularaxis=dict(
                    gridcolor='#F1F5F9',
                    linecolor='#CBD5E1',
                    tickfont=dict(size=12, color='#0F172A', family='Inter', weight='bold')
                )
            ),
            showlegend=False,
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            margin=dict(l=45, r=45, t=15, b=25),
            height=315,
        )

        st.markdown("""
        <div class="wellness-result-card" style="padding-bottom:12px;">
          <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:8px;">
            Wellness Component Breakdown
          </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 2 — 5 METRIC COMPONENT CARDS
    # ═══════════════════════════════════════════════════════════════════════════
    m1, m2, m3, m4, m5 = st.columns(5)

    metrics_list = [
        (m1, "Sleep", sleep_score, "#6366F1", "🌙"),
        (m2, "Activity", activity_score, "#10B981", "🏃"),
        (m3, "Hydration", hydration_score, "#0EA5E9", "💧"),
        (m4, "Stress", stress_score, "#F59E0B", "🛡️"),
        (m5, "Heart Rate", hr_score, "#EC4899", "❤️"),
    ]

    for col, m_name, m_val, m_color, m_icon in metrics_list:
        with col:
            st.markdown(f"""
            <div class="metric-strip-card">
              <div class="metric-strip-header">
                <div class="metric-strip-left">
                  <span>{m_icon}</span>
                  <span>{m_name}</span>
                </div>
                <div class="metric-strip-score">
                  {m_val} <span>/100</span>
                </div>
              </div>
              <div class="metric-strip-bar-bg">
                <div class="metric-strip-bar-fill" style="width:{m_val}%;background:{m_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROW 3 — TRIPLE CARDS: WHY SCORE / WELLNESS INSIGHT / WHAT IF?
    # ═══════════════════════════════════════════════════════════════════════════
    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown(f"""
        <div class="bottom-insight-card">
          <div class="bottom-card-title">
            <span>🔬</span> Why did you get {prediction:.0f}?
          </div>
          <div class="driver-bar-row">
            <div class="driver-label"><span>🌙</span> Sleep</div>
            <div class="driver-val" style="color:#6366F1;">+18</div>
            <div class="driver-track"><div class="driver-fill" style="width:85%;background:#6366F1;"></div></div>
          </div>
          <div class="driver-bar-row">
            <div class="driver-label"><span>🏃</span> Activity</div>
            <div class="driver-val" style="color:#10B981;">+15</div>
            <div class="driver-track"><div class="driver-fill" style="width:72%;background:#10B981;"></div></div>
          </div>
          <div class="driver-bar-row">
            <div class="driver-label"><span>💧</span> Hydration</div>
            <div class="driver-val" style="color:#0EA5E9;">+10</div>
            <div class="driver-track"><div class="driver-fill" style="width:58%;background:#0EA5E9;"></div></div>
          </div>
          <div class="driver-bar-row">
            <div class="driver-label"><span>❤️</span> Heart Rate</div>
            <div class="driver-val" style="color:#EC4899;">+8</div>
            <div class="driver-track"><div class="driver-fill" style="width:48%;background:#EC4899;"></div></div>
          </div>
          <div class="driver-bar-row">
            <div class="driver-label"><span>⚡</span> Stress</div>
            <div class="driver-val" style="color:#F97316;">-7</div>
            <div class="driver-track"><div class="driver-fill" style="width:38%;background:#F97316;"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="bottom-insight-card">
          <div class="bottom-card-title">
            <span>💡</span> Your Wellness Insight
          </div>
          <div style="font-size:12.5px;color:#475569;line-height:1.6;margin-bottom:14px;">
            Your activity level is strong today. Your biggest opportunity is <strong>hydration</strong>.
          </div>
          <div style="background:#F0F9FF;border-left:3.5px solid #0EA5E9;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
            <div style="font-size:12px;font-weight:700;color:#0284C7;">💧 Today's Focus</div>
            <div style="font-size:12px;color:#334155;margin-top:2px;">
              Increase your water intake to 3.0L for better recovery and energy.
            </div>
          </div>
          <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:10px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:12px;font-weight:700;color:#B45309;">🏆 Daily Challenge</span>
              <span style="font-size:11px;font-weight:700;color:#92400E;">0 / 4</span>
            </div>
            <div style="font-size:11.5px;color:#451A03;margin:3px 0 6px;">Drink 500 ml more water today</div>
            <div style="width:100%;height:5px;background:#FEF3C7;border-radius:9999px;overflow:hidden;">
              <div style="width:25%;height:100%;background:#F59E0B;border-radius:9999px;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        potential_score = min(100.0, prediction + 8.0)
        st.markdown(f"""
        <div class="bottom-insight-card">
          <div class="bottom-card-title">
            <span>✨</span> What If?
          </div>
          <div style="font-size:11.5px;color:#64748B;margin-bottom:10px;font-weight:600;">
            Estimated improvement with small changes
          </div>
          <div style="display:flex;align-items:center;justify-content:space-around;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px;margin-bottom:12px;">
            <div style="text-align:center;">
              <div style="font-size:11px;color:#64748B;font-weight:600;">Current</div>
              <div style="font-size:1.6rem;font-weight:900;color:#10B981;">{prediction:.0f}</div>
            </div>
            <div style="font-size:1.4rem;color:#94A3B8;">→</div>
            <div style="text-align:center;">
              <div style="font-size:11px;color:#059669;font-weight:700;">With Improvements</div>
              <div style="font-size:1.6rem;font-weight:900;color:#059669;">{potential_score:.0f}</div>
            </div>
          </div>
          <div style="font-size:12px;color:#334155;line-height:1.7;">
            <div style="display:flex;align-items:center;gap:6px;"><span>🌙</span> + Sleep (7.5h)</div>
            <div style="display:flex;align-items:center;gap:6px;"><span>🏃</span> + Activity (10k steps)</div>
            <div style="display:flex;align-items:center;gap:6px;"><span>💧</span> + Hydration (3.0L)</div>
          </div>
          <div style="font-size:10px;color:#94A3B8;margin-top:8px;font-style:italic;">
            * This is a model-based estimate and not a guarantee of results.
          </div>
        </div>
        """, unsafe_allow_html=True)





# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  PS15 — Wellness Intelligence Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_BASE_DIR, "data", "processed")
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "wellness_predictor.pkl")
_REPORT_DIR = os.path.join(_BASE_DIR, "reports")
_FIG_DIR    = os.path.join(_REPORT_DIR, "figures")


@st.cache_data(show_spinner=False)
def _load_wellness_data():
    ws  = pd.read_csv(os.path.join(_DATA_DIR, "daily_wellness_scores.csv"), parse_dates=["Date"])
    seg = pd.read_csv(os.path.join(_DATA_DIR, "user_lifestyle_profiles.csv"))
    return ws, seg


@st.cache_resource(show_spinner=False)
def _load_ml_model():
    if os.path.exists(_MODEL_PATH):
        return joblib.load(_MODEL_PATH)
    return None


def _load_metrics():
    p = os.path.join(_REPORT_DIR, "ml_model_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#0F172A"}},
        number={"font": {"color": "#0F172A", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e", "size": 9}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#F8FAFC",
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 40],  "color": "#FEF2F2"},
                {"range": [40, 70], "color": "#FFFBEB"},
                {"range": [70, 100],"color": "#F0FDF4"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font_color="#0F172A", height=200,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


def render_wellness_dashboard():
    if not _ANALYTICS_OK:
        st.error("pandas / joblib not available. Install requirements.")
        return

    # ── Header ──
    st.markdown("""
    <div style='background:linear-gradient(135deg,#EEF2FF 0%,#F4F6FC 100%);
                border:1px solid #C7D2FE;border-radius:16px;
                padding:28px 32px;margin-bottom:24px;'>
      <div style='display:flex;align-items:center;gap:16px;'>
        <div style='font-size:3rem;'>🧠</div>
        <div>
          <h1 style='margin:0;color:#0F172A;font-size:1.8rem;font-weight:800;'>
            Wellness Intelligence Dashboard
          </h1>
          <p style='margin:4px 0 0;color:#475569;font-size:0.95rem;'>
            PS15 · Lifestyle Analytics · ML Prediction · Behaviour Segmentation
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    try:
        ws, seg = _load_wellness_data()
    except Exception as e:
        st.error(f"Could not load analytics data: {e}")
        return

    ml_payload = _load_ml_model()
    metrics    = _load_metrics()

    # ── Tabs ──
    tabs = st.tabs([
        "📊 Overview",
        "📈 Lifestyle Analytics",
        "🤖 ML Prediction",
        "👥 Behaviour Segments",
        "🔮 Predict My Score",
    ])

    # ══════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════
    with tabs[0]:
        st.markdown("### 📊 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{len(ws):,}")
        with c2:
            st.metric("Unique Users", ws["User_ID"].nunique())
        with c3:
            st.metric("Avg Wellness Score", f"{ws['Wellness_Score'].mean():.1f}")
        with c4:
            st.metric("Date Range", f"{ws['Date'].dt.year.min()}–{ws['Date'].dt.year.max()}")

        st.markdown("---")
        st.markdown("### 🎯 Population Wellness Score Distribution")
        fig = px.histogram(
            ws, x="Wellness_Score", nbins=60,
            color_discrete_sequence=["#58a6ff"],
            labels={"Wellness_Score": "Wellness Score (0–100)"},
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            font_color="#0F172A", height=320,
            bargap=0.05,
            xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPI gauges per dimension
        st.markdown("### 🏆 Average Dimension Scores")
        g1, g2, g3, g4, g5 = st.columns(5)
        dims = [
            (g1, "Sleep_Score",        "Sleep",      "#58a6ff"),
            (g2, "Activity_Score",     "Activity",   "#3fb950"),
            (g3, "Hydration_Score",    "Hydration",  "#79c0ff"),
            (g4, "Stress_Score",       "Stress",     "#d2a8ff"),
            (g5, "Heart_Rate_Score",   "Heart Rate", "#f78166"),
        ]
        for col, field, label, color in dims:
            avg = ws[field].mean() if field in ws.columns else 0
            with col:
                st.plotly_chart(_gauge(avg, label, color), use_container_width=True)

        # Wellness score stats table
        st.markdown("### 📋 Descriptive Statistics")
        desc = ws["Wellness_Score"].describe().rename(
            {"count":"Count","mean":"Mean","std":"Std Dev",
             "min":"Min","25%":"25th Pct","50%":"Median",
             "75%":"75th Pct","max":"Max"}
        )
        st.dataframe(
            desc.to_frame("Wellness Score").style.format("{:.2f}"),
            use_container_width=True
        )

    # ══════════════════════════════════════
    # TAB 2 — LIFESTYLE ANALYTICS
    # ══════════════════════════════════════
    with tabs[1]:
        st.markdown("### 📈 Lifestyle Analytics")

        # Activity Patterns
        st.markdown("#### 🏃 Activity Patterns by Day of Week")
        if "Day_of_Week" in ws.columns:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_df = ws.groupby("Day_of_Week")[["Steps_Taken","Active_Minutes","Calories_Burned"]].mean().reindex(day_order).reset_index()
            fig = px.bar(
                day_df, x="Day_of_Week", y="Steps_Taken",
                color="Steps_Taken", color_continuous_scale="Blues",
                labels={"Steps_Taken": "Avg Steps", "Day_of_Week": "Day"},
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Sleep patterns
        st.markdown("#### 😴 Sleep Patterns")
        c1, c2 = st.columns(2)
        with c1:
            sleep_bins = pd.cut(ws["Hours_Slept"], bins=[0,5,6,7,8,9,24],
                                labels=["<5h","5-6h","6-7h","7-8h","8-9h",">9h"])
            sleep_counts = sleep_bins.value_counts().sort_index()
            fig = px.bar(
                x=sleep_counts.index.astype(str),
                y=sleep_counts.values,
                color=sleep_counts.values,
                color_continuous_scale="Blues",
                labels={"x":"Sleep Duration","y":"Record Count"},
                title="Sleep Duration Distribution",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"),
                              showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                ws.sample(min(2000, len(ws))),
                x="Hours_Slept", y="Wellness_Score",
                opacity=0.35, color_discrete_sequence=["#58a6ff"],
                labels={"Hours_Slept":"Hours Slept","Wellness_Score":"Wellness Score"},
                title="Sleep vs Wellness Score",
                trendline="ols", trendline_color_override="#3fb950",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)

        # Calorie & hydration
        st.markdown("#### 🔥 Calorie & Hydration Patterns")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                ws.sample(min(2000, len(ws))),
                x="Steps_Taken", y="Calories_Burned",
                opacity=0.3, color="Wellness_Score",
                color_continuous_scale="Viridis",
                labels={"Steps_Taken":"Steps","Calories_Burned":"Calories Burned"},
                title="Steps vs Calories (coloured by Wellness)",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                ws.sample(min(2000, len(ws))),
                x="Water_Intake (Liters)", y="Wellness_Score",
                opacity=0.3, color_discrete_sequence=["#3fb950"],
                labels={"Water_Intake (Liters)":"Water (Litres)",
                        "Wellness_Score":"Wellness Score"},
                title="Hydration vs Wellness Score",
                trendline="ols", trendline_color_override="#f78166",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig, use_container_width=True)

        # Stress & unhealthy trends
        st.markdown("#### ⚠️ Stress Levels & Unhealthy Trends")
        c1, c2 = st.columns(2)
        with c1:
            stress_df = ws.groupby("Stress_Level (1-10)")["Wellness_Score"].mean().reset_index()
            fig = px.bar(
                stress_df, x="Stress_Level (1-10)", y="Wellness_Score",
                color="Wellness_Score", color_continuous_scale="RdYlGn",
                labels={"Stress_Level (1-10)":"Stress Level (1-10)",
                        "Wellness_Score":"Avg Wellness Score"},
                title="Stress Level vs Average Wellness Score",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Flag unhealthy: low sleep (<6h), high stress (>7), low steps (<3000)
            ws["Unhealthy"] = (
                (ws["Hours_Slept"] < 6) |
                (ws["Stress_Level (1-10)"] > 7) |
                (ws["Steps_Taken"] < 3000)
            )
            unhealthy_rate = ws.groupby("User_ID")["Unhealthy"].mean().mul(100)
            top_unhealthy = unhealthy_rate.sort_values(ascending=False).head(10).reset_index()
            top_unhealthy.columns = ["User_ID", "Unhealthy Day %"]
            fig = px.bar(
                top_unhealthy, x="User_ID", y="Unhealthy Day %",
                color="Unhealthy Day %", color_continuous_scale="Reds",
                labels={"User_ID":"User ID","Unhealthy Day %":"% Unhealthy Days"},
                title="Top 10 Users by Unhealthy Day Prevalence",
            )
            fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                              font_color="#0F172A", height=280,
                              xaxis=dict(gridcolor="#F1F5F9"),
                              yaxis=dict(gridcolor="#F1F5F9"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### 🔗 Lifestyle Factor Correlation Matrix")
        corr_cols = ["Hours_Slept","Steps_Taken","Active_Minutes","Calories_Burned",
                     "Water_Intake (Liters)","Heart_Rate (bpm)","Stress_Level (1-10)",
                     "BMI","Wellness_Score"]
        corr = ws[corr_cols].corr()
        fig = px.imshow(
            corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
            text_auto=".2f",
            labels={"color": "Correlation"},
            title="Pearson Correlation Matrix",
        )
        fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                          font_color="#0F172A", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 3 — ML PREDICTION
    # ══════════════════════════════════════
    with tabs[2]:
        st.markdown("### 🤖 ML Model Evaluation")
        if not metrics:
            st.warning("Run analytics/ml_wellness_predictor.py first to generate model metrics.")
        else:
            best = metrics.get("best_model", "")
            model_names = [k for k in metrics if k != "best_model"]

            # Model comparison table
            rows = []
            for name in model_names:
                m = metrics[name]
                rows.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                             "R² Test": m["R2"], "CV R² (5-fold)": m["CV_R2"],
                             "Best": "⭐" if name == best else ""})
            df_metrics = pd.DataFrame(rows)
            st.dataframe(
                df_metrics.style.highlight_max(subset=["R² Test","CV R² (5-fold)"],
                                               color="#EEF2FF")
                                .highlight_min(subset=["MAE","RMSE"],
                                               color="#EEF2FF"),
                use_container_width=True, hide_index=True
            )

            # Summary boxes
            if best and best in metrics:
                bm = metrics[best]
                st.markdown(f"**Best Model: {best}**")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("MAE",  f"{bm['MAE']:.3f}")
                c2.metric("RMSE", f"{bm['RMSE']:.3f}")
                c3.metric("R²",   f"{bm['R2']:.4f}")
                c4.metric("CV R²",f"{bm['CV_R2']:.4f}")

            # Show saved plots
            plot_files = [
                ("phase4_model_comparison.png",    "Model Comparison — MAE, R², CV-R²"),
                ("phase4_actual_vs_predicted.png", "Actual vs Predicted + Residuals"),
                ("phase4_feature_importance.png",  "Feature Importance"),
                ("phase4_error_distribution.png",  "Error Distribution"),
                ("phase4_cv_comparison.png",       "Test R² vs CV R²"),
            ]
            for fname, caption in plot_files:
                fpath = os.path.join(_FIG_DIR, fname)
                if os.path.exists(fpath):
                    st.markdown(f"**{caption}**")
                    st.image(fpath, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 4 — BEHAVIOUR SEGMENTS
    # ══════════════════════════════════════
    with tabs[3]:
        st.markdown("### 👥 User Behaviour Segmentation")

        if seg.empty:
            st.warning("No segmentation data found.")
        else:
            # Segment summary
            seg_summary = seg.groupby("Cluster_Label").agg(
                Users=("User_ID", "count"),
                Avg_Wellness=("Avg_Wellness_Score", "mean"),
                Avg_Sleep=("Avg_Sleep_Hours", "mean"),
                Avg_Steps=("Avg_Steps", "mean"),
                Avg_Stress=("Avg_Stress", "mean"),
            ).reset_index()
            seg_summary = seg_summary.rename(columns={"Cluster_Label":"Segment"})

            st.markdown("#### Segment Overview")
            st.dataframe(
                seg_summary.style.format({"Avg_Wellness":"{:.1f}","Avg_Sleep":"{:.1f}h",
                                          "Avg_Steps":"{:,.0f}","Avg_Stress":"{:.1f}"}),
                use_container_width=True, hide_index=True
            )

            # PCA scatter
            st.markdown("#### 🗺️ PCA User Map (Coloured by Segment)")
            if "PCA1" in seg.columns and "PCA2" in seg.columns:
                fig = px.scatter(
                    seg, x="PCA1", y="PCA2",
                    color="Cluster_Label",
                    hover_data=["Full_Name","Age","Avg_Wellness_Score"],
                    title="User Behaviour Segments (PCA)",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                fig.update_traces(marker=dict(size=9, opacity=0.85))
                fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                                  font_color="#0F172A", height=450,
                                  xaxis=dict(gridcolor="#F1F5F9"),
                                  yaxis=dict(gridcolor="#F1F5F9"),
                                  legend=dict(bgcolor="#FFFFFF",bordercolor="#E2E8F0"))
                st.plotly_chart(fig, use_container_width=True)

            # Radar chart for each segment
            st.markdown("#### 📡 Segment Lifestyle Profiles")
            radar_cols = ["Avg_Sleep_Score","Avg_Activity_Score","Avg_Hydration_Score",
                          "Avg_Stress_Score","Avg_Heart_Rate_Score"]
            radar_labels = ["Sleep","Activity","Hydration","Stress","Heart Rate"]
            radar_avgs = seg.groupby("Cluster_Label")[radar_cols].mean()

            fig = go.Figure()
            colours = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657"]
            for i, (label, row) in enumerate(radar_avgs.iterrows()):
                vals = list(row.values) + [list(row.values)[0]]
                labs = radar_labels + [radar_labels[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=labs, name=label,
                    line_color=colours[i % len(colours)],
                    fill="toself", fillcolor=colours[i % len(colours)],
                    opacity=0.2,
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#F8FAFC",
                    angularaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                                     color="#0F172A"),
                    radialaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
                                    color="#0F172A", range=[0,100]),
                ),
                paper_bgcolor="#FFFFFF",
                font_color="#0F172A",
                legend=dict(bgcolor="#F8FAFC", bordercolor="#E2E8F0"),
                height=450,
                title="Segment Lifestyle Dimension Profiles",
            )
            st.plotly_chart(fig, use_container_width=True)

            # User-level drill-down
            st.markdown("#### 🔍 User-Level Details")
            st.dataframe(
                seg[["User_ID","Full_Name","Age","Gender","Cluster_Label",
                      "Avg_Wellness_Score","Avg_Steps","Avg_Sleep_Hours",
                      "Avg_Stress"]].rename(columns={
                    "Full_Name":"Name","Cluster_Label":"Segment",
                    "Avg_Wellness_Score":"Avg Wellness",
                    "Avg_Steps":"Avg Steps","Avg_Sleep_Hours":"Avg Sleep",
                    "Avg_Stress":"Avg Stress"
                }).style.format({"Avg Wellness":"{:.1f}",
                                  "Avg Steps":"{:,.0f}",
                                  "Avg Sleep":"{:.1f}h",
                                  "Avg Stress":"{:.1f}"}),
                use_container_width=True, hide_index=True
            )

    # ══════════════════════════════════════
    # TAB 5 — PREDICT MY SCORE
    # ══════════════════════════════════════
    with tabs[4]:
        st.markdown("### 🔮 Predict My Wellness Score")
        st.info("For the full prediction experience, use the dedicated **Predict My Score** page from the sidebar.")
        if st.button("🔮 Go to Predict My Score", type="primary", key="wdash_predict_btn"):
            nav_to("predict_score"); st.rerun()
        st.markdown("---")
        st.markdown("**Quick prediction (embedded):**")
        st.markdown("*Note: For personalised recommendations, use the standalone Predict My Score page.*")
        # ── original predict content follows ──
        st.markdown(
            "Enter your daily lifestyle metrics below and the ML model will "
            "predict your Wellness Score (0–100)."
        )

        if ml_payload is None:
            st.warning("ML model not found. Run analytics/ml_wellness_predictor.py first.")
        else:
            pipeline     = ml_payload["pipeline"]
            feature_names = ml_payload["feature_names"]
            model_name   = ml_payload.get("model_name", "ML Model")

            with st.form("wellness_predict_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age    = st.number_input("Age", 10, 100, 30, key="wi_age")
                    gender = st.selectbox("Gender", ["Male", "Female"], key="wi_gender")
                    bmi    = st.number_input("BMI", 10.0, 50.0, 22.5, step=0.1, key="wi_bmi")
                    stress = st.slider("Stress Level (1–10)", 1, 10, 4, key="wi_stress")
                with c2:
                    steps      = st.number_input("Steps Taken", 0, 50000, 8000, step=500, key="wi_steps")
                    active_min = st.number_input("Active Minutes", 0, 480, 45, step=5, key="wi_active")
                    calories   = st.number_input("Calories Burned", 0, 5000, 500, step=50, key="wi_cal")
                with c3:
                    sleep  = st.number_input("Hours Slept", 0.0, 16.0, 7.0, step=0.5, key="wi_sleep")
                    water  = st.number_input("Water Intake (Litres)", 0.0, 8.0, 2.5, step=0.25, key="wi_water")
                    hr     = st.number_input("Heart Rate (bpm)", 40, 200, 72, key="wi_hr")
                    weekend = st.checkbox("Is Weekend?", value=False, key="wi_weekend")

                submitted = st.form_submit_button("🧠 Predict Wellness Score", use_container_width=True)

            if submitted:
                gender_enc = 1 if gender == "Male" else 0
                input_row  = [age, gender_enc, bmi, steps, active_min, calories,
                               sleep, water, hr, stress, int(weekend)]
                try:
                    prediction = float(pipeline.predict([input_row])[0])
                    prediction = max(0.0, min(100.0, prediction))

                    score_color = (
                        "#DC2626" if prediction < 40 else
                        "#D97706" if prediction < 60 else
                        "#16A34A" if prediction < 80 else
                        "#4F46E5"
                    )
                    score_label = (
                        "⚠️ Poor" if prediction < 40 else
                        "🔶 Fair" if prediction < 60 else
                        "✅ Good" if prediction < 80 else
                        "🌟 Excellent"
                    )

                    st.markdown(f"""
                    <div style='background:#FFFFFF;border:2px solid {score_color};
                                border-radius:16px;padding:28px;text-align:center;
                                margin:16px 0;'>
                      <div style='font-size:3.5rem;font-weight:900;
                                  color:{score_color};'>{prediction:.1f}</div>
                      <div style='font-size:1.4rem;color:#0F172A;
                                  font-weight:700;margin-top:8px;'>Wellness Score</div>
                      <div style='font-size:1rem;color:#475569;
                                  margin-top:4px;'>{score_label}</div>
                      <div style='font-size:0.85rem;color:#475569;
                                  margin-top:8px;'>Predicted by: {model_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gauge
                    st.plotly_chart(
                        _gauge(prediction, "Predicted Wellness Score", score_color),
                        use_container_width=True
                    )

                    # Interpretation
                    st.markdown("#### 💡 Key Lifestyle Recommendations")
                    tips = []
                    if sleep < 7:   tips.append(("😴", "Sleep",    "Aim for 7–9 hours of sleep per night."))
                    if stress > 6:  tips.append(("🧘", "Stress",   "Try mindfulness or breathing exercises to reduce stress."))
                    if water < 2.0: tips.append(("💧", "Hydration", "Drink at least 2–2.5 litres of water daily."))
                    if steps < 7000:tips.append(("🚶", "Activity",  "Target 7,000–10,000 steps per day."))
                    if hr > 100:    tips.append(("❤️", "Heart Rate","Elevated resting HR — consult a healthcare provider."))
                    if not tips:    tips.append(("🌟", "Keep it up!","Your lifestyle metrics look excellent!"))

                    for icon, title, tip in tips:
                        st.markdown(f"""
                        <div style='background:#EEF2FF;border-left:4px solid #4F46E5;
                                    border-radius:8px;padding:12px 16px;margin:6px 0;
                                    display:flex;align-items:flex-start;gap:12px;'>
                          <span style='font-size:1.4rem;'>{icon}</span>
                          <div>
                            <strong style='color:#0F172A;'>{title}</strong>
                            <div style='color:#475569;font-size:0.9rem;margin-top:2px;'>{tip}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")


def main():
    load_css()
    init_state()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "fitness_coach":
        render_fitness_coach()
    elif page == "workout":
        render_workout()
    elif page == "yoga":
        render_yoga()
    elif page == "upload":            # backward compat
        render_yoga()
    elif page == "summary":
        render_summary()
    elif page == "history":
        render_history()
    elif page == "wellness_dashboard":
        render_wellness_dashboard()
    elif page == "predict_score":
        render_predict_score()
    elif page == "recommendations":
        render_recommendations()
    else:
        render_home()


if __name__ == "__main__":
    main()
