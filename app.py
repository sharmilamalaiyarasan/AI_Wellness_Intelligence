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
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

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
from core.voice_coach       import (VoiceCoach, get_voice_audio_html, translate_text,
                                    get_ui_text, UI_STRINGS, BODY_PARTS_TA)
from database.database      import init_db, save_workout, get_all_workouts, get_aggregate_stats

# ── App config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness & Yoga Coach",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(APP_DIR, "assets")

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

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #5B51D8 0%, #4338CA 55%, #312E81 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        width: 100% !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-align: left !important;
        margin-bottom: 6px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        color: #FFFFFF !important;
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
        st.markdown("""
        <div style='padding:24px 20px 16px; border-bottom:1px solid rgba(255,255,255,0.15);'>
          <div style='display:flex;align-items:center;gap:12px;'>
            <div style='background:rgba(255,255,255,0.15);border-radius:12px;
                        width:40px;height:40px;display:flex;align-items:center;
                        justify-content:center;font-size:20px;'>🧘</div>
            <div>
              <div style='font-size:16px;font-weight:800;color:#fff;'>AI Fitness & Yoga Coach</div>
              <div style='font-size:12px;color:rgba(255,255,255,0.75);'>Trained Dataset ({count} Poses)</div>
            </div>
          </div>
        </div>
        """.replace("{count}", str(len(SUPPORTED_EXERCISES))), unsafe_allow_html=True)

        st.markdown("<div style='padding:16px 8px 0;'>", unsafe_allow_html=True)

        if st.button("🏠  Dashboard", use_container_width=True):
            _safe_close_cam()
            nav_to("home"); st.rerun()

        if st.button("▶️  Live Workout", use_container_width=True):
            nav_to("workout"); st.rerun()

        if st.button("📷  Upload Image Analysis", use_container_width=True):
            _safe_close_cam()
            nav_to("upload"); st.rerun()

        if st.button("📊  History", use_container_width=True):
            _safe_close_cam()
            nav_to("history"); st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:14px 8px;border-color:rgba(255,255,255,0.15);'>",
                    unsafe_allow_html=True)

        # ── Voice Coach Controls Panel ──
        st.markdown("""
        <div style='padding:0 8px;'>
          <div style='font-size:11px;font-weight:800;color:rgba(255,255,255,0.85);
                      text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px;'>
            <span>🔊</span> AI Voice Coach (Multilingual)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Language Selector
        lang_options = {
            "🇬🇧 English (Default)": "en",
            "🇮🇳 தமிழ் (Tamil)": "ta"
        }
        current_lang = st.session_state.get("voice_lang", "en")
        default_idx = 1 if current_lang == "ta" else 0

        selected_label = st.selectbox(
            "Select Voice & Hint Language:",
            options=list(lang_options.keys()),
            index=default_idx,
            key="voice_lang_select"
        )
        new_lang = lang_options[selected_label]
        if new_lang != st.session_state.voice_lang:
            st.session_state.voice_lang = new_lang
            if "voice_coach" in st.session_state and st.session_state.voice_coach:
                st.session_state.voice_coach.set_language(new_lang)

        # Enable Voice Toggle
        voice_on = st.checkbox(
            "🔊 Enable Spoken Voice Commands",
            value=st.session_state.get("voice_enabled", True),
            key="voice_enabled_chk"
        )
        st.session_state.voice_enabled = voice_on
        if "voice_coach" in st.session_state and st.session_state.voice_coach:
            st.session_state.voice_coach.set_enabled(voice_on)

        # Quick Voice Test Button
        if st.button("🔊 Test Voice Guidance", use_container_width=True):
            test_phrase = "Lift your arm higher." if new_lang == "en" else "உங்கள் கையை மேலே உயர்த்தவும்."
            st.session_state.test_speech_html = get_voice_audio_html(test_phrase, new_lang)
            st.rerun()

        if st.session_state.get("test_speech_html"):
            st.markdown(st.session_state.test_speech_html, unsafe_allow_html=True)
            st.session_state.test_speech_html = ""

        st.markdown("<hr style='margin:14px 8px;border-color:rgba(255,255,255,0.15);'>",
                    unsafe_allow_html=True)

        stats = get_aggregate_stats()
        st.markdown(f"""
        <div style='padding:0 8px 16px;'>
          <div style='font-size:11px;font-weight:700;color:rgba(255,255,255,0.6);
                      text-transform:uppercase;margin-bottom:10px;'>
            Trained Dataset Stats
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
            <div style='background:rgba(255,255,255,0.12);border-radius:10px;padding:10px;text-align:center;'>
              <div style='font-size:1.4rem;font-weight:800;color:#fff;'>{len(SUPPORTED_EXERCISES)}</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.75);'>Pose Classes</div>
            </div>
            <div style='background:rgba(255,255,255,0.12);border-radius:10px;padding:10px;text-align:center;'>
              <div style='font-size:1.4rem;font-weight:800;color:#fff;'>{stats.get('total_reps',0)}</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.75);'>Total Reps</div>
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
#  PAGE 1 — HOME / EXERCISE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def render_home():
    col_title, col_start = st.columns([3, 1])
    with col_title:
        st.markdown("<div class='page-title'>Welcome! 👋</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-sub'>AI Fitness & Yoga Coach </div>",
                    unsafe_allow_html=True)
    with col_start:
        if st.button("▶  Start Workout", type="primary", use_container_width=True):
            nav_to("workout"); st.rerun()

    stats = get_aggregate_stats()
    total_min = (stats.get("total_seconds", 0) or 0) // 60
    avg_acc   = stats.get("avg_accuracy", 0) or 0

    s1, s2, s3, s4 = st.columns(4)
    for col, icon, val, lbl in [
        (s1, "🧘", len(SUPPORTED_EXERCISES),       "Poses"),
        (s2, "🔁", stats.get("total_reps", 0),     "Total Reps"),
        (s3, "⏱️", f"{total_min}m",                "Workout Time"),
        (s4, "✅", f"{avg_acc:.0f}%",               "Avg Accuracy"),
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
        st.markdown(f"<div style='font-size:1.1rem;font-weight:800;color:#0F172A;margin-bottom:10px;'>🎯 Select Pose / Exercise ({len(SUPPORTED_EXERCISES)} Available)</div>",
                    unsafe_allow_html=True)

        disp_options = {POSE_DISPLAY_NAMES.get(ex, ex): ex for ex in SUPPORTED_EXERCISES}
        selected_disp = st.selectbox(
            "Search or select a pose :",
            options=list(disp_options.keys()),
            index=0
        )
        st.session_state.selected_exercise = disp_options[selected_disp]

        ex = st.session_state.selected_exercise
        info = EXERCISE_INFO[ex]

        asset_img_name = ex.replace(" ", "_").lower() + ".jpg"
        asset_img_path = os.path.join(ASSETS_DIR, asset_img_name)

        st.markdown("<br>", unsafe_allow_html=True)
        c_img, c_meta = st.columns([1, 2])
        with c_img:
            if os.path.exists(asset_img_path):
                st.image(asset_img_path, caption=info['display_name'], use_container_width=True)
            else:
                st.markdown("<div style='font-size:5rem;text-align:center;'>🧘</div>", unsafe_allow_html=True)
        with c_meta:
            st.markdown(f"""
            <div class='fitness-card' style='border-top:4px solid #4F46E5;'>
              <div style='font-size:20px;font-weight:900;color:#0F172A;'>{info['display_name']}</div>
              <div style='font-size:12px;color:#4338CA;font-weight:700;margin:4px 0 10px;'>{info['category']}</div>
              <div style='font-size:13px;color:#334155;line-height:1.5;'>{info['description']}</div>
              <div style='margin-top:12px;font-size:13px;font-weight:700;color:#3730A3;background:#EEF2FF;padding:6px 12px;border-radius:10px;display:inline-block;'>
                {info['sets']}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button(f"▶  Start {info['display_name']} Workout", type="primary", use_container_width=True):
            nav_to("workout"); st.rerun()

    with right_col:
        st.markdown(f"""
        <div class="fitness-card">
          <div style='font-size:1rem;font-weight:800;color:#0F172A;margin-bottom:10px;'>
            🤖 AI Model Info
          </div>
          <div style='font-size:13px;color:#334155;line-height:1.6;'>
            <strong style='color:#0F172A;'>Classifier:</strong> Random Forest Regressor/Classifier<br>
            <strong style='color:#0F172A;'>Trained On:</strong> Kaggle Yoga Posture Dataset<br>
            <strong style='color:#0F172A;'>Pose Classes:</strong> {len(SUPPORTED_EXERCISES)} Poses<br>
            <strong style='color:#0F172A;'>Features:</strong> 8 3D Anatomical Joint Angles<br>
            <strong style='color:#0F172A;'>Landmarks:</strong> MediaPipe 33 Keypoints
          </div>
        </div>
        """, unsafe_allow_html=True)

        recent = get_all_workouts()[:3]
        if recent:
            st.markdown("<div style='font-size:1rem;font-weight:800;color:#0F172A;margin:8px 0;'>🕐 Recent Sessions</div>",
                        unsafe_allow_html=True)
            for w in recent:
                acc = w.get("form_accuracy", 0) or 0
                color = _score_color(acc)
                st.markdown(f"""
                <div style='background:#fff;border-radius:12px;padding:10px 14px;
                            margin-bottom:8px;border:1px solid #E2E8F0;
                            display:flex;justify-content:space-between;align-items:center;'>
                  <div>
                    <div style='font-size:13px;font-weight:700;color:#0F172A;'>
                      🧘 {w["exercise"]}
                    </div>
                    <div style='font-size:11px;color:#64748B;'>{w["date"]}</div>
                  </div>
                  <div style='text-align:right;'>
                    <div style='font-size:14px;font-weight:800;color:{color};'>{acc:.0f}%</div>
                    <div style='font-size:11px;color:#64748B;'>{w["total_reps"]} reps</div>
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
                    v_html = vc.trigger_speech(rep_msg)
                    voice_placeholder.markdown(v_html, unsafe_allow_html=True)
            elif vc.can_speak():
                en_cmd, loc_cmd = vc.extract_voice_command(feedback, score)
                if loc_cmd:
                    v_html = vc.trigger_speech(loc_cmd)
                    voice_placeholder.markdown(v_html, unsafe_allow_html=True)

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
def render_upload():
    lang = st.session_state.get("voice_lang", "en")
    st.markdown("<div class='page-title'>📷 Upload Photo Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Upload any yoga photo to auto-detect exercise class and listen to spoken posture corrections in English & Tamil.</div>",
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
                if st.button("🇬🇧 Listen in English", use_container_width=True):
                    audio_html = get_voice_audio_html(top_cue_en, "en")
                    st.markdown(audio_html, unsafe_allow_html=True)
            with v_btn2:
                if st.button("🇮🇳 தமிழில் கேட்கவும்", use_container_width=True):
                    audio_html = get_voice_audio_html(top_cue_ta, "ta")
                    st.markdown(audio_html, unsafe_allow_html=True)

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
        if st.button("🇬🇧 Listen in English", use_container_width=True):
            audio_html = get_voice_audio_html(summary_cue_en, "en")
            st.markdown(audio_html, unsafe_allow_html=True)
    with s_col2:
        if st.button("🇮🇳 தமிழில் கேட்கவும்", use_container_width=True):
            audio_html = get_voice_audio_html(summary_cue_ta, "ta")
            st.markdown(audio_html, unsafe_allow_html=True)

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
#  PAGE 5 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
def render_history():
    st.markdown("<div class='page-title'>📊 Workout History</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Track your progress over time.</div>", unsafe_allow_html=True)

    workouts = get_all_workouts()
    if not workouts:
        st.info("No workout records saved yet.")
        if st.button("▶ Start First Workout", type="primary"):
            nav_to("home"); st.rerun()
        return

    st.markdown("""
    <div style='display:grid;grid-template-columns:2fr 2fr 1fr 1fr 1fr 1fr 1fr;
                gap:8px;background:#E2E8F0;border-radius:12px;
                padding:10px 16px;margin-bottom:6px;'>
      <div style='font-size:11px;font-weight:700;color:#334155;'>Date</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>Exercise</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>Reps</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>✅ Correct</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>❌ Bad</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>Accuracy</div>
      <div style='font-size:11px;font-weight:700;color:#334155;'>Duration</div>
    </div>
    """, unsafe_allow_html=True)

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
          <div style='font-size:14px;font-weight:700;color:#0F172A;'>{w["total_reps"]}</div>
          <div style='font-size:14px;font-weight:700;color:#16A34A;'>{w["correct_reps"]}</div>
          <div style='font-size:14px;font-weight:700;color:#DC2626;'>{w["incorrect_reps"]}</div>
          <div style='font-size:14px;font-weight:800;color:{color};'>{acc:.0f}%</div>
          <div style='font-size:13px;color:#475569;'>{dur}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("← Back to Dashboard"):
        nav_to("home"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    load_css()
    init_state()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "workout":
        render_workout()
    elif page == "upload":
        render_upload()
    elif page == "summary":
        render_summary()
    elif page == "history":
        render_history()
    else:
        render_home()


if __name__ == "__main__":
    main()
