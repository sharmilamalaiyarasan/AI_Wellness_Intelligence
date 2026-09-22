"""Reusable HTML/CSS component functions for the Streamlit app."""
from __future__ import annotations
from ui.theme import COLORS


# ── Primitive helpers ────────────────────────────────────────────────────────

def _pct(value: float, max_val: float = 100.0) -> float:
    return min(100.0, max(0.0, value / max_val * 100)) if max_val else 0.0


def progress_bar(value: float, max_val: float = 100.0,
                 color: str = "#4F46E5", height: int = 6) -> str:
    pct = _pct(value, max_val)
    return (
        f"<div style='background:#E2E8F0;border-radius:{height}px;"
        f"height:{height}px;overflow:hidden;margin-top:5px;'>"
        f"<div style='background:{color};height:100%;width:{pct:.0f}%;"
        f"border-radius:{height}px;'></div></div>"
    )


# ── Card components ──────────────────────────────────────────────────────────

def metric_card(icon: str, value, label: str,
                color: str | None = None,
                subtitle: str | None = None) -> str:
    color = color or COLORS["text"]
    sub = (f"<div style='font-size:11px;color:{COLORS['muted']};margin-top:2px;'>"
           f"{subtitle}</div>") if subtitle else ""
    return f"""
<div class="metric-card">
  <div style="font-size:1.5rem;margin-bottom:4px;">{icon}</div>
  <div class="metric-value" style="color:{color};">{value}</div>
  <div class="metric-label">{label}</div>
  {sub}
</div>"""


def kpi_card(icon: str, value, label: str, color: str = "#4F46E5") -> str:
    return f"""
<div class="kpi-card">
  <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
  <div class="kpi-value" style="color:{color};">{value}</div>
  <div class="kpi-label">{label}</div>
</div>"""


def snap_pill(icon: str, value, label: str) -> str:
    return f"""
<div class="snap-pill">
  <div style="font-size:1.4rem;">{icon}</div>
  <div class="snap-val">{value}</div>
  <div class="snap-lbl">{label}</div>
</div>"""


# ── Page structure ───────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str | None = None,
                badge: str | None = None) -> str:
    badge_html = (
        f"<span style='background:#EEF2FF;color:#4338CA;font-size:11px;"
        f"font-weight:700;padding:3px 10px;border-radius:20px;"
        f"margin-left:12px;'>{badge}</span>"
    ) if badge else ""
    sub_html = (f"<div class='page-sub'>{subtitle}</div>") if subtitle else ""
    return f"""
<div style="margin-bottom:24px;">
  <div class='page-title'>{title}{badge_html}</div>
  {sub_html}
</div>"""


def section_header(title: str, subtitle: str | None = None,
                   icon: str | None = None) -> str:
    prefix = f"<span style='margin-right:8px;'>{icon}</span>" if icon else ""
    sub_html = (
        f"<div class='sec-sub'>{subtitle}</div>"
    ) if subtitle else ""
    return f"""
<div class="sec-head">{prefix}{title}</div>
{sub_html}"""


# ── Content cards ────────────────────────────────────────────────────────────

def insight_card(icon: str, title: str, content: str,
                 color: str = "#4F46E5") -> str:
    return f"""
<div style="background:#FFFFFF;border-left:4px solid {color};border-radius:12px;
            padding:16px 20px;margin:8px 0;border:1px solid #E2E8F0;
            box-shadow:0 2px 8px rgba(15,23,42,0.04);">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span style="font-size:1.3rem;">{icon}</span>
    <strong style="font-size:14px;color:#0F172A;">{title}</strong>
  </div>
  <div style="font-size:13px;color:#475569;line-height:1.6;">{content}</div>
</div>"""


def recommendation_card(icon: str, category: str, problem: str,
                        insight: str, action: str,
                        color: str = "#4F46E5") -> str:
    sev_color = (
        "#DC2626" if "⚠️" in problem else
        "#16A34A" if any(s in problem for s in ("✅", "🌟")) else
        "#D97706"
    )
    return f"""
<div class="rec-card">
  <div style="display:flex;align-items:flex-start;gap:14px;">
    <div class="rec-icon-box" style="background:#F0F4FF;">{icon}</div>
    <div style="flex:1;">
      <div style="font-size:16px;font-weight:700;color:#0F172A;">{category}</div>
      <div style="font-size:12px;font-weight:600;color:{sev_color};margin-top:2px;">{problem}</div>
      <div style="font-size:13px;color:#475569;margin-top:10px;line-height:1.6;">{insight}</div>
      <div class="rec-action">
        <span style="font-size:10px;font-weight:700;color:#16A34A;text-transform:uppercase;letter-spacing:.5px;">Recommended Action</span>
        <div style="font-size:13px;color:#0F172A;font-weight:500;margin-top:3px;">{action}</div>
      </div>
    </div>
  </div>
</div>"""


def empty_state(icon: str, title: str, subtitle: str,
                cta: str | None = None) -> str:
    cta_html = (
        f"<p style='font-size:13px;color:#4F46E5;font-weight:600;margin-top:12px;'>{cta}</p>"
    ) if cta else ""
    return f"""
<div style="text-align:center;padding:60px 24px;background:#FFFFFF;
            border-radius:20px;border:2px dashed #E2E8F0;margin:20px 0;">
  <div style="font-size:3rem;margin-bottom:12px;">{icon}</div>
  <div style="font-size:16px;font-weight:700;color:#0F172A;margin-bottom:6px;">{title}</div>
  <div style="font-size:13px;color:#64748B;">{subtitle}</div>
  {cta_html}
</div>"""


# ── Wellness-specific ────────────────────────────────────────────────────────

def wellness_score_display(score: float, label: str = "Wellness Score") -> str:
    if score >= 80:
        color, status = "#16A34A", "Excellent"
    elif score >= 65:
        color, status = "#0891B2", "Good"
    elif score >= 50:
        color, status = "#D97706", "Fair"
    else:
        color, status = "#DC2626", "Needs Improvement"
    return f"""
<div style="text-align:center;padding:24px;">
  <div style="font-size:4rem;font-weight:900;color:{color};line-height:1;">{score:.0f}</div>
  <div style="font-size:14px;color:#475569;font-weight:600;margin-top:4px;">/ 100 — {label}</div>
  <div style="display:inline-block;background:#EEF2FF;color:{color};
              font-size:12px;font-weight:700;padding:4px 14px;border-radius:20px;margin-top:8px;">
    {status}
  </div>
</div>"""


def dimension_row(icon: str, label: str, score: float,
                  color: str = "#4F46E5", note: str = "") -> str:
    bar = progress_bar(score, color=color, height=5)
    return f"""
<div class="dim-row">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span>{icon}</span>
      <span style="font-size:13px;font-weight:600;color:#0F172A;">{label}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:11px;color:#94A3B8;">{note}</span>
      <span style="font-size:14px;font-weight:800;color:{color};">{score:.0f}</span>
    </div>
  </div>
  {bar}
</div>"""
