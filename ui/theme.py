"""Design system — color tokens and Plotly chart theme factory."""

COLORS = {
    "primary":    "#4F46E5",
    "primary_lt": "#6366F1",
    "bg":         "#F4F6FC",
    "card":       "#FFFFFF",
    "border":     "#E2E8F0",
    "grid":       "#F1F5F9",
    "text":       "#0F172A",
    "text2":      "#475569",
    "muted":      "#94A3B8",
    "success":    "#16A34A",
    "warning":    "#D97706",
    "danger":     "#DC2626",
    # Dimension accents
    "sleep":      "#6366F1",
    "activity":   "#0891B2",
    "stress":     "#E11D48",
    "hydration":  "#059669",
    "heart":      "#DC2626",
}


def chart_layout(title: str = "", height: int = 320,
                 showlegend: bool = True, **kwargs) -> dict:
    """Return a Plotly layout dict matching the app's light theme."""
    layout: dict = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=20, r=20, t=44 if title else 20, b=20),
        showlegend=showlegend,
        legend=dict(
            bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1,
            font=dict(color="#0F172A", size=11),
        ),
        xaxis=dict(
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            tickfont=dict(color="#475569", size=11),
            title_font=dict(color="#475569"),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9", linecolor="#E2E8F0",
            tickfont=dict(color="#475569", size=11),
            title_font=dict(color="#475569"),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF", bordercolor="#E2E8F0",
            font=dict(color="#0F172A"),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(color="#0F172A", size=14, family="Inter"),
            x=0.02, pad=dict(l=4),
        )
    layout.update(kwargs)
    return layout
