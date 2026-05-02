# ==========================================================
# AI CRICKET PERFORMANCE PREDICTION - PRODUCTION VERSION
# ==========================================================

from pathlib import Path

# ── Resolve project root so file paths work from any CWD ──
ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import time
from src.api.cricket_api import (
    fetch_live_matches,
    filter_ipl_matches,
    fetch_match_score,
    extract_score_from_match,
    transform_api_to_features,
    format_match_label,
)

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI Cricket Performance",
    page_icon="",
    layout="wide"
)

# ----------------------------------------------------------
# PREMIUM DARK THEME – GLASSMORPHISM / SAAS DASHBOARD
# ----------------------------------------------------------
st.markdown("""
<style>
/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #f1f5f9;
    --border-subtle: rgba(0, 0, 0, 0.08);
    --accent-blue: #0f172a;
    --success-green: #22c55e;
    --danger-red: #ef4444;
    --text-primary: #000000;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.08);
    --transition-smooth: all 0.2s ease;
}

@media (prefers-color-scheme: light) {
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-card: #f1f5f9;
        --border-subtle: rgba(0, 0, 0, 0.08);
        --accent-blue: #0f172a;
        --success-green: #22c55e;
        --danger-red: #ef4444;
        --text-primary: #000000;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
}

/* ===== GLOBAL ===== */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.015em;
    font-weight: 600 !important;
}

h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }
h4 { font-size: 1rem !important; }

p, span, label, div {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text-secondary);
}

hr { border-color: var(--border-subtle) !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(134, 134, 139, 0.3); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(134, 134, 139, 0.5); }

/* ===== GLASS CARD -> FLAT CARD ===== */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 32px;
    text-align: left;
    box-shadow: var(--shadow-soft);
    margin-bottom: 24px;
}

/* ===== HERO KPI CARD ===== */
.kpi-hero {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    padding: 32px;
    border-radius: 16px;
    text-align: left;
    box-shadow: var(--shadow-soft);
}
.kpi-hero .kpi-label {
    font-size: 13px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.kpi-hero .kpi-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-primary);
}
.kpi-hero .kpi-value {
    font-size: 56px;
    font-weight: 600;
    letter-spacing: -1.5px;
    line-height: 1;
    margin: 16px 0;
    color: var(--text-primary);
}
.kpi-hero .kpi-sub {
    font-size: 14px;
    color: var(--text-secondary);
    font-weight: 400;
}

/* ===== RUNS & WICKETS CLASSES (NOW UNIFIED) ===== */
.kpi-runs, .kpi-wickets {
    /* Removed gradients, unified under flat card style */
}

/* ===== CI CARD ===== */
.ci-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    text-align: left;
    box-shadow: var(--shadow-soft);
}
.ci-card .ci-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 8px;
}
.ci-card .ci-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
}

/* ===== CONTEXT METRICS ===== */
.metric-mini {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: left;
}
.metric-mini .metric-icon {
    display: none; /* Hide icons to reduce noise */
}
.metric-mini .metric-label {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 8px;
}
.metric-mini .metric-val {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}

/* ===== SECTION HEADER ===== */
.section-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}
.section-header .section-icon {
    display: none; /* Hide flashy icons */
}
.section-header .section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.section-header .section-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 4px;
}

/* ===== DIVIDER ===== */
.section-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 48px 0;
}

/* ===== BUTTONS ===== */
[data-testid="stButton"] > button {
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    transition: var(--transition-smooth) !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button * {
    color: #ffffff !important;
}
[data-testid="stButton"] > button:hover {
    filter: brightness(0.9);
}

/* ===== DOWNLOAD BUTTON ===== */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    transition: var(--transition-smooth) !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,0,0,0.05) !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(255,255,255,0.05) !important;
    }
}

/* ===== INPUTS ===== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div {
    background: var(--bg-card) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    color: var(--text-primary) !important;
    background-color: transparent !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}
[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    color: var(--text-primary) !important;
    border: none !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within {
    border-color: var(--accent-blue) !important;
}

/* ===== TABS ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    gap: 24px;
    border-bottom: 1px solid var(--border-subtle);
    padding: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    color: var(--text-secondary) !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--text-primary) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none;
}

/* ===== RADIO BUTTONS ===== */
[data-testid="stRadio"] > div {
    display: flex;
    gap: 12px;
}
[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 20px;
    cursor: pointer;
    font-weight: 400;
}
[data-testid="stRadio"] label:hover {
    background: rgba(0,0,0,0.02);
}
@media (prefers-color-scheme: dark) {
    [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.02);
    }
}

/* ===== WARNING / INFO ===== */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}

/* ===== OVERVIEW FEATURES ===== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 32px;
}
.feature-item {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    text-align: left;
    box-shadow: var(--shadow-soft);
}
.feature-item .feat-icon {
    display: none; /* Hide icons for minimalism */
}
.feature-item .feat-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}
.feature-item .feat-desc {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ===== HERO BANNER ===== */
.hero-banner {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: left;
    margin-bottom: 32px;
    box-shadow: var(--shadow-soft);
}
.hero-banner .hero-icon { display: none; }
.hero-banner .hero-title {
    font-size: 32px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}
.hero-banner .hero-sub {
    font-size: 16px;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 600px;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ===== PLOTLY CHARTS WRAPPER ===== */
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    background: var(--bg-card);
    padding: 16px;
}

/* ===== BLOCK CONTAINER ===== */
.block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* ===== LABEL STYLING ===== */
[data-testid="stWidgetLabel"] p {
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ===== LIVE MODE INDICATOR ===== */
.live-pulse {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success-green);
    margin-right: 8px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    color: var(--success-green);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.api-status-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-soft);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">AI Powered</div>
    <div class="hero-icon">📈</div>
    <div class="hero-title">Cricket Performance Intelligence</div>
    <div class="hero-sub">
        Advanced machine learning models with SHAP explainability,
        uncertainty quantification, and real-time match context analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    batsman_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "batsman_model_data.csv")
    bowler_df = pd.read_csv(ROOT_DIR / "data" / "processed" / "bowler_model_data.csv")
    return batsman_df, bowler_df

batsman_df, bowler_df = load_data()

# ----------------------------------------------------------
# LOAD MODELS & PIPELINES
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    runs_model = joblib.load(ROOT_DIR / "models" / "rf_batsman_model.joblib")
    runs_pipeline = joblib.load(ROOT_DIR / "models" / "feature_pipeline_batsman.pkl")

    wickets_model = joblib.load(ROOT_DIR / "models" / "rf_wickets_model.joblib")
    wickets_pipeline = joblib.load(ROOT_DIR / "models" / "feature_pipeline_bowler.pkl")

    return runs_model, runs_pipeline, wickets_model, wickets_pipeline

runs_model, runs_pipeline, wickets_model, wickets_pipeline = load_models()

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def compute_uncertainty(model, pipeline, df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_processed = pipeline.transform(X)
    y_pred = model.predict(X_processed)
    residuals = y - y_pred
    return np.std(residuals)


def generate_llm_explanation(shap_df, prediction, label):
    lines = []
    lines.append(f" The model predicts <strong>{round(prediction,2)} {label.lower()}</strong> based on:")
    lines.append("")

    for i, row in shap_df.head(3).iterrows():
        feature = row["Feature"]
        impact = row["Impact"]
        impact_val = round(abs(impact), 3)

        if impact > 0:
            lines.append(f"<strong>{feature.replace('_',' ').title()}</strong> is positively influencing performance <span style='color:#10b981;font-size:13px;'>(+{impact_val})</span>")
        else:
            lines.append(f"<strong>{feature.replace('_',' ').title()}</strong> is negatively impacting performance <span style='color:#f87171;font-size:13px;'>(-{impact_val})</span>")

    lines.append("")
    lines.append("Overall, the prediction reflects current form, historical trends, and match conditions.")

    return "<br>".join(lines)


def shap_local_explanation(model, pipeline, X_processed):
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    feature_names = pipeline.get_feature_names_out()

    numeric_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("num__")
    ]

    shap_vals = shap_values[0][numeric_indices]
    clean_names = [
        feature_names[i].replace("num__", "")
        for i in numeric_indices
    ]

    shap_df = pd.DataFrame({
        "Feature": clean_names,
        "Impact": shap_vals
    }).sort_values(by="Impact", key=abs, ascending=False).head(10)

    fig = px.bar(
        shap_df,
        x="Impact",
        y="Feature",
        orientation="h",
        template="plotly_white",
        title="Top Feature Contributions",
        color="Impact",
        color_continuous_scale=["#ec4899", "#64748b", "#3b82f6"]
    )
    fig.update_layout(
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )

    return fig, shap_df


def probability_distribution(prediction, std_dev, label):
    x_range = np.linspace(
        prediction - 4 * std_dev,
        prediction + 4 * std_dev,
        500
    )

    pdf = norm.pdf(x_range, prediction, std_dev)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf,
        mode="lines",
        fill="tozeroy",
        name="Density",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59, 130, 246, 0.15)"
    ))

    fig.add_vline(
        x=prediction,
        line_dash="dash",
        line_color="#10b981",
        annotation_text="Prediction",
        annotation_font=dict(color="#10b981", size=13, family="Inter")
    )

    lower = prediction - 1.96 * std_dev
    upper = prediction + 1.96 * std_dev

    fig.add_vrect(
        x0=lower,
        x1=upper,
        fillcolor="rgba(16, 185, 129, 0.1)",
        line_width=0,
        annotation_text="95% CI",
        annotation_font=dict(color="#10b981", size=12, family="Inter")
    )

    fig.update_layout(
        template="plotly_white",
        title=f"{label} Prediction Distribution",
        xaxis_title=label,
        yaxis_title="Density",
        height=500,
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig, lower, upper


def generate_match_context(score, overs, wickets):
    if overs == 0:
        run_rate = 0
    else:
        run_rate = score / overs
    wickets_left = 10 - wickets
    wickets_left = max(wickets_left, 1)
    pressure_index = run_rate / wickets_left
    return run_rate, wickets, wickets_left, pressure_index

# ----------------------------------------------------------
# RESIDUAL PLOT
# ----------------------------------------------------------
def residual_plot(model, pipeline, df, target_col):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_processed = pipeline.transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    preds = model.predict(X_processed)
    residuals = y.values - preds

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=preds,
        y=residuals,
        mode="markers",
        opacity=0.6,
        name="Residuals",
        marker=dict(color="#8b5cf6", size=5)
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#ef4444"
    )

    fig.update_layout(
        template="plotly_white",
        title="Residual Analysis (Predicted vs Residuals)",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=450,
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

# ----------------------------------------------------------
# REUSABLE LIVE SCORE COMPONENT
# ----------------------------------------------------------
def render_live_score(data):
    """
    Renders a unified Live Score card via st.components.v1.html.
    data dict keys: team_a, team_b, stadium, runs, wickets, overs, run_rate
    """
    team_a = data.get("team_a", "Team A")
    team_b = data.get("team_b", "Team B")
    stadium = data.get("stadium", "Unknown Stadium")
    runs = data.get("runs", 0)
    wickets = data.get("wickets", 0)
    overs_val = data.get("overs", 0)
    run_rate = data.get("run_rate", 0.0)

    # Match status heuristic
    if runs == 0 and overs_val == 0 and wickets == 0:
        match_status = "Waiting for first ball"
        momentum = "Neutral"
        momentum_color = "#64748b"
        rr_eval = "N/A"
        rr_color = "#64748b"
        win_prob = 50
        lose_prob = 50
        
        score_html = f"""
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
                    {data.get('status', 'Match starting...')}
                </div>
                <div style="font-size: 13px; color: #64748b; font-weight: 500;">
                    Scorecard not yet available from provider
                </div>
            </div>
        """
    else:
        if run_rate > 8:
            match_status = f"{team_a} is in control"
        elif run_rate > 6:
            match_status = "Match is balanced"
        else:
            match_status = f"{team_b} gaining advantage"
    
        baseline_rr = 8.0
        rr_eval = "Above par" if run_rate >= baseline_rr else "Below par"
    
        if run_rate >= 8 and wickets <= 4:
            momentum = "Strong"
        elif run_rate >= 6 and wickets <= 6:
            momentum = "Stable"
        else:
            momentum = "Dropping"
            
        momentum_color = "#22c55e" if momentum == "Strong" else ("#f59e0b" if momentum == "Stable" else "#ef4444")
        rr_color = "#22c55e" if rr_eval == "Above par" else "#f59e0b"
        
        # Win probability
        remaining_overs = max(20 - overs_val, 1)
        wickets_in_hand = max(10 - wickets, 0)
        win_prob = min(100, max(0, round(
            40 + (run_rate - 7) * 5 + wickets_in_hand * 3 - (20 - remaining_overs) * 0.8, 1
        )))
        lose_prob = round(100 - win_prob, 1)

        score_html = f"""
            <div>
                <div style="font-size: 52px; font-weight: 800; color: #000000;
                    letter-spacing: -2px; line-height: 1;
                    background: linear-gradient(135deg, #000000 0%, #334155 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {runs}/{wickets}
                </div>
                <div style="font-size: 13px; color: #64748b; margin-top: 8px; font-weight: 400;">
                    After {overs_val} overs
                </div>
            </div>
        """

    st.components.v1.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @keyframes livePulse {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }}
            50% {{ opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>

    <div style="
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 20px;
        padding: 0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #000000;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.02) inset;
        animation: fadeIn 0.4s ease;">

        <!-- MATCH HEADER -->
        <div style="
            display: flex; align-items: center; justify-content: space-between;
            padding: 16px 24px;
            background: linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(139,92,246,0.05) 100%);
            border-bottom: 1px solid rgba(0,0,0,0.06);">
            <div>
                <div style="
                    display: flex; align-items: center; gap: 8px;
                    font-size: 10px; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1.5px;
                    color: #475569; margin-bottom: 6px;">
                    ⚡ <span>Live Match • {{match_status}}</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #000000; letter-spacing: -0.5px;">
                    {team_a} <span style="color:#64748b; font-weight:400;">vs</span> {team_b}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 5px; display:flex; align-items:center; gap:4px;">
                    📍 {stadium}
                </div>
            </div>
            <div style="
                display: flex; align-items: center; gap: 8px;
                background: rgba(34,197,94,0.1);
                border: 1px solid rgba(34,197,94,0.2);
                border-radius: 20px; padding: 6px 14px;">
                <span style="display: inline-block; width: 8px; height: 8px;
                    border-radius: 50%; background: #22c55e;
                    animation: livePulse 2s ease-in-out infinite;"></span>
                <span style="font-size: 11px; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 1.2px;
                    color: #22c55e;">Live</span>
            </div>
        </div>

        <!-- SCORE PANEL -->
        <div style="
            display: flex; align-items: center; justify-content: space-between;
            padding: 28px 28px 24px 28px;
            border-bottom: 1px solid rgba(0,0,0,0.04);">
            {score_html}
            <div style="display: flex; gap: 6px;">
                <div style="text-align: center; background: rgba(0,0,0,0.02);
                    border: 1px solid rgba(0,0,0,0.06); border-radius: 14px;
                    padding: 14px 20px; min-width: 80px;">
                    <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
                        color: #64748b; font-weight: 600; margin-bottom: 6px;">Overs</div>
                    <div style="font-size: 22px; font-weight: 700; color: #000000;">{overs_val}</div>
                </div>
                <div style="text-align: center; background: rgba(0,0,0,0.02);
                    border: 1px solid rgba(0,0,0,0.06); border-radius: 14px;
                    padding: 14px 20px; min-width: 80px;">
                    <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
                        color: #64748b; font-weight: 600; margin-bottom: 6px;">Wickets</div>
                    <div style="font-size: 22px; font-weight: 700; color: #000000;">{wickets}</div>
                </div>
                <div style="text-align: center; background: rgba(0,0,0,0.02);
                    border: 1px solid rgba(0,0,0,0.06); border-radius: 14px;
                    padding: 14px 20px; min-width: 120px;">
                    <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
                        color: #64748b; font-weight: 600; margin-bottom: 6px;">Run Rate</div>
                    <div style="font-size: 22px; font-weight: 700; color: #000000;">
                        {run_rate}
                    </div>
                    <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 4px;">
                        <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                            background:{rr_color};"></span>
                        <span style="font-size: 10px; color: {rr_color}; font-weight: 500;">{rr_eval}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- MOMENTUM STRIP -->
        <div style="display:flex; align-items:center; justify-content:center; gap:8px;
            padding: 10px 24px; background: rgba(0,0,0,0.015);
            border-bottom: 1px solid rgba(0,0,0,0.04);">
            <span style="font-size:10px; text-transform:uppercase; letter-spacing:1.5px;
                color:#64748b; font-weight:600;">Momentum</span>
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                background:{momentum_color};"></span>
            <span style="font-size:12px; font-weight:600; color:{momentum_color};">{momentum}</span>
        </div>

        <!-- MATCH PROBABILITY -->
        <div style="padding: 20px 28px 24px 28px;">
            <div style="display: flex; align-items: center; gap: 8px;
                font-size: 10px; font-weight: 600;
                text-transform: uppercase; letter-spacing: 1.5px;
                color: #64748b; margin-bottom: 12px;">
                📊 <span>Match Probability</span>
            </div>
            <div style="display: flex; width: 100%; height: 8px;
                border-radius: 10px; overflow: hidden; background: #e2e8f0;">
                <div style="width: {win_prob}%; background: linear-gradient(90deg, #22c55e, #4ade80);
                    border-radius: 10px 0 0 10px; transition: width 0.6s cubic-bezier(0.4,0,0.2,1);"></div>
                <div style="width: {lose_prob}%; background: linear-gradient(90deg, #f87171, #ef4444);
                    border-radius: 0 10px 10px 0; transition: width 0.6s cubic-bezier(0.4,0,0.2,1);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                        background: linear-gradient(135deg, #22c55e, #4ade80);"></span>
                    <span style="font-size: 13px; font-weight: 600; color: #22c55e;">{team_a}</span>
                    <span style="font-size: 13px; font-weight: 700; color: #22c55e;">{win_prob}%</span>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-size: 13px; font-weight: 600; color: #ef4444;">{team_b}</span>
                    <span style="font-size: 13px; font-weight: 700; color: #ef4444;">{lose_prob}%</span>
                    <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                        background: linear-gradient(135deg, #f87171, #ef4444);"></span>
                </div>
            </div>
        </div>
    </div>
    """.replace("{{match_status}}", match_status), height=380)


# ----------------------------------------------------------
# SHAP GLOBAL IMPORTANCE (BEESWARM STYLE)
# ----------------------------------------------------------
def shap_beeswarm_plot(model, pipeline, df, target_col):

    X = df.drop(columns=[target_col])
    X_processed = pipeline.transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    feature_names = pipeline.get_feature_names_out()

    numeric_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("num__")
    ]

    shap_array = shap_values[:, numeric_indices]
    clean_names = [
        feature_names[i].replace("num__", "")
        for i in numeric_indices
    ]

    mean_importance = np.abs(shap_array).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": clean_names,
        "Mean_Impact": mean_importance
    }).sort_values(by="Mean_Impact", ascending=True)

    fig = px.bar(
        shap_df,
        x="Mean_Impact",
        y="Feature",
        orientation="h",
        template="plotly_white",
        title="Global Feature Importance (Mean |SHAP|)",
        color="Mean_Impact",
        color_continuous_scale=["#1e293b", "#3b82f6", "#06b6d4"]
    )

    fig.update_layout(
        height=600,
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )

    return fig



# ----------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Prediction",
    "Analytics"
])

# ==========================================================
# OVERVIEW
# ==========================================================
with tab1:
    st.markdown("")  # spacer

    st.markdown("""
    <div class="glass-card" style="margin-bottom: 24px;">
        <div class="section-header">
            <div class="section-icon">🏏</div>
            <div>
                <div class="section-title">Intelligent Cricket Performance Forecasting</div>
                <div class="section-subtitle">Powered by ensemble ML models with real-time match context</div>
            </div>
        </div>
        <div style="margin-top: 16px; color: #94a3b8; font-size: 14px; line-height: 1.8;">
            This system uses advanced machine learning pipelines to deliver accurate, explainable
            performance predictions for both batsmen and bowlers — factoring in historical trends,
            venue-specific data, rolling averages, and live match context.
        </div>
    </div>
    <script>

    </script>
    """, unsafe_allow_html=True)

    # ==============================
    #  FEATURE GRID (static content only — no live score)
    # ==============================
    st.markdown("""

    <div class="feature-grid">
        <div class="feature-item">
            <div class="feat-icon">📊</div>
            <div class="feat-title">Runs Prediction</div>
            <div class="feat-desc">XGBoost / Random Forest ensemble for batsman scoring</div>
        </div>
        <div class="feature-item">
            <div class="feat-icon">🎯</div>
            <div class="feat-title">Wickets Prediction</div>
            <div class="feat-desc">Random Forest model for bowler wicket-taking</div>
        </div>
        <div class="feature-item">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">Feature Engineering</div>
            <div class="feat-desc">Rolling stats, venue averages, player matchups</div>
        </div>
        <div class="feature-item">
            <div class="feat-icon">📑</div>
            <div class="feat-title">SHAP Explainability</div>
            <div class="feat-desc">Local & global feature impact analysis</div>
        </div>
        <div class="feature-item">
            <div class="feat-icon">📊</div>
            <div class="feat-title">Uncertainty Modeling</div>
            <div class="feat-desc">95% confidence intervals with probability density</div>
        </div>
        <div class="feature-item">
            <div class="feat-icon">📄</div>
            <div class="feat-title">PDF Reports</div>
            <div class="feat-desc">One-click downloadable prediction reports</div>
        </div>
    </div>
    <script>

    </script>
    """, unsafe_allow_html=True)


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from io import BytesIO

# ==========================================================
# PREDICTION TAB
# ==========================================================

with tab2:

    st.markdown("")  # spacer

    # ==============================
    #  MODE SELECTOR
    # ==============================
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚡</div>
        <div>
            <div class="section-title">Select Mode</div>
            <div class="section-subtitle">Choose manual entry or live API data source</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Initialize session state
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Manual Mode"
    if "api_score_data" not in st.session_state:
        st.session_state["api_score_data"] = None
    if "live_matches_cache" not in st.session_state:
        st.session_state["live_matches_cache"] = None
    if "selected_match_idx" not in st.session_state:
        st.session_state["selected_match_idx"] = 0
    if "_live_auto_refresh" not in st.session_state:
        st.session_state["_live_auto_refresh"] = False

    mode = st.radio("Select Mode", ["Manual Mode", "Live API Mode"], horizontal=True, key="mode_radio")

    # Handle mode switching cleanly
    if mode != st.session_state.get("input_mode"):
        st.session_state["input_mode"] = mode
        st.session_state["api_score_data"] = None
        st.session_state["live_matches_cache"] = None
        st.session_state["selected_match_idx"] = 0
        st.session_state["pred_clicked"] = False
        st.session_state["_live_auto_refresh"] = False
    else:
        st.session_state["input_mode"] = mode

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  PREPARE LIVE SCORE DATA
    # ==============================
    # Default zero-state values (shown on initial load)
    live_score_data = {
        "team_a": "Team A",
        "team_b": "Team B",
        "stadium": "Unknown Stadium",
        "runs": 0,
        "wickets": 0,
        "overs": 0,
        "run_rate": 0.0,
    }

    api_features = None  # Will hold transformed features if live mode

    # --------------------------------------------------
    # CASE 1: LIVE API MODE
    # --------------------------------------------------
    if st.session_state["input_mode"] == "Live API Mode":

        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📡</div>
            <div>
                <div class="section-title">IPL Live Match Detection</div>
                <div class="section-subtitle">Fetching live IPL matches from CricketData API</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Fetch live matches (with button to refresh)
        fetch_col1, fetch_col2 = st.columns([3, 1])
        with fetch_col2:
            refresh_btn = st.button("🔄 Refresh Matches", key="refresh_api")

        if refresh_btn or st.session_state["live_matches_cache"] is None:
            with st.spinner("Fetching live IPL matches..."):
                all_live = fetch_live_matches()
                ipl_matches = filter_ipl_matches(all_live) if all_live else []
                st.session_state["live_matches_cache"] = ipl_matches if ipl_matches else None

        live_matches = st.session_state["live_matches_cache"]

        if live_matches and len(live_matches) > 0:

            st.markdown("""
            <div style="margin-bottom:8px;">
                <span class="live-badge"><span class="live-pulse"></span> IPL LIVE</span>
            </div>
            """, unsafe_allow_html=True)

            if len(live_matches) == 1:
                selected_match = live_matches[0]
                st.info(f"⚡ Auto-selected: **{format_match_label(selected_match)}**")
            else:
                match_labels = [format_match_label(m) for m in live_matches]
                selected_label = st.selectbox(
                    "Select IPL Match",
                    match_labels,
                    key="live_match_select"
                )
                selected_idx = match_labels.index(selected_label)
                selected_match = live_matches[selected_idx]

            match_id = selected_match.get("id", "")
            score_data = None
            if match_id:
                score_data = fetch_match_score(match_id)

            if not score_data:
                score_data = extract_score_from_match(selected_match)

            if score_data:
                st.session_state["api_score_data"] = score_data
                st.session_state["_live_auto_refresh"] = True

                # Populate live_score_data from API
                api_overs = float(score_data.get("overs", 0) or 0)
                api_runs = int(score_data.get("runs", 0) or 0)
                api_rr = round(api_runs / max(api_overs, 0.1), 2)

                live_score_data = {
                    "team_a": score_data.get("team1", "Team A"),
                    "team_b": score_data.get("team2", "Team B"),
                    "stadium": score_data.get("venue", "Unknown Stadium"),
                    "runs": api_runs,
                    "wickets": int(score_data.get("wickets", 0) or 0),
                    "overs": api_overs,
                    "run_rate": api_rr,
                    "status": score_data.get("status", "Match is live"),
                }

                api_features = transform_api_to_features(score_data)

                if api_features:
                    st.success(f"✅ IPL live data loaded: {score_data['runs']}/{score_data['wickets']} in {score_data['overs']} overs → features ready for prediction")
                else:
                    st.warning("⚠️ Could not transform API data. Falling back to Manual Mode.")
            else:
                st.warning("⚠️ Could not fetch score for selected match.")
                st.session_state["api_score_data"] = None
                st.session_state["_live_auto_refresh"] = False
        else:
            st.warning("⚠️ No IPL live match currently available.")
            st.session_state["api_score_data"] = None
            st.session_state["_live_auto_refresh"] = False

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # CASE 2: MANUAL MODE
    # --------------------------------------------------
    else:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🏏</div>
            <div>
                <div class="section-title">Match Details</div>
                <div class="section-subtitle">Enter team names, stadium, and current match state</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Team & stadium inputs
        tm1, tm2, tm3 = st.columns(3, gap="medium")
        with tm1:
            manual_team_a = st.text_input("Team A Name", value="Team A", key="manual_team_a")
        with tm2:
            manual_team_b = st.text_input("Team B Name", value="Team B", key="manual_team_b")
        with tm3:
            all_venues = sorted(list(set(batsman_df["venue"].unique()) | set(bowler_df["venue"].unique())))
            venue_options = ["Unknown Stadium"] + [v for v in all_venues if v != "Unknown Stadium"]
            manual_stadium = st.selectbox("Stadium Name", options=venue_options, key="manual_stadium")

        st.markdown("")

        # Score inputs
        mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
        with mc1:
            manual_runs = st.number_input("Runs", 0, 500, 0, key="manual_runs")
        with mc2:
            manual_wickets = st.number_input("Wickets", 0, 10, 0, key="manual_wickets")
        with mc3:
            manual_overs = st.number_input("Overs", 0, 20, 0, key="manual_overs")
        with mc4:
            manual_rr = st.number_input("Run Rate", 0.0, 36.0, 0.0, step=0.1, key="manual_rr")

        # Auto-compute run rate if user hasn't manually changed it
        if manual_rr == 0.0 and manual_overs > 0:
            manual_rr = round(manual_runs / max(manual_overs, 0.1), 2)

        live_score_data = {
            "team_a": manual_team_a or "Team A",
            "team_b": manual_team_b or "Team B",
            "stadium": manual_stadium or "Unknown Stadium",
            "runs": manual_runs,
            "wickets": manual_wickets,
            "overs": manual_overs,
            "run_rate": manual_rr,
            "status": "Live Match (Manual Entry)",
        }

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  RENDER LIVE SCORE (BOTH MODES)
    # ==============================
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔴</div>
        <div>
            <div class="section-title">Live Score</div>
            <div class="section-subtitle">Real-time match overview</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_live_score(live_score_data)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  MATCH SETUP
    # ==============================
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏏</div>
        <div>
            <div class="section-title">Match Setup</div>
            <div class="section-subtitle">Configure role, player, and venue</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        role = st.selectbox("Role", ["Batsman", "Bowler"])

    if role == "Batsman":
        player_list = sorted(batsman_df["batter"].unique())
        venue_list = sorted(batsman_df["venue"].unique())
    else:
        player_list = sorted(bowler_df["bowler"].unique())
        venue_list = sorted(bowler_df["venue"].unique())

    with col2:
        player = st.selectbox("Player", player_list)

    with col3:
        venue = st.selectbox("Venue", venue_list)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  RESOLVE MATCH CONTEXT VALUES
    # ==============================
    if st.session_state["input_mode"] == "Live API Mode" and api_features:
        current_score = int(api_features["current_score"])
        overs = int(api_features["overs"])
        wickets = int(api_features["wickets"])
    else:
        current_score = live_score_data["runs"]
        overs = live_score_data["overs"]
        wickets = live_score_data["wickets"]
        # Ensure overs is at least 1 for downstream match context calculation
        if overs == 0:
            overs = 1

    # Store in session state
    st.session_state["live_score"] = current_score
    st.session_state["live_overs"] = overs
    st.session_state["live_wickets"] = wickets

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    # CENTERED BUTTON
    # ==============================
    _, center, _ = st.columns([1,2,1])
    with center:
        predict_btn = st.button("Run AI Prediction", use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  PREDICTION LOGIC
    # ==============================
    if "pred_clicked" not in st.session_state:
        st.session_state["pred_clicked"] = False

    if predict_btn:
        st.session_state["pred_clicked"] = True

    if st.session_state["pred_clicked"]:

        if role == "Batsman":
            df = batsman_df[
                (batsman_df["batter"] == player) &
                (batsman_df["venue"] == venue)
            ]
            target_col = "runs_next_match"
            model = runs_model
            pipeline = runs_pipeline
            label = "Runs"
            card_title = "Predicted Runs"
            gradient = "linear-gradient(135deg,#2563eb,#06b6d4)"
            kpi_class = "kpi-runs"

        else:
            df = bowler_df[
                (bowler_df["bowler"] == player) &
                (bowler_df["venue"] == venue)
            ]
            target_col = "wickets_next_match"
            model = wickets_model
            pipeline = wickets_pipeline
            label = "Wickets"
            card_title = "Predicted Wickets"
            gradient = "linear-gradient(135deg,#7c3aed,#ec4899)"
            kpi_class = "kpi-wickets"

        if not df.empty:

            latest = df.iloc[-1:].copy()
            X_input = latest.drop(columns=[target_col]).copy()

            # ==============================
            #  MATCH CONTEXT FEATURES
            # ==============================
            run_rate, wickets_fallen, wickets_left, pressure_index = generate_match_context(
                current_score, overs, wickets
            )

            X_input["match_run_rate"] = run_rate
            X_input["wickets_fallen"] = wickets_fallen
            X_input["wickets_left"] = wickets_left
            X_input["pressure_index"] = pressure_index

            # ==============================
            # PIPELINE FEATURE ALIGNMENT
            # ==============================
            expected_cols = pipeline.feature_names_in_

            for col in expected_cols:
                if col not in X_input.columns:
                    X_input[col] = 0

            X_input = X_input[expected_cols]

            # ==============================
            #  PREDICTION
            # ==============================
            X_processed = pipeline.transform(X_input)
            raw_pred = model.predict(X_processed)[0]
            display_pred = round(float(raw_pred), 2)

            # ==============================
            #  SIMULATE PRESSURE SCENARIO
            # ==============================
            simulate_btn = st.button(" Simulate Pressure Scenario", key="simulate")

            if simulate_btn:
                base_pred = raw_pred
                # Simulate pressure scenario
                sim_score = current_score + 30
                sim_wickets = min(wickets + 3, 10)
                sim_overs = overs + 2
                sim_run_rate, sim_wf, sim_wl, sim_pressure = generate_match_context(
                    sim_score, sim_overs, sim_wickets
                )
                # Create simulated input
                X_sim = latest.drop(columns=[target_col]).copy()
                X_sim["match_run_rate"] = sim_run_rate
                X_sim["wickets_fallen"] = sim_wf
                X_sim["wickets_left"] = sim_wl
                X_sim["pressure_index"] = sim_pressure
                # Align with pipeline
                sim_expected_cols = pipeline.feature_names_in_
                for col in sim_expected_cols:
                    if col not in X_sim.columns:
                        X_sim[col] = 0
                X_sim = X_sim[sim_expected_cols]
                # Prediction under pressure
                X_sim_processed = pipeline.transform(X_sim)
                sim_pred = model.predict(X_sim_processed)[0]
                impact = round(base_pred - sim_pred, 2)

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">🏏</div>
                    <div>
                        <div class="section-title">Match Impact Analysis</div>
                        <div class="section-subtitle">Performance comparison under simulated pressure</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                imp1, imp2, imp3 = st.columns(3, gap="medium")
                with imp1:
                    st.markdown(f"""
                    <div class="metric-mini">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-label">Current Prediction</div>
                        <div class="metric-val">{round(base_pred, 2)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with imp2:
                    st.markdown(f"""
                    <div class="metric-mini">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-label">Under Pressure</div>
                        <div class="metric-val">{round(sim_pred, 2)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with imp3:
                    st.markdown(f"""
                    <div class="metric-mini">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-label">Impact Drop</div>
                        <div class="metric-val">{impact}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">🏏</div>
                    <div>
                        <div class="section-title">AI Insight</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if impact > 10:
                    st.error("High pressure collapse detected")
                elif impact > 5:
                    st.warning("Moderate performance impact")
                else:
                    st.success("Stable under pressure")

            std_dev = compute_uncertainty(model, pipeline, df, target_col)
            fig_dist, lower, upper = probability_distribution(raw_pred, std_dev, label)

            # ==============================
            # MATCH CONTEXT METRICS
            # ==============================
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">🏏</div>
                <div>
                    <div class="section-title">Match Context Analysis</div>
                    <div class="section-subtitle">Derived metrics from live match state</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            mx1, mx2, mx3, mx4 = st.columns(4, gap="medium")
            with mx1:
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="metric-icon">📈</div>
                    <div class="metric-label">Run Rate</div>
                    <div class="metric-val">{round(run_rate, 2)}</div>
                </div>
                """, unsafe_allow_html=True)
            with mx2:
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-label">Wickets Fallen</div>
                    <div class="metric-val">{wickets_fallen}</div>
                </div>
                """, unsafe_allow_html=True)
            with mx3:
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-label">Wickets Left</div>
                    <div class="metric-val">{wickets_left}</div>
                </div>
                """, unsafe_allow_html=True)
            with mx4:
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-label">Pressure Index</div>
                    <div class="metric-val">{round(pressure_index, 3)}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ==============================
            # OUTPUT SECTION
            # ==============================
            row1_col1, row1_col2 = st.columns([1.5,1], gap="large")

            with row1_col1:
                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">📊</div>
                    <div>
                        <div class="section-title">""" + label + """ Prediction Distribution</div>
                        <div class="section-subtitle">Probability density with 95% confidence interval</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig_dist, use_container_width=True, theme=None)

            with row1_col2:
                # MODEL CONFIDENCE
                if display_pred >= 75 or display_pred <= 25:
                    confidence = "High"
                elif display_pred >= 60 or display_pred <= 40:
                    confidence = "Medium"
                else:
                    confidence = "Low"

                st.markdown(f"""
                <div class="kpi-hero {kpi_class}" style="text-align: left;">
                    <div class="kpi-label">AI Prediction</div>
                    <div class="kpi-title">{card_title}</div>
                    <div class="kpi-value">{display_pred}</div>
                    <div class="kpi-sub">Context-aware ensemble model output</div>
                    <div style="margin-top: 12px; font-size: 14px; font-weight: 600; color: #94a3b8; border-top: 1px solid #1f2937; padding-top: 8px;">
                        Model Confidence: {confidence}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                st.markdown(f"""
                <div class="ci-card">
                    <div class="ci-label">95% Confidence Interval</div>
                    <div class="ci-value">{round(lower,2)} — {round(upper,2)}</div>
                </div>
                """, unsafe_allow_html=True)

            # ==============================
            #  ANALYTICS
            # ==============================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            row2_col1, row2_col2 = st.columns(2, gap="large")

            with row2_col1:
                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">🏏</div>
                    <div>
                        <div class="section-title">Feature Impact (Local SHAP)</div>
                        <div class="section-subtitle">How each feature influenced this prediction</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                shap_fig, shap_df = shap_local_explanation(model, pipeline, X_processed)
                st.plotly_chart(shap_fig, use_container_width=True, theme=None)
                
                # KEY DRIVER INSIGHT
                top_feature = shap_df.iloc[0]["Feature"].replace("_", " ").title()
                impact_dir = "increasing" if shap_df.iloc[0]["Impact"] > 0 else "decreasing"
                
                st.markdown(f"""
                <div style="margin-top: 8px; padding: 12px 16px; background: #f8fafc; border: 1px solid rgba(0,0,0,0.08); border-radius: 8px; font-size: 14px; color: #334155;">
                    <strong>Key Driver:</strong> {top_feature} is {impact_dir} win probability.
                </div>
                """, unsafe_allow_html=True)

            with row2_col2:
                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">🏏</div>
                    <div>
                        <div class="section-title">Residual Analysis</div>
                        <div class="section-subtitle">Model error distribution across predictions</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                residual_fig = residual_plot(model, pipeline, df, target_col=target_col)
                st.plotly_chart(residual_fig, use_container_width=True, theme=None)

            # ==============================
            #  AI EXPLANATION (LLM STYLE)
            # ==============================
            explanation = generate_llm_explanation(shap_df, raw_pred, label)

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffffff, #f1f5f9);
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 20px;
                padding: 32px 36px;
                margin-top: 8px;
                position: relative;
                overflow: hidden;">
                <div style="
                    position: absolute;
                    top: 0; left: 0; right: 0;
                    height: 3px;
                    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
                    border-radius: 20px 20px 0 0;"></div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 18px;">
                    <span style="
                        font-size: 22px;
                        width: 40px;
                        height: 40px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 12px;
                        background: rgba(139,92,246,0.15);
                        border: 1px solid rgba(139,92,246,0.25);">📍</span>
                    <span style="
                        font-size: 18px;
                        font-weight: 700;
                        color: #000000;
                        letter-spacing: -0.01em;">AI Explanation</span>
                </div>
                <div style="
                    color: #334155;
                    font-size: 15px;
                    line-height: 2;
                    font-family: 'Inter', sans-serif;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header">
                <div class="section-icon">🏏</div>
                <div>
                    <div class="section-title">Global Feature Importance</div>
                    <div class="section-subtitle">Mean |SHAP| values across all predictions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            beeswarm_fig = shap_beeswarm_plot(model, pipeline, df, target_col=target_col)
            st.plotly_chart(beeswarm_fig, use_container_width=True, theme=None)

            # =============================
            #  PDF DOWNLOAD
            # =============================
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            elements = []

            styles = getSampleStyleSheet()
            title_style = styles["Heading1"]

            elements.append(Paragraph("Performance Prediction Report", title_style))
            elements.append(Spacer(1, 0.5 * inch))

            report_data = [
                ["Role", role],
                ["Player", player],
                ["Venue", venue],
                [card_title, str(display_pred)],
                ["Confidence Interval", f"{round(lower,2)} to {round(upper,2)}"]
            ]

            table = Table(report_data, colWidths=[2.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("ALIGN", (0,0), (-1,-1), "CENTER")
            ]))

            elements.append(table)
            doc.build(elements)

            pdf = buffer.getvalue()
            buffer.close()

            _, dl_center, _ = st.columns([1, 2, 1])
            with dl_center:
                st.download_button(
                    label=" Download Prediction Report (PDF)",
                    data=pdf,
                    file_name="prediction_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        else:
            st.warning("No data available for selected player and venue.")
   
# ==========================================================
# ANALYTICS TAB
# ==========================================================
with tab3:

    st.markdown("")  # spacer

    st.markdown("""
    <div class="section-header" style="margin-bottom: 16px;">
        <div class="section-icon">📈</div>
        <div>
            <div class="section-title">Global Model Analysis</div>
            <div class="section-subtitle">Performance diagnostics and feature importance across the full dataset</div>
        </div>
    </div>
    <script>

    </script>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Model",
        ["Runs Model", "Wickets Model"]
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if model_choice == "Runs Model":
        df = batsman_df
        target = "runs_next_match"
        model = runs_model
        pipeline = runs_pipeline
    else:
        df = bowler_df
        target = "wickets_next_match"
        model = wickets_model
        pipeline = wickets_pipeline

    X = df.drop(columns=[target])
    y = df[target]
    X_processed = pipeline.transform(X)
    y_pred = model.predict(X_processed)

    # Actual vs Predicted
    fig1 = px.scatter(
        x=y,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        template="plotly_white",
        title="Actual vs Predicted",
        opacity=0.5
    )
    fig1.update_traces(marker=dict(color="#3b82f6", size=5))
    fig1.update_layout(
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True, theme=None)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Residuals
    residuals = y - y_pred

    fig2 = px.histogram(
        residuals,
        nbins=40,
        template="plotly_white",
        title="Residual Distribution",
        color_discrete_sequence=["#8b5cf6"]
    )
    fig2.update_layout(
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Global SHAP
    sample = df.sample(300, random_state=42)
    X_sample = sample.drop(columns=[target])
    X_sample_processed = pipeline.transform(X_sample)

    if hasattr(X_sample_processed, "toarray"):
        X_sample_processed = X_sample_processed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_processed)

    feature_names = pipeline.get_feature_names_out()
    numeric_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("num__")
    ]

    shap_values_num = shap_values[:, numeric_indices]
    clean_names = [
        feature_names[i].replace("num__", "")
        for i in numeric_indices
    ]

    shap_df = pd.DataFrame(shap_values_num, columns=clean_names)
    shap_long = shap_df.melt(var_name="Feature", value_name="SHAP Value")

    fig_bee = px.strip(
        shap_long,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        template="plotly_white",
        title="Global SHAP Beeswarm"
    )

    fig_bee.update_traces(marker=dict(color="#8b5cf6", size=4, opacity=0.5))
    fig_bee.update_layout(
        height=700,
        font=dict(family="Inter", size=13, color="#000000"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_bee, use_container_width=True, theme=None)