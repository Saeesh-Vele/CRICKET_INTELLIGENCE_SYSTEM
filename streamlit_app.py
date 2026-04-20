# ==========================================================
# AI CRICKET PERFORMANCE PREDICTION - PRODUCTION VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import time
from cricket_api import (
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.65);
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-purple: #8b5cf6;
    --accent-pink: #ec4899;
    --accent-emerald: #10b981;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --shadow-glow-blue: 0 0 40px rgba(59, 130, 246, 0.15);
    --shadow-glow-purple: 0 0 40px rgba(139, 92, 246, 0.15);
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== GLOBAL ===== */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(10, 14, 26, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-glass);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-glass);
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.025em;
}

h1 { font-weight: 800 !important; font-size: 2.25rem !important; }
h2 { font-weight: 700 !important; font-size: 1.75rem !important; }
h3 { font-weight: 600 !important; font-size: 1.35rem !important; }
h4 { font-weight: 600 !important; font-size: 1.1rem !important; }

p, span, label, div {
    font-family: 'Inter', sans-serif;
    color: var(--text-secondary);
}

hr { border-color: var(--border-glass) !important; opacity: 0.5 !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ===== GLASS CARD ===== */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 24px;
    text-align: left;
    border-top: 1px solid #1f2937;
    transition: var(--transition-smooth);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.12);
    
    transform: translateY(-2px);
}

/* ===== HERO KPI CARD ===== */
.kpi-hero {
    padding: 38px 32px;
    border-top: 1px solid #1f2937;
    text-align: left;
    border-radius: 24px;
    text-align: left;
    color: white;
    position: relative;
    overflow: hidden;
    
    transition: var(--transition-smooth);
}
.kpi-hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.kpi-hero:hover {
    transform: translateY(-4px) scale(1.01);
    
}
.kpi-hero .kpi-label {
    font-size: 14px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.85;
    margin-bottom: 4px;
}
.kpi-hero .kpi-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
    color: white;
}
.kpi-hero .kpi-value {
    font-size: 72px;
    font-weight: 900;
    letter-spacing: -3px;
    line-height: 1;
    margin: 16px 0;
    background: linear-gradient(135deg, #fff 30%, rgba(255,255,255,0.7));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
}
.kpi-hero .kpi-sub {
    font-size: 13px;
    opacity: 0.75;
    font-weight: 400;
}

/* ===== RUNS GRADIENT ===== */
.kpi-runs {
    background: #1e40af;
}
/* ===== WICKETS GRADIENT ===== */
.kpi-wickets {
    background: #6d28d9;
}

/* ===== CI CARD ===== */
.ci-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 18px;
    padding: 22px;
    border-top: 1px solid #1f2937;
    text-align: left;
    transition: var(--transition-smooth);
}
.ci-card:hover {
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
}
.ci-card .ci-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 8px;
}
.ci-card .ci-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
}

/* ===== CONTEXT METRICS ===== */
.metric-mini {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 20px 20px;
    text-align: left;
    transition: border-color 0.2s ease;
}
.metric-mini:hover {
    border-color: #374151;
}
.metric-mini .metric-icon {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-bottom: 10px;
}
.metric-mini .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9ca3af;
    font-weight: 500;
    margin-bottom: 6px;
}
.metric-mini .metric-val {
    font-size: 28px;
    font-weight: 700;
    color: #e5e7eb;
    letter-spacing: -0.5px;
}

/* ===== SECTION HEADER ===== */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.section-header .section-icon {
    font-size: 20px;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
}
.section-header .section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.section-header .section-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    font-weight: 400;
}

/* ===== DIVIDER ===== */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
    margin: 32px 0;
}

/* ===== BUTTONS ===== */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.02em !important;
    padding: 14px 28px !important;
    transition: var(--transition-smooth) !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.45) !important;
    filter: brightness(1.08);
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ===== DOWNLOAD BUTTON ===== */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1e293b, #334155) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    transition: var(--transition-smooth) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #334155, #475569) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* ===== INPUTS ===== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: var(--transition-smooth) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* ===== TABS ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-glass);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid var(--border-glass);
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    color: var(--text-muted) !important;
    transition: var(--transition-smooth) !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.04);
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    color: var(--accent-blue) !important;
    border-bottom: none !important;
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
    gap: 8px;
}
[data-testid="stRadio"] label {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 10px 20px;
    cursor: pointer;
    transition: var(--transition-smooth);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
[data-testid="stRadio"] label:hover {
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.06);
}

/* ===== WARNING / INFO ===== */
[data-testid="stAlert"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    color: var(--text-primary) !important;
}

/* ===== OVERVIEW FEATURES ===== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 24px;
}
.feature-item {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 18px 16px;
    border-top: 1px solid #1f2937;
    text-align: left;
    transition: var(--transition-smooth);
}
.feature-item:hover {
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}
.feature-item .feat-icon {
    font-size: 32px;
    margin-bottom: 12px;
}
.feature-item .feat-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
}
.feature-item .feat-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ===== HERO BANNER ===== */
.hero-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.12), rgba(236,72,153,0.08));
    border: 1px solid var(--border-glass);
    border-radius: 24px;
    padding: 38px 32px;
    border-top: 1px solid #1f2937;
    text-align: left;
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.hero-banner .hero-icon { font-size: 48px; margin-bottom: 16px; }
.hero-banner .hero-title {
    font-size: 32px;
    font-weight: 900;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 8px;
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
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: white;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ===== PLOTLY CHARTS WRAPPER ===== */
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    overflow: hidden;
}

/* ===== BLOCK CONTAINER ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ===== LABEL STYLING ===== */
[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.75px !important;
    color: var(--text-muted) !important;
}

/* ===== LIVE MODE INDICATOR ===== */
.live-pulse {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse-glow 1.5s ease-in-out infinite;
    vertical-align: middle;
    margin-right: 8px;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #22c55e; }
    50% { opacity: 0.5; box-shadow: 0 0 12px #22c55e; }
}
.live-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.25);
    color: #22c55e;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.api-status-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(6, 182, 212, 0.08));
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
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
    batsman_df = pd.read_csv("data/processed/batsman_model_data.csv")
    bowler_df = pd.read_csv("data/processed/bowler_model_data.csv")
    return batsman_df, bowler_df

batsman_df, bowler_df = load_data()

# ----------------------------------------------------------
# LOAD MODELS & PIPELINES
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    runs_model = joblib.load("model/rf_batsman_model.joblib")
    runs_pipeline = joblib.load("model/feature_pipeline_batsman.pkl")

    wickets_model = joblib.load("model/rf_wickets_model.joblib")
    wickets_pipeline = joblib.load("model/feature_pipeline_bowler.pkl")

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
        template="plotly_dark",
        title="Top Feature Contributions",
        color="Impact",
        color_continuous_scale=["#ec4899", "#64748b", "#3b82f6"]
    )
    fig.update_layout(
        font=dict(family="Inter", size=13),
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
        template="plotly_dark",
        title=f"{label} Prediction Distribution",
        xaxis_title=label,
        yaxis_title="Density",
        height=500,
        font=dict(family="Inter", size=13),
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
        template="plotly_dark",
        title="Residual Analysis (Predicted vs Residuals)",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=450,
        font=dict(family="Inter", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

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
        template="plotly_dark",
        title="Global Feature Importance (Mean |SHAP|)",
        color="Mean_Impact",
        color_continuous_scale=["#1e293b", "#3b82f6", "#06b6d4"]
    )

    fig.update_layout(
        height=600,
        font=dict(family="Inter", size=13),
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
    #  LIVE API SCORE CARD (Tab 1)
    # ==============================
    if st.session_state.get("input_mode") == "Live Match (API)" and st.session_state.get("api_score_data"):
        api_sc = st.session_state["api_score_data"]
        st.markdown(f"""
        <div class="api-status-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span class="live-badge"><span class="live-pulse"></span> LIVE API</span>
                <span style="font-size:13px; color:#94a3b8;">{api_sc.get('inning_label','')}</span>
            </div>
            <div style="font-size:22px; font-weight:700; color:#f1f5f9; margin-bottom:4px;">
                {api_sc.get('team1','Team A')} vs {api_sc.get('team2','Team B')}
            </div>
            <div style="display:flex; gap:32px; margin-top:12px;">
                <div>
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#6b7280; font-weight:500;">Score</div>
                    <div style="font-size:32px; font-weight:700; color:#e5e7eb;">{api_sc.get('runs',0)}/{api_sc.get('wickets',0)}</div>
                </div>
                <div>
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#6b7280; font-weight:500;">Overs</div>
                    <div style="font-size:32px; font-weight:700; color:#e5e7eb;">{api_sc.get('overs',0)}</div>
                </div>
                <div>
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#6b7280; font-weight:500;">Run Rate</div>
                    <div style="font-size:32px; font-weight:700; color:#e5e7eb;">{round(api_sc.get('runs',0) / max(api_sc.get('overs',1), 0.1), 2)}</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#64748b;">Status: {api_sc.get('status','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh every 15 seconds when live
        time.sleep(0.1)  # minimal non-blocking
        if st.session_state.get("_live_auto_refresh", False):
            time.sleep(15)
            st.rerun()

    # ==============================
    #  DYNAMIC LIVE MATCH CARD
    # ==============================
    score = st.session_state.get("live_score", 120)
    overs_val = st.session_state.get("live_overs", 12)
    wickets_val = st.session_state.get("live_wickets", 3)

    run_rate = round(score / max(overs_val, 1), 2)

    # ==============================
    # MATCH STATUS & MOMENTUM
    # ==============================
    if run_rate > 8:
        match_status = "Team A is in control"
    elif run_rate > 6:
        match_status = "Match is balanced"
    else:
        match_status = "Team B gaining advantage"
        
    baseline_rr = 8.0
    rr_eval = "Above par" if run_rate >= baseline_rr else "Below par"
    
    if run_rate >= 8 and wickets_val <= 4:
        momentum = "Strong"
    elif run_rate >= 6 and wickets_val <= 6:
        momentum = "Stable"
    else:
        momentum = "Dropping"

    # ==============================
    # WIN PROBABILITY (compute before rendering)
    # ==============================
    remaining_overs = max(20 - overs_val, 1)
    wickets_in_hand = max(10 - wickets_val, 0)
    win_prob = min(100, max(0, round(
        40 + (run_rate - 7) * 5 + wickets_in_hand * 3 - (20 - remaining_overs) * 0.8, 1
    )))
    lose_prob = round(100 - win_prob, 1)

    st.components.v1.html(f"""

    <style>
    i[data-lucide] {{
        width: 18px;
        height: 18px;
        stroke-width: 1.8;
        color: #9ca3af;
        vertical-align: middle;
        margin-right: 8px;
    }}
    </style>
    <div style="
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0;
        font-family: Inter, system-ui, -apple-system, sans-serif;
        color: #e5e7eb;
        overflow: hidden;">

        <!-- ── MATCH HEADER ── -->
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 18px;
            border-bottom: 1px solid #1f2937;">
            <div>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 11px;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 1.2px;
                    color: #6b7280;
                    margin-bottom: 4px;">
                    ⚡
                    <span>Live Match • {match_status}</span>
                </div>
                <div style="
                    font-size: 18px;
                    font-weight: 600;
                    color: #e5e7eb;
                    letter-spacing: -0.3px;">Team A vs Team B</div>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 6px;">
                <span style="
                    display: inline-block;
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background: #22c55e;
                    box-shadow: 0 0 4px #22c55e;"></span>
                <span style="
                    font-size: 11px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    color: #22c55e;">Live</span>
            </div>
        </div>

        <!-- ── SCORE PANEL ── -->
        <div style="padding-left: 24px; padding-top: 16px;">⏱️</div>
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 24px;
            border-bottom: 1px solid #1f2937;">
            <!-- LEFT: Dominant Score -->
            <div>
                <div style="
                    font-size: 44px;
                    font-weight: 700;
                    color: #e5e7eb;
                    letter-spacing: -1.5px;
                    line-height: 1;">{score}/{wickets_val}</div>
                <div style="
                    font-size: 13px;
                    color: #6b7280;
                    margin-top: 6px;
                    font-weight: 400;">After {overs_val} overs</div>
            </div>
            <!-- RIGHT: Compact Stats -->
            <div style="
                display: flex;
                gap: 28px;">
                <div style="text-align: right;">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 500; margin-bottom: 4px;">Overs</div>
                    <div style="font-size: 20px; font-weight: 600; color: #e5e7eb;">{overs_val}</div>
                </div>
                <div style="
                    width: 1px;
                    background: #1f2937;
                    align-self: stretch;"></div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 500; margin-bottom: 4px;">Wickets</div>
                    <div style="font-size: 20px; font-weight: 600; color: #e5e7eb;">{wickets_val}</div>
                </div>
                <div style="
                    width: 1px;
                    background: #1f2937;
                    align-self: stretch;"></div>
                <div style="text-align: left;">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 500; margin-bottom: 4px;">Run Rate</div>
                    <div style="font-size: 20px; font-weight: 600; color: #e5e7eb;">{run_rate} <span style="font-size:12px;color:#9ca3af;">({rr_eval})</span></div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Momentum: {momentum}</div>
                </div>
            </div>
        </div>

        <!-- ── MATCH PROBABILITY ── -->
        <div style="padding: 16px 24px;">
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: #6b7280;
                margin-bottom: 10px;">
                📈
                <span>Match Probability</span>
            </div>
            <!-- Progress Bar -->
            <div style="
                display: flex;
                width: 100%;
                height: 6px;
                border-radius: 3px;
                overflow: hidden;
                background: #1f2937;">
                <div style="
                    width: {win_prob}%;
                    background: #22c55e;
                    border-radius: 3px 0 0 3px;
                    transition: width 0.4s ease;"></div>
                <div style="
                    width: {lose_prob}%;
                    background: #ef4444;
                    border-radius: 0 3px 3px 0;
                    transition: width 0.4s ease;"></div>
            </div>
            <!-- Labels -->
            <div style="
                display: flex;
                justify-content: space-between;
                margin-top: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: #22c55e;">Team A {win_prob}%</span>
                <span style="font-size: 13px; font-weight: 600; color: #ef4444;">Team B {lose_prob}%</span>
            </div>
        </div>
    </div>
    <script>

    </script>
    """, height=315)

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
    #  INPUT MODE TOGGLE
    # ==============================
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚡</div>
        <div>
            <div class="section-title">Input Mode</div>
            <div class="section-subtitle">Choose manual entry or live API data source</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Initialize session state for input mode
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Manual Input"
    if "api_score_data" not in st.session_state:
        st.session_state["api_score_data"] = None
    if "live_matches_cache" not in st.session_state:
        st.session_state["live_matches_cache"] = None
    if "selected_match_idx" not in st.session_state:
        st.session_state["selected_match_idx"] = 0
    if "_live_auto_refresh" not in st.session_state:
        st.session_state["_live_auto_refresh"] = False

    input_mode = st.selectbox(
        "Data Source",
        ["Manual Input", "Live Match (API)"],
        key="input_mode_radio",
    )

    # Handle mode switching cleanly
    if input_mode != st.session_state.get("input_mode"):
        st.session_state["input_mode"] = input_mode
        st.session_state["api_score_data"] = None
        st.session_state["live_matches_cache"] = None
        st.session_state["selected_match_idx"] = 0
        st.session_state["pred_clicked"] = False
        st.session_state["_live_auto_refresh"] = False
    else:
        st.session_state["input_mode"] = input_mode

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ==============================
    #  LIVE MATCH API SECTION
    # ==============================
    api_features = None  # Will hold transformed features if live mode

    if st.session_state["input_mode"] == "Live Match (API)":

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
                # Filter for IPL matches only
                ipl_matches = filter_ipl_matches(all_live) if all_live else []
                st.session_state["live_matches_cache"] = ipl_matches if ipl_matches else None

        live_matches = st.session_state["live_matches_cache"]

        if live_matches and len(live_matches) > 0:

            st.markdown("""
            <div style="margin-bottom:8px;">
                <span class="live-badge"><span class="live-pulse"></span> IPL LIVE</span>
            </div>
            """, unsafe_allow_html=True)

            # Auto-select if only one IPL match, else show dropdown
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

            # Extract score from match object (already contains score data)
            score_data = extract_score_from_match(selected_match)

            if score_data is None:
                # Fallback: try fetching score via match_id
                match_id = selected_match.get("id", "")
                if match_id:
                    with st.spinner("Fetching live score..."):
                        score_data = fetch_match_score(match_id)

            if score_data:
                st.session_state["api_score_data"] = score_data
                st.session_state["_live_auto_refresh"] = True

                # Display live score card
                st.markdown(f"""
                <div class="api-status-card">
                    <div style="font-size:18px; font-weight:700; color:#f1f5f9; margin-bottom:8px;">
                        {score_data['team1']} vs {score_data['team2']}
                    </div>
                    <div style="display:flex; gap:32px; align-items:flex-end;">
                        <div>
                            <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#6b7280; font-weight:500;">Score</div>
                            <div style="font-size:36px; font-weight:700; color:#e5e7eb;">{score_data['runs']}/{score_data['wickets']}</div>
                        </div>
                        <div>
                            <div style="font-size:11px; text-transform:uppercase; letter-spacing:1.2px; color:#6b7280; font-weight:500;">Overs</div>
                            <div style="font-size:36px; font-weight:700; color:#e5e7eb;">{score_data['overs']}</div>
                        </div>
                        <div style="flex:1;">
                            <div style="font-size:13px; color:#94a3b8;">{score_data.get('inning_label','')}</div>
                            <div style="font-size:12px; color:#64748b; margin-top:4px;">{score_data.get('status','')}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Transform API data to model features
                api_features = transform_api_to_features(score_data)

                if api_features:
                    st.success(f"✅ IPL live data loaded: {score_data['runs']}/{score_data['wickets']} in {score_data['overs']} overs → features ready for prediction")
                else:
                    st.warning("⚠️ Could not transform API data. Falling back to manual input.")
                    st.session_state["input_mode"] = "Manual Input"

            else:
                st.warning("⚠️ Could not fetch score for selected match. Falling back to manual input.")
                st.session_state["api_score_data"] = None
                st.session_state["_live_auto_refresh"] = False

        else:
            # No IPL live matches – fallback to manual
            st.warning("⚠️ No IPL live match currently available. Switching to manual input.")
            st.session_state["input_mode"] = "Manual Input"
            st.session_state["api_score_data"] = None
            st.session_state["_live_auto_refresh"] = False

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
    <script>

    </script>
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
    #  LIVE MATCH CONTEXT
    # ==============================
    if st.session_state["input_mode"] == "Live Match (API)" and api_features:
        # ── LIVE MODE: hide manual inputs, use API data silently ──
        current_score = int(api_features["current_score"])
        overs = int(api_features["overs"])
        wickets = int(api_features["wickets"])

        st.markdown("""
        <div style="padding:14px 20px; background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.18); border-radius:14px; display:flex; align-items:center; gap:12px;">
            <span class="live-badge"><span class="live-pulse"></span> LIVE</span>
            <span style="color:#94a3b8; font-size:13px; font-weight:500;">Live data is being auto-fetched. Manual input disabled.</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── MANUAL MODE: show full input section ──
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🏏</div>
            <div>
                <div class="section-title">Live Match Context</div>
                <div class="section-subtitle">Current match state for context-aware predictions</div>
            </div>
        </div>
        <script>

        </script>
        """, unsafe_allow_html=True)

        st.markdown("")

        mc1, mc2, mc3 = st.columns(3, gap="medium")

        with mc1:
            current_score = st.number_input("Score", 0, 300, 50)

        with mc2:
            overs = st.number_input("Overs", 1, 20, 5)

        with mc3:
            wickets = st.number_input("Wickets", 0, 10, 2)

    # Store in session state so Overview tab can read live values
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
                st.plotly_chart(fig_dist, use_container_width=True)

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
                st.plotly_chart(shap_fig, use_container_width=True)
                
                # KEY DRIVER INSIGHT
                top_feature = shap_df.iloc[0]["Feature"].replace("_", " ").title()
                impact_dir = "increasing" if shap_df.iloc[0]["Impact"] > 0 else "decreasing"
                
                st.markdown(f"""
                <div style="margin-top: 8px; padding: 12px 16px; background: #111827; border: 1px solid #1f2937; border-radius: 8px; font-size: 14px; color: #e5e7eb;">
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
                st.plotly_chart(residual_fig, use_container_width=True)

            # ==============================
            #  AI EXPLANATION (LLM STYLE)
            # ==============================
            explanation = generate_llm_explanation(shap_df, raw_pred, label)

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(30,41,59,0.8));
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255,255,255,0.08);
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
                        color: #f1f5f9;
                        letter-spacing: -0.01em;">AI Explanation</span>
                </div>
                <div style="
                    color: #cbd5e1;
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
            st.plotly_chart(beeswarm_fig, use_container_width=True)

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
        template="plotly_dark",
        title="Actual vs Predicted",
        opacity=0.5
    )
    fig1.update_traces(marker=dict(color="#3b82f6", size=5))
    fig1.update_layout(
        font=dict(family="Inter", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Residuals
    residuals = y - y_pred

    fig2 = px.histogram(
        residuals,
        nbins=40,
        template="plotly_dark",
        title="Residual Distribution",
        color_discrete_sequence=["#8b5cf6"]
    )
    fig2.update_layout(
        font=dict(family="Inter", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

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
        template="plotly_dark",
        title="Global SHAP Beeswarm"
    )

    fig_bee.update_traces(marker=dict(color="#8b5cf6", size=4, opacity=0.5))
    fig_bee.update_layout(
        height=700,
        font=dict(family="Inter", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_bee, use_container_width=True)