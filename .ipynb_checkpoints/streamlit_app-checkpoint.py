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

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI Cricket Performance",
    page_icon="🏏",
    layout="wide"
)

# ----------------------------------------------------------
# DARK PROFESSIONAL UI
# ----------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}
h1, h2, h3, h4 {
    color: #e2e8f0;
}
.card {
    padding: 30px;
    border-radius: 18px;
    color: white;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.05);
}
.runs-card {
    background: linear-gradient(135deg,#2563eb,#06b6d4);
}
.wicket-card {
    background: linear-gradient(135deg,#7c3aed,#ec4899);
}
.info-box {
    background:#111827;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏏 AI Cricket Player Performance System")

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
    runs_model = joblib.load("model/xgb_batsman_model.joblib")
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
        title="Top Feature Contributions"
    )

    return fig


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
        name="Density"
    ))

    fig.add_vline(
        x=prediction,
        line_dash="dash",
        annotation_text="Prediction"
    )

    lower = prediction - 1.96 * std_dev
    upper = prediction + 1.96 * std_dev

    fig.add_vrect(
        x0=lower,
        x1=upper,
        fillcolor="rgba(0,255,150,0.15)",
        line_width=0,
        annotation_text="95% CI"
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"{label} Prediction Distribution",
        xaxis_title=label,
        yaxis_title="Density",
        height=500
    )

    return fig, lower, upper


# ----------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏠 Overview",
    "📊 Prediction",
    "📈 Analytics"
])

# ==========================================================
# OVERVIEW
# ==========================================================
with tab1:
    st.markdown("""
    ### Intelligent Cricket Performance Forecasting System

    This dashboard predicts:

    - 🔵 Runs (XGBoost Model)
    - 🟣 Wickets (Random Forest Model)

    Includes:
    - Feature engineered rolling stats
    - Venue averages
    - Player vs Player stats
    - SHAP explainability
    - Prediction uncertainty modeling
    """)

# ==========================================================
# PREDICTION TAB
# ==========================================================
with tab2:

    col1, col2 = st.columns([1, 2])

    with col1:
        role = st.selectbox("Select Role", ["Batsman", "Bowler"])

        if role == "Batsman":
            player = st.selectbox("Player", sorted(batsman_df["batter"].unique()))
            venue = st.selectbox("Venue", sorted(batsman_df["venue"].unique()))
        else:
            player = st.selectbox("Player", sorted(bowler_df["bowler"].unique()))
            venue = st.selectbox("Venue", sorted(bowler_df["venue"].unique()))

        predict_btn = st.button("🚀 Predict Performance")

    with col2:
        if predict_btn:

            if role == "Batsman":

                df = batsman_df[
                    (batsman_df["batter"] == player) &
                    (batsman_df["venue"] == venue)
                ]

                if not df.empty:
                    latest = df.iloc[-1:]
                    X_input = latest.drop(columns=["runs_next_match"])
                    X_processed = runs_pipeline.transform(X_input)
                    
                    raw_pred = runs_model.predict(X_processed)[0]
                    display_pred = round(float(raw_pred), 1)

                    st.markdown(
                        f'<div class="card runs-card">🔵 {display_pred} Runs</div>',
                        unsafe_allow_html=True
                    )

                    std_dev = compute_uncertainty(
                        runs_model, runs_pipeline,
                        batsman_df, "runs_next_match"
                    )

                    fig_dist, lower, upper = probability_distribution(
                        raw_pred, std_dev, "Runs"
                    )

                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.markdown(
                        f'<div class="info-box">95% CI: {round(lower,1)} to {round(upper,1)}</div>',
                        unsafe_allow_html=True
                    )

                    shap_fig = shap_local_explanation(
                        runs_model, runs_pipeline, X_processed
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)

            else:

                df = bowler_df[
                    (bowler_df["bowler"] == player) &
                    (bowler_df["venue"] == venue)
                ]

                if not df.empty:
                    latest = df.iloc[-1:]
                    X_input = latest.drop(columns=["wickets_next_match"])
                    X_processed = wickets_pipeline.transform(X_input)
                    raw_pred = wickets_model.predict(X_processed)[0]
                    display_pred = round(float(raw_pred), 1)

                    st.markdown(
                        f'<div class="card wicket-card">🟣 {display_pred} Wickets</div>',
                        unsafe_allow_html=True
                    )

                    std_dev = compute_uncertainty(
                        wickets_model, wickets_pipeline,
                        bowler_df, "wickets_next_match"
                    )

                    fig_dist, lower, upper = probability_distribution(
                        raw_pred, std_dev, "Wickets"
                    )

                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.markdown(
                        f'<div class="info-box">95% CI: {round(lower,2)} to {round(upper,2)}</div>',
                        unsafe_allow_html=True
                    )

                    shap_fig = shap_local_explanation(
                        wickets_model, wickets_pipeline, X_processed
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)


# ==========================================================
# ANALYTICS TAB
# ==========================================================
with tab3:

    st.header("Global Model Analysis")

    model_choice = st.radio(
        "Select Model",
        ["Runs Model", "Wickets Model"]
    )

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
        title="Actual vs Predicted"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Residuals
    residuals = y - y_pred

    fig2 = px.histogram(
        residuals,
        nbins=40,
        template="plotly_dark",
        title="Residual Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

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

    fig_bee.update_layout(height=700)

    st.plotly_chart(fig_bee, use_container_width=True)