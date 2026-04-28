# AI Cricket Performance Prediction

Advanced machine learning system for cricket performance prediction with SHAP explainability, uncertainty quantification, and real-time IPL match context analysis.

## Project Structure

```
AI_Cricket_Prediction/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── api/
│   │   └── cricket_api.py        # Live match API integration
│   └── preprocessing/
│       └── data_cleaning.py      # Data cleaning pipeline
├── models/
│   ├── rf_batsman_model.joblib   # Random Forest batsman model
│   ├── rf_wickets_model.joblib   # Random Forest bowler model
│   ├── xgb_batsman_model.joblib  # XGBoost batsman model
│   ├── feature_pipeline_batsman.pkl
│   └── feature_pipeline_bowler.pkl
├── data/
│   ├── raw/                      # Raw match & delivery data
│   ├── cleaned/                  # Cleaned ball-by-ball data
│   └── processed/                # Feature-engineered model data
├── notebooks/                    # EDA & training notebooks
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

## Features

- **Runs Prediction** — XGBoost / Random Forest ensemble for batsman scoring
- **Wickets Prediction** — Random Forest model for bowler wicket-taking
- **SHAP Explainability** — Local & global feature impact analysis
- **Uncertainty Modeling** — 95% confidence intervals with probability density
- **Live Match Context** — Real-time IPL data via CricAPI
- **PDF Reports** — One-click downloadable prediction reports


python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
