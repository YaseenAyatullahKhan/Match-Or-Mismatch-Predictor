import streamlit as st
import joblib
import pandas as pd
from views.overview  import show_overview
from views.eda       import show_eda
from views.models    import show_models
from views.predictor import show_predictor

# Page config
st.set_page_config(page_title="Match or Mismatch", layout="wide", page_icon="💘")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dating_app_behavior_dataset_extended1.csv")
    return df

# Load models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("pkls/logistic_regression.pkl"),
        "Decision Tree": joblib.load("pkls/decision_tree.pkl"),
        "XGBoost": joblib.load("pkls/xgboost.pkl"),
    }
    scaler = joblib.load("pkls/scaler.pkl")
    target_encoder = joblib.load("pkls/target_encoder.pkl")
    model_columns  = joblib.load("pkls/model_columns.pkl")
    return models, scaler, target_encoder, model_columns

df = load_data()
models, scaler, target_enc, model_cols = load_models()

# Hardcoded Random Forest and K-Nearest Neighbors results
# (model .pkl excluded due to very large file sizes; stats taken directly from Google Colab output)
RF_STATS = {
    "Accuracy": 0.0929,
    "Weighted F1 Score": 0.0922,
    "ROC-AUC": 0.4939,
}
KNN_STATS = {
    "Accuracy": 0.0948,
    "Weighted F1 Score": 0.0897,
    "ROC-AUC": 0.4997,
}

# Sidebar navigation
st.sidebar.image("sidebar_icon.png", width=100)
st.sidebar.title("Match or Mismatch")
st.sidebar.markdown("*WIA1006 — Group Assignment*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "📋 Overview",
    "🔍 Data Explorer",
    "📊 Model Performance",
    "💘 Predict My Outcome"
])

# Page routing
if page == "📋 Overview":
    show_overview(df)
elif page == "🔍 Data Explorer":
    show_eda(df)
elif page == "📊 Model Performance":
    show_models(df, models, scaler, target_enc, model_cols, RF_STATS, KNN_STATS)
elif page == "💘 Predict My Outcome":
    show_predictor(df, models, scaler, target_enc, model_cols)