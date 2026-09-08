"""Streamlit dashboard for the insurance fraud detection system.

Run with: streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, RESULTS_DIR, TRAIN_PATH  # noqa: E402
from src.data.clean import clean_data  # noqa: E402
from src.data.features import (  # noqa: E402
    FREQUENCY_COLS,
    ONEHOT_COLS,
    engineer_features,
)
from src.explain.shap_analysis import explain_instance  # noqa: E402

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def load_pipeline_and_meta():
    pipeline = joblib.load(MODELS_DIR / "final_model.pkl")
    with open(MODELS_DIR / "final_model_meta.json") as f:
        meta = json.load(f)
    return pipeline, meta


@st.cache_data
def load_results():
    with open(RESULTS_DIR / "model_results.json") as f:
        return json.load(f)


@st.cache_data
def load_global_importance():
    path = RESULTS_DIR / "shap_global_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_raw_options():
    # Reads from the processed (engineered) split rather than data/raw/, since
    # raw data is gitignored per the project spec and won't exist in a
    # deployed copy of this repo. engineer_features() never touches the
    # values of these categorical columns, only adds/drops other columns,
    # so the category values here are identical to the raw source.
    df = pd.read_csv(TRAIN_PATH)
    options = {}
    for col in ONEHOT_COLS + FREQUENCY_COLS:
        if col in df.columns:
            options[col] = sorted(df[col].dropna().unique().tolist())
    return options


pipeline, meta = load_pipeline_and_meta()
results = load_results()
importance_df = load_global_importance()
raw_options = load_raw_options()

st.title("Insurance Claims Fraud Detection")
st.caption(f"Deployed model: `{meta['final_key']}`  |  Operating threshold: {meta['threshold']:.3f}")

tab_predict, tab_compare, tab_importance, tab_confusion = st.tabs(
    ["Score a Claim", "Model Comparison", "Global Feature Importance", "Confusion Matrix"]
)

with tab_predict:
    st.subheader("Enter claim details")
    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            months_as_customer = st.number_input("Months as customer", min_value=0, value=200)
            age = st.number_input("Age", min_value=16, max_value=100, value=40)
            policy_bind_date = st.date_input("Policy bind date")
            policy_state = st.selectbox("Policy state", raw_options.get("policy_state", ["OH"]))
            policy_csl = st.selectbox("Policy CSL", raw_options.get("policy_csl", ["250/500"]))
            policy_deductable = st.number_input("Policy deductible", min_value=0.0, value=1000.0)
            policy_annual_premium = st.number_input("Policy annual premium", min_value=0.01, value=1300.0)
            umbrella_limit = st.number_input("Umbrella limit", min_value=0.0, value=0.0)
            insured_sex = st.selectbox("Insured sex", raw_options.get("insured_sex", ["MALE", "FEMALE"]))
            insured_education_level = st.selectbox("Education level", raw_options.get("insured_education_level", ["High School"]))

        with col2:
            insured_occupation = st.selectbox("Occupation", raw_options.get("insured_occupation", ["sales"]))
            insured_hobbies = st.selectbox("Hobbies", raw_options.get("insured_hobbies", ["reading"]))
            insured_relationship = st.selectbox("Relationship", raw_options.get("insured_relationship", ["husband"]))
            capital_gains = st.number_input("Capital gains", value=0.0)
            capital_loss = st.number_input("Capital loss", value=0.0)
            incident_date = st.date_input("Incident date")
            incident_type = st.selectbox("Incident type", raw_options.get("incident_type", ["Single Vehicle Collision"]))
            collision_type = st.selectbox("Collision type", raw_options.get("collision_type", ["Rear Collision"]))
            incident_severity = st.selectbox("Incident severity", raw_options.get("incident_severity", ["Minor Damage"]))
            authorities_contacted = st.selectbox("Authorities contacted", raw_options.get("authorities_contacted", ["Police"]))

        with col3:
            incident_state = st.selectbox("Incident state", raw_options.get("incident_state", ["OH"]))
            incident_city = st.selectbox("Incident city", raw_options.get("incident_city", ["Columbus"]))
            incident_hour_of_the_day = st.number_input("Incident hour (0-23)", min_value=0, max_value=23, value=12)
            number_of_vehicles_involved = st.number_input("Vehicles involved", min_value=1, value=1)
            property_damage = st.selectbox("Property damage", raw_options.get("property_damage", ["NO", "YES"]))
            bodily_injuries = st.number_input("Bodily injuries", min_value=0, value=0)
            witnesses = st.number_input("Witnesses", min_value=0, value=1)
            police_report_available = st.selectbox("Police report available", raw_options.get("police_report_available", ["NO", "YES"]))
            auto_make = st.selectbox("Auto make", raw_options.get("auto_make", ["Toyota"]))
            auto_model = st.selectbox("Auto model", raw_options.get("auto_model", ["Camry"]))

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            total_claim_amount = st.number_input("Total claim amount", min_value=0.0, value=10000.0)
        with col5:
            injury_claim = st.number_input("Injury claim", min_value=0.0, value=2000.0)
        with col6:
            property_claim = st.number_input("Property claim", min_value=0.0, value=3000.0)
        with col7:
            vehicle_claim = st.number_input("Vehicle claim", min_value=0.0, value=5000.0)
        auto_year = st.number_input("Auto year", min_value=1980, max_value=2100, value=2015)

        submitted = st.form_submit_button("Score claim")

    if submitted:
        row = {
            "months_as_customer": months_as_customer, "age": age,
            "policy_bind_date": str(policy_bind_date), "policy_state": policy_state,
            "policy_csl": policy_csl, "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium, "umbrella_limit": umbrella_limit,
            "insured_sex": insured_sex, "insured_education_level": insured_education_level,
            "insured_occupation": insured_occupation, "insured_hobbies": insured_hobbies,
            "insured_relationship": insured_relationship, "capital-gains": capital_gains,
            "capital-loss": capital_loss, "incident_date": str(incident_date),
            "incident_type": incident_type, "collision_type": collision_type,
            "incident_severity": incident_severity, "authorities_contacted": authorities_contacted,
            "incident_state": incident_state, "incident_city": incident_city,
            "incident_hour_of_the_day": incident_hour_of_the_day,
            "number_of_vehicles_involved": number_of_vehicles_involved,
            "property_damage": property_damage, "bodily_injuries": bodily_injuries,
            "witnesses": witnesses, "police_report_available": police_report_available,
            "total_claim_amount": total_claim_amount, "injury_claim": injury_claim,
            "property_claim": property_claim, "vehicle_claim": vehicle_claim,
            "auto_make": auto_make, "auto_model": auto_model, "auto_year": auto_year,
        }
        raw_df = pd.DataFrame([row])
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)

        proba = float(pipeline.predict_proba(engineered)[0, 1])
        prediction = int(proba >= meta["threshold"])

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Fraud probability", f"{proba:.1%}")
        with col_b:
            st.metric("Prediction", "FRAUD" if prediction else "Not fraud",
                       delta=f"threshold={meta['threshold']:.3f}", delta_color="off")

        top_features = explain_instance(pipeline, engineered, top_n=8)
        contrib_df = pd.DataFrame(top_features)
        st.subheader("Why this claim was scored this way")
        st.bar_chart(contrib_df.set_index("feature")["shap_value"])
        st.caption("Positive SHAP values push the prediction toward fraud; negative values push toward not-fraud.")

with tab_compare:
    st.subheader("All trained models (test set metrics)")
    rows = []
    for key, m in results.items():
        if key == "ensemble":
            algo, strategy = "ensemble", "soft_voting"
        else:
            algo, strategy = key.split("__")
        rows.append({
            "model": algo, "strategy": strategy,
            "precision_fraud": m["precision_fraud"], "recall_fraud": m["recall_fraud"],
            "f1_fraud": m["f1_fraud"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        })
    comparison_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption(f"Deployed model: `{meta['final_key']}` at threshold {meta['threshold']:.3f} "
               "(chosen for a recall floor, not the highest F1 -- see results/evaluation_report.md).")

with tab_importance:
    st.subheader("Global SHAP feature importance (final model)")
    if importance_df is not None:
        top_n = st.slider("Number of features", 5, min(30, len(importance_df)), 15)
        chart_df = importance_df.head(top_n).set_index("feature")["mean_abs_shap"]
        st.bar_chart(chart_df)
    else:
        st.warning("Run `python -m src.explain.shap_analysis` first to generate this data.")

with tab_confusion:
    st.subheader("Confusion matrix by model")
    selected_key = st.selectbox("Model", list(results.keys()))
    m = results[selected_key]
    cm = np.array(m["confusion_matrix"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{selected_key} @ threshold={m['threshold']:.2f}")
    st.pyplot(fig)

    st.write(f"Precision (fraud): {m['precision_fraud']:.3f} | "
             f"Recall (fraud): {m['recall_fraud']:.3f} | F1 (fraud): {m['f1_fraud']:.3f}")
