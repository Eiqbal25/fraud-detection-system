"""FastAPI service for fraud-probability scoring.

Loads the final tuned pipeline (preprocessing + classifier) once at startup.
A request is run through the exact same cleaning/feature-engineering code
used at training time, so there is only one place these rules are defined.
"""
import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ClaimInput, FeatureContribution, HealthResponse, PredictionResponse
from src.config import MODELS_DIR
from src.data.clean import clean_data
from src.data.features import engineer_features

ml_state = {}


def claim_to_raw_dataframe(claim: ClaimInput) -> pd.DataFrame:
    row = claim.model_dump(by_alias=True)
    return pd.DataFrame([row])


@asynccontextmanager
async def lifespan(app: FastAPI):
    import joblib

    from src.explain.shap_analysis import explain_instance  # noqa: F401  (import validated at startup)

    ml_state["pipeline"] = joblib.load(MODELS_DIR / "final_model.pkl")
    with open(MODELS_DIR / "final_model_meta.json") as f:
        meta = json.load(f)
    ml_state["model_key"] = meta["final_key"]
    ml_state["threshold"] = meta["threshold"]
    yield
    ml_state.clear()


app = FastAPI(title="Insurance Claims Fraud Detection API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    if "pipeline" not in ml_state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", model_key=ml_state["model_key"], threshold=ml_state["threshold"])


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput):
    from src.explain.shap_analysis import explain_instance

    pipeline = ml_state["pipeline"]
    threshold = ml_state["threshold"]

    raw_df = claim_to_raw_dataframe(claim)
    try:
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process claim features: {exc}")

    proba = float(pipeline.predict_proba(engineered)[0, 1])
    prediction = int(proba >= threshold)

    top_features = explain_instance(pipeline, engineered, top_n=5)

    return PredictionResponse(
        fraud_probability=proba,
        prediction=prediction,
        prediction_label="fraud" if prediction == 1 else "not_fraud",
        threshold_used=threshold,
        top_contributing_features=[FeatureContribution(**f) for f in top_features],
    )
