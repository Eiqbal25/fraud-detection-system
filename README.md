# Insurance Claims Fraud Detection System

A production-style classical ML system for flagging fraudulent auto insurance claims: leakage-safe data pipeline, three tuned models compared under two imbalance-handling strategies, an honestly-evaluated ensemble, SHAP explainability, a served FastAPI endpoint, and a Streamlit dashboard — all backed by a 50-test pytest suite.

This project exists to demonstrate classical ML competency (imbalanced classification, hyperparameter tuning, explainability, deployment) alongside existing LLM/RAG/agent work.

## Dataset

[`insurance_claims.csv`](data/raw/insurance_claims.csv) — the standard "Auto Insurance Claims Fraud" dataset (1,000 claims, 39 raw columns: policy details, insured demographics, incident details, claim amounts, vehicle info, and the `fraud_reported` target).

- **Class balance**: 247 fraud / 753 not-fraud = **24.7% fraud rate** (minority class)
- **Missingness** (encoded as `?` in the raw export): `collision_type` (178 rows), `property_damage` (360 rows), `police_report_available` (343 rows), `authorities_contacted` (91 rows)
- **Data quality issues found and fixed**: 1 row with a negative `umbrella_limit` (impossible for a coverage cap), 1 row with `incident_date` before `policy_bind_date` (impossible sequence) — both are treated as missing and imputed downstream rather than silently trusted
- Raw data is **not committed to git** (see `.gitignore`) — the file lives only under `data/raw/` locally

## Architecture

```
fraud-detection-system/
├── src/
│   ├── config.py            # paths, seed, column lists
│   ├── data/
│   │   ├── load.py          # CSV -> DataFrame, missing-token normalization, target binarization
│   │   ├── clean.py         # dedupe + impossible-value fixes (NOT imputation -- see below)
│   │   ├── features.py      # ratio features, ColumnTransformer (impute+scale+encode)
│   │   └── split.py         # stratified 80/20 split, orchestrates the full pipeline
│   ├── models/
│   │   ├── train.py         # 3 algorithms x 2 imbalance strategies = 6 tuned pipelines
│   │   ├── tune.py          # RandomizedSearchCV, 5-fold stratified, scored on PR-AUC
│   │   ├── ensemble.py      # soft-voting ensemble, evaluated against the best single model
│   │   ├── evaluate.py      # metrics, cost-sensitive threshold selection, confusion matrix
│   │   └── select_final.py  # picks final model + deployment threshold, writes the report
│   ├── explain/
│   │   └── shap_analysis.py # global + per-claim SHAP explanations
│   └── api/
│       ├── main.py          # FastAPI app (/predict, /health)
│       └── schemas.py       # Pydantic request/response models
├── dashboard/app.py          # Streamlit: live scoring, model comparison, SHAP, confusion matrix
├── tests/                    # 50 pytest tests across pipeline, features, models, imbalance, API
├── docker/                   # Dockerfile + docker-compose.yml (API + dashboard)
├── data/{raw,processed}/
├── models/                   # saved .pkl pipelines + final_model_meta.json
├── results/                  # evaluation_report.md, model_results.json, SHAP plots
└── notebooks/eda.ipynb       # exploration only, not the pipeline
```

### Why cleaning and feature engineering are split the way they are

`clean.py` only does **structural** fixes (dedup rows, null out impossible values). It deliberately does **not** impute with dataset-wide statistics. Imputation, scaling, and encoding all live inside `features.py`'s `ColumnTransformer`, which is fit **only on the training split** and just `.transform()`-ed on test. Computing a median/mode across the full dataset before splitting would leak test-set information into training — this is exactly what `tests/test_data_pipeline.py::TestNoLeakage` checks for.

SMOTE has the same property: it's wired into an `imblearn.pipeline.Pipeline` (not plain sklearn), whose samplers are automatically no-ops at `.predict()`/`.transform()` time. So `pipeline.fit(X_train, y_train)` oversamples the training fold, but `pipeline.predict(X_test)` never touches SMOTE at all — verified in `tests/test_imbalance_handling.py`.

## Feature engineering

Beyond imputation/scaling/encoding (one-hot for low-cardinality categoricals, frequency encoding for high-cardinality ones like `auto_model`, `insured_hobbies`), five ratio/derived features are engineered:

- `days_to_incident` — days between policy bind date and incident date
- `claim_to_premium_ratio` — total claim vs. annual premium
- `injury_claim_ratio`, `property_claim_ratio`, `vehicle_claim_ratio` — how the claim breaks down
- `claim_per_vehicle` — total claim normalized by vehicles involved

These are pure row-wise arithmetic (no fitted statistics), so they're computed before the train/test split with no leakage risk.

## Models & imbalance handling

Three algorithms, each trained under **both** imbalance strategies (6 pipelines total), tuned with `RandomizedSearchCV` (25 iterations, 5-fold stratified CV, optimizing **PR-AUC** — not accuracy, which is misleading at 24.7% fraud prevalence):

| Model | Strategy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | class_weight='balanced' | 0.556 | 0.714 | 0.625 | 0.820 | 0.561 |
| Logistic Regression | SMOTE | 0.556 | 0.714 | 0.625 | 0.804 | 0.580 |
| Random Forest | class_weight='balanced' | 0.556 | 0.816 | 0.661 | 0.820 | 0.546 |
| Random Forest | SMOTE | 0.583 | 0.714 | 0.642 | 0.816 | 0.588 |
| XGBoost | scale_pos_weight | 0.533 | 0.490 | 0.511 | 0.793 | 0.497 |
| **XGBoost** | **SMOTE** | **0.632** | **0.735** | **0.679** | **0.822** | **0.599** |
| Soft-voting ensemble (all 3, SMOTE variants) | — | 0.583 | 0.714 | 0.642 | 0.826 | 0.601 |

*(Full numbers, best hyperparameters, and CV scores: [`results/model_results.json`](results/model_results.json). Full writeup: [`results/evaluation_report.md`](results/evaluation_report.md).)*

### Does the ensemble win? Honestly, no.

The soft-voting ensemble edges out the best single model on PR-AUC by **+0.002** — noise, not a real improvement — while its recall (0.714) is actually *lower* than XGBoost+SMOTE alone (0.735). Averaging in the weaker logistic regression and random forest members drags recall down for a marginal, unreliable PR-AUC gain. Given the cost-sensitive framing below, **`xgboost__smote` is the deployed model**, not the ensemble — same spirit as documenting a component that doesn't earn its complexity rather than quietly shipping it anyway.

### Cost-sensitive threshold

A missed fraud (false negative) costs far more than a legitimate claim getting flagged for manual review (false positive) — a flagged claim just costs a reviewer's time; a missed fraud pays out in full. So the deployed threshold isn't the default 0.5 (tuned for balanced F1); it's chosen to guarantee **≥85% recall** on fraud:

| Threshold | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|
| 0.5 (default) | 0.632 | 0.735 | 0.679 |
| **0.379 (deployed)** | 0.412 | **0.857** | 0.556 |

This is a deliberate trade: at the deployed threshold, 42/49 fraud cases in the test set are caught (vs. 36/49 at the default threshold), at the cost of more legitimate claims routed to manual review. That's the correct trade for a fraud-triage system, not a fully-automated denial system.

**Key result**: switching XGBoost's imbalance strategy from class-weighting to SMOTE improved fraud-class F1 from **0.511 to 0.679 (+32.9%)**; tuning the deployment threshold then traded some of that F1 back for a **+16.7 point recall gain** (0.735 → 0.857), matching the stated cost asymmetry.

## Explainability (SHAP)

Global feature importance and a worked local example are in [`results/shap_plots/global_importance.png`](results/shap_plots/global_importance.png) and [`results/shap_local_example.json`](results/shap_local_example.json). Top global drivers of a fraud prediction: `incident_severity == Major Damage` (by a wide margin), followed by `insured_hobbies` (frequency-encoded).

**Honest caveat**: `insured_hobbies` being a strong signal is a known quirk of this specific dataset (certain hobbies like chess/cross-fit happen to correlate with the fraud label here) rather than a generalizable real-world fraud indicator — flagged in [`notebooks/eda.ipynb`](notebooks/eda.ipynb) and worth re-validating against real client data before trusting it in production. This is exactly the kind of thing SHAP is for: surfacing what a model actually learned so a human can sanity-check it, not just trusting the probability score.

## API

```bash
uvicorn src.api.main:app --reload
```

- `POST /predict` — accepts raw claim fields (see [`src/api/schemas.py`](src/api/schemas.py)), runs them through the same `clean_data`/`engineer_features` code used at training time, returns fraud probability, the binary prediction at the deployed threshold, and the top 5 SHAP-contributing features
- `GET /health` — returns the loaded model key and threshold

## Dashboard

```bash
streamlit run dashboard/app.py
```

Live claim scoring form, all-model comparison table, global SHAP importance chart, and a per-model confusion matrix viewer.

## Testing

**50 tests, all passing** (`pytest tests/ -q`):

- `test_data_pipeline.py` (14) — no leakage (scaler/encoder fit only on train, verified by checking fitted statistics against a train-only mean vs. a full-dataset mean), correct split shapes/ratios, no nulls post-preprocessing
- `test_features.py` (10) — ratio-feature arithmetic on synthetic rows, divide-by-zero guards, frequency-encoder behavior on unseen categories
- `test_models.py` (10) — probabilities in [0,1], `predict`/`predict_proba` agreement, saved artifacts load and score correctly
- `test_imbalance_handling.py` (6) — class weights actually get set on the classifiers, SMOTE measurably balances the training distribution (24.7% → 50%) and is a no-op at predict time
- `test_api.py` (10) — valid claims return 200 with the correct schema, missing/malformed/out-of-range fields return 422, an unseen auto make/model doesn't crash the frequency encoder

## Deployment

```bash
docker compose -f docker/docker-compose.yml up --build
```

Runs the FastAPI service (port 8000) and Streamlit dashboard (port 8501) as two containers from one image.

## Reproducing this from scratch

```bash
pip install -r requirements.txt
python -m src.data.split            # build data/processed/{train,test}.csv
python -m src.models.train          # train + tune all 6 pipelines
python -m src.models.ensemble       # build + evaluate the soft-voting ensemble
python -m src.models.select_final   # pick final model/threshold, write results/evaluation_report.md
python -m src.explain.shap_analysis # generate SHAP plots
pytest tests/ -q
```

`RANDOM_SEED = 42` is fixed in `src/config.py`, so re-running the pipeline reproduces the exact numbers above.

## Tech stack

`scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE), `shap`, `pandas`, `numpy`, `FastAPI`, `Pydantic`, `Streamlit`, `pytest`, `Docker` / `Docker Compose`.
