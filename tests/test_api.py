import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import RAW_DATA_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def valid_payload():
    df = pd.read_csv(RAW_DATA_PATH)
    row = df.iloc[0].to_dict()
    for col in ["policy_number", "insured_zip", "incident_location", "fraud_reported"]:
        row.pop(col, None)
    # Serialize through JSON to coerce numpy scalar types (int64/float64) to
    # plain Python types, matching what a real HTTP client would send.
    import json
    return json.loads(json.dumps(row, default=str))


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert "model_key" in body
        assert 0 <= body["threshold"] <= 1


class TestPredict:
    def test_valid_claim_returns_200(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_valid_claim_response_schema(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["prediction"] in (0, 1)
        assert body["prediction_label"] in ("fraud", "not_fraud")
        assert (body["prediction"] == 1) == (body["prediction_label"] == "fraud")
        assert isinstance(body["top_contributing_features"], list)
        assert len(body["top_contributing_features"]) > 0
        for item in body["top_contributing_features"]:
            assert "feature" in item and "shap_value" in item

    def test_missing_required_field_returns_422(self, client, valid_payload):
        payload = copy.deepcopy(valid_payload)
        del payload["age"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client, valid_payload):
        payload = copy.deepcopy(valid_payload)
        payload["age"] = "not_a_number"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_out_of_range_age_returns_422(self, client, valid_payload):
        payload = copy.deepcopy(valid_payload)
        payload["age"] = 200
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_claim_amount_returns_422(self, client, valid_payload):
        payload = copy.deepcopy(valid_payload)
        payload["total_claim_amount"] = -100
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_unknown_category_handled_gracefully(self, client, valid_payload):
        """A never-seen auto_make/model should not crash -- the frequency
        encoder must map it to 0 rather than raising."""
        payload = copy.deepcopy(valid_payload)
        payload["auto_make"] = "TotallyMadeUpBrand"
        payload["auto_model"] = "MadeUpModelX"
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0.0 <= response.json()["fraud_probability"] <= 1.0
