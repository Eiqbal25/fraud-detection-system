"""Pydantic request/response models for the fraud-detection API."""
from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ClaimInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    months_as_customer: int = Field(..., ge=0, description="Tenure with the insurer, in months")
    age: int = Field(..., ge=16, le=100)
    policy_bind_date: date
    policy_state: str
    policy_csl: str = Field(..., description="Combined single limit, e.g. '250/500'")
    policy_deductable: float = Field(..., ge=0)
    policy_annual_premium: float = Field(..., gt=0)
    umbrella_limit: float = Field(..., ge=0)
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: float = Field(0, alias="capital-gains")
    capital_loss: float = Field(0, alias="capital-loss")
    incident_date: date
    incident_type: str
    collision_type: str | None = None
    incident_severity: str
    authorities_contacted: str | None = None
    incident_state: str
    incident_city: str
    incident_hour_of_the_day: int = Field(..., ge=0, le=23)
    number_of_vehicles_involved: int = Field(..., ge=1)
    property_damage: str | None = None
    bodily_injuries: int = Field(..., ge=0)
    witnesses: int = Field(..., ge=0)
    police_report_available: str | None = None
    total_claim_amount: float = Field(..., ge=0)
    injury_claim: float = Field(..., ge=0)
    property_claim: float = Field(..., ge=0)
    vehicle_claim: float = Field(..., ge=0)
    auto_make: str
    auto_model: str
    auto_year: int = Field(..., ge=1980, le=2100)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    prediction_label: str
    threshold_used: float
    top_contributing_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_key: str
    threshold: float
