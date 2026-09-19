from typing import Optional
from pydantic import BaseModel, Field


class CropPredictionInput(BaseModel):
    area: str = Field(
        ...,
        description="Country or geographical area (e.g., Albania, Egypt, India, United States of America)",
        json_schema_extra={"example": "Albania"},
        min_length=2,
    )
    item: str = Field(
        ...,
        description="Crop type item (e.g., Maize, Wheat, Potatoes, Rice, paddy)",
        json_schema_extra={"example": "Maize"},
        min_length=2,
    )
    year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Harvest year (e.g., 2013, 2024)",
        json_schema_extra={"example": 2013},
    )
    average_rain_fall_mm_per_year: float = Field(
        ...,
        ge=0,
        description="Average annual rainfall in millimeters",
        json_schema_extra={"example": 1485.0},
    )
    pesticides_tonnes: float = Field(
        ...,
        ge=0,
        description="Total pesticides usage in tonnes",
        json_schema_extra={"example": 121.0},
    )
    avg_temp: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Average annual temperature in Celsius",
        json_schema_extra={"example": 16.37},
    )


class CropPredictionOutput(BaseModel):
    request_id: str = Field(..., description="Unique correlation request ID")
    area: str = Field(..., description="Target area / country")
    item: str = Field(..., description="Target crop item")
    predicted_yield_hg_ha: float = Field(..., description="Predicted yield in hectograms per hectare (hg/ha)")
    predicted_yield_tons_ha: float = Field(..., description="Predicted yield in metric tons per hectare (tons/ha)")
    status: str = Field(default="success", description="Prediction status")


class FeedbackInput(BaseModel):
    request_id: str = Field(..., description="Correlation ID of prediction request")
    actual_yield_hg_ha: float = Field(..., ge=0, description="Observed yield in hg/ha")
    comments: Optional[str] = Field(None, description="Optional feedback comments")


class FeedbackResponse(BaseModel):
    status: str = Field(default="recorded", description="Feedback status")
    message: str = Field(..., description="Acknowledgement message")
    request_id: str = Field(..., description="Request ID associated with feedback")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health state")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model pipeline readiness indicator")
