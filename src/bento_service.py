import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class PredictRequest(BaseModel):
    distance_km: float = Field(
        ...,
        gt=0,
        description="Trip distance in kilometers",
        json_schema_extra={"example": 12.5},
    )
    passengers: int = Field(
        ...,
        ge=1,
        le=8,
        description="Number of passengers",
        json_schema_extra={"example": 2},
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the day (0-23)",
        json_schema_extra={"example": 14},
    )


class PredictResponse(BaseModel):
    predicted_duration_minutes: float = Field(..., description="Predicted trip duration in minutes")
    status: str = Field(default="success", description="Prediction status")


# Modern BentoML 1.2+ Service Definition
@bentoml.service(
    name="ride_duration_service",
    resources={"cpu": "2"},
    traffic={"timeout": 10}
)
class RideDurationService:
    """
    BentoML Service for Ride Duration Prediction with micro-batching support.
    """

    def __init__(self):
        # Load model pipeline or fallback
        try:
            import mlflow.pyfunc
            model_uri = "models:/RideDurationModel/Production"
            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"[BentoML] Successfully loaded model from MLflow Registry: {model_uri}")
        except Exception as e:
            print(f"[BentoML] MLflow load fallback ({e}). Using baseline Scikit-Learn predictor.")
            from sklearn.ensemble import RandomForestRegressor
            X_dummy = np.array([[5.0, 2, 14], [10.0, 1, 8], [2.5, 3, 18]])
            y_dummy = np.array([16.5, 28.0, 10.2])
            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(X_dummy, y_dummy)
            self.model = rf

    @bentoml.api(batchable=True, batch_dim=0)
    async def predict(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """
        Async micro-batched prediction endpoint.
        Receives a batch of requests and executes inference concurrently.
        """
        data = [req.model_dump() for req in requests]
        df = pd.DataFrame(data)

        if hasattr(self.model, "predict"):
            predictions = self.model.predict(df)
        else:
            predictions = self.model(df)

        results = []
        for pred in predictions:
            val = float(np.round(pred, 2))
            results.append(PredictResponse(predicted_duration_minutes=val, status="success"))

        return results

    @bentoml.api
    async def healthz(self) -> Dict[str, Any]:
        """
        Health probe endpoint for load balancing readiness.
        """
        return {"status": "healthy", "service": "ride_duration_service"}
