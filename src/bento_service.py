from typing import Any, Dict, List

import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class CropPredictRequest(BaseModel):
    area: str = Field(..., description="Country name", json_schema_extra={"example": "Egypt"})
    item: str = Field(..., description="Crop item", json_schema_extra={"example": "Potatoes"})
    year: int = Field(..., ge=1990, le=2030, description="Crop year", json_schema_extra={"example": 2023})
    average_rain_fall_mm_per_year: float = Field(
        ..., ge=0.0, description="Average rainfall (mm/year)", json_schema_extra={"example": 760.5}
    )
    pesticides_tonnes: float = Field(
        ..., ge=0.0, description="Pesticides used (tonnes)", json_schema_extra={"example": 91.3}
    )
    avg_temp: float = Field(
        ..., ge=-10.0, le=60.0, description="Average temperature (C)", json_schema_extra={"example": 24.5}
    )


class CropPredictResponse(BaseModel):
    predicted_yield_hg_ha: float = Field(..., description="Predicted crop yield in hg/ha")
    predicted_yield_tons_ha: float = Field(..., description="Predicted crop yield in tons/ha")
    status: str = Field(default="success", description="Prediction status")


@bentoml.service(
    name="crop_yield_service",
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CropYieldService:
    """
    BentoML Service for Crop Yield Prediction with adaptive micro-batching support.
    """

    def __init__(self):
        try:
            import mlflow.pyfunc

            model_uri = "models:/CropYieldModel/Production"
            self.model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            import os

            import joblib

            model_path = "models/model.pkl"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                from prodml.train import train_model_pipeline

                self.model = train_model_pipeline()

    @bentoml.api(batchable=True, batch_dim=0)
    async def predict(self, requests: List[CropPredictRequest]) -> List[CropPredictResponse]:
        """
        Async micro-batched prediction endpoint for crop yield inference.
        """
        input_dicts = []
        for req in requests:
            input_dicts.append(
                {
                    "Area": req.area,
                    "Item": req.item,
                    "Area_Item": f"{req.area}_{req.item}",
                    "Year": req.year,
                    "average_rain_fall_mm_per_year": req.average_rain_fall_mm_per_year,
                    "pesticides_tonnes": req.pesticides_tonnes,
                    "avg_temp": req.avg_temp,
                }
            )

        df = pd.DataFrame(input_dicts)

        if hasattr(self.model, "predict"):
            preds = self.model.predict(df)
        else:
            preds = self.model(df)

        results = []
        for pred in preds:
            val_hg = float(np.round(np.clip(pred, 0, None), 2))
            val_tons = float(np.round(val_hg / 10000.0, 4))
            results.append(
                CropPredictResponse(
                    predicted_yield_hg_ha=val_hg,
                    predicted_yield_tons_ha=val_tons,
                    status="success",
                )
            )

        return results

    @bentoml.api
    async def healthz(self) -> Dict[str, Any]:
        """
        Health probe endpoint for service readiness.
        """
        return {"status": "healthy", "service": "crop_yield_service"}
