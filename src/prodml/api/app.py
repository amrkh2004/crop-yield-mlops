import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, status

from prodml.api.middleware import LoggingAndCorrelationMiddleware
from prodml.api.schemas import (
    BatchCropPredictionInput,
    BatchCropPredictionOutput,
    CropPredictionInput,
    CropPredictionOutput,
    FeedbackInput,
    FeedbackResponse,
    HealthResponse,
    MetadataResponse,
)
from prodml.config import settings
from prodml.logging import get_logger, setup_logging
from prodml.model import CropYieldModel

setup_logging(settings.LOG_LEVEL)
logger = get_logger("prodml.app")

ml_model = CropYieldModel(
    model_path=settings.MODEL_PATH,
    onnx_path=(
        os.path.join(os.path.dirname(settings.MODEL_PATH), "model.onnx")
        if hasattr(settings, "MODEL_PATH")
        else "models/model.onnx"
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler: Loads ML models (Pickle & ONNX) during startup.
    """
    logger.info("app_startup", message="Loading machine learning model pipeline...")
    ml_model.load_or_create()
    app.state.model = ml_model
    logger.info("app_startup_complete", model_path=settings.MODEL_PATH)
    yield
    logger.info("app_shutdown", message="Shutting down API server...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(LoggingAndCorrelationMiddleware)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health():
    """
    Health check endpoint for service readiness and load balancer probes.
    """
    model_loaded = hasattr(app.state, "model") and app.state.model.pipeline is not None
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=model_loaded,
    )


@app.get("/metadata", response_model=MetadataResponse, status_code=status.HTTP_200_OK)
async def metadata():
    """
    Metadata endpoint returning model version, schema features, and supported backends.
    """
    model: CropYieldModel = getattr(app.state, "model", ml_model)
    return MetadataResponse(
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        model_path=settings.MODEL_PATH,
        onnx_path="models/model.onnx",
        feature_names=model.FEATURE_NAMES,
        categorical_features=model.CATEGORICAL_FEATURES,
        numeric_features=model.NUMERIC_FEATURES,
        supported_backends=["pickle", "onnx"],
    )


@app.post(
    "/predict",
    response_model=CropPredictionOutput,
    status_code=status.HTTP_200_OK,
)
async def predict(
    payload: CropPredictionInput,
    request: Request,
    backend: str = Query("pickle", description="Backend engine to use: 'pickle' or 'onnx'"),
):
    """
    Predict crop yield (hg/ha and tons/ha) for a single input item.
    """
    request_id = getattr(request.state, "request_id", "N/A")
    input_dict = {
        "Area": payload.area,
        "Item": payload.item,
        "Year": payload.year,
        "average_rain_fall_mm_per_year": payload.average_rain_fall_mm_per_year,
        "pesticides_tonnes": payload.pesticides_tonnes,
        "avg_temp": payload.avg_temp,
    }

    model: CropYieldModel = app.state.model
    prediction_result = model.predict(input_dict, backend=backend)

    logger.info(
        "prediction_generated",
        request_id=request_id,
        backend=backend,
        area=payload.area,
        item=payload.item,
        predicted_yield_hg_ha=prediction_result["predicted_yield_hg_ha"],
    )

    return CropPredictionOutput(
        request_id=request_id,
        area=payload.area,
        item=payload.item,
        predicted_yield_hg_ha=prediction_result["predicted_yield_hg_ha"],
        predicted_yield_tons_ha=prediction_result["predicted_yield_tons_ha"],
        status="success",
    )


@app.post(
    "/predict/batch",
    response_model=BatchCropPredictionOutput,
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    payload: BatchCropPredictionInput,
    request: Request,
    backend: str = Query("pickle", description="Backend engine to use: 'pickle' or 'onnx'"),
):
    """
    Batch prediction endpoint for processing multiple crop items simultaneously.
    """
    request_id = getattr(request.state, "request_id", "N/A")
    input_dicts = [
        {
            "Area": item.area,
            "Item": item.item,
            "Year": item.year,
            "average_rain_fall_mm_per_year": item.average_rain_fall_mm_per_year,
            "pesticides_tonnes": item.pesticides_tonnes,
            "avg_temp": item.avg_temp,
        }
        for item in payload.inputs
    ]

    model: CropYieldModel = app.state.model
    batch_results = model.predict(input_dicts, backend=backend)

    outputs = []
    for item_input, result in zip(payload.inputs, batch_results):
        outputs.append(
            CropPredictionOutput(
                request_id=request_id,
                area=item_input.area,
                item=item_input.item,
                predicted_yield_hg_ha=result["predicted_yield_hg_ha"],
                predicted_yield_tons_ha=result["predicted_yield_tons_ha"],
                status="success",
            )
        )

    logger.info(
        "batch_prediction_generated",
        request_id=request_id,
        backend=backend,
        total_items=len(outputs),
    )

    return BatchCropPredictionOutput(
        request_id=request_id,
        predictions=outputs,
        total_items=len(outputs),
        status="success",
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
)
async def feedback(payload: FeedbackInput, request: Request):
    """
    Receive and log actual crop yield feedback/corrections for model monitoring.
    """
    logger.info(
        "feedback_received",
        correlation_request_id=payload.request_id,
        actual_yield_hg_ha=payload.actual_yield_hg_ha,
        comments=payload.comments,
    )

    return FeedbackResponse(
        status="recorded",
        message="Feedback successfully recorded for model evaluation",
        request_id=payload.request_id,
    )
