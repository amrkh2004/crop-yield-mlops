from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from prodml.config import settings
from prodml.logging import setup_logging, get_logger
from prodml.model import CropYieldModel
from prodml.api.schemas import (
    CropPredictionInput,
    CropPredictionOutput,
    FeedbackInput,
    FeedbackResponse,
    HealthResponse,
)
from prodml.api.middleware import LoggingAndCorrelationMiddleware

setup_logging(settings.LOG_LEVEL)
logger = get_logger("prodml.app")

ml_model = CropYieldModel(model_path=settings.MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler: Loads the ML model ONCE during server startup.
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


@app.post(
    "/predict",
    response_model=CropPredictionOutput,
    status_code=status.HTTP_200_OK,
)
async def predict(payload: CropPredictionInput, request: Request):
    """
    Predict crop yield (hg/ha and tons/ha) based on notebook trained model features:
    Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
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
    prediction_result = model.predict(input_dict)

    logger.info(
        "prediction_generated",
        request_id=request_id,
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
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
)
async def feedback(payload: FeedbackInput, request: Request):
    """
    Receive and log actual crop yield feedback/corrections to improve model tracking.
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
