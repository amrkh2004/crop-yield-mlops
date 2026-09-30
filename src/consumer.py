import json
import os
import time

import joblib
import pandas as pd

from prodml.data import FEATURE_NAMES
from prodml.logging import get_logger

logger = get_logger("prodml.consumer")

MODEL = None


def get_model(model_name: str = "CropYieldModel"):
    """
    Loads and caches the ML model ONCE in memory.
    """
    global MODEL
    if MODEL is None:
        try:
            import mlflow.pyfunc

            model_uri = f"models:/{model_name}/Production"
            logger.info("loading_mlflow_model", uri=model_uri)
            MODEL = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning("mlflow_load_fallback", error=str(e))
            model_path = "models/model.pkl"
            if os.path.exists(model_path):
                MODEL = joblib.load(model_path)
            else:
                from prodml.train import train_model_pipeline

                MODEL = train_model_pipeline()
    return MODEL


def store_result(
    event_id: str,
    payload: dict,
    predicted_yield_hg_ha: float,
    latency_ms: float,
    output_dir: str = "data/events/results",
):
    """
    Stores prediction result and processing latency metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_record = {
        "event_id": event_id,
        "payload": payload,
        "predicted_yield_hg_ha": float(predicted_yield_hg_ha),
        "predicted_yield_tons_ha": round(float(predicted_yield_hg_ha) / 10000.0, 4),
        "latency_ms": float(latency_ms),
        "processed_at": time.time(),
    }

    file_path = os.path.join(output_dir, "processed_events.jsonl")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_record) + "\n")


def predict_on_event(event_id: str, payload: dict) -> dict:
    """
    Executes model inference on a single crop yield event payload.
    """
    start_time = time.perf_counter()
    model = get_model()

    input_data = {
        "Area": payload.get("Area", "Egypt"),
        "Item": payload.get("Item", "Potatoes"),
        "Area_Item": f"{payload.get('Area', 'Egypt')}_{payload.get('Item', 'Potatoes')}",
        "Year": payload.get("Year", 2023),
        "average_rain_fall_mm_per_year": payload.get("average_rain_fall_mm_per_year", 760.5),
        "pesticides_tonnes": payload.get("pesticides_tonnes", 91.3),
        "avg_temp": payload.get("avg_temp", 24.5),
    }

    df = pd.DataFrame([input_data])[FEATURE_NAMES]
    if hasattr(model, "predict"):
        preds = model.predict(df)
    else:
        preds = model(df)

    pred_val = round(float(preds[0]), 2)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

    store_result(event_id, payload, pred_val, latency_ms)
    return {"event_id": event_id, "predicted_yield_hg_ha": pred_val, "latency_ms": latency_ms}


def run_consumer(
    redis_host: str = None,
    redis_port: int = 6379,
    stream_name: str = "crop_events",
    group_name: str = "crop_consumer_group",
    consumer_name: str = "worker_1",
):
    """
    Listens to Redis Streams and processes incoming crop yield events.
    Falls back to mock event loop if Redis server is unreachable.
    """
    redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
    logger.info("initializing_event_consumer", stream=stream_name, host=redis_host)

    try:
        import redis

        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()

        try:
            r.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except redis.exceptions.ResponseError:
            pass

        while True:
            events = r.xreadgroup(group_name, consumer_name, {stream_name: ">"}, count=10, block=1000)
            if events:
                for stream, message_list in events:
                    for msg_id, payload in message_list:
                        raw_data = json.loads(payload.get("data", "{}"))
                        result = predict_on_event(msg_id, raw_data)
                        r.xack(stream_name, group_name, msg_id)
                        logger.info(
                            "event_processed",
                            msg_id=msg_id,
                            yield_hg_ha=result["predicted_yield_hg_ha"],
                            latency_ms=result["latency_ms"],
                        )
    except Exception as e:
        logger.warning("redis_unavailable_fallback", error=str(e))
        mock_payload = {
            "Area": "Egypt",
            "Item": "Potatoes",
            "Year": 2023,
            "average_rain_fall_mm_per_year": 760.5,
            "pesticides_tonnes": 91.3,
            "avg_temp": 24.5,
        }
        for i in range(1, 101):
            predict_on_event(f"mock_evt_{i:04d}", mock_payload)
        logger.info("mock_events_processed", total=100)


if __name__ == "__main__":
    run_consumer()
