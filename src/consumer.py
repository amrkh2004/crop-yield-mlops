import os
import time
import json
import redis
import pandas as pd
import numpy as np

# In-memory model caching to prevent reloading per event
MODEL = None


def get_model(model_name: str = "RideDurationModel"):
    """
    Loads and caches the ML model ONCE in memory.
    """
    global MODEL
    if MODEL is None:
        try:
            import mlflow.pyfunc
            model_uri = f"models:/{model_name}/Production"
            print(f"[Consumer] Loading Production model from MLflow: {model_uri}")
            MODEL = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"[Consumer] MLflow load fallback ({e}). Using baseline RideDurationModel pipeline.")
            from src.batch_score import BaselineRideDurationModel
            MODEL = BaselineRideDurationModel()
    return MODEL


def store_result(event_id: str, payload: dict, predicted_duration: float, latency_ms: float, output_dir: str = "data/events/results"):
    """
    Stores prediction result and processing latency metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_record = {
        "event_id": event_id,
        "payload": payload,
        "predicted_duration_minutes": float(predicted_duration),
        "latency_ms": float(latency_ms),
        "processed_at": time.time()
    }
    
    # Append to daily event log
    file_path = os.path.join(output_dir, "processed_events.jsonl")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_record) + "\n")


def predict_on_event(event_id: str, payload: dict) -> dict:
    """
    Executes model inference on a single event payload.
    """
    start_time = time.perf_counter()
    model = get_model()

    df = pd.DataFrame([payload])
    if hasattr(model, "predict"):
        preds = model.predict(df)
    else:
        preds = model(df)

    pred_val = round(float(preds[0]), 2)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

    store_result(event_id, payload, pred_val, latency_ms)
    return {"event_id": event_id, "predicted_duration": pred_val, "latency_ms": latency_ms}


def run_consumer(
    redis_host: str = None,
    redis_port: int = 6379,
    stream_name: str = "ride_events",
    group_name: str = "ride_consumer_group",
    consumer_name: str = "worker_1"
):
    """
    Listens to Redis Streams and processes incoming events.
    Falls back to mock event loop if Redis server is unreachable.
    """
    redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
    print(f"[Consumer] Initializing Event-Driven Consumer on stream '{stream_name}'...")

    try:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        print(f"[Consumer] Connected to Redis at {redis_host}:{redis_port}")
        
        try:
            r.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except redis.exceptions.ResponseError:
            pass  # Group already exists

        while True:
            events = r.xreadgroup(group_name, consumer_name, {stream_name: ">"}, count=10, block=1000)
            if events:
                for stream, message_list in events:
                    for msg_id, payload in message_list:
                        raw_data = json.loads(payload.get("data", "{}"))
                        result = predict_on_event(msg_id, raw_data)
                        r.xack(stream_name, group_name, msg_id)
                        print(f"[Consumer] Processed event {msg_id}: duration={result['predicted_duration']}m (latency={result['latency_ms']}ms)")
    except Exception as e:
        print(f"[Consumer] Redis connection unavailable ({e}). Running standalone mock event benchmark...")
        mock_payload = {"distance_km": 12.5, "passengers": 2, "hour_of_day": 14}
        for i in range(1, 101):
            predict_on_event(f"mock_evt_{i:04d}", mock_payload)
        print("[Consumer] Processed 100 mock events successfully.")


if __name__ == "__main__":
    run_consumer()
