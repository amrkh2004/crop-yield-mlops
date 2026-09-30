import random
import time

import numpy as np

try:
    from src.consumer import predict_on_event
except (ImportError, ModuleNotFoundError):
    from consumer import predict_on_event  # type: ignore[import-not-found,import-untyped]

from prodml.logging import get_logger

logger = get_logger("prodml.event_producer")


def run_latency_benchmark(target_rate: int = 100, num_events: int = 500):
    """
    Simulates sending crop yield events at target_rate (e.g. 100 events/sec) and measures latency.
    """
    logger.info("starting_event_benchmark", target_rate=target_rate, num_events=num_events)

    areas = ["Egypt", "India", "Brazil", "United States of America", "Albania"]
    items = ["Potatoes", "Wheat", "Maize", "Rice, paddy", "Sorghum"]

    latencies = []
    interval = 1.0 / target_rate
    start_batch = time.perf_counter()

    for i in range(1, num_events + 1):
        t0 = time.perf_counter()

        payload = {
            "Area": random.choice(areas),
            "Item": random.choice(items),
            "Year": random.randint(1990, 2024),
            "average_rain_fall_mm_per_year": round(random.uniform(300.0, 1800.0), 2),
            "pesticides_tonnes": round(random.uniform(10.0, 500.0), 2),
            "avg_temp": round(random.uniform(12.0, 35.0), 2),
        }

        res = predict_on_event(f"bench_evt_{i:04d}", payload)
        latencies.append(res["latency_ms"])

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))

    total_time = time.perf_counter() - start_batch
    actual_rate = num_events / total_time

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_lat = np.mean(latencies)

    results = {
        "achieved_rate": round(actual_rate, 2),
        "avg_latency_ms": round(avg_lat, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "p99_latency_ms": round(p99, 3),
    }

    logger.info("benchmark_results", **results)
    return results


if __name__ == "__main__":
    run_latency_benchmark()
