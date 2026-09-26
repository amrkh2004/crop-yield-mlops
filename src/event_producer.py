import time
import json
import random
import numpy as np

try:
    from src.consumer import predict_on_event
except (ImportError, ModuleNotFoundError):
    from consumer import predict_on_event  # type: ignore[import-not-found,import-untyped]


def run_latency_benchmark(target_rate: int = 100, num_events: int = 500):
    """
    Simulates sending events at target_rate (e.g. 100 events/sec) and measures latency.
    """
    print(f"[EventProducer] Starting throughput benchmark at {target_rate} events/sec ({num_events} total events)...")

    latencies = []
    interval = 1.0 / target_rate

    start_batch = time.perf_counter()

    for i in range(1, num_events + 1):
        t0 = time.perf_counter()

        payload = {
            "distance_km": round(random.uniform(1.0, 35.0), 2),
            "passengers": random.randint(1, 5),
            "hour_of_day": random.randint(0, 23)
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

    print("\n" + "="*50)
    print("EVENT CONSUMER LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Events Processed : {num_events}")
    print(f"Target Throughput      : {target_rate} events/sec")
    print(f"Achieved Throughput    : {actual_rate:.2f} events/sec")
    print(f"Average Latency        : {avg_lat:.3f} ms")
    print(f"p50 Latency            : {p50:.3f} ms")
    print(f"p95 Latency            : {p95:.3f} ms")
    print(f"p99 Latency            : {p99:.3f} ms")
    print("="*50 + "\n")

    return {
        "achieved_rate": round(actual_rate, 2),
        "avg_latency_ms": round(avg_lat, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "p99_latency_ms": round(p99, 3)
    }


if __name__ == "__main__":
    run_latency_benchmark()
