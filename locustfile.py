import random
import json
from locust import HttpUser, task, between


class RideDurationLoadTestUser(HttpUser):
    """
    Locust load test user simulating client traffic against Ride Duration service.
    """
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict_endpoint(self):
        """
        Sends realistic ride payload to /predict endpoint.
        """
        payload = {
            "distance_km": round(random.uniform(1.0, 45.0), 2),
            "passengers": random.randint(1, 6),
            "hour_of_day": random.randint(0, 23)
        }
        headers = {"Content-Type": "application/json"}

        # BentoML batch endpoint accepts a list or single object based on schema
        with self.client.post(
            "/predict",
            data=json.dumps([payload]),
            headers=headers,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        response.success()
                    else:
                        response.failure(f"Unexpected JSON structure: {data}")
                except Exception as exc:
                    response.failure(f"JSON decode failed: {exc}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def health_endpoint(self):
        """
        Health probe request with lower task weight.
        """
        with self.client.get("/healthz", catch_response=True, name="/healthz") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with code {response.status_code}")
