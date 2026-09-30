import json
import random

from locust import HttpUser, between, task


class CropYieldLoadTestUser(HttpUser):
    """
    Locust load test user simulating client traffic against Crop Yield service.
    """

    wait_time = between(0.1, 1.0)

    AREAS = ["Egypt", "India", "Brazil", "United States of America", "Albania"]
    ITEMS = ["Potatoes", "Wheat", "Maize", "Rice, paddy", "Sorghum"]

    @task(4)
    def predict_endpoint(self):
        """
        Sends realistic crop yield payload to /predict endpoint.
        """
        payload = {
            "area": random.choice(self.AREAS),
            "item": random.choice(self.ITEMS),
            "year": random.randint(1990, 2024),
            "average_rain_fall_mm_per_year": round(random.uniform(300.0, 1800.0), 2),
            "pesticides_tonnes": round(random.uniform(10.0, 500.0), 2),
            "avg_temp": round(random.uniform(12.0, 35.0), 2),
        }
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            "/predict",
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predicted_yield_hg_ha" in data or (isinstance(data, list) and len(data) > 0):
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
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with code {response.status_code}")
