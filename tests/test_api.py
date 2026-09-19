def test_health_endpoint(client):
    """
    Test /health endpoint returns 200 OK with expected health schema.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict_endpoint_success(client, valid_prediction_payload):
    """
    Test /predict endpoint returns 200 OK with valid payload matching notebook schema.
    """
    response = client.post("/predict", json=valid_prediction_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["area"] == "Albania"
    assert data["item"] == "Maize"
    assert data["predicted_yield_hg_ha"] >= 0.0
    assert data["predicted_yield_tons_ha"] >= 0.0
    assert data["status"] == "success"
    assert "X-Request-ID" in response.headers


def test_predict_endpoint_custom_request_id(client, valid_prediction_payload):
    """
    Test /predict endpoint preserves custom X-Request-ID header.
    """
    custom_id = "custom-correlation-id-999"
    headers = {"X-Request-ID": custom_id}
    response = client.post("/predict", json=valid_prediction_payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == custom_id
    assert response.headers["X-Request-ID"] == custom_id


def test_predict_endpoint_validation_error(client, invalid_prediction_payload):
    """
    Test /predict endpoint rejects invalid payload with 422 Unprocessable Entity.
    """
    response = client.post("/predict", json=invalid_prediction_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_feedback_endpoint_success(client, sample_feedback_payload):
    """
    Test /feedback endpoint returns 200 OK for valid feedback data.
    """
    response = client.post("/feedback", json=sample_feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert data["request_id"] == sample_feedback_payload["request_id"]


def test_feedback_endpoint_validation_error(client):
    """
    Test /feedback endpoint rejects negative actual yield with 422.
    """
    invalid_feedback = {
        "request_id": "test-id",
        "actual_yield_hg_ha": -10.0,
    }
    response = client.post("/feedback", json=invalid_feedback)
    assert response.status_code == 422
