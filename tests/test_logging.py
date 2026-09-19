from prodml.logging import setup_logging, get_logger


def test_setup_logging_and_get_logger():
    """
    Tests that setup_logging executes without error and returns bound structlog logger.
    """
    setup_logging("DEBUG")
    logger = get_logger("test_logger")
    assert logger is not None


def test_logging_middleware_generated_request_id(client):
    """
    Tests that LoggingAndCorrelationMiddleware automatically attaches a UUID request_id.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0
