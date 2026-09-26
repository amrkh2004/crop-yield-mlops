import asyncio
from src.bento_service import RideDurationService, PredictRequest, PredictResponse


def test_bento_service_predict():
    async def _test():
        service = RideDurationService()
        requests = [
            PredictRequest(distance_km=10.0, passengers=2, hour_of_day=14),
            PredictRequest(distance_km=5.0, passengers=1, hour_of_day=8),
        ]
        responses = await service.predict(requests)
        assert len(responses) == 2
        assert isinstance(responses[0], PredictResponse)
        assert responses[0].predicted_duration_minutes > 0.0

    asyncio.run(_test())


def test_bento_service_healthz():
    async def _test():
        service = RideDurationService()
        health = await service.healthz()
        assert health["status"] == "healthy"

    asyncio.run(_test())
