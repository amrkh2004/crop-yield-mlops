import asyncio

from src.bento_service import CropPredictRequest, CropPredictResponse, CropYieldService


def test_bento_service_predict():
    async def _test():
        service = CropYieldService()
        requests = [
            CropPredictRequest(
                area="Egypt",
                item="Potatoes",
                year=2023,
                average_rain_fall_mm_per_year=760.5,
                pesticides_tonnes=91.3,
                avg_temp=24.5,
            ),
            CropPredictRequest(
                area="India",
                item="Wheat",
                year=2022,
                average_rain_fall_mm_per_year=1100.0,
                pesticides_tonnes=250.0,
                avg_temp=28.0,
            ),
        ]
        responses = await service.predict(requests)
        assert len(responses) == 2
        assert isinstance(responses[0], CropPredictResponse)
        assert responses[0].predicted_yield_hg_ha > 0.0

    asyncio.run(_test())


def test_bento_service_healthz():
    async def _test():
        service = CropYieldService()
        health = await service.healthz()
        assert health["status"] == "healthy"

    asyncio.run(_test())
