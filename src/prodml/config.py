import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Crop Yield Prediction API"
    VERSION: str = "0.1.0"
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.pkl")

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
