FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . redis pandas numpy mlflow

ENV PYTHONUNBUFFERED=1 \
    REDIS_HOST=redis

# Run consumer with restart unless-stopped flag
CMD ["python", "src/consumer.py"]
