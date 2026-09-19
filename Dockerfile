# ==========================================
# Stage 1: Build & Dependency Installation
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build tools if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files and install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ==========================================
# Stage 2: Minimal Runtime Stage
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create unprivileged non-root user (appuser) for security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Copy installed virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and models directory
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup pyproject.toml README.md ./

# Create directories for models and logs with non-root ownership
RUN mkdir -p /app/models /app/logs && chown -R appuser:appgroup /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    MODEL_PATH=/app/models/model.pkl

# Switch to non-root user
USER appuser

EXPOSE 8000

# Run FastAPI server with Uvicorn
CMD ["uvicorn", "prodml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
