# Multi-stage Dockerfile for TacticaLM
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code, then install runtime dependencies.
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# --- API Stage ---
FROM base as api
EXPOSE 8000
CMD ["uvicorn", "tacticalm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Streamlit Stage ---
FROM base as ui
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Default to API if no target is specified
FROM api
