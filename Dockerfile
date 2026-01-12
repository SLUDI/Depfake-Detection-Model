# Multi-stage build for optimized production image
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Set environment variables for caching
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/datasets
ENV TORCH_HOME=/app/.cache/torch
ENV MPLCONFIGDIR=/app/.cache/matplotlib


# Copy application code
COPY app.py .
COPY config.py .
COPY video_processor.py .
COPY model/ ./model/


# Create non-root user for security (optional but recommended)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run with optimized settings
CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--workers", "1", \
    "--timeout-keep-alive", "60", \
    "--log-level", "info"]
