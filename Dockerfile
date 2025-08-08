# Use Python 3.11 slim image for better performance and smaller size
FROM python:3.11-slim

# Set environment variables for better Docker and Render compatibility
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies (minimal for faster builds)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
        libffi-dev \
        libssl-dev \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations for Render
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (required for document processing)
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application code (exclude unnecessary files)
COPY main.py .
COPY supabase_utils.py .

# Remove local storage directories since we're using Supabase
# No need for uploads, vector_db, logs directories

# Create non-root user for security (Render recommended practice)
RUN useradd --create-home --shell /bin/bash --uid 1000 app \
    && chown -R app:app /app
USER app

# Expose the port (Render will use $PORT environment variable)
EXPOSE $PORT

# Health check for better monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/v1/health || exit 1

# Use CMD instead of ENTRYPOINT for Render compatibility
# Render will set the PORT environment variable automatically
CMD uvicorn main:app --host $HOST --port $PORT --workers 1 --log-level info --access-log