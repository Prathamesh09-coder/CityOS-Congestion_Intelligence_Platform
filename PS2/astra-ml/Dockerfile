FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy lock files and pyproject.toml
COPY pyproject.toml uv.lock ./

# Install python dependencies with uv.
# We override the pytorch index to download CPU wheels specifically to keep the image slim.
RUN uv pip install --system --no-cache -e .[deep] --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code and model artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-u", "src/astra_ml/api/main.py"]
