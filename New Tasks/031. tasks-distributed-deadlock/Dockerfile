# Use a lightweight Python base
FROM python:3.10-slim

# Install system utilities needed for debugging (optional but helpful for agents)
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install torch to reduce runtime overhead
# Using CPU-only torch to keep image size reasonable for benchmarking
RUN pip install --no-cache-dir torch pytest

# Copy task files
COPY . /app

# Make scripts executable
RUN chmod +x run-tests.sh

# Environment variables to force unbuffered output (helps with logging concurrency)
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]