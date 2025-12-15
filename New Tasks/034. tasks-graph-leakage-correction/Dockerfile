FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install torch to save time during agent run
# CPU version is much lighter for benchmarking
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install pytest

COPY . /app

RUN chmod +x run-tests.sh

# Stop python from buffering stdout/stderr to see logs immediately
ENV PYTHONUNBUFFERED=1