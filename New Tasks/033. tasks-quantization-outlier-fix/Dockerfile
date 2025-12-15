FROM python:3.10-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install dependencies to speed up agent runtime
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pytest numpy

COPY . /app

# Ensure executable permissions
RUN chmod +x run-tests.sh

ENV PYTHONUNBUFFERED=1