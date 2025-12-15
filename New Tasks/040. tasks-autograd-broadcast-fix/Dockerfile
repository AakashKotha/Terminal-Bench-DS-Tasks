FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir numpy pytest

# Set working directory
WORKDIR /app

# Copy task files
COPY task-deps/nanograd.py /app/nanograd.py
COPY task-deps/train.py /app/train.py

# Env vars
ENV PYTHONPATH=/app