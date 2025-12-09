# Stage 1: Setup
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pytest uv

# Copy Code
COPY task-deps/grid_env.py /app/src/grid_env.py
COPY task-deps/train.py /app/src/train.py

# Create output dir
RUN mkdir -p /app/models

CMD ["/bin/bash"]