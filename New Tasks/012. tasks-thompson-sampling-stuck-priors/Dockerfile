# Stage 1: Agent Environment
FROM python:3.11-slim

WORKDIR /app

# System Utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# ML Libraries
RUN pip install --no-cache-dir numpy pytest uv

# Copy Source Code (The buggy agent)
COPY task-deps/bandit.py /app/src/bandit.py
COPY task-deps/simulation.py /app/src/simulation.py

# Create Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]