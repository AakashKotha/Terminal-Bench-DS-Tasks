# Stage 1: Data Generator
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install pandas pyarrow numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano && \
    rm -rf /var/lib/apt/lists/*

# Python deps
# pyarrow/fastparquet required for parquet operations
RUN pip install --no-cache-dir pandas numpy pyarrow fastparquet pytest uv

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/broken_backfill.py /app/src/create_training_set.py

# Setup output dir
RUN mkdir -p /app/data

CMD ["/bin/bash"]