# Stage 1: Data Generation
FROM python:3.11-slim as generator

WORKDIR /build
# Install dependencies for generator
RUN pip install pandas pyarrow numpy

# Copy generator script
COPY task-deps/generate_data.py .

# Generate data (Outputs to /build/app/data)
# We move it to correct path in next stage
RUN mkdir -p /app/data && python generate_data.py && cp /app/data/tick_data.parquet /build/tick_data.parquet

# Stage 2: Final Agent Environment
FROM python:3.11-slim

WORKDIR /app

# Install System Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python Tools (Agent needs these)
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    fastparquet \
    pytest \
    uv

# Copy Generated Data from Stage 1
COPY --from=generator /build/tick_data.parquet /app/data/tick_data.parquet

# Copy the BROKEN script to the source directory for the agent to fix
COPY task-deps/broken_pipeline.py /app/src/feature_pipeline.py

# Ensure output directory for any tests
RUN mkdir -p /app/tests

CMD ["/bin/bash"]