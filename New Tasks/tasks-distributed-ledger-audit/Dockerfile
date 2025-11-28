# Stage 1: Generator
# We use this stage to run the secret generation script. 
# The script itself will NOT be available in the final image, only its output.
FROM python:3.11-slim as generator

WORKDIR /build
# Install deps for generation
RUN pip install pandas pyarrow lxml numpy

# Copy generator
COPY task-deps/generate_data.py .

# Run generator (Produces /build/output/raw and /build/output/golden)
RUN python generate_data.py

# Stage 2: Final Agent Environment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for the Agent (and for tests)
# We pre-install pandas/pyarrow so the agent doesn't waste time installing them, 
# but they still need to import and use them correctly.
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    fastparquet \
    lxml \
    pytest \
    uv

# Copy the RAW data to the user's data directory
COPY --from=generator /build/output/raw /app/data/raw

# Copy the GOLDEN data to the hidden task-deps directory (for validation only)
COPY --from=generator /build/output/golden /app/task-deps

# Create output directory ensuring permissions
RUN mkdir -p /app/audit

CMD ["/bin/bash"]