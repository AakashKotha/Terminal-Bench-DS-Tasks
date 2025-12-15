# Stage 1: Data Gen
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pytest uv

# Setup dirs
COPY --from=generator /app/task_file/input_data /app/task_file/input_data
COPY task-deps/broken_packer.py /app/task_file/scripts/baseline_packer.py
COPY task-deps/generate_data.py /app/task_file/scripts/cost_model.py 

# Copy Tests for internal validation
COPY tests /app/tests

# Ensure output dir exists
RUN mkdir -p /app/output

CMD ["/bin/bash"]