# Stage 1: Data Gen
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install torch pandas
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch pandas pytest uv

COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/broken_dataset.py /app/src/dataset.py
COPY task-deps/debug_training.py /app/src/debug_training.py

RUN mkdir -p /app/output

CMD ["/bin/bash"]