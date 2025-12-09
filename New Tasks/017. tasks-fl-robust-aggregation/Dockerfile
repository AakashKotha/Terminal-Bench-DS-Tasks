# Stage 1: Data
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy scikit-learn
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy scikit-learn pytest uv

COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/fl_server.py /app/src/fl_server.py

# Output Dir
RUN mkdir -p /app/models

CMD ["/bin/bash"]