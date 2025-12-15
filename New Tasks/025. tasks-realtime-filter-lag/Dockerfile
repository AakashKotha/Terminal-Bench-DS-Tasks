# Stage 1: Data Gen
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy pandas
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas scipy pytest uv

COPY --from=generator /app/data /app/data
COPY task-deps/control_loop.py /app/src/control_loop.py

RUN mkdir -p /app/output

CMD ["/bin/bash"]