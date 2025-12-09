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
    curl git vim nano && \
    rm -rf /var/lib/apt/lists/*

# Install Data Science Stack
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    pyarrow \
    fastparquet \
    pytest \
    uv

# Copy Data
COPY --from=generator /app/data /app/data
COPY --from=generator /app/tests /app/tests

# Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]