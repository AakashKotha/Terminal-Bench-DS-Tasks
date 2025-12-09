# Stage 1: Generator
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install torch numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (smaller footprint)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    pytest \
    uv

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/leaky_gnn.py /app/src/train_gnn.py

# Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]