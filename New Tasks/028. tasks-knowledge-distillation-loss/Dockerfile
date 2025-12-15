# Stage 1: Data Generation
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install torch numpy scikit-learn
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install ML stack
RUN pip install --no-cache-dir \
    torch \
    numpy \
    scikit-learn \
    pytest \
    uv

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/models.py /app/src/models.py
COPY task-deps/broken_train.py /app/src/train_distillation.py

# Create output directory
RUN mkdir -p /app/output

CMD ["/bin/bash"]