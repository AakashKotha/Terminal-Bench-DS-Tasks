# Stage 1: Generator
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy scipy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

# Install System Utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install ML/Audio Stack
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pytest \
    uv

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/broken_train.py /app/src/train_model.py

# Output Dir
RUN mkdir -p /app/models

CMD ["/bin/bash"]