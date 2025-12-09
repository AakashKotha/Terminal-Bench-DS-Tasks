# Stage 1: Data Generation
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install pandas numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib pytest uv

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source Code (The Broken Extractor and The Train Script)
COPY task-deps/broken_extractor.py /app/src/feature_extractor.py
COPY task-deps/train.py /app/src/train.py

# Create Output Dir
RUN mkdir -p /app/models

CMD ["/bin/bash"]