# Stage 1: Generate Data
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

# Install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install python libs
RUN pip install --no-cache-dir \
    numpy \
    pytest \
    uv

# Copy Data from generator
COPY --from=generator /app/data /app/data

# Copy Source Code
COPY task-deps/broken_search.py /app/src/broken_search.py

# Create Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]