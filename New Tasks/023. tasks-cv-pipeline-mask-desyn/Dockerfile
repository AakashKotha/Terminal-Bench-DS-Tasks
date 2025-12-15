# Stage 1: Setup
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Pillow/OpenCV if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and Vision
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    numpy \
    pytest \
    uv

# Copy Source
COPY task-deps/dataset.py /app/src/dataset.py
COPY task-deps/validate_alignment.py /app/src/validate_alignment.py

# Create output
RUN mkdir -p /app/output

CMD ["/bin/bash"]