FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch numpy psutil pytest uv

# Copy Data
COPY task-deps/train.py /app/task-deps/train.py

# Create output dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]