FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install torch cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install pytest numpy

COPY . /app

RUN chmod +x run-tests.sh

ENV PYTHONUNBUFFERED=1