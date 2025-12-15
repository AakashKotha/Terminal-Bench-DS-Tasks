FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install common libs to save time
RUN pip install pytest pytest-asyncio

COPY . /app

RUN chmod +x run-tests.sh

ENV PYTHONUNBUFFERED=1