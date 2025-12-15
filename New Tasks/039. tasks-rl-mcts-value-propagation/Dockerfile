FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install numpy pytest

COPY . /app
RUN chmod +x run-tests.sh

ENV PYTHONUNBUFFERED=1