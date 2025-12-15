FROM python:3.10-slim

# Install basic utilities
RUN apt-get update && apt-get install -y \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install testing dependencies
RUN pip install pytest

# Set working directory
WORKDIR /app

# Copy task dependencies
COPY task-deps /app/task-deps

# Ensure the log file is writable if created by the user, 
# though usually the agent runs as root in these benchmarks.
RUN touch /app/usage_log.txt && chmod 777 /app/usage_log.txt