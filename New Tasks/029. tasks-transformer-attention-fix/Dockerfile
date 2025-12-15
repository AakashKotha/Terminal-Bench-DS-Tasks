FROM python:3.10-slim

WORKDIR /app

# Install System Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python Deps
# Using a specific torch version for stability
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    numpy \
    pytest \
    uv

# Copy Task Files
COPY task-deps /app/task-deps
COPY tests /app/tests
COPY run-tests.sh /app/run-tests.sh

# Permissions
RUN chmod +x /app/run-tests.sh

CMD ["/bin/bash"]