FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy pytest uv

# Copy Source Code
COPY task-deps/grid_env.py /app/src/grid_env.py
COPY task-deps/train.py /app/src/train.py

# Setup Output Directory
RUN mkdir -p /app/output

CMD ["/bin/bash"]