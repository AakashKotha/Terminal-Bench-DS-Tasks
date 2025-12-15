# Stage 1: Data Generation
FROM python:3.11-slim as generator
WORKDIR /build
# Install sqlite3 for verification (optional, python has it built-in)
RUN pip install numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest uv

# Copy Data
COPY --from=generator /app/db /app/db
# Copy Problem
COPY task-deps/my-sql-query.sql /app/task-deps/my-sql-query.sql
# Copy Test helpers
COPY tests/golden.sql /app/tests/golden.sql

CMD ["/bin/bash"]