# Stage 1: Data Generation
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pytest uv

# Copy Data (Input and Hidden Truth)
COPY --from=generator /app/data /app/data
COPY --from=generator /app/tests /app/tests

# Copy Source
COPY task-deps/naive_hmm.py /app/src/gene_finder.py

# Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]