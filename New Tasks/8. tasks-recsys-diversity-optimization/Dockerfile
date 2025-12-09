# Stage 1: Data Generation
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install pandas numpy pyarrow scipy scikit-learn
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Environment
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas numpy scipy scikit-learn pytest uv pyarrow

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
# We seed the "Naive" script as the starting point for the agent
COPY task-deps/naive_recommend.py /app/src/recommend.py

# Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]