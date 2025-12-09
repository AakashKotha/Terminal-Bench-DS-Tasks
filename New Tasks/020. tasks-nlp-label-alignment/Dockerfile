# Stage 1: Data
FROM python:3.11-slim as generator
WORKDIR /build
RUN pip install numpy
COPY task-deps/generate_data.py .
RUN python generate_data.py

# Stage 2: Agent Env
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install ML Stack (Heavy)
RUN pip install --no-cache-dir \
    torch \
    transformers \
    numpy \
    pytest \
    uv

# Pre-download BERT tokenizer to avoid network latency/timeout during test
RUN python -c "from transformers import BertTokenizerFast; BertTokenizerFast.from_pretrained('bert-base-uncased')"

# Copy Data
COPY --from=generator /app/data /app/data

# Copy Source
COPY task-deps/broken_processor.py /app/src/processor.py

# Output Dir
RUN mkdir -p /app/output

CMD ["/bin/bash"]