# Use a multi-stage build to generate data and keep the final image clean

# --- Stage 1: Data and Script Builder ---
# This stage creates the data and test files
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies needed for data generation
RUN pip install pandas scikit-learn numpy

# Copy the data generation script and the "broken" training script
COPY task-deps/generate_data.py .
COPY task-deps/train.py .

# Run the script to create the data files in the /app directory
# This creates:
# 1. employees.csv (for the agent to use)
# 2. hidden_test_set.npz (for the test harness to use)
# 3. train.py (the "broken" script for the agent)
RUN python3 generate_data.py

# --- Stage 2: Final Task Image ---
# This is the image the agent will actually use
FROM python:3.11-slim

# Install curl (used by run-tests.sh)
# Also install core ML libraries the agent will need
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Install python packages the agent is likely to use
RUN pip install pandas scikit-learn numpy joblib

WORKDIR /app

# Create directories for data, models, src, and reports
RUN mkdir -p /app/data /app/models /app/src /app/reports

# Copy the "broken" script and "dirty" data from the builder
COPY --from=builder /app/train.py /app/src/train.py
COPY --from=builder /app/employees.csv /app/data/employees.csv

# Copy the test runner script and the tests directory
COPY run-tests.sh /app/run-tests.sh
COPY tests /app/tests

# Copy the hidden test set for the validation script
COPY --from=builder /app/hidden_test_set.npz /app/tests/hidden_test_set.npz

# Make the test runner script executable
RUN chmod +x /app/run-tests.sh