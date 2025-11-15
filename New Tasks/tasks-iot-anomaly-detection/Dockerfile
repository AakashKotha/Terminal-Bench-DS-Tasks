# Use a multi-stage build to generate data and keep the final image clean

# --- Stage 1: Data Builder ---
# This stage creates the database and JSON files
FROM python:3.11-slim AS builder

WORKDIR /app

# We need pandas in the builder to create the data easily
RUN pip install pandas

# Copy the data generation script
COPY task-deps/generate_data.py .

# Run the script to create the data files in the /app directory
RUN python3 generate_data.py

# --- Stage 2: Final Task Image ---
# This is the image the agent will actually use
FROM python:3.11-slim

# Install curl (used by run-tests.sh to install uv) and sqlite3 (for the agent)
RUN apt-get update && apt-get install -y curl sqlite3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create directories for data, reports, and tests
RUN mkdir -p /app/data /app/reports /app/tests

# Copy the generated data files from the builder stage to their final location
COPY --from=builder /app/sensor_metadata.json /app/data/sensor_metadata.json
COPY --from=builder /app/sensor_readings.db /app/data/sensor_readings.db

# Copy the test runner script and the tests
COPY run-tests.sh /app/run-tests.sh
COPY tests /app/tests

# Make the test runner script executable
RUN chmod +x /app/run-tests.sh