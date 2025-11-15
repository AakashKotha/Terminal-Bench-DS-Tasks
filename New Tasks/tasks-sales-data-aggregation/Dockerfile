# Use a standard Python base image
FROM python:3.11-slim

# Install curl (used by run-tests.sh to install uv)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create directories for data and reports
# The agent will be instructed to read from /app/data and write to /app/reports
RUN mkdir -p /app/data /app/reports

# Copy the "new data set" (the dirty CSV) into the image
COPY task-deps/sales_records.csv /app/data/sales_records.csv

# Copy the test runner script and the tests
COPY run-tests.sh /app/run-tests.sh
COPY tests /app/tests

# Make the test runner script executable
RUN chmod +x /app/run-tests.sh