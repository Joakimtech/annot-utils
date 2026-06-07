# Dockerfile - Containerize annot-utils for easy deployment

# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for image processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install annot-utils in development mode
RUN pip install -e .

# Create a non-root user for security
RUN useradd -m -s /bin/bash annotuser && chown -R annotuser:annotuser /app
USER annotuser

# Set the entry point to annot-utils CLI
ENTRYPOINT ["annot-utils"]

# Default command (shows help)
CMD ["--help"]
