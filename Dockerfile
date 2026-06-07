# Dockerfile - Minimal working version
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files first
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Copy the package directory
COPY annot_utils/ annot_utils/

# Install dependencies and package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Set entrypoint
ENTRYPOINT ["annot-utils"]
CMD ["--help"]
