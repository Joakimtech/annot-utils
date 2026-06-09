FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -s /bin/bash annotuser && chown -R annotuser:annotuser /app
USER annotuser

# Entry point
ENTRYPOINT ["annot-utils"]
CMD ["--help"]
