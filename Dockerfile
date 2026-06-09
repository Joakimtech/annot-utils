FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files first (for better caching)
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY annot_utils/ annot_utils/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -s /bin/bash annotuser && chown -R annotuser:annotuser /app
USER annotuser

# Set entrypoint
ENTRYPOINT ["annot-utils"]
CMD ["--help"]
