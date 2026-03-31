FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project source
COPY . .

# Create output directory
RUN mkdir -p /app/outputs

# Default command: run the full pipeline
CMD ["python", "main.py"]
