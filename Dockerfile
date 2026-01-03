FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run via uvicorn pointing to module symbol (avoids __main__ module duality)
CMD ["uvicorn", "server:asgi_app", "--host", "0.0.0.0", "--port", "8080"]
