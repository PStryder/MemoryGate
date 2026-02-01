FROM python:3.11-slim

WORKDIR /app

# Default to no auto-migrations in container runtime
ENV AUTO_MIGRATE_ON_STARTUP=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app \
    && chmod +x /app/entrypoint.sh

USER app

# Expose port
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, os, urllib.request; url=os.environ.get('MEMORYGATE_MCP_URL','http://localhost:8080/mcp'); token=os.environ.get('MEMORYGATE_API_KEY'); payload={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'memorygate.health','arguments':{}}}; req=urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}); token and req.add_header('Authorization','Bearer '+token); resp=urllib.request.urlopen(req, timeout=5); data=json.load(resp); assert 'result' in data" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
