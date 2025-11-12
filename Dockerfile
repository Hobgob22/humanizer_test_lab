# Dockerfile – Production image with React + FastAPI

############################
# 1. Frontend Builder Stage
############################
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build


############################
# 2. Python Builder Stage
############################
FROM python:3.11-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps into /root/.local
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


############################
# 3. Final Runtime Stage
############################
FROM python:3.11-slim

# Install only minimal runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Workdir for the application
WORKDIR /app

# Copy Python dependencies from python-builder stage
COPY --from=python-builder /root/.local /home/appuser/.local

# Copy React build from frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy application source (preserve ownership)
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser .env* /app/

# Create writable directories and adjust ownership
RUN mkdir -p cache logs results \
           data/ai_texts data/human_texts data/ai_paras data/human_paras \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# PATH and PYTHONPATH so "src" package is always importable
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:${PYTHONPATH}

# Production environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI server
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
