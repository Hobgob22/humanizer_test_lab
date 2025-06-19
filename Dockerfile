# Dockerfile – Production image (optimized)

############################
# 1. Builder stage
############################
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps into /root/.local
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


############################
# 2. Final runtime stage
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

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source (preserve ownership)
COPY --chown=appuser:appuser . .

# Create writable directories and adjust ownership
RUN mkdir -p cache logs results \
           data/ai_texts data/human_texts data/ai_paras data/human_paras \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# PATH and PYTHONPATH so “src” package is always importable
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:${PYTHONPATH}

# Production environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit
CMD ["streamlit", "run", "src/ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=200"]
