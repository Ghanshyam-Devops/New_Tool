FROM python:3.10-slim
 
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && \
    apt-get install -y tzdata curl && \
    rm -rf /var/lib/apt/lists/*
 
ENV TZ=Asia/Kolkata
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Expose default port (informational)
EXPOSE 8000
 
# Health check using Azure's injected port
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8501}/ || exit 1
 
# 🚀 RUN WITH UVICORN (IMPORTANT)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8501}"]