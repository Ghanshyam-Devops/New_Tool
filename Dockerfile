FROM python:3.10-slim

WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y tzdata curl && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Kolkata

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Healthcheck for Azure App Service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/ || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
