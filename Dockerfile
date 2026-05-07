FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    UVICORN_WORKERS=4 \
    FUNCTION_ID=b702f636-f60c-4a3d-a6f4-f3568c13bd7d \
    RIVA_SERVER=grpc.nvcf.nvidia.com:443 \
    RIVA_USE_SSL=true \
    RIVA_MAX_MESSAGE_LENGTH=67108864

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -c "import riva.client"

COPY . .
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
