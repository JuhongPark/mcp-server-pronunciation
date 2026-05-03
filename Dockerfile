FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MCP_PRONUNCIATION_MODEL=tiny.en \
    MCP_PRONUNCIATION_PRELOAD=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

ENTRYPOINT ["mcp-server-pronunciation"]
CMD ["serve"]
