# CPU only image: the default configuration needs no GPU and downloads nothing.
FROM python:3.12-slim

LABEL org.opencontainers.image.authors="晨星"
LABEL org.opencontainers.image.title="phosphor-ai-stack"
LABEL org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PHOSPHOR_API_HOST=0.0.0.0 \
    PHOSPHOR_API_PORT=8080

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY pyproject.toml README.md LICENSE ./

ENV PYTHONPATH=/app/src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health').read()" || exit 1

CMD ["python", "-m", "phosphor.cli", "serve"]
