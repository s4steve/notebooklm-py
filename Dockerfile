FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

RUN pip install --no-cache-dir -e ".[mcp]"

# Auth file is mounted at runtime (not baked into image)
ENV NOTEBOOKLM_HOME=/data
ENV NOTEBOOKLM_DOWNLOAD_DIR=/downloads

RUN groupadd --system appgroup && useradd --system --gid appgroup appuser
USER appuser

EXPOSE 8765

CMD ["notebooklm-mcp", "--host", "0.0.0.0", "--port", "8765"]
