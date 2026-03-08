FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[mcp]"

# Auth file is mounted at runtime (not baked into image)
ENV NOTEBOOKLM_HOME=/data
ENV NOTEBOOKLM_DOWNLOAD_DIR=/downloads

EXPOSE 8765

CMD ["notebooklm-mcp", "--host", "0.0.0.0", "--port", "8765"]
