# Image for the extraction service, referenced by ops/docker-compose.yml.
# Like that file, this has never been built -- the host had no Docker access.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# arize-phoenix is only needed to *run* the collector, not to export to it;
# the service image just needs the OTel SDK and the OTLP exporter.
RUN grep -v '^arize-phoenix' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

COPY llmobs/ ./llmobs/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

EXPOSE 8100
CMD ["uvicorn", "llmobs.app:app", "--host", "0.0.0.0", "--port", "8100"]
