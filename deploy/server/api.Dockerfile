# FastAPI app behind the web UI. Build from the repo root (docker-compose.yml does this).
FROM python:3.12-slim

WORKDIR /workspace

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# only what the API reads: serving code, model artifacts, KG snapshot
COPY app/common.py app/
COPY app/artifacts/ app/artifacts/
COPY scripts/gen_app_snapshot.py scripts/
COPY data/kg_snapshots/app_snapshot.json data/kg_snapshots/
COPY api/ api/

ENV PYTHONUNBUFFERED=1
USER nobody
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
