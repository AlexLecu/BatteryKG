# BatteryKG demo app (Streamlit)

Four pages: **Cell Explorer** (claims + measured distribution + discrepancy
verdict), **Prediction with Abstention** (coverage-gated cycle-life
prediction), **Graph Neighborhood** (interactive SIMILAR_TO network),
**Pipeline Status** (KG counts + extraction metrics).

**Honesty rule:** every displayed number comes from the knowledge graph, a
model artifact in `app/artifacts/`, or a report file in `outputs/`. When a
source is unavailable the app says so — nothing is mocked.

## Configuration

All config is environment variables with `.env` fallback (repo root); no
hardcoded hosts: `NEO4J_URI` (default `bolt://localhost:7687`), `NEO4J_USER`,
`NEO4J_PASSWORD`.

## Run locally

```bash
# one-time: data + KG + artifacts
docker compose up -d neo4j
python -m src.ingestion.build_all && python -m src.ingestion.features
python -m src.kg.load && python -m src.kg.coverage
python -m src.kg.claims && python -m src.kg.discrepancy
python -m src.models.train_final          # -> app/artifacts/

streamlit run app/main.py                 # http://localhost:8501
```

## Run the full stack with Docker

```bash
# .env must contain NEO4J_USER / NEO4J_PASSWORD (see .env.example)
docker compose up --build                 # Neo4j (7474/7687) + app (8501)
```

The app container talks to Neo4j at `bolt://neo4j:7687` (compose network);
local runs default to `bolt://localhost:7687`. Note: the image copies
`app/artifacts/` and `data/processed/` at build time — retrain/rebuild to
refresh them, and load the KG from the host (`python -m src.kg.load ...`)
since the graph volume persists in `./neo4j/data`.

## Deployment paths

**1. Streamlit Community Cloud + Neo4j Aura Free**
- Create a free Aura instance; load the KG against it by setting
  `NEO4J_URI=neo4j+s://<id>.databases.neo4j.io`, `NEO4J_USER`,
  `NEO4J_PASSWORD` and running the `src.kg.*` loaders once from your machine.
- Push the repo to GitHub (artifacts in `app/artifacts/` are tracked; the
  processed parquet used by the app is small).
- On share.streamlit.io: new app -> `app/main.py`; set the three `NEO4J_*`
  secrets in the app settings (Streamlit injects them as env vars).

**2. Docker Compose on a VPS**
- Copy the repo (or a release bundle) to the VPS, create `.env` with real
  credentials, `docker compose up -d --build`.
- Put a reverse proxy (Caddy/nginx) in front of :8501 for TLS; keep 7474/7687
  firewalled to localhost unless you need remote DB access.

## Tests

```bash
pytest tests/test_app.py -q     # headless AppTest smoke tests per page,
                                # artifact loading, env-var resolution
```
