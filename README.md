# BatteryKG

BatteryKG builds a knowledge graph of commercial battery cells from
deliberately conflicting sources — manufacturer datasheet **claims**, cycling
dataset **measurements**, and independent tests — and reconciles them with
full provenance. On top of the graph it predicts cycle life for unseen cells
via graph-mediated transfer (leave-one-cell-out), and **abstains** when graph
coverage around a query cell is too sparse to trust a prediction. An
LLM-agent pipeline (extractor → validator → human-gated promotion) keeps the
claim side of the graph updatable from new documents without letting a model
write to the graph unreviewed.


## Quickstart

Requires Docker and Python ≥ 3.11.

```bash
# 1. Credentials (no defaults ship — set your own Neo4j password)
cp .env.example .env         # edit NEO4J_PASSWORD

# 2. Start Neo4j (and optionally the app) — http://localhost:7474
docker compose up -d neo4j

# 3. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # requirements.lock.txt has exact pins

# 4. Data + KG load pipeline (reproducible, no manual steps)
python -m src.ingestion.download all   # raw datasets, ~8.6 GB, resumable
python -m src.ingestion.build_all      # tidy Parquet -> data/processed/
python -m src.ingestion.features       # early-cycle features
python -m src.kg.load                  # cells/measurements + SIMILAR_TO edges -> Neo4j
python -m src.kg.coverage              # graph-coverage metrics on the edges
python -m src.kg.claims                # gold-standard claims -> Neo4j
python -m src.kg.discrepancy           # claim-vs-measured verdicts
python -m src.ingestion.lygte          # fetch + parse lygte-info.dk independent tests
python -m src.kg.independent           # third source -> Neo4j (reconciliation panel)

# 5. Demo app — http://localhost:8501
streamlit run app/main.py
```

The trained serving artifacts ship in `app/artifacts/`, so the **Prediction
with Abstention** page works immediately after step 3 — the graph pages light
up once the KG is loaded. Alternatively `docker compose up --build` runs the
full stack (Neo4j + app) in containers. Every page follows an honesty rule:
numbers come from the graph, the artifacts, or a report file — when a source
is unavailable the page says so instead of mocking data.

## Released gold standard

`data/claims/` is a citable contribution of the paper:

- `*.yaml` — hand-extracted manufacturer claims for the A123 APR18650M1A
  (LFP), Panasonic NCR18650B (NCA), and LG 18650HG2 (NMC), with unit, page
  number, and the datasheet's *stated conditions* per claim (an explicitly
  recorded `"unspecified"` when the datasheet states none — that omission is
  itself a finding). Format and property vocabulary: `data/claims/README.md`.

The manufacturer PDFs themselves are copyrighted and are not redistributed;
source URLs and retrieval dates are in `data/README.md`. The extraction-input
text snapshots (`data/claims/extracted_text/`) are likewise not redistributed —
they are regenerated from the publicly available datasheet PDFs via the
ingestion script: download the PDFs to `data/claims/datasheets/` (URLs in
`data/README.md`), then run `python -m src.agents.pdf_text`.

## Repository structure

```
src/ingestion/   dataset loaders (Severson/MIT, Sandia, HUST, lygte parser), features, QC
src/kg/          Neo4j schema, loaders, SIMILAR_TO graph features, discrepancy detection
src/models/      graph-mediated predictor, graph-free baseline, coverage-gated abstention
src/agents/      LLM claim extraction, validator, literature monitor (human-gated)
src/viz/         publication figure scripts
app/             Streamlit demo (4 pages) + trained serving artifacts
tests/           full suite; fixtures stand in for network/copyrighted sources
data/claims/     released gold standard (see above)
data/README.md   provenance for every dataset (URLs, download dates)
```

## Repo map — what produces what

Shipped artifacts and the script that regenerates each:

| shipped file | produced by |
|---|---|
| `app/artifacts/*` (models, meta, neighbor bank) | `python -m src.models.train_final` |
| `outputs/experiment_01_prediction.md` | `python -m src.models.experiment_01` (prediction + abstention study) |
| `outputs/experiment_02_extraction.md` | `python -m src.agents.experiment_02` / `experiment_02b` (LLM claim extraction vs. gold standard; needs `GROQ_API_KEY`) |
| `data/eval/questions.jsonl` | `python -m src.agents.experiment_03` (one spec question per gold claim; shipped so the eval is exactly reproducible) |
| `data/kg_snapshots/severson_edges_publication.json` | frozen export of the Severson `SIMILAR_TO` edge lists from the publication KG; read by `src/viz/make_paper_figures.py` so the figures rebuild without a live database |
| `data/claims/README.md` vocabulary table | `python scripts/gen_claims_vocab.py` (derived from the YAMLs) |

## Tests

```bash
pytest -q
```

Tests requiring a live Neo4j or unbuilt local data skip with an explicit
reason; everything else runs self-contained from shipped fixtures.

## License

MIT — see [LICENSE](LICENSE). Third-party datasets retain their own licenses
(sources in `data/README.md`); `data/hust_discharge_rates.json` is vendored
from BatteryML (MIT).
