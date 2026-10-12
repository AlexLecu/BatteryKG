# Deploying BatteryKG

The public instance runs at **<https://batterykg.utcluj.ro>** on a server of the
Technical University of Cluj-Napoca (CloudUT). Everything it needs is in
[`deploy/server/`](deploy/server/): one `docker compose` project behind the
server's nginx, with the university reverse proxy in front for HTTPS.

```
browser ──HTTPS──▶ university reverse proxy (TLS) ──HTTP──▶ host nginx :80
                                                             │  Host: batterykg.utcluj.ro
                                                             ▼
                                         batterykg-web  (127.0.0.1:3003)
                                         nginx: Vue UI + same-origin proxy
                                                             │ /api/*
                                                             ▼
                                         batterykg-api  FastAPI (uvicorn)
                                         app/common.py serving code, xgboost artifacts,
                                         knowledge-graph snapshot
```

| Container | Image source | Role |
|---|---|---|
| `batterykg-web` | [`deploy/server/web.Dockerfile`](deploy/server/web.Dockerfile) | Built Vue UI ([`web/`](web/)) + nginx routing `/api/*` |
| `batterykg-api` | [`deploy/server/api.Dockerfile`](deploy/server/api.Dockerfile) | [`api/main.py`](api/main.py): predictions and knowledge-graph answers as JSON |

There is **no database and no secret**. The API answers knowledge-graph questions
from [`data/kg_snapshots/app_snapshot.json`](data/kg_snapshots/app_snapshot.json),
the recorded results of every query the pages can issue, and predictions from the
model artifacts in [`app/artifacts/`](app/artifacts/), through the same
`app/common.py` functions as the Streamlit app. Only `batterykg-web` is published,
and only on `127.0.0.1`.

## Installing on a server

Requirements: Docker with the compose plugin and nginx on the host.

```bash
git clone https://github.com/AlexLecu/BatteryKG.git && cd BatteryKG/deploy/server
docker compose up -d --build        # first build takes a few minutes
```

Then the host nginx site (needs sudo):

```bash
sudo cp nginx-host.conf /etc/nginx/sites-available/batterykg
sudo ln -s /etc/nginx/sites-available/batterykg /etc/nginx/sites-enabled/batterykg
sudo nginx -t && sudo systemctl reload nginx
```

The host block listens on ports 80 and 3000 (the ones the university proxy uses)
and matches only `server_name batterykg.utcluj.ro`, so other sites on the server
are not affected. TLS is terminated by the university proxy; do not add
certificates locally. Set `WEB_PORT` in a `.env` next to the compose file if
3003 is taken (and change `nginx-host.conf` to match).

## Updating

```bash
cd ~/BatteryKG && git pull && cd deploy/server && docker compose up -d --build
```

After rebuilding the knowledge graph, regenerate the snapshot against the live
Neo4j (`python -m scripts.gen_app_snapshot`), commit it, and update the server.
After retraining (`python -m src.models.train_final`), commit `app/artifacts/`
the same way.

## Things to know about the university proxy

* It does **not forward WebSockets**, which Streamlit needs. That is why the hosted
  site is a plain-HTTP web app; the Streamlit app in [`app/`](app/) is for local use.
* While connected to the CloudUT VPN, its DNS does not resolve the public
  subdomains; disconnect the VPN to open the site from the same machine.

## Running locally

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000          # from the repository root
cd web && npm install && npm run dev      # http://localhost:5173, proxies /api to :8000
```

With `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` set (environment or `.env`),
the API queries the live graph instead of the snapshot.
