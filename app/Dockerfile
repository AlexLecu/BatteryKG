# BatteryKG Streamlit app.
# Build from the REPO ROOT (the compose file does this):
#   docker build -f app/Dockerfile -t batterykg-app .
FROM python:3.12-slim

WORKDIR /workspace

# single source of truth for dependencies — no hand-picked subset (drift risk)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code + the data/artifacts the app reads (honesty rule: real sources only)
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY data/processed/ data/processed/
COPY data/claims/*.yaml data/claims/
COPY outputs/*.md outputs/

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
