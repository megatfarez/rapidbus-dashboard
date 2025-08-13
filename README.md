# Rapid Bus Bunching (Leaflet, no Google)

Streamlit dashboard for detecting Rapid Bus bunching.

## Run locally
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a **GitHub repo** (public or private).
2. In Streamlit Cloud: *New app* → choose the repo/branch → `app.py` → Deploy.

### Optional: Along-path distances
If you have **GTFS static** (trips.txt + shapes.txt), place a zip at `data/gtfs_static.zip` or set env var `GTFS_STATIC_ZIP` to its path. The app will auto-use it.
