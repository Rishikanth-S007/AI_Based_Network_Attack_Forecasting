# AI-Based Network Attack Forecasting

A Streamlit dashboard for forecasting network attack activity from flow and packet-level features. The application combines a trained world model with MITRE ATT&CK stage mapping, response recommendations, and explainability output.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app includes sample scenarios under `data/sample_scenarios.json` and accepts feature CSV uploads through the dashboard. The trained checkpoint is stored at `project_package/backend/ctu13_world_model.pt`.

## Repository layout

- `app.py` - Streamlit application
- `project_package/backend/` - inference, model, ingestion, and training modules
- `assets/` - dashboard styling and network visualization assets
- `data/sample_scenarios.json` - lightweight demo data
- `requirements.txt` - runtime dependencies

Large raw captures and training datasets are intentionally excluded from Git through `.gitignore`. They can be supplied separately when retraining is required.
