# Streamlit Dashboard

Reads forecasts, features, SHAP drivers, alerts, and model metrics from
Hopsworks Feature Groups.

## Local run

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Set `HOPSWORKS_API_KEY` in `.env` (project root) or in
`.streamlit/secrets.toml`:

```toml
HOPSWORKS_API_KEY = "your_key_here"
```
