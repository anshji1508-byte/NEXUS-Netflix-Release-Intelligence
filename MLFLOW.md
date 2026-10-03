# Netflix forecasting MLOps runbook

This project now has a repeatable training-to-serving path:

1. `train_netflix_model.py` loads the local catalog, creates monthly time features, evaluates the chronological holdout, and saves a compressed Joblib bundle.
2. `artifacts/netflix_forecaster.joblib` stores the trained estimator, feature contract, history, and evaluation metadata needed for inference.
3. `netflix_api.py` exposes `/health`, `/metadata`, and `/forecast` for service integration.
4. `netflix_forecast.py` remains the Streamlit analyst interface, including visual evaluation and CSV export.

## Train and serve

```bash
python train_netflix_model.py
uvicorn netflix_api:app --reload --port 8000
```

The API is designed for container or CI deployment. `/health` can be used as a readiness probe, while the metrics JSON beside the artifact is a simple local model registry record. In a team environment, the artifact directory can be promoted to object storage and the same endpoint can load a pinned model version.
