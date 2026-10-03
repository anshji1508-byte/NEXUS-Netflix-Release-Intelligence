"""Reusable training and inference code for the Netflix release forecaster."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["month_num", "quarter", "year_index", "month_sin", "month_cos", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "rolling_3", "rolling_6"]

def resolve_data(path=None):
    candidates = [Path(path)] if path else []
    base = Path(__file__).resolve().parent
    candidates += [base / "dataset.csv", base / "Auspify" / "Dataset.csv"]
    for candidate in candidates:
        if candidate and candidate.exists(): return candidate
    raise FileNotFoundError("dataset.csv or Auspify/Dataset.csv was not found")

def load_catalog(path=None):
    source = resolve_data(path)
    df = pd.read_csv(source)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["date_added"]).copy()
    return df.sort_values("date_added").reset_index(drop=True), source

def monthly_series(catalog):
    result = catalog.set_index("date_added").resample("MS").size().rename("releases").reset_index()
    return result.rename(columns={"date_added": "month"})

def make_features(series):
    z = series.copy()
    z["month_num"] = z.month.dt.month
    z["quarter"] = z.month.dt.quarter
    z["year_index"] = z.month.dt.year - z.month.dt.year.min()
    z["month_sin"] = np.sin(2 * np.pi * z.month.dt.month / 12)
    z["month_cos"] = np.cos(2 * np.pi * z.month.dt.month / 12)
    for lag in [1, 2, 3, 6, 12]: z[f"lag_{lag}"] = z.releases.shift(lag)
    z["rolling_3"] = z.releases.shift(1).rolling(3).mean()
    z["rolling_6"] = z.releases.shift(1).rolling(6).mean()
    return z

def train_bundle(data_path=None, holdout_months=12):
    catalog, source = load_catalog(data_path)
    monthly = monthly_series(catalog)
    series = monthly.set_index("month")["releases"].astype(float)
    if len(series) <= holdout_months:
        raise ValueError("The dataset is too short to reserve a holdout window.")

    train_end = len(series) - holdout_months
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    predicted = fitted.forecast(steps=len(test))
    seasonal_naive = series.shift(12).iloc[train_end:]

    test_frame = pd.DataFrame({
        "month": test.index,
        "releases": test.to_numpy(),
        "predicted": predicted.to_numpy(),
        "seasonal_naive": seasonal_naive.to_numpy(),
    })

    metrics = {
        "mae": float(mean_absolute_error(test_frame["releases"], test_frame["predicted"])),
        "rmse": float(mean_squared_error(test_frame["releases"], test_frame["predicted"]) ** 0.5),
        "baseline_mae": float(mean_absolute_error(test_frame["releases"], test_frame["seasonal_naive"])),
        "holdout_months": int(holdout_months),
        "rows": int(len(catalog)),
        "monthly_observations": int(len(monthly)),
        "data_source": str(source),
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "feature_count": len(FEATURES),
    }
    metrics["baseline_lift_pct"] = float((1 - metrics["mae"] / metrics["baseline_mae"]) * 100)

    bundle = {
        "model": fitted,
        "history": monthly,
        "features": FEATURES,
        "metrics": metrics,
        "source": str(source),
        "model_type": "ETS",
    }
    return bundle, test_frame

def future_forecast(bundle, periods=12):
    periods = int(periods)
    start_month = pd.Timestamp(bundle["history"]["month"].max()) + pd.offsets.MonthBegin(1)
    months = pd.date_range(start=start_month, periods=periods, freq="MS")
    model = bundle["model"]

    if hasattr(model, "get_forecast"):
        forecast = model.get_forecast(steps=periods)
        mean = forecast.predicted_mean
        interval = forecast.conf_int(alpha=0.2)
        series = zip(months, mean, interval.iloc[:, 0], interval.iloc[:, 1])
    else:
        mean = model.forecast(steps=periods)
        residuals = np.asarray(getattr(model, "resid", []), dtype=float)
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        lower = np.maximum(0.0, mean.to_numpy() - 1.28 * residual_std)
        upper = mean.to_numpy() + 1.28 * residual_std
        series = zip(months, mean, lower, upper)

    rows = []
    for month, value, lower, upper in series:
        rows.append({
            "month": month.strftime("%Y-%m-%d"),
            "forecast": round(float(max(0.0, value)), 2),
            "low": round(float(max(0.0, lower)), 2),
            "high": round(float(max(0.0, upper)), 2),
        })
    return rows

def save_bundle(bundle, artifact_path):
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    artifact.with_suffix(".metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))
    return artifact

def load_bundle(artifact_path):
    return joblib.load(artifact_path)
