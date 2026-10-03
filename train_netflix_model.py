"""Train, evaluate and version the Netflix forecaster artifact."""
import argparse
import json
from pathlib import Path
from netflix_ml import train_bundle, save_bundle, future_forecast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--artifact", default="artifacts/netflix_forecaster.joblib")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    bundle, holdout = train_bundle(args.data)
    artifact = save_bundle(bundle, args.artifact)
    output = Path(args.artifact).with_name("future_forecast.csv")
    import pandas as pd
    pd.DataFrame(future_forecast(bundle, args.horizon)).to_csv(output, index=False)
    print(json.dumps({"artifact": str(artifact), "forecast_csv": str(output), **bundle["metrics"]}, indent=2))

if __name__ == "__main__":
    main()
