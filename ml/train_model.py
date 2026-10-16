import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Paths
DATA_PATH = "room_requests.csv"
MODEL_PATH = "models/demand_model.joblib"


def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Basic validation
    required = ["request_date", "start_hour", "status"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. Clean data
    df["request_date"] = pd.to_datetime(
        df["request_date"], errors="coerce"
    )
    df["start_hour"] = pd.to_numeric(
        df["start_hour"], errors="coerce"
    )

    df = df.dropna(subset=["request_date", "start_hour"])
    df = df[df["start_hour"].between(0, 23)]

    # Count requests per date and hour.
    # All recorded requests are included, regardless of status.
    hourly = (
        df.groupby([
            df["request_date"].dt.normalize(),
            "start_hour"
        ])
        .size()
        .reset_index(name="request_count")
    )

    hourly.columns = ["date", "hour", "request_count"]

    # 4. Feature engineering
    hourly["weekday"] = hourly["date"].dt.dayofweek
    hourly["month"] = hourly["date"].dt.month
    hourly["day_of_month"] = hourly["date"].dt.day

    # Cyclical encoding preserves the relationship between
    # the beginning and end of the daily and weekly cycles.
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)

    hourly["weekday_sin"] = np.sin(
        2 * np.pi * hourly["weekday"] / 7
    )
    hourly["weekday_cos"] = np.cos(
        2 * np.pi * hourly["weekday"] / 7
    )

    features = [
        "hour",
        "weekday",
        "month",
        "day_of_month",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]

    X = hourly[features]
    y = hourly["request_count"]

    if len(hourly) < 10:
        raise ValueError("Not enough hourly observations to train.")

    # 5. Chronological train/test split
    split = int(len(hourly) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 6. Baseline: training-set mean
    baseline_predictions = np.full(
        len(y_test), y_train.mean()
    )

    # 7. Train the model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 8. Evaluate
    predictions = model.predict(X_test)

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )
    model_mae = mean_absolute_error(y_test, predictions)

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )
    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print("\n========== CSPACE DEMANDCAST ==========")
    print("Hourly observations:", len(hourly))
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    print("\nBaseline MAE:", round(baseline_mae, 4))
    print("Random Forest MAE:", round(model_mae, 4))

    print("\nBaseline RMSE:", round(baseline_rmse, 4))
    print("Random Forest RMSE:", round(model_rmse, 4))

    print("\nFeature importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(importance.to_string())

    # 9. Save trained model and metadata
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "trained_through": str(
                hourly["date"].max().date()
            ),
            "target": "request_count",
        },
        MODEL_PATH,
    )

    print("\nModel saved to:", MODEL_PATH)
    print("=======================================\n")


if __name__ == "__main__":
    main()