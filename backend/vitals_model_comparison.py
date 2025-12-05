"""
vitals_model_comparison.py
Compare Random Forest, Linear Regression and Gradient Boosting
for multi-output vitals prediction on the same dataset.

Metrics: R2, MAE, RMSE (averaged across all 6 vitals)
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# ===== Paths (same as your other script) =====
CSV_PATH = os.path.join("raw_data", "human_vital_signs_dataset_2024.csv")

def load_features_targets():
    if not os.path.exists(CSV_PATH):
        print("ERROR: dataset not found at", CSV_PATH)
        sys.exit(1)

    print("Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # targets
    target_cols = [
        "Heart Rate",
        "Respiratory Rate",
        "Body Temperature",
        "Oxygen Saturation",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure"
    ]

    # basic cleaning
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    df = df.dropna(subset=target_cols)

    # gender to numeric
    df["Gender"] = df.get("Gender").map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    # BMI
    if "Derived_BMI" in df.columns:
        df["BMI"] = df["Derived_BMI"]
    else:
        df["Height (m)"] = df.get("Height (m)", np.nan)
        df["Weight (kg)"] = df.get("Weight (kg)", np.nan)
        df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2 + 1e-6)

    feature_cols = ["Age", "Gender", "BMI", "Derived_HRV", "Derived_Pulse_Pressure", "Derived_MAP"]
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0.0

    X = df[feature_cols].fillna(0)
    y = df[target_cols].astype(float)

    return X, y, feature_cols, target_cols


def evaluate_model(name, model, X_train, X_test, y_train, y_test, target_cols):
    print("\n==============================")
    print("Training:", name)
    print("==============================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # global metrics
    r2_avg  = r2_score(y_test, y_pred, multioutput="uniform_average")
    mae_avg = mean_absolute_error(y_test, y_pred)       # averaged across outputs
    rmse_avg = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{name} – Overall:")
    print(f"  R2   : {r2_avg:.4f}")
    print(f"  MAE  : {mae_avg:.4f}")
    print(f"  RMSE : {rmse_avg:.4f}")

    # per-vital metrics
    per_target = []
    for i, col in enumerate(target_cols):
        r2_i   = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae_i  = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        per_target.append({
            "Vital": col,
            "R2": r2_i,
            "MAE": mae_i,
            "RMSE": rmse_i
        })
        print(f"  {col:25s} | R2={r2_i:.4f}  MAE={mae_i:.4f}  RMSE={rmse_i:.4f}")

    metrics = {
        "name": name,
        "r2_avg": r2_avg,
        "mae_avg": mae_avg,
        "rmse_avg": rmse_avg,
        "per_target": per_target
    }
    return metrics


def main():
    X, y, feature_cols, target_cols = load_features_targets()
    print("Feature matrix shape:", X.shape)
    print("Targets shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    print("Training on", X_train.shape[0], "rows")

    # ===== Define models =====
    models = [
        (
            "Random Forest (MultiOutput)",
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=15,
                    n_jobs=-1,
                    random_state=42
                )
            )
        ),
        (
            "Linear Regression (MultiOutput)",
            MultiOutputRegressor(
                LinearRegression()
            )
        ),
        (
            "Gradient Boosting (MultiOutput)",
            MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        )
    ]

    summary_rows = []

    for name, model in models:
        m = evaluate_model(name, model, X_train, X_test, y_train, y_test, target_cols)
        summary_rows.append({
            "Model": name,
            "R2_avg": m["r2_avg"],
            "MAE_avg": m["mae_avg"],
            "RMSE_avg": m["rmse_avg"]
        })

    # Nice table for report
    summary_df = pd.DataFrame(summary_rows)
    print("\n\n=== COMPARISON TABLE (AVERAGE OVER ALL VITALS) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Optionally save to CSV for using in dashboard / report
    out_path = os.path.join("models", "model_comparison_metrics.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print("\nSaved comparison metrics to:", out_path)


if __name__ == "__main__":
    main()
