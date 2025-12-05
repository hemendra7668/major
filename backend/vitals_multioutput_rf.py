"""
vitals_multioutput_rf.py
Train a MultiOutput RandomForest model and save to models/vitals_multioutput_rf.pkl
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

CSV_PATH = os.path.join("raw_data", "human_vital_signs_dataset_2024.csv")
OUT_MODEL = os.path.join("models", "vitals_multioutput_rf.pkl")

def main():
    if not os.path.exists(CSV_PATH):
        print("ERROR: dataset not found at", CSV_PATH)
        sys.exit(1)

    print("Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Preprocess
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    target_cols = [
        "Heart Rate",
        "Respiratory Rate",
        "Body Temperature",
        "Oxygen Saturation",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure"
    ]
    df = df.dropna(subset=target_cols)
    df["Gender"] = df.get("Gender").map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    if "Derived_BMI" in df.columns:
        df["BMI"] = df["Derived_BMI"]
    else:
        # Avoid division by zero
        df["Height (m)"] = df.get("Height (m)", np.nan)
        df["Weight (kg)"] = df.get("Weight (kg)", np.nan)
        df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2 + 1e-6)

    features = ["Age", "Gender", "BMI", "Derived_HRV", "Derived_Pulse_Pressure", "Derived_MAP"]
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].fillna(0)
    y = df[target_cols].astype(float)

    print("Feature matrix shape:", X.shape)
    print("Targets shape:", y.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print("Training on", X_train.shape[0], "rows")

    # Train model
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42))
    model.fit(X_train, y_train)
    print("Training complete")

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Save
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(model, OUT_MODEL)
    print("Model saved to:", OUT_MODEL)

    # Quick sample predict
    sample = np.array([[45, 0, 24.5, 5, 40, 90]])
    print("Sample input:", sample.tolist())
    print("Sample prediction:", model.predict(sample).tolist())

if __name__ == "__main__":
    main()
