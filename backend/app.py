from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os
import sys
from werkzeug.utils import secure_filename

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vitals_multioutput_rf.pkl")

# Load model with error handling
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("ERROR: Failed to load model:", MODEL_PATH)
    print(type(e).__name__, e)
    sys.exit(1)

# Features expected by model
FEATURES = ["Age","Gender","BMI","Derived_HRV","Derived_Pulse_Pressure","Derived_MAP"]

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit
ALLOWED_EXT = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def representative_from_window(df_window):
    """
    Given a dataframe with time-window rows, produce a single representative vector
    matching FEATURES. Strategy:
      - For Age, Gender, BMI: take last observed value (assumed static).
      - For Derived_* features: compute EWMA (span = window length) to give more weight to recent rows.
    Returns: dict {feature:value}
    """
    out = {}
    n = max(1, len(df_window))
    span = max(2, n)  # span used for EWMA (bigger span -> smoother)
    for f in FEATURES:
        if f not in df_window.columns:
            # if missing, default to 0.0
            out[f] = 0.0
            continue

        # treat constants: Age, Gender, BMI -> use last observed (most recent row)
        if f in ("Age", "Gender", "BMI"):
            try:
                out[f] = float(df_window[f].dropna().iloc[-1])
            except Exception:
                # fallback to mean or 0
                vals = df_window[f].dropna()
                out[f] = float(vals.iloc[-1]) if len(vals)>0 else 0.0
        else:
            # dynamic features: use EWMA (recent rows weighted heavier)
            try:
                series = pd.to_numeric(df_window[f], errors="coerce").fillna(method="ffill").fillna(0.0)
                ewma_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
                out[f] = float(ewma_val)
            except Exception:
                out[f] = float(df_window[f].dropna().mean()) if df_window[f].dropna().shape[0] > 0 else 0.0
    return out

def predict_from_rep(rep):
    # rep is dict with FEATURES -> single floats
    row_df = pd.DataFrame([rep], columns=FEATURES)
    preds = model.predict(row_df)[0].tolist()
    resp = {
        "Heart Rate": round(preds[0],2),
        "Respiratory Rate": round(preds[1],2),
        "Body Temperature": round(preds[2],2),
        "Oxygen Saturation": round(preds[3],2),
        "Systolic Blood Pressure": round(preds[4],2),
        "Diastolic Blood Pressure": round(preds[5],2)
    }
    return resp

@app.route("/")
def index():
    return send_from_directory("frontend", "landing.html")

@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Accepts JSON with single-row features (Age,Gender,BMI,Derived_HRV,Derived_Pulse_Pressure,Derived_MAP)
    Returns a single prediction (one set of vitals).
    """
    data = request.json or {}
    try:
        row = {f: float(data.get(f, 0.0)) for f in FEATURES}
        resp = predict_from_rep(row)
        return jsonify({"success": True, "predictions": resp})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/predict_window", methods=["POST"])
def predict_window():
    """
    Accepts either:
      - file upload form 'file' (CSV) containing 1..N rows (CSV must include at least the columns used)
      - JSON body with key 'rows' which is a list of row-objects [{...}, {...}, ...]
    Returns: single prediction after aggregating the window.
    """
    # 1) try file upload
    if "file" in request.files:
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        if not allowed_file(f.filename):
            return jsonify({"success": False, "error": "Only CSV files allowed"}), 400
        try:
            # Read CSV into pandas
            df = pd.read_csv(f)
        except Exception as e:
            return jsonify({"success": False, "error": "Failed to read CSV: " + str(e)}), 400
    else:
        # 2) try JSON array in body under key 'rows', or raw array body
        try:
            js = request.get_json(force=True)
        except Exception:
            js = None
        if not js:
            return jsonify({"success": False, "error": "No file or JSON body found. Upload CSV as 'file' or send JSON array under 'rows'"}), 400

        # support both { "rows": [...] } and raw array [...]
        if isinstance(js, dict) and "rows" in js:
            rows = js["rows"]
        elif isinstance(js, list):
            rows = js
        else:
            return jsonify({"success": False, "error": "JSON body must be an array or an object with 'rows' key"}), 400

        try:
            df = pd.DataFrame(rows)
        except Exception as e:
            return jsonify({"success": False, "error": "Failed to parse rows into DataFrame: " + str(e)}), 400

    if df is None or df.shape[0] == 0:
        return jsonify({"success": False, "error": "No rows found in input"}), 400

    # Optionally enforce a window size range (e.g., expect 1..20 rows)
    if df.shape[0] < 1:
        return jsonify({"success": False, "error": "Need at least 1 row"}), 400
    if df.shape[0] > 200:
        # safety limit
        df = df.tail(200)

    # Build representative single vector
    rep = representative_from_window(df)

    # Predict single aggregate vitals
    try:
        resp = predict_from_rep(rep)
        meta = {
            "rows_used": int(df.shape[0]),
            "representative_features": rep
        }
        return jsonify({"success": True, "predictions": resp, "meta": meta})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
