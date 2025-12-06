
from turtle import pd
from flask import app, jsonify

from flask import Blueprint, request, jsonify
from app import FEATURES, allowed_file, predict_from_rep
from backend.Helper.Representwindow import representative_from_window
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