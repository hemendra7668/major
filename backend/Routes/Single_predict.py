from flask import Blueprint, request, jsonify
from app import FEATURES, predict_from_rep

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict_single():
    data = request.json or {}

    try:
        row = {f: float(data.get(f, 0.0)) for f in FEATURES}
        resp = predict_from_rep(row)
        return jsonify({"success": True, "predictions": resp})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
