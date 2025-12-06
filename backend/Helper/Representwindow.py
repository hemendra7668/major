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