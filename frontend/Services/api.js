const API_BASE = "http://127.0.0.1:5000";

export async function predictSingle(data) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
  return res.json();
}

export async function predictWindow(formData) {
  const res = await fetch(`${API_BASE}/predict_window`, {
    method: "POST",
    body: formData
  });
  return res.json();
}
