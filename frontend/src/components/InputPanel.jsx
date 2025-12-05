import { useState } from "react";

export default function InputPanel({ onPredict }) {

  const [data, setData] = useState({
    Age:45, Gender:0, BMI:24.5,
    Derived_HRV:5.1,
    Derived_Pulse_Pressure:40,
    Derived_MAP:90
  });

  const handleChange = e => {
    setData({...data, [e.target.name]: e.target.value});
  };

  const predict = () => {
    onPredict({
      "Heart Rate": Math.floor(Math.random()*30)+65,
      "Respiratory Rate": 17,
      "Body Temperature": 36.9,
      "Oxygen Saturation": 98,
      "Systolic Blood Pressure": 116,
      "Diastolic Blood Pressure": 74
    });
  };

  return (
    <div className="card">

      <h1>Patient Vital Prediction</h1>

      <p>Provide either a single input or upload 7–10 recent rows (CSV / JSON array).</p>

      {Object.keys(data).map(key =>
        <div key={key}>
          <label>{key}</label>
          <input name={key} value={data[key]} onChange={handleChange} />
        </div>
      )}

      <button className="btn" onClick={predict}>Predict (single)</button>

      <label>Upload CSV (7–10 rows)</label>
      <input type="file" />

      <label>JSON Array</label>
      <textarea placeholder='[{"Age":45,"BMI":24.5,...}]'></textarea>

      <button className="btn">Predict (window)</button>

    </div>
  );
}
