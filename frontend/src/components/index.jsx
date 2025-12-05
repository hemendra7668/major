import { useState } from "react";
import InputPanel from "./InputPanel.jsx";
import Prediction from "./Prediction.jsx";
import RobotPanel from "./Robo.jsx";
import "./index.css";
export default function App() {

  const [prediction, setPrediction] = useState(null);

  return (
    <div className="page">

      <div className="panel">

        <RobotPanel prediction={prediction} />

        <InputPanel onPredict={setPrediction} />

      </div>

      {prediction && <Prediction data={prediction} />}

    </div>
  );
}
