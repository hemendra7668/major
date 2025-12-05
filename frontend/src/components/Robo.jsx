import Heartbeat from "./HeartBeat.jsx";
// import robotImg from "../assets/robot.png"; // Optional if you add image

export default function RobotPanel({ prediction }) {

  const bpm = prediction?.["Heart Rate"] || 72;

  return (
    <div className="card">

      <h2>Patient Vital Sign Prediction and Analysis App</h2>
      <p>Clinical assistant — analyzes recent vitals</p>

      <div style={{display:"flex",gap:"10px",alignItems:"center"}}>
        <div className="status">Ready</div>
        <button className="btn ghost">🔊 Voice: On</button>
      </div>

      <p style={{fontSize:".85rem",color:"#9feefd",marginTop:"10px"}}>
        Robot animates while computing — shows a single prediction for your uploaded window.
      </p>

      <Heartbeat bpm={bpm} />

      <div style={{display:"flex", gap:"10px"}}>
        <button className="btn ghost">Wave</button>
        <button className="btn">Ping</button>
      </div>

    </div>
  );
}
