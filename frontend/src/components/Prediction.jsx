export default function Prediction({ data }) {

  return (
    <div className="card" style={{maxWidth:"1100px",marginTop:"15px"}}>

      <h2>Prediction Result</h2>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"10px"}}>
        {Object.entries(data).map(([k,v]) => (
          <div key={k} style={{
            background:"rgba(0,255,255,.03)",
            padding:"10px",
            borderRadius:"10px"
          }}>
            <strong>{k}</strong>
            <div style={{fontSize:"1.3rem"}}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
