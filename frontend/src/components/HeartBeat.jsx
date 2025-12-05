import { useEffect, useRef } from "react";

export default function Heartbeat({ bpm }) {

  const canvasRef = useRef();

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    let t = 0;

    function draw() {
      ctx.clearRect(0,0,260,80);
      ctx.beginPath();

      for (let x=0; x<260; x++) {
        let y = 40 + Math.sin((x+t)*bpm/400) * 18;
        ctx.lineTo(x,y);
      }

      ctx.strokeStyle = "#00f0ff";
      ctx.lineWidth = 2;
      ctx.stroke();

      t++;
      requestAnimationFrame(draw);
    }

    draw();
  }, [bpm]);

  return (
    <div style={{marginTop:"10px"}}>
      <canvas ref={canvasRef} width="260" height="80"></canvas>
      <div style={{textAlign:"center",color:"#9feefd"}}>Estimated BPM</div>
      <div style={{textAlign:"center",fontSize:"1.3rem"}}>{bpm} bpm</div>
    </div>
  );
}
