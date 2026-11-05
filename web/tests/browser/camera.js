// Deterministic non-face video. Edge's fake hardware device can disappear on restart.
export async function useCanvasCamera(page,firstError=null){
  await page.addInitScript(({firstError})=>{
    let first=true;window.cameraCalls=0;
    navigator.mediaDevices.getUserMedia=async()=>{
      window.cameraCalls++;
      if(first&&firstError){first=false;throw new DOMException('Simulated camera error',firstError);}
      first=false;
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      const ctx=canvas.getContext('2d');let frame=0;
      const draw=()=>{for(let y=0;y<8;y++)for(let x=0;x<8;x++){
        ctx.fillStyle=(x+y)%2?'#607080':'#b0c0d0';ctx.fillRect(x*80,y*60,80,60);
      }ctx.fillStyle='#405060';ctx.fillRect((frame++%100)*5,0,20,20);};
      draw();const timer=setInterval(draw,60),stream=canvas.captureStream(15);
      for(const track of stream.getTracks()){
        const stop=track.stop.bind(track);track.stop=()=>{clearInterval(timer);stop();};
      }
      return stream;
    };
  },{firstError});
}
