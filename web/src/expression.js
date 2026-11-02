import * as ort from 'onnxruntime-web/wasm';
export const LABELS=['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt'];
ort.env.wasm.wasmPaths='/ort/';
ort.env.wasm.numThreads=1;

export function decode(scores,timestamp) {
  if(scores.length!==8 || !Array.from(scores).every(Number.isFinite)) throw new Error('Invalid FER+ output');
  const max=Math.max(...scores), weights=Array.from(scores,x=>Math.exp(x-max)), total=weights.reduce((a,b)=>a+b,0);
  const index=weights.indexOf(Math.max(...weights));
  return {label:LABELS[index],score:weights[index]/total,timestamp};
}
export class ExpressionCNN {
  static async create() {
    const session=await ort.InferenceSession.create('/models/emotion-ferplus-8.onnx',{executionProviders:['wasm']});
    return new ExpressionCNN(session);
  }
  constructor(session) {
    this.session=session;this.canvas=document.createElement('canvas');this.canvas.width=this.canvas.height=64;
    this.ctx=this.canvas.getContext('2d',{willReadFrequently:true});
  }
  async infer(video,points,timestamp) {
    const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
    const x=Math.max(0,Math.min(...xs)), y=Math.max(0,Math.min(...ys));
    const w=Math.min(video.videoWidth,Math.max(...xs))-x, h=Math.min(video.videoHeight,Math.max(...ys))-y;
    if(w<=0||h<=0) return null;
    this.ctx.drawImage(video,x,y,w,h,0,0,64,64);
    const pixels=this.ctx.getImageData(0,0,64,64).data, input=new Float32Array(4096);
    for(let i=0;i<input.length;i++) input[i]=.299*pixels[i*4]+.587*pixels[i*4+1]+.114*pixels[i*4+2];
    const result=await this.session.run({[this.session.inputNames[0]]:new ort.Tensor('float32',input,[1,1,64,64])});
    return decode(result[this.session.outputNames[0]].data,timestamp);
  }
  async close() {await this.session.release();}
}
