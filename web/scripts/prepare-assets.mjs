// Serve all runtime assets from the same deployment. No external CDN at runtime.
import { mkdir, copyFile, cp, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const web = fileURLToPath(new URL('../', import.meta.url));
const modelDir=path.join(web,'public/models');
await mkdir(modelDir,{recursive:true});
const model=path.join(modelDir,'face_landmarker.task');
const sha='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff';
const hash=buffer=>createHash('sha256').update(buffer).digest('hex');
if(!existsSync(model) || hash(await readFile(model))!==sha) {
  const candidates=[path.join(web,'../models/face_landmarker.task'),path.join(web,'../research_upgrade/MBU PROJECT/models/face_landmarker.task')];
  const local=candidates.find(p=>existsSync(p));
  if(local && hash(await readFile(local))===sha) await copyFile(local,model);
  else {
    console.log('Downloading the official version 1 face landmark model…');
    const response=await fetch('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',{signal:AbortSignal.timeout(120000)});
    if(!response.ok) throw new Error(`Face model download failed: ${response.status}`);
    const bytes=Buffer.from(await response.arrayBuffer());
    if(hash(bytes)!==sha) throw new Error('Face model checksum mismatch; refusing to build.');
    await writeFile(model,bytes);
  }
}
const wasmCandidates=[path.join(web,'node_modules/@mediapipe/tasks-vision/wasm'),path.join(web,'../node_modules/@mediapipe/tasks-vision/wasm')];
const wasm=wasmCandidates.find(p=>existsSync(p));
if(!wasm) throw new Error('MediaPipe WASM assets missing. Run npm install at the repository root.');
await cp(wasm,path.join(web,'public/wasm'),{recursive:true});
console.log('Verified face model and copied matching MediaPipe WASM assets.');

const ortDir=path.join(web,'../node_modules/onnxruntime-web/dist');
await mkdir(path.join(web,'public/ort'),{recursive:true});
for(const file of ['ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.mjs']) {
  await copyFile(path.join(ortDir,file),path.join(web,'public/ort',file));
}
const expression=path.join(modelDir,'emotion-ferplus-8.onnx');
const expressionSha='a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c';
if(!existsSync(expression) || hash(await readFile(expression))!==expressionSha) {
  const local=path.join(web,'../models/emotion-ferplus-8.onnx');
  let bytes;
  if(existsSync(local)) bytes=await readFile(local);
  else {
    const response=await fetch('https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx',{signal:AbortSignal.timeout(120000)});
    if(!response.ok) throw new Error('FER+ download failed: '+response.status);
    bytes=Buffer.from(await response.arrayBuffer());
  }
  if(hash(bytes)!==expressionSha) throw new Error('FER+ checksum mismatch');
  await writeFile(expression,bytes);
}
console.log('Verified FER+ CNN and copied local ONNX Runtime assets.');
