// Geometry uses pixel coordinates, matching the Python EAR/MAR definitions.
export const LEFT = [33, 160, 158, 133, 153, 144];
export const RIGHT = [362, 385, 387, 263, 373, 380];
export const MOUTH = [78, 82, 13, 312, 308, 317, 14, 87];
export const DEFAULTS = Object.freeze({ ear: .21, mar: .55, eyeSeconds: 1.5, mouthSeconds: 1.5, poseSeconds: 2, yaw: 25, pitch: 20 });
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export function ear(points, ids) {
  const [a,b,c,d,e,f] = ids.map(i => points[i]);
  return (distance(b,f) + distance(c,e)) / Math.max(2 * distance(a,d), 1e-8);
}
export function poseFromMatrix(matrix) {
  const m = matrix?.data;
  if (!m || m.length !== 16 || !Array.from(m).every(Number.isFinite)) return null;
  // MediaPipe canonical-face transform is column-major. Normalize columns to remove scale.
  const x = Math.hypot(m[0], m[1], m[2]);
  const y = Math.hypot(m[4], m[5], m[6]);
  const z = Math.hypot(m[8], m[9], m[10]);
  if (Math.min(x,y,z) < 1e-8) return null;
  const degrees = r => r * 180 / Math.PI;
  return { yaw: degrees(Math.asin(Math.max(-1, Math.min(1, -m[2]/x)))),
    pitch: degrees(Math.atan2(m[6]/y, m[10]/z)), roll: degrees(Math.atan2(m[1]/x, m[0]/x)),
    axes: [[m[0]/x, -m[1]/x], [m[4]/y, -m[5]/y], [m[8]/z, -m[9]/z]] };
}
export function measure(landmarks, matrix, width, height) {
  if (!landmarks || landmarks.length < 468) return null;
  const points = landmarks.map(p => ({ x: p.x*width, y:p.y*height, z:p.z*width }));
  const pose = poseFromMatrix(matrix);
  if (!pose || !points.every(p => [p.x,p.y,p.z].every(Number.isFinite))) return null;
  const left = ear(points, LEFT), right = ear(points, RIGHT);
  const mar = distance(points[13], points[14]) / Math.max(distance(points[78], points[308]), 1e-8);
  const xs = points.map(p => p.x);
  const inside = points.filter(p => p.x>=0 && p.x<width && p.y>=0 && p.y<height).length/points.length;
  const faceWidth = (Math.max(...xs)-Math.min(...xs))/width;
  return { left, right, mar, ...pose, points, inside, faceWidth, framed: inside>.95 && faceWidth>.15 };
}
export function validateSettings(config) {
  for (const [key, value] of Object.entries(config)) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${key}: enter a positive finite number.`);
  }
  if (config.ear>1 || config.mar>2 || config.yaw>90 || config.pitch>90) throw new Error('Use EAR ≤ 1, MAR ≤ 2 and angles ≤ 90°.');
  return config;
}
export class Monitor {
  constructor(config = DEFAULTS) { this.config={...config}; this.starts={}; this.last=null; this.yawns=0; this.yawnActive=false; }
  reset() { this.starts={}; this.last=null; this.yawnActive=false; }
  update(now, values) {
    if (this.last!==null && (now-this.last>.75 || now<this.last)) this.reset();
    this.last=now;
    if (!values || !['left','right','mar','yaw','pitch'].every(k=>Number.isFinite(values[k]))) {
      this.reset(); this.last=now;
      return { state:'UNAVAILABLE', alerts:[], durations:{}, candidates:{} };
    }
    const c=this.config;
    const candidates={ 'Eyes closed':Math.max(values.left,values.right)<c.ear,
      'Mouth open / yawn proxy':values.mar>c.mar, 'Head away':Math.abs(values.yaw)>c.yaw || Math.abs(values.pitch)>c.pitch };
    const limits=[c.eyeSeconds,c.mouthSeconds,c.poseSeconds], durations={}, alerts=[];
    Object.entries(candidates).forEach(([name, active],i)=>{
      if(active) { this.starts[name]??=now; durations[name]=Math.max(0,now-this.starts[name]); if(durations[name]>=limits[i]) alerts.push(name); }
      else { delete this.starts[name]; durations[name]=0; }
    });
    const yawn=alerts.includes('Mouth open / yawn proxy');
    if(yawn && !this.yawnActive) this.yawns++;
    this.yawnActive=yawn;
    return { state:alerts.length?'ALERT':'MONITORING', alerts, durations, candidates };
  }
}
const median = values => { const a=[...values].sort((a,b)=>a-b), n=a.length; return n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2; };
export class Calibration {
  constructor() { this.started=null; this.samples=[]; this.baseline={yaw:0,pitch:0}; this.calibrated=false; }
  begin(now) { this.started=now; this.samples=[]; }
  cancel() { this.started=null; this.samples=[]; }
  add(now, values, config) {
    if(this.started===null) return null;
    if(values) this.samples.push(values);
    if(now-this.started<5) return { done:false, message:`Face forward, eyes open, mouth closed. ${Math.ceil(5-now+this.started)}s · ${this.samples.length} valid frames` };
    this.started=null;
    if(this.samples.length<30) return {done:true,success:false,message:'Calibration needs 30 healthy frames. Improve lighting and try again.'};
    config.ear=Math.max(.08,Math.min(.35,median(this.samples.map(v=>(v.left+v.right)/2))*.72));
    config.mar=Math.max(.35,Math.min(.9,median(this.samples.map(v=>v.mar))+.25));
    this.baseline={yaw:median(this.samples.map(v=>v.yaw)),pitch:median(this.samples.map(v=>v.pitch))};
    this.calibrated=true;
    return {done:true,success:true,message:'Calibrated for this session. Thresholds updated below.'};
  }
  correct(v) { return {...v,yaw:v.yaw-this.baseline.yaw,pitch:v.pitch-this.baseline.pitch}; }
}
