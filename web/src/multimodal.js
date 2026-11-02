export function gaze(points, eyesOpen=true) {
  if (!eyesOpen || !points || points.length<478) return null;
  const offsets=[];
  for (const [a,b,i] of [[33,133,468],[362,263,473]]) {
    let start=points[a], end=points[b];
    if(end.x<start.x) [start,end]=[end,start];
    const dx=end.x-start.x, dy=end.y-start.y, w=Math.hypot(dx,dy), iris=points[i];
    if(w<4) return null;
    const ratio=((iris.x-start.x)*dx+(iris.y-start.y)*dy)/(w*w);
    const vertical=Math.abs(dx*(iris.y-start.y)-dy*(iris.x-start.x))/(w*w);
    if(!Number.isFinite(ratio) || ratio<.05 || ratio>.95 || vertical>.2) return null;
    offsets.push(ratio-.5);
  }
  return Math.abs(offsets[0]-offsets[1])>.18?null:(offsets[0]+offsets[1])/2;
}
export function visibility(data, light, contrast) {
  if(light<20 || light>245 || contrast<3) return 'CAMERA VIEW UNAVAILABLE';
  if(!data) return 'FACE NOT DETECTED';
  if(!data.framed || light<=35 || light>=235) return 'POSSIBLE PARTIAL OBSTRUCTION / POOR QUALITY';
  return 'MONITORING';
}
export function fresh(cue, now) {
  return cue && Number.isFinite(cue.timestamp) && now-cue.timestamp>=0 && now-cue.timestamp<=1.5 &&
    Number.isFinite(cue.score) && cue.score>=.6 && cue.score<=1 ? cue:null;
}
export class Fusion {
  constructor() {this.reset();}
  reset() {this.last=null;this.away=null;this.lost=null;this.exp=null;this.expStart=null;this.expStamp=null;}
  update(now,state,health,gazeValue,expression,paused=false) {
    if(this.last!==null && (now<this.last || now-this.last>.75)) this.reset();
    this.last=now;
    if(paused) {this.reset();return {alerts:[],sound:null,summary:'Calibrating · alerts paused'};}
    if(health!=='MONITORING') {
      this.away=null;this.exp=null;this.expStart=null;this.expStamp=null;this.lost??=now;
      return {alerts:[],sound:now-this.lost>=2?'availability':null,summary:health};
    }
    this.lost=null;
    const alerts=[...state.alerts];
    if(Number.isFinite(gazeValue) && Math.abs(gazeValue)>.12) {
      this.away??=now;if(now-this.away>=2) alerts.push('Eye gaze away');
    } else this.away=null;
    const exp=fresh(expression,now);
    if(exp?.label!==this.exp || !exp) {this.exp=exp?.label;this.expStart=exp?.timestamp;}
    this.expStamp=exp?.timestamp;
    const stable=exp && this.expStamp-this.expStart>=1;
    const fatigue=alerts.some(x=>x==='Eyes closed'||x==='Mouth open / yawn proxy');
    const distraction=alerts.some(x=>x==='Head away'||x==='Eye gaze away');
    return {alerts,sound:fatigue?'fatigue':distraction?'distraction':null,
      summary:(fatigue?'Sustained fatigue cues':distraction?'Sustained distraction cues':'No sustained observable alerts')+
        (stable?` · Expression context: ${exp.label} (uncalibrated)`:'')};
  }
}
export const PATTERNS={distraction:[660,880],fatigue:[1040,780,1040],availability:[440,330]};
