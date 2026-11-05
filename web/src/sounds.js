// Offline Web Audio synthesis. Gains are application limits, not acoustic levels.
export const MAX_MASTER_GAIN=.12;
export const DEFAULT_VOLUME=50;
export const FADE_SECONDS=.015;
export const PATTERNS={
  fatigue:{label:'Fatigue · falling pulses',wave:'sine',from:440,to:330,duration:.35,gap:.15,count:2,level:1},
  distraction:{label:'Distraction · rising notes',wave:'triangle',from:660,to:880,duration:.15,gap:0,count:2,level:1},
  availability:{label:'Monitoring unavailable · soft tone',wave:'sine',from:520,to:520,duration:.4,gap:0,count:1,level:.35},
};
export function clampVolume(value){return Number.isFinite(value)?Math.max(0,Math.min(100,value)):DEFAULT_VOLUME;}
export function setMasterVolume(context,master,value){
  const at=context.currentTime;
  master.gain.cancelScheduledValues(at);
  master.gain.setTargetAtTime(MAX_MASTER_GAIN*clampVolume(value)/100,at,.015);
}
export function createMasterGain(context,value=DEFAULT_VOLUME){
  const master=context.createGain();
  master.gain.setValueAtTime(MAX_MASTER_GAIN*clampVolume(value)/100,context.currentTime);
  master.connect(context.destination);return master;
}
export function canPlay({state,enabled,preview=false,pattern,now,last=-Infinity,volume=DEFAULT_VOLUME}){
  return state==='running'&&Object.hasOwn(PATTERNS,pattern)&&Number.isFinite(now)&&
    clampVolume(volume)>0&&(preview||(enabled&&now-last>=3));
}
export function synthesize(context,master,key){
  const pattern=PATTERNS[key];if(!Object.hasOwn(PATTERNS,key))return;
  const started=context.currentTime;
  for(let i=0;i<pattern.count;i++){
    const oscillator=context.createOscillator(),envelope=context.createGain();
    const at=started+i*(pattern.duration+pattern.gap),end=at+pattern.duration;
    oscillator.type=pattern.wave;
    oscillator.frequency.setValueAtTime(pattern.from,at);
    oscillator.frequency.linearRampToValueAtTime(pattern.to,end);
    envelope.gain.setValueAtTime(0,at);
    envelope.gain.linearRampToValueAtTime(pattern.level,at+FADE_SECONDS);
    envelope.gain.setValueAtTime(pattern.level,end-FADE_SECONDS);
    envelope.gain.linearRampToValueAtTime(0,end);
    oscillator.connect(envelope);envelope.connect(master);
    oscillator.onended=()=>{oscillator.disconnect();envelope.disconnect();};
    oscillator.start(at);oscillator.stop(end);
  }
}
