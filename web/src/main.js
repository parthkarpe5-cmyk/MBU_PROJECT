import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { DEFAULTS, LEFT, RIGHT, MOUTH, Monitor, Calibration, measure, validateSettings } from './core.js';
import './style.css';
import {gaze,visibility,fresh,Fusion,PATTERNS} from './multimodal.js';
import {ExpressionCNN} from './expression.js';

const fields = [['ear','EAR below',.01],['mar','MAR above',.01],['eyeSeconds','Eye closure · seconds',.1],['mouthSeconds','Mouth opening · seconds',.1],['poseSeconds','Head away · seconds',.1],['yaw','Yaw limit · degrees',1],['pitch','Pitch limit · degrees',1]];
const switches = [['mesh','Face mesh',false],['landmarks','Eye & mouth points',true],['iris','Iris',true],['lines','Measurement lines',true],['axes','Pose axes',true],['ids','Point IDs',false],['mirror','Mirror preview',true],['sound','Sound',false],['debug','Debug',false],['expression','Emotion CNN',true]];
document.querySelector('#app').innerHTML=`
<main class="shell">
  <header class="topbar"><div class="brand"><div class="brandmark" aria-hidden="true">DG</div><div><p class="eyebrow">Intelligent Driver Monitoring</p><h1>DriveGuard AI</h1></div></div><div class="privacy"><span class="dot"></span>On-device processing · No recording<button id="endSession" disabled>Stop session</button></div></header>
  <div class="intro"><div><h2>Every cue. One clear view.</h2><p>Live facial geometry, temporal alerts, and transparent measurements.</p></div><span class="tag">BROWSER RESEARCH PROTOTYPE</span></div>
  <div id="message" class="message" role="status" hidden></div>
  <nav class="viewbar" aria-label="Workspace view"><div class="segmented"><button data-view="split" aria-pressed="true">Split view</button><button data-view="camera" aria-pressed="false">Camera only</button><button data-view="dashboard" aria-pressed="false">Dashboard only</button></div><label class="splitcontrol">Camera share <input id="splitSize" aria-label="Camera panel size" type="range" min="35" max="70" value="60"></label><button id="fullscreen">Fullscreen</button></nav><div class="layout" data-view="split">
    <section class="panel" aria-label="Webcam and overlays"><div class="panelhead"><h2>Camera workspace</h2><span id="cameraState">CAMERA OFF</span></div>
      <div class="viewport mirrored" id="viewport"><video id="video" autoplay muted playsinline aria-label="Local webcam preview"></video><canvas id="overlay" aria-label="Facial landmark overlays"></canvas><div class="placeholder" id="placeholder"><div class="lens" aria-hidden="true">◎</div><strong>Your camera stays on your device.</strong><p>Start a session to see facial landmarks and live measurements. Camera access begins only with your permission.</p></div><div class="camera-badge" id="liveBadge" hidden>● LIVE · LOCAL</div></div>
      <div class="legend">Cyan: EAR / MAR lines · Yellow: eye & mouth points · Pink: iris<br>Pose axes: X red, Y green, Z blue. Pose refers to camera coordinates.</div>
      <div class="controls"><button class="primary" id="start">Start camera</button><button id="calibrate" disabled>Calibrate · 5 seconds</button><button id="stop" class="danger" disabled>Stop camera</button></div>
      <div class="selectrow"><label for="camera">Camera</label><select id="camera"><option value="">Default camera</option></select></div>
      <div class="switches">${switches.map(([id,label,on])=>`<label><input id="${id}" type="checkbox" ${on?'checked':''}>${label}</label>`).join('')}</div>
    </section>
    <aside class="right" aria-label="Monitoring dashboard">
      <section class="panel"><div class="statusblock"><div class="statusline"><p class="eyebrow">Monitoring status</p><span id="status" class="statuspill">OFFLINE</span></div><h3 id="headline">Ready when you are</h3><p id="health">Start the camera, then calibrate while facing forward.</p></div>
      <div class="metricgrid"><div class="metric"><label>Eye aspect ratio · L / R</label><strong id="earValue">— / —</strong><small id="eyeState">Waiting for observation</small></div><div class="metric"><label>Mouth aspect ratio</label><strong id="marValue">—</strong><small id="mouthState">Yawn proxy only</small></div><div class="metric"><label>Head yaw / pitch</label><strong id="poseValue">— / —</strong><small id="poseState">Approximate orientation</small></div><div class="metric"><label>Eye gaze · calibrated proxy</label><strong id="gazeValue">—</strong><small>Independent of head orientation</small></div><div class="metric"><label>Facial expression · CNN</label><strong id="expressionValue">—</strong><small id="expressionScore">Awaiting model</small></div><div class="metric"><label>Processing rate</label><strong id="fpsValue">—</strong><small id="countValue">0 mouth-open events</small></div></div><div class="fusion" id="fusion">MULTIMODAL FUSION · Waiting for observations</div><div class="alertlist" id="alerts">No active session. No alert assessment.</div></section>
      <section class="panel"><div class="panelhead"><h2>Session calibration</h2><span id="calibrationTag">NOT CALIBRATED</span></div><div class="explanation" id="calibrationInfo">Face forward with eyes naturally open and mouth closed for five seconds. Calibration adjusts eye/mouth thresholds and neutral head pose.</div></section>
      <section class="panel"><div class="panelhead"><h2>Models & visibility</h2><span>TRANSPARENT BY DESIGN</span></div><div class="explanation"><strong id="modelState">FER+ expression CNN · waiting for session.</strong><br>Expressions add uncertain context; they never trigger alerts alone. Learned driver behavior requires a trained Python research bundle; unavailable here.<br>Partial obstruction is reported as a possibility when framing or image quality is unreliable, not as an identified object.</div></section>
    </aside>
  </div>
  <details class="panel settings"><summary>Alert sounds & previews</summary><div class="soundgrid">${['distraction','fatigue','availability'].map(event=>`<div class="soundrow"><label for="sound-${event}">${event==='availability'?'Monitoring unavailable':event==='fatigue'?'Fatigue cues':'Distraction cues'}</label><select id="sound-${event}">${Object.keys(PATTERNS).map(name=>`<option ${name===event?'selected':''}>${name}</option>`).join('')}<option>Mute</option></select><button data-preview="${event}">Preview</button></div>`).join('')}<p>Enable Sound for alerts. Availability sounds after 2 seconds. Events repeat at most every 3 seconds.</p></div></details>
  <details class="panel settings"><summary>Thresholds & timing</summary><form id="settings" class="settingsinner">${fields.map(([key,label,step])=>`<div class="field"><label for="setting-${key}">${label}</label><input id="setting-${key}" name="${key}" type="number" min="${step}" step="${step}" value="${DEFAULTS[key]}" required></div>`).join('')}<div class="settingsfooter"><button type="submit">Apply settings</button><p>Settings stay in memory. Calibration updates these values.</p></div></form></details>
  <details class="panel settings" id="debugPanel" hidden><summary>Why did the monitor report this?</summary><pre id="debugText" class="debug">No observations.</pre></details>
  <div class="notice">Stationary testing only. Mouth opening does not prove a yawn, gaze is an approximate calibrated iris proxy, and these cues do not diagnose fatigue. This prototype is not validated for driving or safety decisions.</div>
  <footer><span>Frames are processed in your browser. No images, video, or measurements are uploaded or stored by this app.</span><span>MBU PROJECT · DriveGuard AI · Capstone edition</span></footer>
</main>`;

const $=id=>document.getElementById(id);
const video=$('video'), canvas=$('overlay'), ctx=canvas.getContext('2d');
const lightCanvas=document.createElement('canvas'); lightCanvas.width=16; lightCanvas.height=16;
const lightCtx=lightCanvas.getContext('2d',{willReadFrequently:true});
let monitor=new Monitor(), calibration=new Calibration(), fusion=new Fusion();
let cnn=null,cnnLoading=false,cnnBusy=false,cnnTask=null,expressionCue=null,lastExpression=-Infinity,observationEpoch=0;
let gazeBaseline=null,gazeSamples=[],currentGaze=null,currentHealth='MONITORING';
const soundTimes={};
let stream=null, detector=null, running=false, loading=false, generation=0, raf=0;
let previousVideo=-1, lastFrameAt=0, lastProcessAt=0, previousFpsAt=0, lastBeep=-Infinity, fps=0, audio=null, stalled=false;
let runtimeLabel='';
function notify(text,error=false) { $('message').textContent=text; $('message').hidden=!text; $('message').classList.toggle('error',error); }
function status(kind,headline,detail) { $('status').textContent=kind; $('status').className='statuspill '+(kind==='MONITORING'?'good':kind==='ALERT'?'alert':kind==='OFFLINE'?'':'warn'); $('headline').textContent=headline; $('health').textContent=detail; }
function clearMetrics() { for(const id of ['earValue','marValue','poseValue','fpsValue']) $(id).textContent='—'; $('eyeState').textContent='No current measurement'; $('mouthState').textContent='No current measurement'; $('poseState').textContent='No current measurement'; }
async function cameras() {
  if(!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const selected=$('camera').value;
    const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    $('camera').replaceChildren(new Option('Default camera',''),...devices.map((d,i)=>new Option(d.label||`Camera ${i+1}`,d.deviceId)));
    if(devices.some(d=>d.deviceId===selected)) $('camera').value=selected;
  } catch { /* Enumeration is optional; default camera can still work. */ }
}
function unlockAudio() {
  if(!$('sound').checked) return;
  try { audio??=new (window.AudioContext||window.webkitAudioContext)(); void audio.resume().catch(()=>{}); } catch { notify('Audio is unavailable in this browser. Visual alerts still work.'); }
}
function beep(now,event,preview=false) {
  const pattern=PATTERNS[$('sound-'+event).value];
  if(!pattern||!audio||audio.state!=='running'||(!preview&&(!$('sound').checked||now-(soundTimes[event]??-Infinity)<3)))return;
  pattern.forEach((frequency,i)=>{const o=audio.createOscillator(),g=audio.createGain(),at=audio.currentTime+i*.23;o.frequency.value=frequency;g.gain.setValueAtTime(0,at);g.gain.linearRampToValueAtTime(.08,at+.01);g.gain.linearRampToValueAtTime(0,at+.15);o.connect(g);g.connect(audio.destination);o.start(at);o.stop(at+.16);});soundTimes[event]=now;
}
async function loadExpression() {
  if(cnn||cnnLoading||!$('expression').checked)return;
  cnnLoading=true;const token=generation;$('modelState').textContent='Loading FER+ expression CNN…';
  try{const model=await ExpressionCNN.create();if(token!==generation||!running){await model.close();return;}cnn=model;$('modelState').textContent='FER+ CNN ready · eight facial expression classes';}
  catch(error){if(token===generation)$('modelState').textContent='Expression unavailable: '+error.message;}
  finally{cnnLoading=false;if(token!==generation&&running)void loadExpression();}
}
function requestExpression(now,data) {
  if(!cnn||cnnBusy||!$('expression').checked||now-lastExpression<.35)return;
  const token=generation,epoch=observationEpoch,model=cnn;cnnBusy=true;lastExpression=now;
  cnnTask=model.infer(video,data.points,now).then(cue=>{if(token===generation&&epoch===observationEpoch&&running&&$('expression').checked)expressionCue=cue;})
  .catch(error=>{if(token===generation){expressionCue=null;$('modelState').textContent='Expression unavailable: '+error.message;}}).finally(()=>{cnnBusy=false;});
}
function friendlyError(error) {
  const messages={NotAllowedError:'Camera permission was denied. Allow camera access for this site in your browser, then try again.',NotFoundError:'No webcam was found. Connect a camera and try again.',NotReadableError:'The webcam is busy or unavailable. Close other camera apps and retry.',OverconstrainedError:'That camera is unavailable. Select the default camera and retry.'};
  return messages[error.name] || `${error.message || error}. If model loading failed, check the deployment build and reload.`;
}
function stop(reason='Session stopped. Camera released.') {
  ++generation; running=false; loading=false; cancelAnimationFrame(raf);
  if(stream) for(const track of stream.getTracks()) track.stop();
  stream=null; video.pause(); video.srcObject=null;
  detector?.close(); detector=null;
  monitor.reset();calibration.cancel();fusion.reset();expressionCue=null;observationEpoch++;
  const old=cnn;cnn=null;if(old)void Promise.resolve(cnnTask).finally(()=>old.close()).catch(()=>{});
  $('gazeValue').textContent='—';$('expressionValue').textContent='—';$('expressionScore').textContent='No current prediction';$('fusion').textContent='No active session';
  ctx.clearRect(0,0,canvas.width,canvas.height);
  $('placeholder').hidden=false; $('liveBadge').hidden=true;
  $('start').disabled=false; $('start').textContent='Start camera'; $('stop').disabled=true; $('endSession').disabled=true; $('calibrate').disabled=true; $('camera').disabled=false;
  $('cameraState').textContent='CAMERA OFF'; $('alerts').textContent='No active session. No alert assessment.';
  $('calibrationTag').textContent='NOT ACTIVE'; $('calibrationInfo').textContent='Session ended. Start the camera and calibrate again for a new session.';
  clearMetrics(); status('OFFLINE','Session paused',reason); $('debugText').textContent='Camera stopped. Previous observations are invalid.';
}
async function start() {
  if(running||loading) return;
  if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { notify('Webcam access requires HTTPS or localhost and a supported browser.',true); return; }
  const token=++generation; loading=true; notify('Loading the on-device face model. The first start may take a moment.');
  $('start').disabled=true; $('stop').disabled=false; $('endSession').disabled=false; $('camera').disabled=true; $('start').textContent='Loading model…';
  status('LOADING','Preparing your session','Model and analysis assets are loaded from this deployment.'); unlockAudio();
  let localDetector=null, localStream=null;
  try {
    const files=await FilesetResolver.forVisionTasks('/wasm');
    const options={baseOptions:{modelAssetPath:'/models/face_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numFaces:1,minFaceDetectionConfidence:.6,minFacePresenceConfidence:.6,minTrackingConfidence:.6,outputFacialTransformationMatrixes:true};
    try { localDetector=await FaceLandmarker.createFromOptions(files,options); runtimeLabel='GPU'; }
    catch { options.baseOptions.delegate='CPU'; localDetector=await FaceLandmarker.createFromOptions(files,options); runtimeLabel='CPU fallback'; }
    if(token!==generation) { localDetector.close(); return; }
    localStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},...($('camera').value?{deviceId:{exact:$('camera').value}}:{facingMode:'user'})},audio:false});
    if(token!==generation) { localStream.getTracks().forEach(t=>t.stop()); localDetector.close(); return; }
    stream=localStream; detector=localDetector; video.srcObject=stream;
    await video.play();
    if(token!==generation) return;
    monitor=new Monitor(monitor.config); calibration=new Calibration();
    $('calibrationTag').textContent='NOT CALIBRATED'; $('calibrationInfo').textContent='Face forward, then press Calibrate. No calibration has been applied to this session.';
    running=true;loading=false;gazeBaseline=null;gazeSamples=[];fusion.reset();expressionCue=null;lastExpression=-Infinity;Object.keys(soundTimes).forEach(k=>delete soundTimes[k]);void loadExpression(); stalled=false; previousVideo=-1; lastProcessAt=0; fps=0; previousFpsAt=0; lastBeep=-Infinity; lastFrameAt=performance.now();
    $('start').textContent='Camera running'; $('calibrate').disabled=false; $('placeholder').hidden=true; $('liveBadge').hidden=false; $('cameraState').textContent='LIVE · '+runtimeLabel;
    for(const track of stream.getVideoTracks()) track.addEventListener('ended',()=>{if(running)stop('Camera disconnected or permission was revoked.');},{once:true});
    await cameras();
    if(token!==generation) return;
    notify('Camera is active. Calibrate before interpreting head orientation and alert thresholds.');
    raf=requestAnimationFrame(loop);
  } catch(error) {
    localStream?.getTracks().forEach(t=>t.stop());
    if(localDetector && localDetector!==detector) localDetector.close();
    if(token===generation) { stop('Camera could not start.'); notify(friendlyError(error),true); }
  }
}
function brightness() {
  lightCtx.drawImage(video,0,0,16,16);
  const pixels=lightCtx.getImageData(0,0,16,16).data;
  let sum=0,sum2=0;for(let i=0;i<pixels.length;i+=4){const v=.299*pixels[i]+.587*pixels[i+1]+.114*pixels[i+2];sum+=v;sum2+=v*v;}const mean=sum/(pixels.length/4);return {mean,contrast:Math.sqrt(Math.max(0,sum2/(pixels.length/4)-mean*mean))};
}
function draw(data) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!data) return;
  const p=data.points;
  const line=(a,b,color,width=1)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(p[a].x,p[a].y);ctx.lineTo(p[b].x,p[b].y);ctx.stroke();};
  if($('mesh').checked) for(const edge of FaceLandmarker.FACE_LANDMARKS_TESSELATION) line(edge.start,edge.end,'#6cb99855',.7);
  if($('landmarks').checked) for(const i of [...LEFT,...RIGHT,...MOUTH]) {ctx.fillStyle='#f9d773';ctx.beginPath();ctx.arc(p[i].x,p[i].y,2,0,Math.PI*2);ctx.fill();if($('ids').checked){ctx.font='9px monospace';ctx.fillText(String(i),p[i].x+3,p[i].y-3);}}
  if($('iris').checked) for(let i=468;i<Math.min(478,p.length);i++) {ctx.fillStyle='#e991de';ctx.beginPath();ctx.arc(p[i].x,p[i].y,2,0,Math.PI*2);ctx.fill();}
  if($('lines').checked) for(const [a,b] of [[33,133],[160,144],[158,153],[362,263],[385,380],[387,373],[13,14],[78,308]]) line(a,b,'#68d9ed',1.2);
  if($('axes').checked) data.axes.forEach(([x,y],i)=>{ctx.strokeStyle=['#ff746c','#91e1a2','#75a6ff'][i];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[1].x,p[1].y);ctx.lineTo(p[1].x+x*65,p[1].y+y*65);ctx.stroke();});
}
function render(now,data,state,light) {
  $('fpsValue').textContent=fps.toFixed(1)+' FPS'; $('countValue').textContent=`${monitor.yawns} mouth-open events`;
  if(currentHealth!=='MONITORING') {
    clearMetrics(); $('fpsValue').textContent=fps.toFixed(1)+' FPS';
    status('UNAVAILABLE','Waiting for a clear face',!data?'Keep one face in view.':!data.framed?'Center your face and move a little closer.':'Improve the lighting; avoid a bright background.');
    $('alerts').textContent='Tracking is unavailable. Alert timers have been reset.';
  } else {
    $('earValue').textContent=`${data.left.toFixed(3)} / ${data.right.toFixed(3)}`; $('marValue').textContent=data.mar.toFixed(3);
    $('poseValue').textContent=`${data.yaw.toFixed(0)}° / ${data.pitch.toFixed(0)}°`;
    const calibrating=calibration.started!==null;
    $('eyeState').textContent=calibrating?'Calibrating · alerts paused':state.candidates['Eyes closed']?'Both eyes below threshold':'Eyes above closure threshold';
    $('mouthState').textContent=calibrating?'Calibrating · alerts paused':state.candidates['Mouth open / yawn proxy']?'Mouth open · candidate':'Mouth opening below threshold';
    $('poseState').textContent=calibrating?'Calibrating · alerts paused':state.candidates['Head away']?'Outside forward range':'Within forward range';
    status(calibrating?'CALIBRATING':state.state,calibrating?'Learning your neutral position':state.alerts.length?state.alerts.join(' · '):'No sustained alerts',`Framing ${Math.round(data.inside*100)}% · face width ${Math.round(data.faceWidth*100)}% · ${calibration.calibrated?'calibrated':'not calibrated'}`);
    $('alerts').textContent=calibrating?'Alerts paused during calibration.':Object.entries(state.durations).map(([k,v])=>`${k}: ${v.toFixed(1)}s`).join('  ·  ');
  }
  $('debugText').textContent=`Runtime: ${runtimeLabel}. Target processing: 15 FPS.\nHealth: ${data?`in frame ${(data.inside*100).toFixed(0)}%; face width ${(data.faceWidth*100).toFixed(0)}%`:'no face'}. Brightness: ${light.toFixed(0)}/255.\nDetection/presence/tracking minima: 0.6 (settings, not per-frame confidence).\nEAR < ${monitor.config.ear.toFixed(3)} for ${monitor.config.eyeSeconds}s; MAR > ${monitor.config.mar.toFixed(3)} for ${monitor.config.mouthSeconds}s.\nYaw ±${monitor.config.yaw}°, pitch ±${monitor.config.pitch}° for ${monitor.config.poseSeconds}s.\nPose: canonical-face matrix estimate; different method from Python solvePnP.\nRoll: ${data?data.roll.toFixed(1)+'°':'unavailable'}. Lost frames or gaps >0.75s reset timers.`;
}
function loop(time) {
  if(!running) return;
  try {
    if(video.readyState>=2 && video.currentTime!==previousVideo && time-lastProcessAt>=1000/15) {
      previousVideo=video.currentTime; lastProcessAt=time; lastFrameAt=time; stalled=false;
      if(canvas.width!==video.videoWidth || canvas.height!==video.videoHeight) {canvas.width=video.videoWidth;canvas.height=video.videoHeight;}
      const result=detector.detectForVideo(video,time);
      const data=measure(result.faceLandmarks[0],result.facialTransformationMatrixes[0],canvas.width,canvas.height);
      const quality=brightness(),light=quality.mean;currentHealth=visibility(data,light,quality.contrast);const healthy=currentHealth==='MONITORING';
      const rawGaze=healthy?gaze(data.points,Math.min(data.left,data.right)>monitor.config.ear):null;
      const wasCalibrating=calibration.started!==null;if(wasCalibrating&&rawGaze!==null)gazeSamples.push(rawGaze);
      const now=time/1000, note=calibration.add(now,healthy?data:null,monitor.config);
      if(note) {$('calibrationInfo').textContent=note.message;if(note.done){$('calibrationTag').textContent=note.success?'CALIBRATED':'RETRY';notify(note.message,!note.success);if(note.success){syncSettings();if(gazeSamples.length>=30){const sorted=[...gazeSamples].sort((a,b)=>a-b);gazeBaseline=sorted[Math.floor(sorted.length/2)];}else notify('Head calibration complete. Gaze needs 30 visible-eye samples; retry.');}}}
      const values=healthy && calibration.started===null?calibration.correct(data):null;
      const state=monitor.update(now,values);
      currentGaze=rawGaze!==null&&gazeBaseline!==null?rawGaze-gazeBaseline:null;
      if(!healthy||wasCalibrating){expressionCue=null;observationEpoch++;}else requestExpression(now,data);
      const fused=fusion.update(now,state,currentHealth,currentGaze,expressionCue,wasCalibrating);
      state.alerts=fused.alerts;if(healthy&&fused.alerts.length)state.state='ALERT';
      $('fusion').textContent=fused.summary;
      $('gazeValue').textContent=!healthy?'Unavailable':currentGaze===null?'Calibrate first':Math.abs(currentGaze)>.12?'Away':'Forward range';
      const expression=healthy&&!wasCalibrating&&$('expression').checked?fresh(expressionCue,now):null;
      $('expressionValue').textContent=expression?expression.label:'Unavailable';
      $('expressionScore').textContent=expression?Math.round(expression.score*100)+'% · uncalibrated score':$('expression').checked?'Uncertain / stale / awaiting prediction':'Disabled';
      if(previousFpsAt) {const current=1000/(performance.now()-previousFpsAt);fps=fps?fps*.8+current*.2:current;}
      previousFpsAt=performance.now();
      draw(data); render(now,values||data,state,light);
      if(!healthy)status(currentHealth,'Monitoring needs a clear view',currentHealth==='FACE NOT DETECTED'?'No reliable face. Absence and full obstruction cannot be distinguished.':'Image or framing quality is unreliable; obstruction is possible.');
      if(fused.sound)beep(now,fused.sound);
    } else if(time-lastFrameAt>1500 && !stalled) {
      stalled=true;monitor.reset();calibration.cancel();fusion.reset();expressionCue=null;observationEpoch++;$('gazeValue').textContent='Unavailable';$('expressionValue').textContent='Unavailable';$('expressionScore').textContent='Stale observation';$('fusion').textContent='Camera view unavailable · stale frames';beep(time/1000,'availability');draw(null);clearMetrics();status('STALE','Camera data is stale','Previous readings are invalid. Stop and restart the camera.');$('alerts').textContent='Alert assessment unavailable; waiting for fresh frames.';
    }
    raf=requestAnimationFrame(loop);
  } catch(error) {stop('Processing stopped. Camera released.');notify(friendlyError(error),true);}
}
function syncSettings() {for(const [key] of fields) $('setting-'+key).value=monitor.config[key];}
$('start').addEventListener('click',start);
$('stop').addEventListener('click',()=>{stop();notify('Session ended. Camera released.');});
$('calibrate').addEventListener('click',()=>{if(!running)return;calibration.begin(performance.now()/1000);gazeSamples=[];fusion.reset();expressionCue=null;observationEpoch++;monitor.reset();$('calibrationTag').textContent='CALIBRATING';notify('For five seconds, face forward with eyes open and mouth closed.');});
$('settings').addEventListener('submit',event=>{event.preventDefault();try{const config=Object.fromEntries(fields.map(([key])=>[key,Number($('setting-'+key).value)]));monitor.config=validateSettings(config);monitor.reset();fusion.reset();notify('Thresholds applied. Alert timers reset.');}catch(error){notify(error.message,true);}});
$('mirror').addEventListener('change',()=> $('viewport').classList.toggle('mirrored',$('mirror').checked));
$('debug').addEventListener('change',()=>{$('debugPanel').hidden=!$('debug').checked;$('debugPanel').open=$('debug').checked;});
$('sound').addEventListener('change',unlockAudio);
$('camera').addEventListener('change',()=>{if(running||loading)stop('Camera changed. Start a new session.');});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(running||loading)){stop('Stopped because this tab was hidden.');notify('Monitoring stopped when the tab was hidden. Start again when ready.');}});
window.addEventListener('pagehide',()=>stop());
void cameras();

$('expression').addEventListener('change',()=>{expressionCue=null;observationEpoch++;if(running)void loadExpression();});
for(const button of document.querySelectorAll('button[data-view]'))button.addEventListener('click',()=>{document.querySelector('.layout').dataset.view=button.dataset.view;document.querySelectorAll('button[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('splitSize').disabled=button.dataset.view!=='split';});
$('splitSize').addEventListener('input',()=>document.querySelector('.layout').style.setProperty('--camera-share',$('splitSize').value+'%'));
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{notify('Fullscreen unavailable in this browser.');}});
document.addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';});
for(const button of document.querySelectorAll('[data-preview]'))button.addEventListener('click',async()=>{try{audio??=new(window.AudioContext||window.webkitAudioContext)();await audio.resume();beep(performance.now()/1000,button.dataset.preview,true);}catch{notify('Audio unavailable. Visual alerts remain active.');}});

$('endSession').addEventListener('click',()=>{stop();notify('Session ended. Camera released.');});
