import {test} from 'node:test';
import assert from 'node:assert/strict';
import {gaze,visibility,fresh,Fusion,PATTERNS} from '../src/multimodal.js';
import {decode} from '../src/expression.js';
test('gaze uses iris position independently of head pose; closed or missing eyes suppressed',()=>{
  const p=Array.from({length:478},()=>({x:0,y:0}));
  for(const [a,b,i,start] of [[33,133,468,10],[362,263,473,60]]) {
    p[a]={x:start,y:20};p[b]={x:start+30,y:20};p[i]={x:start+21,y:20};
  }
  assert.ok(Math.abs(gaze(p)-.2)<1e-6);assert.equal(gaze(p,false),null);assert.equal(gaze(p.slice(0,468)),null);
  p[468].x+=10;assert.equal(gaze(p),null);
});
test('missing face, unreliable framing and unusable camera remain distinct',()=>{
  assert.equal(visibility(null,0,0),'CAMERA VIEW UNAVAILABLE');
  assert.equal(visibility(null,100,30),'FACE NOT DETECTED');
  assert.match(visibility({framed:false},100,30),/PARTIAL/);
});
test('fusion sustains gaze, prioritizes fatigue, resets on loss, gaps and calibration',()=>{
  const f=new Fusion(), normal={alerts:[]};let r;
  for(const t of [0,.5,1,1.5,2])r=f.update(t,normal,'MONITORING',.2,null);
  assert.deepEqual(r.alerts,['Eye gaze away']);assert.equal(r.sound,'distraction');
  assert.equal(f.update(2.1,{alerts:['Eyes closed']},'MONITORING',.2,null).sound,'fatigue');
  assert.deepEqual(f.update(2.2,normal,'FACE NOT DETECTED',.2,null).alerts,[]);
  for(const t of [2.7,3.2,3.7,4.2])r=f.update(t,normal,'FACE NOT DETECTED',null,null);
  assert.equal(r.sound,'availability');
  assert.equal(f.update(7,normal,'MONITORING',.2,null).sound,null);
  assert.equal(f.update(7.1,normal,'MONITORING',.2,null,true).sound,null);
});
test('CNN scores and temporal expression context never create an alert',()=>{
  const cue=decode([0,0,0,0,8,0,0,0],0),f=new Fusion();
  assert.equal(cue.label,'anger');assert.ok(cue.score>.9);
  assert.equal(fresh(cue,2),null);assert.equal(fresh(cue,-1),null);
  for(const t of [0,.5,1])assert.equal(f.update(t,{alerts:[]},'MONITORING',null,cue).sound,null);
  const r=f.update(1.1,{alerts:[]},'MONITORING',null,{...cue,timestamp:1.1});
  assert.match(r.summary,/anger/);assert.deepEqual(r.alerts,[]);
  assert.throws(()=>decode([0,NaN],0));assert.equal(new Set(Object.values(PATTERNS).map(JSON.stringify)).size,3);
});
