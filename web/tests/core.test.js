import test from 'node:test';
import assert from 'node:assert/strict';
import { Monitor, Calibration, ear, poseFromMatrix, measure, DEFAULTS, validateSettings } from '../src/core.js';
const open={left:.3,right:.3,mar:.1,yaw:0,pitch:0};
const closed={...open,left:.1,right:.1};
function feed(m, v, start=0,end=2) {let result;for(let t=start;t<=end+1e-8;t+=.1) result=m.update(t,v);return result;}
test('EAR is scale invariant and matches Python eye formula',()=>{
  const points=[[0,0],[1,1],[3,1],[4,0],[3,-1],[1,-1]].map(([x,y])=>({x,y}));
  assert.equal(ear(points,[0,1,2,3,4,5]),.5);
  assert.equal(ear(points.map(({x,y})=>({x:x*50,y:y*50})),[0,1,2,3,4,5]),.5);
});
test('ordinary blink resets while continuous closure alerts',()=>{
  const m=new Monitor();m.update(0,closed);assert.deepEqual(m.update(.2,closed).alerts,[]);m.update(.3,open);
  assert.deepEqual(m.update(.4,closed).alerts,[]);
  assert.ok(feed(new Monitor(),closed).alerts.includes('Eyes closed'));
});
test('one eye closure does not trigger bilateral closure',()=>{assert.deepEqual(feed(new Monitor(),{...open,left:.1}).alerts,[]);});
test('missing, invalid, late, and backwards observations break continuity',()=>{
  for(const gap of [null,{...closed,mar:NaN}]) {const m=new Monitor();feed(m,closed,0,1);assert.equal(m.update(1.1,gap).state,'UNAVAILABLE');assert.deepEqual(m.update(1.2,closed).alerts,[]);}
  const m=new Monitor();feed(m,closed,0,1);assert.deepEqual(m.update(3,closed).alerts,[]);assert.deepEqual(m.update(0,closed).alerts,[]);
});
test('mouth events counted once per continuous event; head away uses duration',()=>{
  const m=new Monitor();const values={...open,mar:.8,yaw:40};
  assert.ok(feed(m,values,0,3).alerts.includes('Head away'));assert.equal(m.yawns,1);
  m.update(3.1,open);feed(m,values,3.2,5.3);assert.equal(m.yawns,2);
});
test('calibration updates thresholds and neutral pose, rejects low coverage',()=>{
  const c=new Calibration(), config={...DEFAULTS};c.begin(0);
  let note;for(let i=0;i<=50;i++)note=c.add(i/10,{...open,yaw:8,pitch:-4},config);
  assert.equal(note.success,true);assert.ok(Math.abs(config.ear-.216)<1e-12);assert.equal(config.mar,.35);
  assert.equal(c.correct({...open,yaw:8,pitch:-4}).yaw,0);
  c.begin(10);assert.equal(c.add(15,null,config).success,false);assert.equal(c.baseline.yaw,8);
});
test('column-major transform yields frontal pose and known yaw',()=>{
  const identity={data:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,-50,1]};
  const neutral=poseFromMatrix(identity);assert.ok(Math.abs(neutral.yaw)<1e-10);assert.equal(neutral.pitch,0);
  const a=Math.PI/6,c=Math.cos(a),s=Math.sin(a);
  const result=poseFromMatrix({data:[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,-50,1]});
  assert.ok(Math.abs(result.yaw-30)<1e-10);assert.equal(result.pitch,0);
  assert.equal(poseFromMatrix({data:[]}),null);assert.equal(poseFromMatrix({data:Array(16).fill(0)}),null);
});
test('unavailable face geometry is never scored',()=>{assert.equal(measure(null,null,640,480),null);assert.equal(measure(Array(478).fill({x:.5,y:.5,z:0}),null,640,480),null);});
test('manual settings reject nonfinite and out of bounds',()=>{
  for(const config of [{...DEFAULTS,ear:NaN},{...DEFAULTS,eyeSeconds:0},{...DEFAULTS,yaw:100}])assert.throws(()=>validateSettings(config));
  assert.deepEqual(validateSettings({...DEFAULTS}),DEFAULTS);
});
