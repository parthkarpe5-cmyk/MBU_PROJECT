import test from 'node:test';
import assert from 'node:assert/strict';
import {SessionSummary, serializeReport} from '../src/session.js';

test('summary counts alert onsets, brief blinks and yawn proxies, not repeated frames',()=>{
  const session=new SessionSummary(0,'2026-11-05T10:00:00.000Z');
  const frame=(atMs,alerts=[],eyesClosed=false,yawns=0)=>session.observe({atMs,processingMs:10,alerts,eyesClosed,yawns});
  frame(0);frame(100,[],true);frame(200); // One short blink.
  frame(300,['Eyes closed'],true);frame(600,['Eyes closed'],true);frame(1000,['Eyes closed'],true);
  frame(1100); // Sustained closure is not a blink.
  frame(1200,['Mouth open / yawn proxy'],false,1);frame(1300,['Mouth open / yawn proxy'],false,1);
  frame(1400,[],false,1);frame(1500,['Mouth open / yawn proxy'],false,2);
  const result=session.finish(2000,'Stopped');
  assert.equal(result.durationSeconds,2);assert.equal(result.blinkCount,1);assert.equal(result.yawnCount,2);
  assert.equal(result.alertCounts['Eyes closed'],1);assert.equal(result.alertCounts['Mouth open / yawn proxy'],2);
  assert.equal(result.processedFrames,11);assert.equal(result.averageProcessingMs,10);
  assert.equal(result.minimumFps,2.5);
  assert.ok(Math.abs(result.averageFps-10*1000/1500)<1e-9);
  assert.equal(session.finish(9000,'duplicate stop'),result);
  frame(10000,['Eyes closed']);assert.equal(result.alertCounts['Eyes closed'],1);
});
test('missing observations and long gaps do not manufacture blinks; empty statistics stay null',()=>{
  const session=new SessionSummary(0);
  session.observe({atMs:0,processingMs:5,eyesClosed:true});
  session.observe({atMs:100,processingMs:7,eyesClosed:null});
  session.observe({atMs:200,processingMs:9,eyesClosed:false});
  session.observe({atMs:300,processingMs:3,eyesClosed:true});
  session.observe({atMs:1200,processingMs:1,eyesClosed:false});
  assert.equal(session.finish(1300,'stop').blinkCount,0);
  const empty=new SessionSummary(0).finish(0,'stop');
  assert.equal(empty.averageFps,null);assert.equal(empty.minimumFps,null);assert.equal(empty.averageProcessingMs,null);
  const single=new SessionSummary(0);single.observe({atMs:50,processingMs:7});
  assert.equal(single.finish(100,'stop').averageFps,null);
});
test('availability/stall events count once until cleared; reporting has no storage or network dependency',()=>{
  const session=new SessionSummary(0);
  session.countAlerts(['Monitoring unavailable']);session.countAlerts(['Monitoring unavailable']);
  session.countAlerts(['Camera stream stale']);session.countAlerts(['Camera stream stale']);
  session.countAlerts([]);session.countAlerts(['Monitoring unavailable']);
  const report=session.finish(1000,'stop');
  assert.equal(report.alertCounts['Monitoring unavailable'],2);assert.equal(report.alertCounts['Camera stream stale'],1);
  assert.deepEqual(JSON.parse(serializeReport(report,'json').text),report);
  assert.ok(!Object.hasOwn(report,'frames'));assert.ok(!Object.hasOwn(report,'landmarks'));
  assert.throws(()=>serializeReport(null,'json'));assert.throws(()=>serializeReport(report,'xml'));
});
test('CSV includes aggregate fields, quotes text, handles nulls, and guards formula prefixes',()=>{
  const report=new SessionSummary(0).finish(0,'=1+1,"quoted"\nnext');
  const csv=serializeReport(report,'csv');
  assert.equal(csv.extension,'csv');assert.match(csv.text,/"metric","value"/);
  assert.match(csv.text,/"blinkCount","0"/);assert.match(csv.text,/"averageFps",""/);
  assert.match(csv.text,/"stopReason","'=1\+1,""quoted""\nnext"/);
  assert.match(csv.text,/"alert: Eye gaze away","0"/);
});
