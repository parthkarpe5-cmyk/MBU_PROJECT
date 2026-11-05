import test from 'node:test';
import assert from 'node:assert/strict';
import {PATTERNS,FADE_SECONDS,MAX_MASTER_GAIN,canPlay,clampVolume,createMasterGain,setMasterVolume,synthesize} from '../src/sounds.js';
import {defaultPreferences,validatePreferences} from '../src/preferences.js';

function fakeContext(){
  const nodes=[];
  const param=()=>({calls:[],setValueAtTime(...args){this.calls.push(['set',...args]);},
    linearRampToValueAtTime(...args){this.calls.push(['ramp',...args]);},
    cancelScheduledValues(...args){this.calls.push(['cancel',...args]);},
    setTargetAtTime(...args){this.calls.push(['target',...args]);}});
  return {currentTime:10,destination:{},nodes,
    createGain(){const node={gain:param(),connect(target){this.target=target;},disconnect(){this.disconnected=true;}};nodes.push(node);return node;},
    createOscillator(){const node={frequency:param(),connect(target){this.target=target;},disconnect(){this.disconnected=true;},start(at){this.started=at;},stop(at){this.stopped=at;}};nodes.push(node);return node;},
  };
}
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-9,`${actual} != ${expected}`);
test('fatigue has two falling sine pulses, 350 ms each, separated by 150 ms',()=>{
  const context=fakeContext(),master=createMasterGain(context,100);synthesize(context,master,'fatigue');
  const oscillators=context.nodes.filter(node=>node.frequency);
  assert.equal(oscillators.length,2);
  for(const [i,oscillator] of oscillators.entries()){
    const at=10+i*.5;assert.equal(oscillator.type,'sine');near(oscillator.started,at);near(oscillator.stopped,at+.35);
    assert.deepEqual(oscillator.frequency.calls,[['set',440,at],['ramp',330,at+.35]]);
    assert.equal(oscillator.target.target,master);
    oscillator.onended();assert.equal(oscillator.disconnected,true);assert.equal(oscillator.target.disconnected,true);
  }
});
test('distraction has two rising 150 ms triangle notes; unavailability has a quieter single 400 ms sine',()=>{
  for(const [key,wave,start,end,duration,count] of [['distraction','triangle',660,880,.15,2],['availability','sine',520,520,.4,1]]){
    const context=fakeContext(),master=createMasterGain(context);synthesize(context,master,key);
    const oscillators=context.nodes.filter(node=>node.frequency);assert.equal(oscillators.length,count);
    for(const [i,oscillator] of oscillators.entries()){
      const at=10+i*duration;assert.equal(oscillator.type,wave);near(oscillator.stopped-oscillator.started,duration);
      assert.deepEqual(oscillator.frequency.calls,[['set',start,at],['ramp',end,at+duration]]);
    }
  }
  assert.ok(PATTERNS.availability.level<PATTERNS.fatigue.level);
});
test('every note fades in/out for 15 ms and ends at zero; volume has a fixed capped master',()=>{
  assert.ok(FADE_SECONDS>=.01&&FADE_SECONDS<=.02);
  for(const key of Object.keys(PATTERNS)){
    const context=fakeContext(),master=createMasterGain(context,10000);synthesize(context,master,key);
    assert.deepEqual(master.gain.calls,[['set',MAX_MASTER_GAIN,10]]);
    for(const oscillator of context.nodes.filter(node=>node.frequency)){
      const at=oscillator.started,end=oscillator.stopped,level=PATTERNS[key].level;
      assert.deepEqual(oscillator.target.gain.calls,[['set',0,at],['ramp',level,at+.015],['set',level,end-.015],['ramp',0,end]]);
    }
    setMasterVolume(context,master,-10);assert.deepEqual(master.gain.calls.at(-1),['target',0,10,.015]);
  }
  assert.equal(clampVolume(NaN),50);assert.equal(clampVolume(Infinity),50);assert.equal(clampVolume(101),100);
});
test('cooldown and global mute gate alerts; previews bypass them but never bypass a muted mapping',()=>{
  const request={state:'running',enabled:true,pattern:'fatigue',now:10,last:8};
  assert.equal(canPlay(request),false);assert.equal(canPlay({...request,last:7}),true);
  assert.equal(canPlay({...request,last:7,enabled:false}),false);
  assert.equal(canPlay({...request,enabled:false,preview:true}),true);
  assert.equal(canPlay({...request,pattern:'Mute',preview:true}),false);
  assert.equal(canPlay({...request,volume:0,preview:true}),false);
  assert.equal(canPlay({...request,state:'suspended',preview:true}),false);
  assert.equal(canPlay({...request,now:NaN,preview:true}),false);
});
test('old saved preferences gain default volume; saved volume clamps and reset defaults remain moderate',()=>{
  const old={...defaultPreferences(),masterVolume:undefined};assert.equal(validatePreferences(old).masterVolume,50);
  assert.equal(validatePreferences({...old,masterVolume:500}).masterVolume,100);
  assert.equal(validatePreferences({...old,masterVolume:-1}).masterVolume,0);
});
