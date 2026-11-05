import test from 'node:test';
import assert from 'node:assert/strict';
import {SETTINGS_KEY,defaultPreferences,validatePreferences,loadPreferences,savePreferences,resetPreferences} from '../src/preferences.js';

function storage(){const data=new Map();return {getItem:key=>data.get(key)??null,setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key),data};}
test('thresholds, sound mapping, mute and view persist, reset removes only this application key',()=>{
  const store=storage(),value=defaultPreferences();value.thresholds.ear=.25;value.sounds.fatigue='Mute';value.soundEnabled=true;value.view='dashboard';value.splitSize=55;
  store.setItem('unrelated','preserve');assert.equal(savePreferences(store,value),true);
  assert.deepEqual(loadPreferences(store),value);assert.equal(resetPreferences(store),true);
  assert.deepEqual(loadPreferences(store),defaultPreferences());assert.equal(store.getItem('unrelated'),'preserve');
});
test('corrupt data, unknown versions and invalid saved choices fall back safely',()=>{
  const store=storage();store.setItem(SETTINGS_KEY,'not-json');assert.deepEqual(loadPreferences(store),defaultPreferences());
  assert.deepEqual(validatePreferences({version:9}),defaultPreferences());
  const value=defaultPreferences();value.thresholds.ear=-1;value.sounds.fatigue='invalid';value.view='invalid';value.soundEnabled='true';value.splitSize=500;
  assert.deepEqual(validatePreferences(value),defaultPreferences());
});
test('blocked or unavailable storage never breaks monitoring; saved values exclude session data',()=>{
  const blocked={getItem(){throw Error('blocked');},setItem(){throw Error('quota');},removeItem(){throw Error('blocked');}};
  assert.deepEqual(loadPreferences(blocked),defaultPreferences());assert.equal(savePreferences(blocked,defaultPreferences()),false);assert.equal(resetPreferences(blocked),false);
  assert.deepEqual(loadPreferences(null),defaultPreferences());
  const store=storage();savePreferences(store,{...defaultPreferences(),report:{secret:'no'},video:'no'});
  assert.ok(!store.getItem(SETTINGS_KEY).includes('secret'));assert.ok(!store.getItem(SETTINGS_KEY).includes('video'));
});
