import test from 'node:test';
import assert from 'node:assert/strict';
import {friendlyError} from '../src/errors.js';
test('camera denied, absent, busy and model failures have actionable messages without raw exceptions',()=>{
  assert.match(friendlyError({name:'NotAllowedError'}),/Allow camera access/);
  assert.match(friendlyError({name:'NotFoundError'}),/Connect a camera/);
  assert.match(friendlyError({name:'NotReadableError'}),/Close other camera apps/);
  assert.match(friendlyError(Error('internal stack'),'model'),/face model could not load/);
  assert.match(friendlyError(Error('internal stack'),'expression'),/geometry monitoring/);
  assert.ok(!friendlyError(Error('internal stack'),'model').includes('internal stack'));
  assert.match(friendlyError(null),/try again/);
});
