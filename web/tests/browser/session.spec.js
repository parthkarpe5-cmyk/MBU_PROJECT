import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {useCanvasCamera} from './camera.js';

test('stop shows aggregate summary, performance clears, and JSON/CSV download only on click',async({page},testInfo)=>{
  await useCanvasCamera(page);
  const downloads=[];page.on('download',download=>downloads.push(download));
  await page.goto('/');await expect(page.locator('#sessionSummary')).toBeHidden();
  await page.locator('#expression').uncheck();await page.locator('#start').click();
  await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});
  await expect(page.locator('#livePerformance')).toHaveText(/Performance: \d+\.\d+ FPS · \d+\.\d+ ms\/frame/);
  await page.waitForTimeout(400);
  await page.evaluate(()=>{window.sessionTracks=[...document.querySelector('#video').srcObject.getTracks()];});
  await page.locator('#stop').click();
  await expect(page.locator('#sessionSummary')).toBeVisible();
  await expect(page.locator('#summaryValues')).toContainText('Average processing (ms)');
  await expect(page.locator('#summaryAlerts')).toContainText('Eyes closed: 0');
  await expect(page.locator('#livePerformance')).toHaveText('Performance: — FPS · — ms/frame');
  expect(await page.evaluate(()=>window.sessionTracks.every(track=>track.readyState==='ended'))).toBe(true);
  expect(downloads).toHaveLength(0);
  const network=[];page.on('request',request=>{if(request.url().startsWith('http'))network.push(request.url());});
  const first=page.waitForEvent('download');await page.locator('#saveReport').click();
  const json=await first;expect(json.suggestedFilename()).toMatch(/\.json$/);
  const report=JSON.parse(await readFile(await json.path(),'utf8'));
  expect(report.durationSeconds).toBeGreaterThan(0);expect(report.processedFrames).toBeGreaterThan(1);
  expect(report.averageFps).toBeGreaterThan(0);expect(report.averageProcessingMs).toBeGreaterThanOrEqual(0);
  expect(report.blinkCount).toBe(0);expect(report.yawnCount).toBe(0);
  expect(report).not.toHaveProperty('landmarks');expect(report).not.toHaveProperty('video');
  await page.locator('#reportFormat').selectOption('csv');
  const second=page.waitForEvent('download');await page.locator('#saveReport').click();
  const csv=await second;expect(await readFile(await csv.path(),'utf8')).toContain('"averageProcessingMs"');
  expect(network).toEqual([]);
  expect(await page.evaluate(()=>Object.keys(localStorage).filter(key=>key.includes('session')))).toEqual([]);
  await page.screenshot({path:testInfo.outputPath('session-summary.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:testInfo.outputPath('session-summary-mobile.png'),fullPage:true});
  await page.reload();await expect(page.locator('#sessionSummary')).toBeHidden();
});

test('preferences survive reload; reset restores thresholds, sound mute/map and view without touching other keys',async({page})=>{
  await page.goto('/');await page.getByText('Thresholds & timing',{exact:true}).click();
  await page.locator('#setting-ear').fill('0.25');await page.getByRole('button',{name:'Apply settings',exact:true}).click();
  await page.getByText('Alert sounds & previews',{exact:true}).click();
  await page.locator('#sound-fatigue').selectOption('Mute');await page.locator('#sound').check();
  await page.getByRole('button',{name:'Dashboard only',exact:true}).click();
  await page.evaluate(()=>localStorage.setItem('other-app','keep'));
  await page.reload();
  await expect(page.locator('#setting-ear')).toHaveValue('0.25');await expect(page.locator('#sound-fatigue')).toHaveValue('Mute');
  await expect(page.locator('#sound')).toBeChecked();await expect(page.locator('.layout')).toHaveAttribute('data-view','dashboard');
  await page.getByText('Thresholds & timing',{exact:true}).click();await page.locator('#resetSettings').click();
  await expect(page.locator('#setting-ear')).toHaveValue('0.21');await expect(page.locator('#sound-fatigue')).toHaveValue('fatigue');
  await expect(page.locator('#sound')).not.toBeChecked();await expect(page.locator('.layout')).toHaveAttribute('data-view','split');
  expect(await page.evaluate(()=>localStorage.getItem('other-app'))).toBe('keep');
  await page.reload();await expect(page.locator('#setting-ear')).toHaveValue('0.21');
});

test('blocked storage still allows controls, and About states all limits on a narrow viewport',async({page})=>{
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Blocked','SecurityError');}}));
  await page.goto('/');await page.getByRole('button',{name:'Dashboard only',exact:true}).click();
  await expect(page.locator('.layout')).toHaveAttribute('data-view','dashboard');
  await expect(page.locator('#message')).toContainText('Preferences could not be saved');
  await page.getByText('About / Limitations',{exact:true}).click();
  await expect(page.locator('#about')).toContainText('not a safety or automotive system');
  await expect(page.locator('#about')).toContainText('Frames are processed locally');
  await expect(page.locator('#about')).toContainText('never trigger an alert alone');
  await expect(page.locator('#about')).toContainText('Accuracy has not been validated across people');
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

for(const [name,message] of [['NotAllowedError','Camera permission was denied'],['NotFoundError','No webcam was found']]){
  test(`${name} has a Retry that starts a fresh session`,async({page})=>{
    await useCanvasCamera(page,name);
    await page.goto('/');await page.locator('#expression').uncheck();await page.locator('#start').click();
    await expect(page.locator('#message')).toContainText(message,{timeout:60000});
    await expect(page.locator('#retry')).toBeVisible();
    expect(await page.locator('#video').evaluate(video=>video.srcObject)).toBe(null);
    await expect(page.locator('#sessionSummary')).toBeHidden();
    await page.locator('#retry').click();await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});
    await page.locator('#stop').click();await expect(page.locator('#sessionSummary')).toBeVisible();
  });
}

test('face-model load failure has friendly Retry and does not request the camera',async({page})=>{
  await useCanvasCamera(page);
  await page.route('**/models/face_landmarker.task',route=>route.abort());
  await page.goto('/');await page.locator('#expression').uncheck();await page.locator('#start').click();
  await expect(page.locator('#message')).toContainText('face model could not load',{timeout:60000});
  expect(await page.evaluate(()=>window.cameraCalls)).toBe(0);
  await page.unroute('**/models/face_landmarker.task');await page.locator('#retry').click();
  await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});await page.locator('#stop').click();
});

test('expression-model Retry recovers while the existing camera stream stays active',async({page})=>{
  await useCanvasCamera(page);
  await page.route('**/models/emotion-ferplus-8.onnx',route=>route.abort());
  await page.goto('/');await page.locator('#start').click();
  await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});
  await expect(page.locator('#message')).toContainText('expression model could not load',{timeout:60000});
  await page.evaluate(()=>{window.originalTrack=document.querySelector('#video').srcObject.getVideoTracks()[0];});
  await page.unroute('**/models/emotion-ferplus-8.onnx');await page.locator('#retry').click();
  await expect(page.locator('#modelState')).toContainText('FER+ CNN ready',{timeout:60000});
  expect(await page.evaluate(()=>window.originalTrack===document.querySelector('#video').srcObject.getVideoTracks()[0])).toBe(true);
  await page.locator('#stop').click();expect(await page.evaluate(()=>window.originalTrack.readyState)).toBe('ended');
});
