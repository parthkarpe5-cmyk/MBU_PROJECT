import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import {useCanvasCamera} from './camera.js';

test('loads without webcam permission and adapts to a mobile viewport',async({page})=>{
  let cameraCalls=0;
  await page.exposeFunction('cameraCalled',()=>cameraCalls++);
  await page.addInitScript(()=>{const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=(...args)=>{window.cameraCalled();return original(...args);};});
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'DriveGuard AI',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Start camera',exact:true})).toBeEnabled();
  await expect(page.locator('#status')).toHaveText('OFFLINE');
  expect(cameraCalls).toBe(0);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('permission denial gives an actionable error and allows retry',async({page})=>{
  await page.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Denied','NotAllowedError');};});
  await page.goto('/');await page.locator('#start').click();
  await expect(page.locator('#message')).toContainText('Camera permission was denied',{timeout:60000});
  await expect(page.locator('#start')).toBeEnabled();
  await expect(page.locator('#stop')).toBeDisabled();
  expect(await page.locator('#video').evaluate(v=>v.srcObject)).toBe(null);
});

test('real MediaPipe loads; fake webcam stops and restarts without leaking tracks',async({page})=>{
  await useCanvasCamera(page);
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');await page.locator('#start').click();
  await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});
  // The synthetic feed can be normally exposed or too flat/dark, depending on its frame.
  await expect(page.locator('#status')).toHaveText(/^(FACE NOT DETECTED|CAMERA VIEW UNAVAILABLE)$/,{timeout:30000});
  await expect(page.locator('#earValue')).toHaveText('—');
  await expect(page.locator('#message')).not.toHaveClass(/error/);
  await page.evaluate(()=>{window.testTracks=[...document.querySelector('#video').srcObject.getTracks()];});
  await page.locator('#stop').click();
  expect(await page.evaluate(()=>window.testTracks.every(t=>t.readyState==='ended'))).toBe(true);
  await expect(page.locator('#status')).toHaveText('OFFLINE');
  await page.locator('#start').click();
  await expect(page.locator('#cameraState')).toContainText('LIVE',{timeout:60000});
  await page.locator('#stop').click();
  expect(errors).toEqual([]);
});

test('real face fixture produces geometry and completes calibration',async({page},testInfo)=>{
  test.skip(!process.env.DMS_TEST_FACE,'Set DMS_TEST_FACE to the public-domain scikit-image astronaut.png fixture.');
  const fixture=readFileSync(process.env.DMS_TEST_FACE).toString('base64');
  await page.addInitScript(({fixture})=>{
    navigator.mediaDevices.getUserMedia=async()=>{
      const image=new Image();image.src='data:image/png;base64,'+fixture;await image.decode();
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      const ctx=canvas.getContext('2d');
      const draw=()=>ctx.drawImage(image,100,0,320,300,0,0,640,480);draw();
      const timer=setInterval(draw,60);const stream=canvas.captureStream(15);
      for(const track of stream.getTracks()){const stop=track.stop.bind(track);track.stop=()=>{clearInterval(timer);stop();};}
      return stream;
    };
  },{fixture});
  await page.goto('/');await page.locator('#start').click();
  await expect(page.locator('#status')).toHaveText('MONITORING',{timeout:60000});
  await expect(page.locator('#earValue')).not.toHaveText('—');
  await page.locator('#mesh').check();
  await page.locator('#ids').check();
  await page.locator('#debug').check();
  await expect(page.locator('#debugText')).toBeVisible();
  await page.locator('#calibrate').click();
  await expect(page.locator('#calibrationTag')).toHaveText('CALIBRATED',{timeout:15000});
  await expect(page.locator('#calibrationInfo')).toContainText('Calibrated for this session');
  await expect(page.locator('#modelState')).toContainText('FER+ CNN ready',{timeout:60000});
  await expect(page.locator('#expressionValue')).not.toHaveText('Unavailable',{timeout:30000});
  await page.screenshot({path:testInfo.outputPath('dashboard.png'),fullPage:true});
  await page.locator('#stop').click();
});

test('workspace views, resizing, sound assignments and light theme work without a camera',async({page},testInfo)=>{
  await page.goto('/');
  await page.getByRole('button',{name:'Camera only',exact:true}).click();
  await expect(page.getByLabel('Monitoring dashboard')).toBeHidden();
  await page.getByRole('button',{name:'Dashboard only',exact:true}).click();
  await expect(page.getByLabel('Webcam and overlays')).toBeHidden();
  await expect(page.getByLabel('Monitoring dashboard')).toBeVisible();
  await page.getByRole('button',{name:'Split view',exact:true}).click();
  await page.locator('#splitSize').fill('55');
  await expect(page.locator('.layout')).toHaveAttribute('style',/55%/);
  await page.getByText('Alert sounds & previews',{exact:true}).click();
  await page.locator('#sound-distraction').selectOption('availability');
  await page.locator('[data-preview=distraction]').click();
  expect(await page.evaluate(()=>getComputedStyle(document.documentElement).colorScheme)).toBe('light');
  await page.screenshot({path:testInfo.outputPath('driveguard-light.png'),fullPage:true});
});
