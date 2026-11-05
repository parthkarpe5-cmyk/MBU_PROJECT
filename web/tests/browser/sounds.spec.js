import {test,expect} from '@playwright/test';
import {readFileSync} from 'node:fs';

test('friendly sound choices, real Web Audio previews, capped volume, persistence and reset',async({page})=>{
  await page.addInitScript(()=>{
    window.scheduledTones=[];
    const original=AudioContext.prototype.createOscillator;
    AudioContext.prototype.createOscillator=function(){const oscillator=original.call(this);
      const record={frequencies:[]};window.scheduledTones.push(record);
      for(const method of ['setValueAtTime','linearRampToValueAtTime']){
        const fn=oscillator.frequency[method].bind(oscillator.frequency);
        oscillator.frequency[method]=(...args)=>{record.frequencies.push([method,...args]);return fn(...args);};
      }
      const start=oscillator.start.bind(oscillator);oscillator.start=at=>{record.wave=oscillator.type;record.start=at;return start(at);};
      const stop=oscillator.stop.bind(oscillator);oscillator.stop=at=>{record.end=at;return stop(at);};
      return oscillator;
    };
  });
  await page.goto('/');await page.getByText('Alert sounds & previews',{exact:true}).click();
  await expect(page.locator('#sound-fatigue option:checked')).toHaveText('Fatigue · falling pulses');
  await expect(page.locator('#sound-availability option:checked')).toHaveText('Monitoring unavailable · soft tone');
  await expect(page.locator('#masterVolume')).toHaveAttribute('max','100');
  await expect(page.locator('#sound')).not.toBeChecked();
  const network=[];page.on('request',request=>network.push(request.url()));
  await page.locator('[data-preview=fatigue]').click();
  await expect.poll(()=>page.evaluate(()=>window.scheduledTones.length)).toBe(2);
  const tones=await page.evaluate(()=>window.scheduledTones);
  expect(tones[0].wave).toBe('sine');expect(tones[0].frequencies[0][1]).toBe(440);expect(tones[0].frequencies[1][1]).toBe(330);
  expect(tones[0].end-tones[0].start).toBeCloseTo(.35);
  await page.locator('[data-preview=distraction]').click();
  await expect.poll(()=>page.evaluate(()=>window.scheduledTones.length)).toBe(4);
  await page.locator('[data-preview=availability]').click();
  await expect.poll(()=>page.evaluate(()=>window.scheduledTones.length)).toBe(5);
  const extra=await page.evaluate(()=>window.scheduledTones.slice(2));
  expect(extra.map(tone=>tone.wave)).toEqual(['triangle','triangle','sine']);
  expect(extra[0].frequencies.map(step=>step[1])).toEqual([660,880]);
  expect(extra[2].frequencies.map(step=>step[1])).toEqual([520,520]);
  await page.locator('#sound-fatigue').selectOption('Mute');await page.locator('[data-preview=fatigue]').click();
  expect(await page.evaluate(()=>window.scheduledTones.length)).toBe(5);
  await page.locator('#masterVolume').fill('0');await page.locator('[data-preview=availability]').click();
  expect(await page.evaluate(()=>window.scheduledTones.length)).toBe(5);
  await page.locator('#masterVolume').fill('35');await page.locator('#masterVolume').dispatchEvent('change');
  await expect(page.locator('#volumeValue')).toHaveText('35%');expect(network).toEqual([]);
  await page.reload();await expect(page.locator('#masterVolume')).toHaveValue('35');
  await expect(page.locator('#sound-fatigue')).toHaveValue('Mute');
  await page.getByText('Thresholds & timing',{exact:true}).click();await page.locator('#resetSettings').click();
  await expect(page.locator('#masterVolume')).toHaveValue('70');await expect(page.locator('#volumeValue')).toHaveText('70%');
});

test('an active alert repeats, clears on a normal face, resumes on loss and stops with the session',async({page})=>{
  test.skip(!process.env.DMS_TEST_FACE,'Requires the public-domain face fixture.');
  const fixture=readFileSync(process.env.DMS_TEST_FACE).toString('base64');
  await page.addInitScript(({fixture})=>{
    localStorage.setItem('driveguard.settings.v1',JSON.stringify({version:1,soundEnabled:true,
      thresholds:{eyeSeconds:60,mouthSeconds:60,poseSeconds:60}}));
    window.toneCount=0;window.showFace=false;
    const original=AudioContext.prototype.createOscillator;
    AudioContext.prototype.createOscillator=function(){window.toneCount++;return original.call(this);};
    navigator.mediaDevices.getUserMedia=async()=>{
      const image=new Image();image.src='data:image/png;base64,'+fixture;await image.decode();
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;const ctx=canvas.getContext('2d');
      const draw=()=>{
        if(window.showFace)ctx.drawImage(image,100,0,320,300,0,0,640,480);
        else for(let y=0;y<8;y++)for(let x=0;x<8;x++){ctx.fillStyle=(x+y)%2?'#607080':'#b0c0d0';ctx.fillRect(x*80,y*60,80,60);}
      };
      draw();const timer=setInterval(draw,60),stream=canvas.captureStream(15);window.soundTestTracks=stream.getTracks();
      for(const track of stream.getTracks()){const stop=track.stop.bind(track);track.stop=()=>{clearInterval(timer);stop();};}
      return stream;
    };
  },{fixture});
  await page.goto('/');await page.locator('#expression').uncheck();await page.locator('#start').click();
  await expect.poll(()=>page.evaluate(()=>window.toneCount),{timeout:20000}).toBeGreaterThanOrEqual(4);
  await page.locator('#sound').uncheck();const muted=await page.evaluate(()=>window.toneCount);
  await page.waitForTimeout(700);expect(await page.evaluate(()=>window.toneCount)).toBe(muted);
  await page.locator('#sound').check();
  await expect.poll(()=>page.evaluate(()=>window.toneCount)).toBeGreaterThan(muted);
  await page.evaluate(()=>{window.showFace=true;});
  await expect(page.locator('#status')).toHaveText('MONITORING',{timeout:20000});
  const normal=await page.evaluate(()=>window.toneCount);await page.waitForTimeout(1100);
  expect(await page.evaluate(()=>window.toneCount)).toBe(normal);
  await page.evaluate(()=>{window.showFace=false;});
  await expect.poll(()=>page.evaluate(()=>window.toneCount),{timeout:15000}).toBeGreaterThan(normal);
  await page.locator('#stop').click();const stopped=await page.evaluate(()=>window.toneCount);
  await page.waitForTimeout(1100);expect(await page.evaluate(()=>window.toneCount)).toBe(stopped);
  expect(await page.evaluate(()=>window.soundTestTracks.every(track=>track.readyState==='ended'))).toBe(true);
});
