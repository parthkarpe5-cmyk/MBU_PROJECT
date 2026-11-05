import {test,expect} from '@playwright/test';

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
  await expect(page.locator('#masterVolume')).toHaveValue('50');await expect(page.locator('#volumeValue')).toHaveText('50%');
});
