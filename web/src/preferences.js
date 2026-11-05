import {DEFAULTS, validateSettings} from './core.js';
import {PATTERNS} from './multimodal.js';

export const SETTINGS_KEY='driveguard.settings.v1';
export const defaultPreferences=()=>({version:1,thresholds:{...DEFAULTS},
  sounds:{distraction:'distraction',fatigue:'fatigue',availability:'availability'},
  soundEnabled:false,view:'split',splitSize:60});

export function validatePreferences(value) {
  const defaults=defaultPreferences();
  if(!value||value.version!==1)return defaults;
  const thresholds=Object.fromEntries(Object.keys(DEFAULTS).map(key=>[key,value.thresholds?.[key]??DEFAULTS[key]]));
  try{validateSettings(thresholds);defaults.thresholds=thresholds;}catch{/* Invalid persisted thresholds are not applied. */}
  for(const event of Object.keys(defaults.sounds)){
    const choice=value.sounds?.[event];
    if(Object.hasOwn(PATTERNS,choice)||choice==='Mute')defaults.sounds[event]=choice;
  }
  if(typeof value.soundEnabled==='boolean')defaults.soundEnabled=value.soundEnabled;
  if(['split','camera','dashboard'].includes(value.view))defaults.view=value.view;
  if(Number.isFinite(value.splitSize)&&value.splitSize>=35&&value.splitSize<=70)defaults.splitSize=value.splitSize;
  return defaults;
}
export function loadPreferences(storage) {
  try{return validatePreferences(JSON.parse(storage.getItem(SETTINGS_KEY)));}catch{return defaultPreferences();}
}
export function savePreferences(storage,value) {
  try{storage.setItem(SETTINGS_KEY,JSON.stringify(validatePreferences(value)));return true;}catch{return false;}
}
export function resetPreferences(storage) {
  try{storage.removeItem(SETTINGS_KEY);return true;}catch{return false;}
}
