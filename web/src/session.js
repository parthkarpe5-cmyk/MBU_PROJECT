// Aggregate numbers only: no images, landmarks, expressions or frame history.
export const ALERT_TYPES = ['Eyes closed', 'Mouth open / yawn proxy', 'Head away',
  'Eye gaze away', 'Monitoring unavailable', 'Camera stream stale'];

export class SessionSummary {
  constructor(startedMs, startedAt = new Date().toISOString()) {
    this.startedMs = startedMs; this.startedAt = startedAt; this.finished = null;
    this.alertCounts = Object.fromEntries(ALERT_TYPES.map(type => [type, 0]));
    this.active = new Set(); this.blinks = 0; this.yawns = 0; this.closedAt = null;
    this.frames = 0; this.processingTotal = 0; this.lastFrame = null;
    this.intervalTotal = 0; this.intervals = 0; this.minimumFps = null;
  }
  observe({atMs, processingMs, alerts = [], eyesClosed = null, yawns = this.yawns}) {
    if (this.finished) return;
    if (!Number.isFinite(atMs) || !Number.isFinite(processingMs) || processingMs < 0) return;
    const interval = this.lastFrame === null ? null : atMs - this.lastFrame;
    if (interval !== null && interval > 0) {
      this.intervalTotal += interval; this.intervals++;
      this.minimumFps = Math.min(this.minimumFps ?? Infinity, 1000 / interval);
    }
    if (interval !== null && (interval > 750 || interval <= 0)) {
      this.closedAt = null; this.active.clear();
    }
    this.lastFrame = atMs; this.frames++; this.processingTotal += processingMs;
    this.countAlerts(alerts);
    if (eyesClosed === true) this.closedAt ??= atMs;
    else {
      if (eyesClosed === false && this.closedAt !== null) {
        const duration = atMs - this.closedAt;
        if (duration >= 60 && duration <= 450) this.blinks++;
      }
      this.closedAt = null;
    }
    if (Number.isFinite(yawns) && yawns >= 0) this.yawns = Math.max(this.yawns, yawns);
  }
  countAlerts(alerts) {
    const next = new Set(alerts.filter(type => Object.hasOwn(this.alertCounts, type)));
    for (const type of next) if (!this.active.has(type)) this.alertCounts[type]++;
    this.active = next;
  }
  finish(stoppedMs, reason, endedAt = new Date().toISOString()) {
    this.finished ??= {
      schemaVersion: 1, startedAt: this.startedAt, endedAt, stopReason: reason,
      durationSeconds: Math.max(0, stoppedMs - this.startedMs) / 1000,
      alertCounts: {...this.alertCounts}, blinkCount: this.blinks, yawnCount: this.yawns,
      processedFrames: this.frames,
      averageFps: this.intervals ? this.intervals * 1000 / this.intervalTotal : null,
      minimumFps: this.minimumFps,
      averageProcessingMs: this.frames ? this.processingTotal / this.frames : null,
      processingScope: 'Synchronous frame analysis, overlays and UI updates; excludes async expression inference, camera buffering and display latency.',
      alertCounting: 'One event per onset of each sustained alert, not per frame or sound repeat.',
    };
    return this.finished;
  }
}

export function serializeReport(summary, format) {
  if (!summary) throw new Error('Stop a monitoring session before saving a report.');
  if (format === 'json') return {type: 'application/json', extension: 'json', text: JSON.stringify(summary, null, 2)};
  if (format !== 'csv') throw new Error('Choose JSON or CSV.');
  const rows = [['metric', 'value'], ...Object.entries(summary).filter(([key]) => key !== 'alertCounts'),
    ...Object.entries(summary.alertCounts).map(([key, value]) => [`alert: ${key}`, value])];
  // Quoted text and neutralized formula prefixes keep CSV imports passive.
  const cell = value => {
    let text = value === null ? '' : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  return {type: 'text/csv', extension: 'csv', text: rows.map(row => row.map(cell).join(',')).join('\r\n')};
}
