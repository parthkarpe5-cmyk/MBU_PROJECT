"""Three deterministic PCM patterns. Async Windows playback never stalls capture."""
import io
import math
import struct
import wave

PATTERNS = {'distraction': (660, 880), 'fatigue': (1040, 780, 1040), 'availability': (440, 330)}


def wav_bytes(name, volume=.15):
    if name not in PATTERNS or not 0 <= volume <= 1:
        raise ValueError('Unknown sound or volume outside 0..1')
    rate, samples = 16000, []
    for frequency in PATTERNS[name]:
        for i in range(2400):
            envelope = min(1., i/160, (2399-i)/160)
            samples.append(int(32767*volume*envelope*math.sin(2*math.pi*frequency*i/rate)))
        samples.extend([0]*1200)
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack('<'+'h'*len(samples), *samples))
    return output.getvalue()


class SoundPlayer:
    def __init__(self):
        import tempfile
        from pathlib import Path
        self.directory = tempfile.TemporaryDirectory(prefix='driveguard-sounds-')
        self.paths = {}
        self.last = {}
        for name in PATTERNS:
            path = Path(self.directory.name) / (name+'.wav')
            path.write_bytes(wav_bytes(name))
            self.paths[name] = path

    def play(self, event, now, mapping=None, preview=False):
        import winsound
        name = (mapping or {}).get(event, event)
        if name == 'Mute' or name not in self.paths:
            return
        if not preview and now-self.last.get(event, -math.inf) < 3:
            return
        winsound.PlaySound(str(self.paths[name]), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.last[event] = now

    def close(self):
        import winsound
        winsound.PlaySound(None, 0)
        self.directory.cleanup()
