"""Explicit one-time download of the official Google Face Landmarker model."""
from pathlib import Path
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def main():
    target = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'
    target.parent.mkdir(exist_ok=True)
    if target.is_file():
        print(f'Model already exists: {target}')
        return
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=90) as response, temporary.open('wb') as file:
            import shutil
            shutil.copyfileobj(response, file)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError('Model download is unexpectedly small')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Downloaded official model: {target}')


if __name__ == '__main__':
    main()
