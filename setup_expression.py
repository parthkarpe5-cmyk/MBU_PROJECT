"""Explicit FER+ download, verified against the upstream Git LFS SHA-256."""
import hashlib
from pathlib import Path
import re
import urllib.request

BASE = 'https://raw.githubusercontent.com/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'
URL = BASE.replace('raw.githubusercontent.com', 'media.githubusercontent.com/media')


def main():
    # The small upstream LFS pointer provides the expected digest and byte count.
    pointer = urllib.request.urlopen(BASE, timeout=60).read().decode('utf-8')
    digest = re.search(r'oid sha256:([a-f0-9]{64})', pointer)
    size = re.search(r'size (\d+)', pointer)
    if not digest or not size:
        raise RuntimeError('Cannot establish upstream model integrity')
    target = Path(__file__).resolve().parent / 'models' / 'emotion-ferplus-8.onnx'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=120) as source, temporary.open('wb') as output:
            import shutil
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != int(size[1]) or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest[1]:
            raise RuntimeError('FER+ model integrity check failed')
        temporary.replace(target)
        print(f'FER+ verified: {digest[1]} ({target.stat().st_size} bytes)')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
