"""Small immutable artifact helpers; no implicit resume/overwrite."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def checked_json(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('JSON artifact hash mismatch')
    return json.loads(data)


def write_json_once(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        out.write(payload)


def decoded_sha256(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
