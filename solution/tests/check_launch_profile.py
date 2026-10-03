"""Verify one device task per invocation, including MIX and Vector paths."""
import csv
import math
from pathlib import Path


def records(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('empty kernel profile')
    for row in rows:
        if not any(name in row['Op Name'] for name in ('Baseline', 'Fused', 'Scores', 'Finish')):
            raise ValueError('unexpected kernel in profile: ' + row['Op Name'])
        for key in ('Task Start Time(us)', 'Task Duration(us)'):
            if not math.isfinite(float(row[key])):
                raise ValueError('nonfinite kernel timing')
        if float(row['Task Duration(us)']) <= 0:
            raise ValueError('nonpositive kernel duration')
    return sorted(rows, key=lambda row: float(row['Task Start Time(us)']))


def verify(path, metadata, repeats):
    rows = records(path)
    expected = len(metadata) * repeats
    if len(rows) != expected:
        raise ValueError(f'each iteration must launch exactly 1 kernel: expected {expected}, got {len(rows)}')
    for i, spec in enumerate(metadata):
        b, m, n, k = spec['shape']
        cube = ((k <= 128 and m >= 256 and n >= 256 and m*n*k >= 1 << 24) or
                (k > 128 and m >= 32 and n >= 128 and b*m*n*k >= 1 << 23))
        name = 'Fused' if cube else 'Baseline'
        if any(name not in row['Op Name'] for row in rows[i*repeats:(i+1)*repeats]):
            raise ValueError('unexpected dispatch or per-case launch count')
    return rows
