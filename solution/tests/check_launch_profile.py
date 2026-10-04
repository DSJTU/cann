"""Verify one device task per invocation, including MIX and Vector paths."""
import csv
import math
from pathlib import Path


def records(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('empty kernel profile')
    names = ('fused_kernel', 'bmmms_small_kernel', 'bmmms_dot_kernel')
    for row in rows:
        if not any(name in row['Op Name'] for name in names):
            raise ValueError('unexpected kernel in profile: ' + row['Op Name'])
        for key in ('Task Start Time(us)', 'Task Duration(us)'):
            if not math.isfinite(float(row[key])):
                raise ValueError('nonfinite kernel timing')
        if float(row['Task Duration(us)']) <= 0:
            raise ValueError('nonpositive kernel duration')
    return sorted(rows, key=lambda row: float(row['Task Start Time(us)']))


def dispatch(shape):
    """Map the three shape-based dispatches to their device kernel names."""
    _, m, n, k = shape
    if m == n == 1:
        return 'bmmms_dot_kernel'
    if m <= 32 and n <= 64 and k <= 256 and n*k <= 8192 and m*n*k <= 65536:
        return 'bmmms_small_kernel'
    return 'fused_kernel'


def verify(path, metadata, repeats):
    rows = records(path)
    expected = len(metadata) * repeats
    if len(rows) != expected:
        raise ValueError(f'each iteration must launch exactly 1 kernel: expected {expected}, got {len(rows)}')
    for i, spec in enumerate(metadata):
        name = dispatch(spec['shape'])
        if any(name not in row['Op Name'] for row in rows[i*repeats:(i+1)*repeats]):
            raise ValueError('unexpected dispatch or per-case launch count')
    return rows
