"""Verify one device task per invocation, including MIX and Vector paths."""
import csv
import math
from pathlib import Path


def records(path):
    with Path(path).open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('empty kernel profile')
    names = ('fused_kernel', 'bmmms_small_kernel', 'bmmms_dot_kernel',
             'bmmms_static_dot_kernel')
    for row in rows:
        if not any(name in row['Op Name'] for name in names):
            raise ValueError('unexpected kernel in profile: ' + row['Op Name'])
        for key in ('Task Start Time(us)', 'Task Duration(us)'):
            if not math.isfinite(float(row[key])):
                raise ValueError('nonfinite kernel timing')
        if float(row['Task Duration(us)']) <= 0:
            raise ValueError('nonpositive kernel duration')
    return sorted(rows, key=lambda row: float(row['Task Start Time(us)']))


def verify(path, metadata, repeats):
    if not metadata or repeats < 1:
        raise ValueError('empty launch check')
    rows = records(path)
    expected = len(metadata) * repeats
    if len(rows) != expected:
        raise ValueError(f'each iteration must launch exactly 1 kernel: expected {expected}, got {len(rows)}')
    # The contract constrains launch count, not the reference's thresholds.
    # A candidate may legitimately choose Cube instead of Vector or vice versa.
    for i in range(len(metadata)):
        if len({row['Op Name'] for row in rows[i*repeats:(i+1)*repeats]}) != 1:
            raise ValueError('inconsistent dispatch or per-case launch count')
    return rows
