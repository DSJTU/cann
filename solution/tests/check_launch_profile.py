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


def dispatch(shape, cores=20, tb=False, ta=False):
    """Expected entry for the diagnostic device's available Cube count."""
    _, m, n, k = shape
    if m == n == 1:
        return 'bmmms_static_dot_kernel' if k <= 64 else 'bmmms_dot_kernel'
    n_pad, groups = (n+7)//8*8, (k+63)//64
    fixed_bytes = ((m*k+15)//16 + (n*k+15)//16)*32 + 32 + (
        m*k + n*k + max(m*k,n*k) + 2*k + (2*m+7)//8*8)*4
    if (m <= 64 and n <= 64 and k <= 1024 and m*k <= 8192 and n*k <= 8192
            and m*n*k <= 131072 and n_pad*k <= 16384
            and fixed_bytes + n_pad*(k+groups)*4 <= 180*1024):
        return 'bmmms_small_kernel'
    return 'fused_kernel'


def verify(path, metadata, repeats):
    rows = records(path)
    expected = len(metadata) * repeats
    if len(rows) != expected:
        raise ValueError(f'each iteration must launch exactly 1 kernel: expected {expected}, got {len(rows)}')
    for i, spec in enumerate(metadata):
        name = dispatch(spec['shape'], spec.get('cores', 0) or 20, spec.get('tb', False), spec.get('ta', False))
        if any(name not in row['Op Name'] for row in rows[i*repeats:(i+1)*repeats]):
            raise ValueError('unexpected dispatch or per-case launch count')
    return rows
