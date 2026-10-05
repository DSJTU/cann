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
             'direct_cube_kernel', 'packed_cube_kernel')
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
    b, m, n, k = shape
    ta, tb = ta and m != 1, tb or n == 1
    if m == n == 1:
        return 'bmmms_dot_kernel'
    if (m*n <= 256 or b > cores) and m <= 32 and n <= 64 and k <= 256 and n*k <= 8192 and m*n*k <= 65536:
        return 'bmmms_small_kernel'
    if k >= 4096 and m <= 128 and n <= 256 and b <= 8 and not (not ta and tb and m*n <= 512) and b*((m+15)//16) <= cores:
        return 'packed_cube_kernel'
    if k <= 256 and not (m >= 512 and n >= 1024) and not (tb and m <= 128 and n > 256 and k >= 128):
        tm, tn = min(128, (m+15)//16*16), min(128, (n+15)//16*16)
        while not (m <= 128 and n <= 128) and b*((m+tm-1)//tm)*((n+tn-1)//tn) < cores and (tm > 16 or tn > 16):
            if tm > 16 and (tn == 16 or tm >= tn):
                tm = (tm//2+15)//16*16
            else:
                tn = (tn//2+15)//16*16
        if b*((m+tm-1)//tm) <= cores:
            return 'direct_cube_kernel'
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
