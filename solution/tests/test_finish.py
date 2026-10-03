#!/usr/bin/env python3
"""Check the actual Finish kernel against math.fsum; CPU model only."""
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from test_cpu import ROOT, cpu_source


def main():
    rng = np.random.default_rng(8241)
    specs, expected = [], []
    for m in (1, 2, 3, 7, 8, 9, 17, 31, 32, 33, 127, 128, 129,
              257, 513, 1649, 4095, 4096, 4097, 8191, 8192):
        for batch in (1, 3, 21):
            for mode in ('random', 'negative', 'cancellation', 'magnitude'):
                values = rng.uniform(-100, 100, (batch, m)).astype(np.float32)
                if mode == 'negative':
                    values = -np.abs(values)
                elif mode == 'cancellation':
                    values.fill(0)
                    values[:, :m // 2] = 1
                    values[:, m // 2:2 * (m // 2)] = -1
                    values[:, -1] = 2 ** -10
                elif mode == 'magnitude':
                    values = np.ldexp(rng.uniform(-1, 1, (batch, m)),
                                      rng.integers(-40, 30, (batch, m))).astype(np.float32)
                for groups in (1, 2, 7):
                    # Each row's winner belongs to a different N partition.
                    # Preserve the adversarial M sum while exercising max first.
                    partitions = np.broadcast_to(values[:, None, :], (batch, groups, m)).copy()
                    partitions -= rng.uniform(1, 100, partitions.shape).astype(np.float32)
                    for row in range(m):
                        partitions[:, row % groups, row] = values[:, row]
                    maxima = partitions.max(axis=1)
                    specs.append((batch, m, min(batch, 20), groups, partitions))
                    expected.extend(np.float32(math.fsum(map(float, row))) for row in maxima)
    payload = bytearray(struct.pack('<I', len(specs)))
    for batch, m, cores, groups, values in specs:
        payload += struct.pack('<4I', batch, m, cores, groups) + values.tobytes()
    with tempfile.TemporaryDirectory(prefix='bmmms-finish-') as tmp:
        tmp = Path(tmp)
        (tmp / 'kernel_cpu.inc').write_text(cpu_source(ROOT / 'kernel.asc'))
        executable = tmp / 'finish_runner'
        subprocess.run([
            'g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
            '-Wno-unused-parameter', '-ffp-contract=off', '-fsanitize=address,undefined',
            '-fno-omit-frame-pointer', '-pthread', '-I', str(ROOT / 'tests'), '-I', str(tmp),
            str(ROOT / 'tests/finish_runner.cpp'), '-o', str(executable),
        ], check=True)
        env = dict(os.environ)
        env['ASAN_OPTIONS'] = env.get('ASAN_OPTIONS', '') + ':detect_leaks=0'
        result = subprocess.run([str(executable)], input=payload, capture_output=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
    actual = np.frombuffer(result.stdout, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape, 'incomplete Finish output'
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scaled = error / (1e-4 + 1e-4 * np.abs(expected.astype(np.float64)))
    assert np.all(np.isfinite(actual)) and np.all(scaled <= 1), f'Finish error/tolerance={scaled.max()}'
    print(f'PASS: {len(specs)} Finish cases, {actual.size} outputs, math.fsum golden')
    print('PASS: tails, batch reuse, cancellation, magnitudes, repeatability, inputs, guards, ASan/UBSan')
    print(f'Worst absolute error: {error.max():.8g}; max error/tolerance: {scaled.max():.8g}')


if __name__ == '__main__':
    main()
