#!/usr/bin/env python3
"""Run actual kernel control flow with synchronous CPU stand-ins.

This does not compile for an NPU or model CANN synchronization, hardware
rounding, or performance. It verifies partitioning, addressing, valid tail
selection, reduction order, output bounds, and dispatch.
NumPy and g++ are needed for the numerical model. The current implementation
also runs a Clang host address-space check before executing numerical cases.
"""
import argparse
import itertools
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np
from check_host_types import main as check_host_types

ROOT = Path(__file__).resolve().parents[1]


def cpu_source(source_path):
    source = source_path.read_text()
    # Submission code must remain free of debug output. This is our regression
    # check, not a claim that the platform's complete validation rules are known.
    assert not re.search(r'\b(?:printf|fprintf|puts|putchar|cout|cerr|DumpTensor)\b', source)
    assert '#include <cstdio>' not in source and '#include <stdio.h>' not in source
    source = re.sub(r'^#include "[^"]+"\n', '', source, flags=re.MULTILINE)
    launch = re.compile(r'(Baseline<[^>]+>)<<<([^,]+), nullptr, stream>>>\(([^;]+)\);')
    source, count = launch.subn(r'sim::Launch(\2, [&] { \1(\3); });', source)
    assert count == 8, f'unexpected launch count: {count}'
    source, count = re.subn(
        r'(Fused<[^>]+>)<<<([^,]+), nullptr, stream>>>\(([^;]+)\);',
        r'sim::LaunchMixed(\2, [&] { \1(\3); });', source)
    assert count == 2, f'unexpected Cube launch count: {count}'
    assert '<<<' not in source
    return '#include "sim_matmul.h"\n' + source


def quantize(values, dtype):
    values = values.astype(np.float32)
    if dtype == 1:
        data = values.astype(np.float16)
        return data.view(np.uint16), data.astype(np.float64)
    # BF16 round-to-nearest-even, matching conversion of these finite values.
    bits = values.view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    stored = bits.astype(np.uint16)
    return stored, (stored.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def cases():
    # Covers batch strides, N tails, K tails, full K tiles followed by a tail,
    # and M/N partition boundaries.
    shapes = [
        (1, 1, 1, 32), (3, 1, 1, 40), (2, 1, 1, 8192),
        (1, 2, 3, 32), (2, 15, 17, 40), (3, 17, 15, 72),
        (1, 33, 129, 64), (1, 65, 257, 40), (2, 127, 63, 32),
        (3, 31, 17, 256), (1, 1, 513, 40), (1, 2, 129, 32),
        (1, 129, 1, 40), (64, 1, 2, 32), (1, 1, 17, 8192),
        (2, 17, 33, 136), (1, 3, 19, 248), (1, 2, 17, 264),
        (1, 7, 15, 56), (1, 9, 16, 96), (1, 5, 17, 112), (1, 3, 33, 120),
        (1, 8192, 1, 32), (1, 65, 1, 40), (64, 3, 2, 32),
    ]
    for i, (shape, dtype, ta, tb) in enumerate(itertools.product(shapes, (1, 2), (False, True), (False, True))):
        yield shape, dtype, ta, tb, (1, 24, 32)[i % 3], 'random'
    # Negative-only rows expose zero initialization and zero-filled N tails.
    for dtype, ta, tb, cores in itertools.product((1, 2), (False, True), (False, True), (1, 24)):
        yield (2, 17, 131, 40), dtype, ta, tb, cores, 'negative'
    # N maxima at different columns for different M rows expose a wrong
    # sum(max(partial_scores)) or sum(partial_scores) merge of N partitions.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (1, 3, 257, 32), dtype, ta, tb, 24, 'partition'
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (2, 33, 19, 40), dtype, ta, tb, 24, 'zero'
        yield (1, 17, 1, 32), dtype, ta, tb, 1, 'cancellation'
        yield (1, 17, 1, 32), dtype, ta, tb, 24, 'cancellation'
    # Exceed the real production parallelism threshold, including multiple
    # adjacent batch outputs and M ranges not divisible by the group count.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (3, 129, 257, 40), dtype, ta, tb, 24, 'negative'
        yield (3, 129, 257, 40), dtype, ta, tb, 24, 'zero'
        yield (1, 129, 257, 32), dtype, ta, tb, 24, 'partition'
        yield (1, 513, 129, 32), dtype, ta, tb, 24, 'parallel-cancellation'


def extended_cases():
    """Long-dot precision and the actual 20-core partition boundaries."""
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        for cores in (1, 20):
            for mode in ('k-cancellation', 'mixed-magnitude', 'close-max'):
                yield (1, 9, 17, 8192), dtype, ta, tb, cores, mode
        for shape, cores in (
            ((3, 21, 129, 392), 7),
            ((3, 21, 129, 392), 19),
            ((3, 21, 129, 392), 20),
            ((7, 9, 257, 456), 20),
            ((1, 2, 4097, 128), 20),
            ((21, 2, 17, 32), 20),
        ):
            yield shape, dtype, ta, tb, cores, 'random'
    # Multi-row long K. (1, 9, 17, 8192) stays on Vector. (1, 32, 64, 8192) meets
    # the long-K Cube gate and packs every K panel of its single 64-column tile.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        for cores in (1, 20):
            for mode in ('k-cancellation', 'mixed-magnitude', 'close-max'):
                yield (1, 32, 64, 8192), dtype, ta, tb, cores, mode
            yield (1, 32, 65, 256), dtype, ta, tb, cores, 'negative'
            yield (1, 32, 64, 256), dtype, ta, tb, cores, 'random'
        yield (2, 40, 80, 256), dtype, ta, tb, 20, 'random'


def performance_cases():
    """Tile-size and A-row block boundaries with independent numerical goldens."""
    shapes = [(1, 15, 31, 128), (1, 16, 32, 128), (1, 17, 63, 40),
              (1, 31, 64, 120), (1, 33, 65, 128), (3, 65, 129, 128),
              (1, 64, 257, 128), (1, 129, 257, 128)]
    for i, (shape, dtype, ta, tb) in enumerate(itertools.product(shapes, (1, 2), (False, True), (False, True))):
        yield shape, dtype, ta, tb, (1, 20)[(i // 2) % 2], 'random'
    # Cross the actual Cube dispatch threshold, including M/N tails and worker reuse.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (1, 257, 513, 128), dtype, ta, tb, 20, 'random'
        yield (1, 1649, 257, 40), dtype, ta, tb, 20, 'negative'
        # Two short-K windows: N crosses 1024, including a short tail window.
        yield (1, 256, 2048, 32), dtype, ta, tb, 20, 'random'
        yield (1, 256, 1031, 64), dtype, ta, tb, 1, 'negative'
    # Short-K A cache holds at most 128 rows. 129 rows must reload A per N tile.
    # N=65 crosses both tile widths; K=64 stays below the new Cube work gate.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (1, 128, 65, 64), dtype, ta, tb, 1, 'random'
        yield (1, 129, 65, 64), dtype, ta, tb, 1, 'random'
        yield (1, 20, 33, 40), dtype, ta, tb, 1, 'negative'
        yield (1, 32, 33, 128), dtype, ta, tb, 20, 'negative'
        yield (1, 64, 257, 128), dtype, ta, tb, 20, 'close-max'


def cube_finish_cases():
    """Exercise the actual Cube dispatch and large-M compensated reduction."""
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (1, 1649, 257, 40), dtype, ta, tb, 20, 'm-cancellation'
        yield (1, 8191, 257, 40), dtype, ta, tb, 20, 'm-magnitude'
        # One core reuses Finish's UB across batches, with different row values.
        yield (3, 8192, 257, 40), dtype, ta, tb, 1, 'm-cancellation'


def short_cube_cases():
    """Short Cube coverage, N partitions, request windows and dispatch edges."""
    definitions = [
        ((1, 32, 256, 128), 'random'),   # Exactly 2^20; now Cube.
        ((1, 32, 255, 128), 'negative'), # Just below the work gate; Vector.
        ((1, 16, 512, 128), 'negative'), # Half of one base-M tile is valid.
        ((1, 15, 1025, 128), 'random'),  # M gate keeps this on Vector.
        ((1, 513, 63, 40), 'negative'),  # N gate keeps this on Vector.
        ((1, 32, 4097, 32), 'random'),   # One worker spans several 1024 windows.
        ((1, 33, 2049, 40), 'negative'), # Both M/N tails, uneven N partitions.
        ((1, 64, 513, 128), 'close-max'),
        ((8, 16, 65, 128), 'random'),    # Aggregate batch work crosses the gate.
        ((3, 65, 257, 40), 'random'),    # Uneven batch/M task count.
        ((1, 32, 1025, 32), 'partition'), # Different rows win in different N groups.
        ((3, 129, 257, 40), 'm-cancellation'),
    ]
    for (shape, mode), dtype, ta, tb, cores in itertools.product(
            definitions, (1, 2), (False, True), (False, True), (1, 20)):
        yield shape, dtype, ta, tb, cores, mode


def long_cube_cases():
    """Cross long-K dispatch, panel windows, reuse and compensation."""
    definitions = [((1, 65, 129, 8192), mode) for mode in
                   ('k-cancellation', 'mixed-magnitude', 'close-max', 'block-cancellation')]
    definitions += [((4, 65, 129, 256), 'random'), ((8, 33, 129, 392), 'random'),
                    # One core owns every 64-column tile of this N, including the K compensation.
                    ((1, 64, 512, 256), 'k-cancellation'),
                    ((1, 64, 520, 256), 'negative'),
                    ((1, 65, 129, 1024), 'random'), ((1, 33, 129, 8192), 'random'),
                    ((4, 65, 129, 392), 'negative'),
                    # (1, 16, 64, 4096) is the long-K Cube product edge, exactly 2^22.
                    # (1, 16, 80, 392) stays on Vector; its product is below that edge.
                    ((1, 16, 64, 4096), 'k-cancellation'), ((1, 16, 80, 392), 'negative')]
    for i, ((shape, mode), dtype, ta, tb) in enumerate(itertools.product(
            definitions, (1, 2), (False, True), (False, True))):
        yield shape, dtype, ta, tb, (1, 20)[i % 2], mode
    # At one core these cover 128/256-column windows, multiple windows, short
    # windows and strided C reads. At 20 cores N partitions may keep width 64.
    windows = [((1, 64, 129, 512), 'random'),
               ((1, 64, 257, 256), 'close-max'),
               ((1, 64, 385, 256), 'negative'),
               ((1, 65, 513, 136), 'random'),
               ((3, 65, 513, 136), 'partition'),
               ((1, 65, 1025, 136), 'm-cancellation')]
    for (shape, mode), dtype, ta, tb, cores in itertools.product(
            windows, (1, 2), (False, True), (False, True), (1, 20)):
        yield shape, dtype, ta, tb, cores, mode
    # All 64 K panels inside a wide request; cancellation remains inside a
    # hypothetical 512-K panel, so widening N must not widen Cube accumulation K.
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        yield (1, 65, 257, 8192), dtype, ta, tb, 1, 'block-cancellation'
        # Forty resident tasks at 20 cores each own four base-N tiles. Covers
        # the 256-column batch path at production parallelism, including M tail.
        yield (1, 257, 2048, 136), dtype, ta, tb, 20, 'random'
    for dtype, ta, tb, cores in itertools.product((1, 2), (False, True), (False, True), (1, 20)):
        # Both narrow and wide windows: a 2^30 panel, a 2^-5 panel,
        # and a -2^30 panel. Kahan loses the small panel in FP32.
        yield (1, 65, 129, 512), dtype, ta, tb, cores, 'panel-magnitude'
        # Multiple windows, uneven rows/columns and a final short K panel.
        yield (1, 65, 513, 520), dtype, ta, tb, cores, 'panel-magnitude-negative'


def make_inputs(shape, mode, rng):
    batch, m, n, k = shape
    a = rng.uniform(-1, 1, (batch, m, k))
    b = rng.uniform(-1, 1, (batch, k, n))
    if mode in ('m-cancellation', 'm-magnitude'):
        a.fill(0)
        b.fill(0)
        b[:, 0, :] = 1
        for bi in range(batch):
            values = a[bi, :, 0]
            if mode == 'm-cancellation':
                half = (m - 1) // 2
                values[:half] = 1 + bi
                values[half:2 * half] = -(1 + bi)
                values[-1] = (bi + 1) * 2 ** -10
            else:
                values[:] = np.ldexp(rng.uniform(-1, 1, m), rng.integers(-12, 12, m))
    elif mode == 'negative':
        a = np.abs(a) + .1
        b = -(np.abs(b) + .1)
    elif mode == 'zero':
        a.fill(0)
    elif mode == 'partition':
        a.fill(0)
        b.fill(0)
        for row, col in enumerate((0, 128, 256)):
            a[0, row, row] = 1
            b[0, row, :] = -1
            b[0, row, col] = row + 1
    elif mode == 'cancellation':
        a.fill(0)
        b.fill(0)
        a[0, :, 0] = np.array([1, -1] * 8 + [2 ** -10])
        b[0, 0, 0] = 1
    elif mode == 'parallel-cancellation':
        a.fill(0)
        b.fill(0)
        a[0, :-1, 0] = np.tile([1, -1], (shape[1] - 1) // 2)
        a[0, -1, 0] = 2 ** -10
        b[0, 0, :] = 1
    elif mode in ('k-cancellation', 'mixed-magnitude'):
        a.fill(1)
        b.fill(0)
        quarter = k // 4
        large = 1 if mode == 'k-cancellation' else 4096
        small = 2 ** -20 if mode == 'k-cancellation' else 2 ** -10
        b[:, :quarter, :] = large
        b[:, quarter:2 * quarter, :] = small
        b[:, 2 * quarter:3 * quarter, :] = -large
    elif mode == 'block-cancellation':
        a.fill(1)
        b.fill(0)
        for start in range(0, k, 512):
            b[:, start:start + 128, :] = 4096
            b[:, start + 128:start + 256, :] = 2 ** -10
            b[:, start + 256:start + 384, :] = -4096
    elif mode in ('panel-magnitude', 'panel-magnitude-negative'):
        a.fill(1024)
        b.fill(0)
        sign = 1 if mode == 'panel-magnitude' else -1
        b[:, :128, :] = sign * 8192
        b[:, 128:256, :] = sign * 2 ** -22
        b[:, 256:384, :] = -sign * 8192
    elif mode == 'close-max':
        a.fill(1)
        b.fill(0)
        b[:, 0, :] = 1
        b[:, 1, :] = np.arange(n) * 2 ** -12
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('all', 'correctness', 'extended', 'performance', 'cube-finish', 'short-cube', 'long-cube'), default='all')
    args = parser.parse_args()
    source_path = ROOT / 'kernel.asc'
    check_host_types()
    with tempfile.TemporaryDirectory(prefix='bmmms-cpu-') as temp:
        temp = Path(temp)
        (temp / 'kernel_cpu.inc').write_text(cpu_source(source_path))
        executable = temp / 'sim_runner'
        subprocess.run([
            'g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-Wno-unused-parameter', '-ffp-contract=off',
            '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
            '-pthread',
            '-I', str(ROOT / 'tests'), '-I', str(temp),
            str(ROOT / 'tests/sim_runner.cpp'), '-o', str(executable),
        ], check=True)
        specs = list(cases()) if args.suite in ('all', 'correctness') else []
        if args.suite in ('all', 'extended'):
            specs.extend(extended_cases())
        if args.suite in ('all', 'performance'):
            specs.extend(performance_cases())
        if args.suite == 'cube-finish':
            specs.extend(cube_finish_cases())
        if args.suite in ('all', 'short-cube'):
            specs.extend(short_cube_cases())
        if args.suite in ('all', 'long-cube'):
            specs.extend(long_cube_cases())
        payload = bytearray(struct.pack('<I', len(specs)))
        expected = []
        rng = np.random.default_rng(20261002)
        for index, (shape, dtype, ta, tb, cores, mode) in enumerate(specs):
            a, b = make_inputs(shape, mode, rng)
            a, logical_a = quantize(a, dtype)
            b, logical_b = quantize(b, dtype)
            golden = (logical_a @ logical_b).max(axis=-1).sum(axis=-1).astype(np.float32)
            physical_a = a.swapaxes(-1, -2) if ta else a
            physical_b = b.swapaxes(-1, -2) if tb else b
            payload += struct.pack('<9I', *shape, dtype, ta, tb, cores, index % 2)
            payload += physical_a.tobytes(order='C') + physical_b.tobytes(order='C')
            expected.append(golden)
        # LeakSanitizer cannot run under this sandbox's ptrace supervision.
        # AddressSanitizer and UBSan remain enabled.
        env = dict(os.environ)
        env['ASAN_OPTIONS'] = env.get('ASAN_OPTIONS', '') + ':detect_leaks=0'
        result = subprocess.run([str(executable)], input=payload, capture_output=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        if os.environ.get('BMMMS_REQUEST_LOG'):
            print(result.stderr.decode(errors='replace'), end='')
        values = np.frombuffer(result.stdout, dtype=np.float32)
        assert len(values) == sum(s[0][0] for s in specs)
        offset = 0
        worst_error = worst_scaled = 0.0
        for spec, golden in zip(specs, expected):
            actual = values[offset:offset + len(golden)]
            offset += len(golden)
            error = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
            scaled = error / (1e-4 + 1e-4 * np.abs(golden.astype(np.float64)))
            worst_error = max(worst_error, float(error.max()))
            worst_scaled = max(worst_scaled, float(scaled.max()))
            if not np.all(np.isfinite(actual)) or not np.all(scaled <= 1):
                raise AssertionError(f'{spec}: actual={actual}, golden={golden}, error={error}')
        print(f'PASS: {len(specs)} CPU cases, FP16/BF16, all four storage layouts')
        print('PASS: threaded collective execution')
        print('PASS: repeatability, input immutability, output guards')
        print('PASS: exactly one simulated kernel launch per invocation in every case')
        print('PASS: AddressSanitizer/UBSan, local initialization/alignment/DMA-position checks')
        print(f'Worst absolute error: {worst_error:.8g}; max error/tolerance: {worst_scaled:.6g}')
        print('CPU model does not validate graph capture, NPU synchronization, performance or the webpage judge.')


if __name__ == '__main__':
    main()
