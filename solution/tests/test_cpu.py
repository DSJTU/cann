#!/usr/bin/env python3
"""Run actual kernel control flow with synchronous CPU stand-ins.

This does not compile for an NPU or model CANN synchronization, internal Matmul
transfers, hardware rounding, or performance. It verifies partitioning,
addressing, valid tail selection, reduction order, output bounds, and dispatch.
NumPy and g++ are needed for the numerical model. The current implementation
also runs a Clang host address-space check before executing numerical cases.
"""
import itertools
import os
import argparse
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
    if source_path.name == 'vector_v4.asc':
        # Preserve the device-tested boundary in this isolated experiment.
        baseline = (ROOT / 'experiments/vector_v2.asc').read_text()
        start = 'extern "C" void run_kernel('
        assert source[source.index(start):] == baseline[baseline.index(start):], 'host dispatch changed from v2'
        signature = re.compile(r'__global__ __vector__ void Baseline\([^{]+')
        assert signature.search(source).group() == signature.search(baseline).group(), 'kernel parameters changed from v2'
    if source_path.name == 'kernel.asc':
        baseline = (ROOT / 'experiments/vector_v4.asc').read_text()
        # Undo exactly the three intentional edits. The entire rest of both
        # host and device source must match the known partial judge success.
        restored = source.replace('uint32_t k, uint32_t groups)', 'uint32_t k)')
        restored = restored.replace('    batch *= groups;\n', '')
        restored = restored.replace('(x1, x2, y, batch, m, n, k, 1u);',
                                    '(x1, x2, y, batch, m, n, k);')
        def normalized(text):
            return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', text))
        assert normalized(restored) == normalized(baseline), 'v8 changed more than its extra scalar argument'
        assert source.count('batch *= groups;') == 1
        assert source.count('(x1, x2, y, batch, m, n, k, 1u);') == 8
    source = re.sub(r'^#include "[^"]+"\n', '', source, flags=re.MULTILINE)
    launch = re.compile(r'(Baseline<[^>]+>|MaxSim<[^>]+>|Dot<[^>]+>|Finish)<<<([^,]+), nullptr, stream>>>\(([^;]+)\);')
    source, count = launch.subn(r'sim::Launch(\2, [&] { \1(\3); });', source)
    expected = {'kernel.asc': 8, 'vector_v2.asc': 8, 'vector_v3.asc': 9,
                'vector_v4.asc': 8, 'vector_v7.asc': 9, 'fused_v1.asc': 7}
    assert count == expected[source_path.name], f"unexpected launch count: {count}"
    assert '<<<' not in source
    return source


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
    # and the archived implementation's M/N partition boundaries.
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


def make_inputs(shape, mode, rng):
    batch, m, n, k = shape
    a = rng.uniform(-1, 1, (batch, m, k))
    b = rng.uniform(-1, 1, (batch, k, n))
    if mode == 'negative':
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
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused-v1', action='store_true', help='check archived fused implementation')
    parser.add_argument('--vector-v2', action='store_true', help='check archived batch-only vector implementation')
    parser.add_argument('--vector-v3', action='store_true', help='check archived M-parallel vector implementation')
    parser.add_argument('--vector-v4', action='store_true', help='check archived device-tested batched dot products')
    parser.add_argument('--vector-v7', action='store_true', help='check archived graph-compatible M-parallel implementation')
    parser.add_argument('--fail-allocation', action='store_true', help='force workspace allocation failure in the CPU model')
    args = parser.parse_args()
    if sum((args.fused_v1, args.vector_v2, args.vector_v3, args.vector_v4, args.vector_v7)) > 1:
        parser.error('select only one archived implementation')
    source_path = (ROOT / 'experiments/fused_v1.asc' if args.fused_v1 else
                   ROOT / 'experiments/vector_v2.asc' if args.vector_v2 else
                   ROOT / 'experiments/vector_v3.asc' if args.vector_v3 else
                   ROOT / 'experiments/vector_v4.asc' if args.vector_v4 else
                   ROOT / 'experiments/vector_v7.asc' if args.vector_v7 else ROOT / 'kernel.asc')
    current = source_path.name == 'kernel.asc'
    if args.fail_allocation and not args.vector_v7:
        parser.error('--fail-allocation applies only to --vector-v7; current v8 does not allocate workspace')
    if current:
        check_host_types()
    with tempfile.TemporaryDirectory(prefix='bmmms-cpu-') as temp:
        temp = Path(temp)
        (temp / 'kernel_cpu.inc').write_text(cpu_source(source_path))
        executable = temp / 'sim_runner'
        subprocess.run([
            'g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
            '-Wno-unused-parameter', '-ffp-contract=off',
            '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
            *(['-DSIM_FUSED_V1'] if args.fused_v1 else []),
            *(['-DSIM_VECTOR_V3'] if args.vector_v3 else []),
            *(['-DSIM_VECTOR_V5'] if args.vector_v7 else []),
            *(['-DSIM_FAIL_ALLOCATION'] if args.fail_allocation else []),
            '-I', str(ROOT / 'tests'), '-I', str(temp),
            str(ROOT / 'tests/sim_runner.cpp'), '-o', str(executable),
        ], check=True)
        specs = list(cases())
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
        if args.fused_v1:
            print('PASS: 4608 launch-plan boundary combinations')
        elif args.vector_v3 or args.vector_v7:
            print('PASS: 4608 M-partition boundary combinations')
        elif args.vector_v4:
            print('PASS: host dispatch and kernel signature exactly match device-tested v2')
        if current:
            print('PASS: entire source matches v4 except the extra scalar argument, use, and eight launch arguments')
        if args.fail_allocation:
            print('PASS: forced allocation failure falls back to batch-only NPU control flow')
        print('PASS: repeatability, input immutability, output guards')
        print('PASS: AddressSanitizer/UBSan, local initialization/alignment/DMA-position checks')
        print(f'Worst absolute error: {worst_error:.8g}; max error/tolerance: {worst_scaled:.6g}')
        print('CPU model does not validate graph capture, NPU synchronization, performance or the webpage judge.')


if __name__ == '__main__':
    main()
