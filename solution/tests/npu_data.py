#!/usr/bin/env python3
"""Prepare reproducible storage values and independent FP64 NPU test goldens."""
import argparse
import itertools
import json
from pathlib import Path
import struct

import numpy as np
from test_cpu import cases, extended_cases, performance_cases, cube_finish_cases, make_inputs, quantize


def specs(suite):
    if suite == 'smoke':
        for dtype, ta, tb, cores in itertools.product((1, 2), (False, True), (False, True), (1, 0)):
            for shape in ((1, 1, 1, 32), (2, 17, 19, 40)):
                yield shape, dtype, ta, tb, cores, 'random'
        return
    if suite == 'correctness':
        yield from cases()
        return
    if suite == 'extended':
        yield from extended_cases()
        return
    if suite == 'performance':
        yield from performance_cases()
        return
    if suite == 'cube-finish':
        yield from cube_finish_cases()
        return
    if suite == 'precision':
        for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
            yield (1, 1, 1, 8192), dtype, ta, tb, 1, 'k-cancellation'
        return
    if suite == 'benchmark':
        shapes = [(1, 64, 257, 128), (1, 129, 257, 136), (1, 9, 17, 8192),
                  (3, 21, 129, 392), (8, 33, 65, 128), (64, 3, 17, 32)]
        for shape, dtype, ta, tb in itertools.product(shapes, (1, 2), (False, True), (False, True)):
            yield shape, dtype, ta, tb, 0, 'random'
        return
    shapes = [(1, 512, 1024, 128), (1, 1024, 4096, 128), (1, 8192, 257, 40),
              (1, 129, 1024, 8192), (8, 129, 257, 136), (64, 17, 33, 64)]
    if suite == 'large-short-k':
        shapes = shapes[:3]
    for shape, dtype, ta, tb in itertools.product(shapes, (1, 2), (False, True), (False, True)):
        yield shape, dtype, ta, tb, 0, 'random'


def golden_rows(a, b):
    batch, m, _ = a.shape
    result = np.empty(batch, dtype=np.float32)
    for bi in range(batch):
        maxima = np.empty(m, dtype=np.float64)
        for begin in range(0, m, 64):
            maxima[begin:begin + 64] = (a[bi, begin:begin + 64] @ b[bi]).max(axis=-1)
        result[bi] = maxima.sum(dtype=np.float64)
    return result


def generate(suite, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    definitions = list(specs(suite))
    rng = np.random.default_rng(20261002)
    outputs, metadata = [], []
    with prefix.with_suffix('.bin').open('wb') as out:
        out.write(struct.pack('<I', len(definitions)))
        for index, (shape, dtype, ta, tb, cores, mode) in enumerate(definitions):
            a, b = make_inputs(shape, mode, rng)
            stored_a, a = quantize(a, dtype)
            stored_b, b = quantize(b, dtype)
            golden = golden_rows(a, b)
            outputs.append(golden)
            pa = stored_a.swapaxes(-1, -2) if ta else stored_a
            pb = stored_b.swapaxes(-1, -2) if tb else stored_b
            out.write(struct.pack('<9I', *shape, dtype, ta, tb, cores, index % 2))
            out.write(pa.tobytes(order='C'))
            out.write(pb.tobytes(order='C'))
            metadata.append(dict(index=index, shape=shape, dtype=dtype, ta=ta, tb=tb, cores=cores, mode=mode))
    np.concatenate(outputs).tofile(prefix.with_suffix('.golden.bin'))
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared {len(definitions)} {suite} cases: {prefix.with_suffix(".bin")}')


def verify(prefix, output):
    prefix = Path(prefix)
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    golden = np.fromfile(prefix.with_suffix('.golden.bin'), dtype=np.float32)
    actual = np.fromfile(output, dtype=np.float32)
    if actual.shape != golden.shape:
        raise RuntimeError(f'incomplete output: {actual.size} / {golden.size} values')
    offset, failures, worst = 0, [], 0.0
    for spec in metadata:
        count = spec['shape'][0]
        x, y = actual[offset:offset + count], golden[offset:offset + count]
        offset += count
        error = np.abs(x.astype(np.float64) - y.astype(np.float64))
        scaled = error / (1e-4 + 1e-4 * np.abs(y.astype(np.float64)))
        worst = max(worst, float(scaled.max()))
        if not np.all(np.isfinite(x)) or not np.all(scaled <= 1):
            failures.append(dict(spec=spec, actual=x.tolist(), golden=y.tolist(), scaled=scaled.tolist()))
    report = dict(cases=len(metadata), passed=len(metadata) - len(failures), worst_error_over_tolerance=worst, failures=failures)
    Path(output).with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('smoke', 'correctness', 'extended', 'performance', 'cube-finish', 'precision', 'benchmark', 'large-short-k', 'stress'), default='smoke')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--verify', help='verify device output instead of generating inputs')
    args = parser.parse_args()
    if args.verify:
        verify(args.prefix, args.verify)
    else:
        generate(args.suite, args.prefix)
