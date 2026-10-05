#!/usr/bin/env python3
"""Prepare reproducible storage values and independent FP64 NPU test goldens."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct

import numpy as np
import sys
sys.dont_write_bytecode = True

from test_cpu import cases, extended_cases, performance_cases, cube_finish_cases, short_cube_cases, long_cube_cases, make_inputs, quantize
from case_catalog import BENCHMARK_SUITES, benchmark_specs, coverage, input_seed, robustness_specs, validate_shape

SUITES = ('baseline', 'native', 'smoke', 'correctness', 'extended', 'performance',
          'cube-finish', 'short-cube', 'long-cube', 'launch-rule', 'precision',
          'benchmark', 'large-short-k', 'stress', 'robustness') + BENCHMARK_SUITES


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def specs(suite):
    if suite not in SUITES:
        raise ValueError(f'unknown suite: {suite}')
    if suite in BENCHMARK_SUITES:
        yield from benchmark_specs(suite)
        return
    if suite == 'robustness':
        yield from robustness_specs()
        return
    if suite == 'baseline':
        shapes = ((3,1,1,32),(3,1,1,40),(3,1,1,64),(3,1,1,72),
                  (3,33,17,40),(64,3,2,32),(1,65,129,520),(1,8191,129,32))
        for shape, dtype, ta, tb in itertools.product(shapes,(1,2),(False,True),(False,True)):
            yield shape,dtype,ta,tb,0,'random'
        return
    if suite == 'native':
        # Current Cube/Vector boundaries, full-K tails and multi-chunk Finish.
        shapes = ((3, 31, 32, 128), (2, 32, 31, 72), (64, 3, 2, 32),
                  (1, 65, 513, 520), (1, 65, 129, 8192), (1, 8192, 257, 40),
                  (3, 17, 33, 520), (1, 128, 256, 512), (8, 33, 65, 1032),
                  (1, 129, 256, 520), (1, 65, 257, 512),
                  (1, 32, 32, 2048), (2, 17, 63, 1024), (1, 31, 17, 512))
        for shape, dtype, ta, tb in itertools.product(shapes, (1, 2), (False, True), (False, True)):
            yield shape, dtype, ta, tb, 0, 'random'
        for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
            yield (1, 9, 17, 520), dtype, ta, tb, 1, 'random'
        for mode, cores in itertools.product(('negative', 'zero'), (1, 0)):
            yield (2, 17, 33, 520), 2, True, True, cores, mode
        for mode in ('negative', 'zero'):
            yield (2, 32, 31, 72), 2, True, True, 1, mode
        for mode in ('m-magnitude', 'm-cancellation'):
            yield (1, 8191, 257, 40), 2, True, True, 0, mode
        yield (1, 1, 1, 8192), 1, False, False, 0, 'random'
        return
    if suite == 'launch-rule':
        # Cover all current dispatches and exactly one task per iteration.
        # These are our diagnostic inputs, not the unknown contest shapes.
        shapes = [(3,1,1,32),(3,1,1,40),(3,1,1,64),(3,1,1,72),
                  (3,33,17,40),(64,3,2,32),(1,65,129,520),(1,8191,129,32),
                  (1,1,257,512),(2,64,129,136)]
        for i, shape in enumerate(shapes):
            yield shape, 1 + i % 2, bool(i % 2), bool(i % 3), 0, 'random'
        return
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
    if suite == 'short-cube':
        yield from short_cube_cases()
        return
    if suite == 'long-cube':
        yield from long_cube_cases()
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


def generate(suite, prefix, seed=20261002):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    definitions = list(specs(suite))
    outputs, metadata = [], []
    with prefix.with_suffix('.bin').open('wb') as out:
        out.write(struct.pack('<I', len(definitions)))
        for index, (shape, dtype, ta, tb, cores, mode) in enumerate(definitions):
            validate_shape(shape)
            case_seed = input_seed(shape, mode, seed)
            rng = np.random.default_rng(case_seed)
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
            metadata.append(dict(index=index, shape=shape, dtype=dtype, ta=ta, tb=tb,
                                 cores=cores, mode=mode, input_seed=case_seed,
                                 flops=2*shape[0]*shape[1]*shape[2]*shape[3],
                                 input_bytes=2*shape[0]*shape[3]*(shape[1]+shape[2]),
                                 **(coverage(shape) if suite in BENCHMARK_SUITES else dict(family='diagnostic', split=suite))))
    np.concatenate(outputs).tofile(prefix.with_suffix('.golden.bin'))
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = dict(suite=suite, seed=seed, cases=len(metadata),
                    geometries=len({tuple(s['shape']) for s in metadata}),
                    input_sha256=sha256(prefix.with_suffix('.bin')),
                    golden_sha256=sha256(prefix.with_suffix('.golden.bin')),
                    metadata_sha256=sha256(prefix.with_suffix('.json')),
                    golden_method='quantized storage -> FP64 dot/max/sum -> FP32')
    prefix.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(definitions)} {suite} cases: {prefix.with_suffix(".bin")}')


def verify(prefix, output, quiet=False):
    prefix = Path(prefix)
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    golden = np.fromfile(prefix.with_suffix('.golden.bin'), dtype=np.float32)
    actual = np.fromfile(output, dtype=np.float32)
    if golden.size != sum(s['shape'][0] for s in metadata) or not np.all(np.isfinite(golden)):
        raise RuntimeError('invalid or incomplete golden')
    manifest_path = prefix.with_suffix('.manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for suffix, key in (('.bin', 'input_sha256'), ('.json', 'metadata_sha256'), ('.golden.bin', 'golden_sha256')):
            if sha256(prefix.with_suffix(suffix)) != manifest[key]:
                raise RuntimeError(f'test data hash mismatch: {suffix}')
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
    report = dict(cases=len(metadata), passed=len(metadata) - len(failures), worst_error_over_tolerance=worst, failures=failures,
                  input_sha256=sha256(prefix.with_suffix('.bin')), output_sha256=sha256(output),
                  golden_sha256=sha256(prefix.with_suffix('.golden.bin')))
    Path(output).with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not quiet:
        print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=SUITES, default='smoke')
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--verify', help='verify device output instead of generating inputs')
    args = parser.parse_args()
    if args.verify:
        verify(args.prefix, args.verify)
    else:
        generate(args.suite, args.prefix, args.seed)
