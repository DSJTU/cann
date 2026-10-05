"""Geometry coverage independent of the implementation's dispatch thresholds.

These synthetic shapes are not the judge's 15 hidden test points. Tune on
benchmark-tune; use benchmark-holdout only to check a completed candidate.
Each family has equal weight, each geometry has eight dtype/layout variants.
"""
import hashlib
import itertools
import json


# (B, M, N, K). Keep geometry disjoint between the two sets.
GEOMETRIES = {
    'dot': (
        ((1, 1, 1, 32), (8, 1, 1, 64), (64, 1, 1, 72), (3, 1, 1, 8192)),
        ((7, 1, 1, 40), (63, 1, 1, 56), (2, 1, 1, 520), (32, 1, 1, 4088))),
    'small': (
        ((1, 3, 7, 40), (2, 16, 16, 64), (8, 17, 31, 128), (1, 9, 7, 512)),
        ((3, 5, 11, 56), (1, 23, 15, 72), (7, 29, 9, 120), (2, 7, 13, 392))),
    'dispatch-boundary': (
        ((1, 32, 32, 128), (1, 33, 32, 128), (1, 64, 64, 32), (1, 65, 64, 32)),
        ((3, 16, 16, 512), (3, 17, 16, 512), (1, 31, 33, 128), (1, 32, 33, 128))),
    'skinny': (
        ((1, 1, 257, 128), (3, 129, 1, 40), (1, 2, 2048, 256), (1, 1024, 3, 512)),
        ((2, 1, 513, 136), (7, 257, 1, 72), (1, 3, 4095, 120), (1, 2047, 2, 520))),
    'batch': (
        ((8, 33, 65, 128), (20, 17, 33, 256), (40, 9, 129, 40), (64, 17, 17, 64)),
        ((7, 49, 97, 136), (19, 31, 63, 248), (41, 7, 193, 56), (63, 15, 19, 72))),
    'rectangular': (
        ((1, 128, 2048, 128), (1, 2048, 128, 128), (1, 64, 4096, 32), (1, 4096, 64, 64)),
        ((1, 127, 2049, 136), (1, 2049, 127, 120), (1, 63, 4097, 40), (1, 4097, 63, 56))),
    'balanced': (
        ((1, 128, 128, 128), (1, 256, 256, 512), (1, 512, 512, 128), (2, 257, 257, 136)),
        ((1, 129, 127, 120), (1, 255, 257, 520), (1, 513, 511, 136), (3, 193, 191, 248))),
    'long-k': (
        ((1, 17, 33, 1024), (1, 65, 129, 2048), (2, 33, 65, 4096), (1, 129, 257, 8192)),
        ((2, 19, 35, 1032), (1, 67, 131, 2056), (3, 31, 63, 4104), (1, 127, 255, 8184))),
    'large-reduction': (
        ((1, 4095, 129, 32), (1, 4096, 129, 32), (1, 4097, 129, 32), (1, 8192, 257, 40)),
        ((1, 4089, 131, 40), (1, 4103, 127, 56), (2, 4111, 65, 72), (1, 8191, 255, 48))),
}
BENCHMARK_SUITES = ('benchmark-tune', 'benchmark-holdout')


def benchmark_specs(suite):
    side = BENCHMARK_SUITES.index(suite)
    for groups in GEOMETRIES.values():
        for shape, dtype, ta, tb in itertools.product(groups[side], (1, 2), (False, True), (False, True)):
            yield shape, dtype, ta, tb, 0, 'random'


def coverage(shape):
    shape = tuple(shape)
    for family, groups in GEOMETRIES.items():
        for side, geometries in enumerate(groups):
            if shape in geometries:
                return dict(family=family, split=BENCHMARK_SUITES[side])
    return dict(family='diagnostic', split='diagnostic')


def validate_shape(shape):
    b, m, n, k = shape
    if not (1 <= b <= 64 and 1 <= m <= 8192 and 1 <= n <= 8192
            and 32 <= k <= 8192 and k % 8 == 0
            and b * m * k <= 2**26 and b * n * k <= 2**26):
        raise ValueError(f'outside contest contract: {shape}')


def input_seed(shape, mode, seed):
    # Exclude dtype/layout so all eight variants share the same logical source.
    # Include geometry, so adding/reordering/filtering cases cannot change it.
    key = json.dumps([list(shape), mode, seed], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')


def robustness_specs():
    """Known semantic failure modes, separate from performance weighting."""
    definitions = (
        ((3, 3, 2, 32), 'row-winners'),
        ((64, 3, 9, 40), 'batch-distinct'),
        ((2, 17, 129, 56), 'last-max'),
        ((3, 65, 257, 120), 'last-max'),
        ((1, 33, 65, 1032), 'negative'),
        ((1, 4095, 17, 32), 'm-cancellation'),
        ((1, 4096, 17, 32), 'm-cancellation'),
        ((1, 4097, 17, 32), 'm-cancellation'),
        ((1, 8192, 17, 32), 'm-magnitude'),
    )
    for (shape, mode), dtype, ta, tb in itertools.product(definitions, (1, 2), (False, True), (False, True)):
        yield shape, dtype, ta, tb, 20, mode
