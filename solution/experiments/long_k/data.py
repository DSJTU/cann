#!/usr/bin/env python3
"""Long-K hardware probes: stored input FP64 golden, true multi-tile tails."""
import argparse
import itertools
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TESTS = HERE / 'baseline/solution/tests'
if not TESTS.exists():
    TESTS = HERE.parents[1] / 'tests'
sys.path.insert(0, str(TESTS))
import npu_data
import numpy as np

BASE_INPUTS = npu_data.make_inputs


def inputs(shape, mode, rng):
    if mode != 'block-cancellation':
        return BASE_INPUTS(shape, mode, rng)
    batch, m, n, k = shape
    a = np.ones((batch, m, k), dtype=np.float32)
    b = np.zeros((batch, k, n), dtype=np.float32)
    # Cancellation occurs inside each 512-K request, not only between them.
    for start in range(0, k, 512):
        b[:, start:start + 128, :] = 4096
        b[:, start + 128:start + 256, :] = 2 ** -10
        b[:, start + 256:start + 384, :] = -4096
    return a, b


def definitions(suite):
    if suite == 'intra-chunk':
        shapes = [((1, 65, 65, 8192), 'block-cancellation')]
    elif suite == 'precision':
        shapes = [((1, 65, 65, 8192), mode) for mode in
                  ('k-cancellation', 'mixed-magnitude', 'close-max', 'block-cancellation')]
        shapes += [((2, 65, 129, 256), 'random'), ((3, 33, 65, 392), 'random'),
                   ((1, 33, 65, 1024), 'random'), ((1, 33, 65, 8192), 'random'),
                   ((2, 65, 129, 392), 'negative')]
    elif suite == 'benchmark':
        shapes = [((1, 129, 1024, 8192), 'random'),
                  ((1, 256, 1024, 1024), 'random'), ((8, 65, 257, 136), 'random')]
    else:
        raise ValueError(suite)
    for (shape, mode), dtype, ta, tb in itertools.product(shapes, (1, 2), (False, True), (False, True)):
        yield shape, dtype, ta, tb, 20, mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix')
    parser.add_argument('--suite', choices=('precision', 'benchmark', 'intra-chunk'), default='precision')
    args = parser.parse_args()
    npu_data.specs = definitions
    npu_data.make_inputs = inputs
    npu_data.generate(args.suite, args.prefix)
