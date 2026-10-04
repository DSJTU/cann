#!/usr/bin/env python3
"""Small explicit-core fixtures for official CPU Twin and GDB."""
import argparse
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import npu_data


def fixtures(suite):
    shapes = {
        'smoke': [(1, 1, 1, 32), (2, 17, 19, 40)],
        'small-matrix': [(3, 31, 32, 128), (2, 32, 31, 72), (64, 3, 2, 32)],
        'short-cube': [(1, 64, 128, 128)],
        'long-cube': [(1, 257, 129, 136), (1, 65, 512, 136)],
    }
    suites = shapes if suite == 'all' else [suite]
    for name in suites:
        for shape, dtype, ta, tb in itertools.product(shapes[name], (1, 2), (False, True), (False, True)):
            yield shape, dtype, ta, tb, 1, 'random'


def precision_fixtures():
    for shape, dtype, ta, tb, cores in itertools.product(
            ((1, 65, 129, 512), (1, 65, 513, 520)), (1, 2),
            (False, True), (False, True), (1, 2)):
        mode = 'panel-magnitude' if shape[2] == 129 else 'panel-magnitude-negative'
        yield shape, dtype, ta, tb, cores, mode


def batch_fixtures():
    for shape, dtype, (ta, tb) in itertools.product(
            ((1, 65, 513, 1160), (1, 65, 129, 1792)),
            (1, 2), ((False, False), (True, True))):
        yield shape, dtype, ta, tb, 1, 'random'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('smoke', 'small-matrix', 'short-cube', 'long-cube', 'long-precision', 'long-batches', 'all'), default='smoke')
    parser.add_argument('--case', type=int, help='select one case for a GDB session')
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    selected = list(precision_fixtures() if args.suite == 'long-precision' else
                    batch_fixtures() if args.suite == 'long-batches' else fixtures(args.suite))
    if args.case is not None:
        if not 0 <= args.case < len(selected):
            parser.error(f'case must be between 0 and {len(selected)-1}')
        selected = [selected[args.case]]
    npu_data.specs = lambda _: iter(selected)
    npu_data.generate(args.suite, args.prefix)
