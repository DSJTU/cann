"""Independent goldens; shared runner also checks guards, inputs and repeats."""
import itertools
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
import npu_data

def specs(suite):
    if suite == 'short-validation':
        for original in ('correctness', 'extended'):
            for case in npu_data_original_specs(original):
                if case[0][3] <= 128:
                    yield case
        return
    if suite == 'large':
        yield from npu_data_original_specs('large-short-k')
        return
    if suite == 'precision':
        for dtype, ta, tb, k, mode in itertools.product(
                (1, 2), (False, True), (False, True), (32, 64, 128),
                ('k-cancellation', 'mixed-magnitude')):
            yield (1, 33, 17, k), dtype, ta, tb, 0, mode
        return
    for dtype, ta, tb in itertools.product((1, 2), (False, True), (False, True)):
        for shape in ((1, 32, 64, 128), (3, 17, 19, 40), (1, 33, 65, 120),
                      (8, 65, 129, 128), (1, 1, 1, 32), (1, 129, 257, 128)):
            yield shape, dtype, ta, tb, 0, 'random'
        for mode in ('negative', 'zero', 'close-max'):
            yield (3, 33, 65, 128), dtype, ta, tb, 1, mode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    parser.add_argument('--suite', choices=('tiles', 'precision', 'large', 'short-validation'), default='tiles')
    args = parser.parse_args()
    npu_data_original_specs = npu_data.specs
    npu_data.specs = specs
    npu_data.generate(args.suite, args.prefix)
