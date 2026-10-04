"""Verify the smoke runner's complete, finite FP32 output."""
import argparse
from pathlib import Path

import numpy as np


def verify_result(output_path, golden_path):
    output_path, golden_path = Path(output_path), Path(golden_path)
    size = golden_path.stat().st_size
    if not size or size % 4 or output_path.stat().st_size != size:
        print('FAILED: incomplete or mismatched FP32 files')
        return False
    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(golden)):
        print('FAILED: nonfinite output or golden')
        return False
    error = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    passed = bool(np.all(error <= 1e-4 + 1e-4 * np.abs(golden.astype(np.float64))))
    print(f'{"PASSED" if passed else "FAILED"}: max absolute error {error.max():.8g}')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id', type=int, choices=(0,), nargs='?', default=0)
    parser.parse_args()
    output, golden = Path('output/y.bin'), Path('output/golden_y.bin')
    if not output.is_file() or not golden.is_file():
        parser.exit(1, 'FAILED: missing output/y.bin or output/golden_y.bin\n')
    raise SystemExit(0 if verify_result(output, golden) else 1)
