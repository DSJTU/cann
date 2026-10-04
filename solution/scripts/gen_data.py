"""Generate the single legal FP16 smoke input used by main.asc."""
from pathlib import Path

import numpy as np
from BatchMatmulMaxSum import impl


def main():
    source = Path('input/case0')
    golden = Path('output/golden_case0')
    source.mkdir(parents=True, exist_ok=True)
    golden.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    x1 = rng.uniform(-1, 1, (1, 1, 32)).astype(np.float16)
    x2 = rng.uniform(-1, 1, (1, 32, 1)).astype(np.float16)
    x1.tofile(source / 'x1.bin')
    x2.tofile(source / 'x2.bin')
    impl(x1, x2).tofile(golden / 'golden_y.bin')
    print('Generated smoke input and FP32 golden.')


if __name__ == '__main__':
    main()
