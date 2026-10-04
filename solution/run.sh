#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]]; then
    echo 'Set ASCEND_HOME_PATH to the CANN toolkit root containing set_env.sh.' >&2
    exit 1
fi
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build
cmake --build build -j4
cd build
python3 ../scripts/gen_data.py
cp input/case0/x1.bin input/x1.bin
cp input/case0/x2.bin input/x2.bin
cp output/golden_case0/golden_y.bin output/golden_y.bin
mkdir -p output
rm -f output/y.bin
timeout 120 ./batch_matmul_max_sum_custom
python3 ../scripts/verify_result.py 0
