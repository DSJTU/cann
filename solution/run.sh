#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]]; then
    echo 'Set ASCEND_HOME_PATH to the CANN toolkit root containing set_env.sh.' >&2
    exit 1
fi
source "$ASCEND_HOME_PATH/set_env.sh"
task_source=$(pwd)
task_work="$task_source/../.private/runtime/smoke"
cmake -S . -B "$task_work"
cmake --build "$task_work" -j4
cd "$task_work"
python3 "$task_source/scripts/gen_data.py"
cp input/case0/x1.bin input/x1.bin
cp input/case0/x2.bin input/x2.bin
cp output/golden_case0/golden_y.bin output/golden_y.bin
mkdir -p output
rm -f output/y.bin
timeout 120 ./batch_matmul_max_sum_custom
python3 "$task_source/scripts/verify_result.py" 0
