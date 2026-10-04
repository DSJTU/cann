#!/usr/bin/env bash
set -eo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
task_sdk=${BMMMS_CANN_ROOT:-${ASCEND_HOME_PATH:-}}
task_python=$(command -v "${BMMMS_TWIN_PYTHON:-python3}")
if [[ ! -f "$task_sdk/set_env.sh" || ! -x "$task_python" ]]; then
    echo 'Set BMMMS_CANN_ROOT and BMMMS_TWIN_PYTHON to the local SDK and Python environment.' >&2
    exit 1
fi
source "$task_sdk/set_env.sh"
task_suite=smoke
task_case=()
task_debug=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) task_suite=$2; shift 2 ;;
        --case) task_case=(--case "$2"); shift 2 ;;
        --gdb) task_debug=1; shift ;;
        *) echo "Usage: $0 [--suite smoke|small-matrix|short-cube|long-cube|long-precision|all] [--case INDEX] [--gdb]" >&2; exit 1 ;;
    esac
done
task_work=${BMMMS_TWIN_WORK:-"$task_root/.private/runtime/cpu-twin"}
mkdir -p "$task_work"
task_work=$(cd "$task_work" && pwd)
mkdir -p "$task_work/run"
task_compiler=${CXX:-g++}
task_compiler=$(command -v "$task_compiler")
cmake -S "$task_root/solution/tests/cpu_twin" -B "$task_work/build" \
    -DCMAKE_CXX_COMPILER="$task_compiler" -DPython3_EXECUTABLE="$task_python" -DCANN_ROOT="$task_sdk" \
    -DKERNEL_SOURCE="$task_root/solution/kernel.asc"
cmake --build "$task_work/build" -j4
task_prefix="$task_work/data/$task_suite${task_case[1]:+-case${task_case[1]}}"
"$task_python" "$task_root/solution/tests/cpu_twin/data.py" --suite "$task_suite" "${task_case[@]}" --prefix "$task_prefix"
cd "$task_work/run"
if [[ $task_debug == 1 ]]; then
    exec gdb -q -nx -ex 'set debuginfod enabled off' -ex 'set print elements 16' \
        -ex 'set follow-fork-mode child' -ex 'set detach-on-fork off' \
        -ex 'set schedule-multiple on' --args "$task_work/build/bmmms_cpu_twin" \
        "$task_prefix.bin" "$task_prefix.out.bin"
fi
"$task_work/build/bmmms_cpu_twin" "$task_prefix.bin" "$task_prefix.out.bin" > "$task_prefix.log" 2>&1
"$task_python" "$task_root/solution/tests/npu_data.py" --prefix "$task_prefix" --verify "$task_prefix.out.bin"
echo "CPU Twin log: $task_prefix.log"
