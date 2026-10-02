#!/usr/bin/env python3
"""Check actual host source with Clang address spaces, without an ASC SDK.

Kernel launches become ordinary declarations and calls. ACL declarations only
provide their C++ interface types. This does not validate ASC launch generation,
device instructions, SDK linking, or NPU operation.
"""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERFACES = r'''
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#define __gm__ __attribute__((address_space(1)))
#define GM_ADDR __gm__ uint8_t*
struct half {};
struct bfloat16_t {};
using aclrtStream = void*;
using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
aclError aclrtMalloc(void**, size_t, int);
aclError aclrtFree(void*);
aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t);
using aclmdlRI = void*;
using aclrtCallback = void (*)(void*);
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED };
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*);
aclError aclmdlRIDestroyRegisterCallback(aclmdlRI, aclrtCallback, void*);
struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };
'''


def host_source(path):
    source = path.read_text()
    # Copy the actual metadata/plan helpers, excluding device-only helpers.
    begin = source.index('namespace bmmms {')
    end = source.index('__aicore__ inline uint32_t Min')
    helpers = source[begin:end]
    signatures = re.findall(r'__global__ (?:__vector__|__mix__\(0, 1\)) void (Baseline|Finish)\(([^{}]+)\)\s*\{', source)
    names = [name for name, _ in signatures]
    assert names in (['Baseline'], ['Baseline', 'Finish'])
    declarations = []
    for name, args in signatures:
        if name == 'Baseline':
            declarations.append('template <typename T, bool transA, bool transB>')
        declarations.append(f'void {name}({args});')
    host = source[source.index('extern "C" void run_kernel('):]
    host, count = re.subn(
        r'(Baseline<[^>]+>|Finish)<<<([^,]+), nullptr, stream>>>\(([^;]+)\);',
        r'(void(stream), void(\2), \1(\3));', host)
    assert count == (9 if 'Finish' in names else 8) and '<<<' not in host
    return INTERFACES + helpers + '\n'.join(declarations) + '\n}\n' + host


def check(path):
    with tempfile.TemporaryDirectory(prefix='bmmms-host-types-') as temp:
        filename = Path(temp) / 'host.cpp'
        filename.write_text(host_source(path))
        return subprocess.run([
            'clang++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
            '-Wno-unused-const-variable', '-fsyntax-only', str(filename),
        ], capture_output=True, text=True)


def main():
    old = check(ROOT / 'experiments/vector_v5.asc')
    assert old.returncode != 0, 'the known v5 address-space error was not reproduced'
    assert 'reinterpret_cast' in old.stderr and 'is not allowed' in old.stderr, old.stderr
    print('PASS: archived v5 reproduces the illegal ordinary-pointer to GM_ADDR cast')
    fixed = check(ROOT / 'kernel.asc')
    if fixed.returncode:
        raise RuntimeError(fixed.stderr)
    print('PASS: current actual Host dispatch passes Clang address-space type checking')
    print('This is a C++ type check, not an ASC/SDK build or an NPU validation.')


if __name__ == '__main__':
    main()
