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
#define __gm__ __attribute__((address_space(1)))
#define GM_ADDR __gm__ uint8_t*
#define __kfc_workspace__
struct half {};
struct bfloat16_t {};
using aclrtStream = void*;
struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };

using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
using aclmdlRI = void*;
using aclrtCallback = void (*)(void*);
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
aclError aclrtMalloc(void**, size_t, int);
aclError aclrtFree(void*);
aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int);
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus*, aclmdlRI*);
aclError aclmdlRIDestroyRegisterCallback(aclmdlRI, aclrtCallback, void*);
namespace AscendC { namespace tiling { struct TCubeTiling { uint32_t baseM, baseN; }; } }
namespace platform_ascendc {
struct PlatformAscendC { size_t GetLibApiWorkSpaceSize() const; };
struct PlatformAscendCManager { static PlatformAscendC* GetInstance(const char*); };
}
namespace matmul_tiling {
enum class TPosition { GM, LCM };
enum class CubeFormat { ND };
enum class DataType { DT_FLOAT16, DT_BF16, DT_FLOAT };
struct MatmulApiTiling {
    explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC&);
    void SetAType(TPosition, CubeFormat, DataType, bool);
    void SetBType(TPosition, CubeFormat, DataType, bool);
    void SetCType(TPosition, CubeFormat, DataType);
    void SetShape(uint32_t, uint32_t, uint32_t);
    void SetOrgShape(uint32_t, uint32_t, uint32_t);
    void SetBias(bool);
    void SetFixSplit(uint32_t, uint32_t, int);
    void SetBufferSpace(int, int, int);
    int GetTiling(AscendC::tiling::TCubeTiling&);
};
}
'''


def host_source(path):
    source = re.sub(r'^#include "[^"]+"\n', '', path.read_text(), flags=re.MULTILINE)
    # Replace actual device bodies with declarations; retain all Host logic,
    # including the capture registry and both dispatch branches.
    pattern = re.compile(
        r'^(?:template\s*<[^>]+>\s*)?(?:__schedmode__\([^\n]+?\)\s+)?'
        r'__(?:aicore|global)__[^{}]+\{', re.MULTILINE)
    names = []
    while match := pattern.search(source):
        opening = match.end() - 1
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        signature = match.group()[:-1]
        if '__global__' in signature:
            signature = re.sub(r'__(?:global|mix|schedmode)__\s*(?:\([^)]*\))?', '', signature)
            names.extend(re.findall(r'void (Baseline|Scores|Finish)\(', signature))
            replacement = signature + ';'
        else:
            replacement = ''
        source = source[:match.start()] + replacement + source[end:]
    assert names == ['Baseline', 'Scores', 'Finish'], names
    for pattern, expected in ((r'Baseline<[^>]+>', 8), (r'Scores<[^>]+>', 2), (r'Finish', 1)):
        source, count = re.subn(
            '(' + pattern + r')<<<([^,]+), nullptr, stream>>>\(([^;]+)\);',
            r'(void(stream), void(\2), \1(\3));', source)
        assert count == expected, (pattern, count)
    assert '<<<' not in source
    return INTERFACES + source


def check(path):
    with tempfile.TemporaryDirectory(prefix='bmmms-host-types-') as temp:
        filename = Path(temp) / 'host.cpp'
        filename.write_text(host_source(path))
        return subprocess.run([
            'clang++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
            '-Wno-unused-const-variable', '-fsyntax-only', str(filename),
        ], capture_output=True, text=True)


def main():
    fixed = check(ROOT / 'kernel.asc')
    if fixed.returncode:
        raise RuntimeError(fixed.stderr)
    print('PASS: current actual Host dispatch passes Clang address-space type checking')
    print('This is a C++ type check, not an ASC/SDK build or an NPU validation.')


if __name__ == '__main__':
    main()
