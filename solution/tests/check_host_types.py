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
using aclrtContext = void*;
struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };

using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
aclError aclrtMalloc(void**, size_t, int);
aclError aclrtFree(void*);
aclError aclrtGetCurrentContext(aclrtContext*);
aclError aclrtSynchronizeStream(aclrtStream);
namespace AscendC { namespace tiling { struct TCubeTiling { uint32_t baseM, baseN; }; } }
namespace platform_ascendc {
struct PlatformAscendC { size_t GetLibApiWorkSpaceSize() const; uint32_t GetCoreNumAic() const; };
struct PlatformAscendCManager { static PlatformAscendC* GetInstance(const char*); };
}
namespace matmul_tiling {
enum class TPosition { GM, LCM };
enum class CubeFormat { ND };
enum class DataType { DT_FLOAT16, DT_BF16, DT_BFLOAT16 = DT_BF16, DT_FLOAT };
struct MatmulApiTiling {
    explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC&);
    int SetAType(TPosition, CubeFormat, DataType, bool);
    int SetBType(TPosition, CubeFormat, DataType, bool);
    void SetCType(TPosition, CubeFormat, DataType);
    void SetBiasType(TPosition, CubeFormat, DataType);
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
    # Remove only the reviewed device-only classes, keeping Schedule/Plan
    # and every Host function intact for the address-space check.
    for name in ('NativeCube', 'SmallMatrix', 'BmmmsDotKernel'):
        match = re.search(r'template <[^>]+>\s*class '+name+r'\s*\{', source)
        assert match, name
        depth, end = 1, match.end()
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        source = source[:match.start()] + source[end + 1:]
    # Replace actual device bodies with declarations; retain all Host logic,
    # including tiling/scratch caches and every dispatch branch.
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
            signature = re.sub(r'__(?:global|vector|mix|schedmode)__\s*(?:\([^)]*\))?', '', signature)
            names.extend(re.findall(r'void (fused_kernel|bmmms_small_kernel|bmmms_dot_kernel)\(', signature))
            replacement = signature + ';'
        else:
            replacement = ''
        source = source[:match.start()] + replacement + source[end:]
    assert names == ['fused_kernel', 'bmmms_small_kernel', 'bmmms_dot_kernel'], names
    for pattern, expected in ((r'fused_kernel<[^>]+>', 1), (r'bmmms_small_kernel<[^>]+>', 1), (r'bmmms_dot_kernel<[^>]+>', 2)):
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
