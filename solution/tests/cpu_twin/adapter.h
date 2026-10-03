#pragma once
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tikicpulib.h"
#include "kern_fwk.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
// The SDK device headers define names that collide with Host tiling enums.
// Parse Host headers first and remove only the three macros used by our Host.
#undef DT_FLOAT
#undef DT_FLOAT16
#undef DT_BF16
#include <new>
#include <stdexcept>
#include <tuple>

struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };

// CPU Twin launches take GM pointers. Put value arguments (including tiling)
// in shared GM instead of exposing their bytes as purported GM addresses.
template <typename F, typename... Args> struct TwinCall {
    F function;
    std::tuple<Args...> arguments;
};
template <typename F, typename... Args> void TwinEntry(uint8_t* address) {
    auto* call = reinterpret_cast<TwinCall<F, Args...>*>(address);
    std::apply(call->function, call->arguments);
}
inline size_t& TwinLaunchCount() { static size_t count = 0; return count; }
template <typename F, typename... Args>
void TwinLaunch(KernelMode mode, const char* name, F function, uint32_t blocks, Args... args) {
    using Call = TwinCall<F, Args...>;
    auto* memory = AscendC::GmAlloc(sizeof(Call));
    if (!memory) throw std::bad_alloc();
    auto* call = new (memory) Call{function, std::make_tuple(args...)};
    AscendC::SetKernelMode(mode);
    ++TwinLaunchCount();
    AscendC::RunKernelFunctionOnCpu(TwinEntry<F, Args...>, name, blocks,
        reinterpret_cast<uint8_t*>(call));
    call->~Call();
    AscendC::GmFree(memory);
}

// Host runtime behavior remains a separate NPU test. These adapters provide
// shared GM for the unchanged kernel bodies, and explicitly reject capture.
inline aclError TwinMalloc(void** memory, size_t bytes, aclrtMemMallocPolicy) {
    *memory = AscendC::GmAlloc(bytes);
    return *memory ? ACL_SUCCESS : ACL_ERROR_BAD_ALLOC;
}
inline aclError TwinFree(void* memory) { AscendC::GmFree(memory); return ACL_SUCCESS; }
inline aclError TwinSync(aclrtStream, int32_t) { return ACL_SUCCESS; }
inline aclError TwinCaptureInfo(aclrtStream, aclmdlRICaptureStatus* status, aclmdlRI* model) {
    *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    *model = nullptr;
    return ACL_SUCCESS;
}
inline aclError TwinRejectCapture(aclmdlRI, aclrtCallback, void*) {
    throw std::runtime_error("CPU Twin runner does not support graph capture");
}
#define aclrtMalloc TwinMalloc
#define aclrtFree TwinFree
#define aclrtSynchronizeStreamWithTimeout TwinSync
#define aclmdlRICaptureGetInfo TwinCaptureInfo
#define aclmdlRIDestroyRegisterCallback TwinRejectCapture
