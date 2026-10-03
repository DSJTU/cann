// Synchronous mathematical model for layout, packing and bounds checks.
// FP64 dot products do not model Cube precision, pipelines or graph lifetime.
#pragma once
#include <atomic>
#include <cstdlib>
#define __kfc_workspace__
using aclError = int;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
using aclmdlRI = void*;
using aclrtCallback = void (*)(void*);
enum aclmdlRICaptureStatus {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED
};
inline aclError aclrtMalloc(void** ptr, size_t size, int) {
    *ptr = std::malloc(size);
    return *ptr ? ACL_SUCCESS : 1;
}
inline aclError aclrtFree(void* ptr) { std::free(ptr); return ACL_SUCCESS; }
inline aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int) { return ACL_SUCCESS; }
inline aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* status, aclmdlRI* model) {
    *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE; *model = nullptr; return ACL_SUCCESS;
}
inline aclError aclmdlRIDestroyRegisterCallback(aclmdlRI, aclrtCallback, void*) {
    throw std::runtime_error("graph ownership is not modeled by the CPU test");
}
namespace bmmms_sim {
inline std::atomic<uint64_t> &MatmulRequests() { static std::atomic<uint64_t> value{0}; return value; }
inline std::atomic<uint64_t> &MatmulTiles() { static std::atomic<uint64_t> value{0}; return value; }
}
enum class CubeFormat { ND };
namespace AscendC { namespace tiling {
struct TCubeTiling {
    uint32_t baseM = 32, baseN = 64, orgM = 0, orgN = 0, orgK = 0;
    uint32_t batchA = 0, batchB = 0, singleCoreM = 0, singleCoreN = 0, singleCoreK = 0;
};
}}
namespace AscendC {
inline void Max(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t i=0;i<count;++i) dst.SetValue(i,std::max(a.GetValue(i),b.GetValue(i)));
}
}
inline void* GetSysWorkSpacePtr() { return nullptr; }
namespace platform_ascendc {
struct PlatformAscendC { size_t GetLibApiWorkSpaceSize() const { return 64; } };
struct PlatformAscendCManager {
    static PlatformAscendC* GetInstance(const char*) { static PlatformAscendC platform; return &platform; }
};
}
namespace matmul_tiling {
enum class TPosition { GM, LCM };
using CubeFormat = ::CubeFormat;
enum class DataType { DT_FLOAT16, DT_BF16, DT_FLOAT };
class MatmulApiTiling {
    AscendC::tiling::TCubeTiling tiling;
public:
    explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC&) {}
    void SetAType(TPosition, CubeFormat, DataType, bool) {}
    void SetBType(TPosition, CubeFormat, DataType, bool) {}
    void SetCType(TPosition, CubeFormat, DataType) {}
    void SetShape(uint32_t m, uint32_t n, uint32_t) { tiling.baseM=m; tiling.baseN=n; }
    void SetOrgShape(uint32_t m, uint32_t n, uint32_t k) { tiling.orgM=m; tiling.orgN=n; tiling.orgK=k; }
    void SetBias(bool) {}
    void SetFixSplit(uint32_t m, uint32_t n, int) { tiling.baseM=m; tiling.baseN=n; }
    void SetBufferSpace(int,int,int) {}
    int SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k) {
        if (batchA < 1 || batchB < 1 || m < 1 || n < 1 || k < 1) return -1;
        tiling.batchA=uint32_t(batchA); tiling.batchB=uint32_t(batchB);
        tiling.singleCoreM=uint32_t(m); tiling.singleCoreN=uint32_t(n); tiling.singleCoreK=uint32_t(k);
        return 0;
    }
    int GetTiling(AscendC::tiling::TCubeTiling& output) const { output=tiling; return 0; }
};
}
namespace matmul {
enum class BatchMode { BATCH_LESS_THAN_L1 = 0, BATCH_LARGE_THAN_L1 = 1, SINGLE_LARGE_THAN_L1 = 2 };
enum class LayoutMode { NONE = 0, BSNGD = 1, SBNGD = 2, BNGS1S2 = 3, NORMAL = 4 };
struct MatmulConfig { int batchMode = 0; };
constexpr MatmulConfig GetNormalConfig(bool = false, bool = false, bool = false,
    BatchMode mode = BatchMode::BATCH_LESS_THAN_L1) {
    return MatmulConfig{int(mode)};
}
constexpr MatmulConfig kDefaultConfig = GetNormalConfig();
template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
    LayoutMode LAYOUT = LayoutMode::NONE>
struct MatmulType { using T = TYPE; static constexpr bool isTrans = ISTRANS; static constexpr LayoutMode layout = LAYOUT; };
template <typename A, typename B, typename C, typename Bias, const MatmulConfig& CFG = kDefaultConfig>
class Matmul {
    AscendC::GlobalTensor<typename A::T> a;
    AscendC::GlobalTensor<typename B::T> b;
    uint32_t orgM=0, orgN=0, orgK=0, m=0, n=0, k=0, baseM=0, baseN=0, tile=0;
    uint32_t singleM=0, singleN=0, singleK=0;
    bool ta=false, tb=false, done=true, pending=false;
public:
    void Init(const AscendC::tiling::TCubeTiling* t) {
        orgM=t->orgM; orgN=t->orgN; orgK=t->orgK; baseM=t->baseM; baseN=t->baseN;
        singleM=t->singleCoreM; singleN=t->singleCoreN; singleK=t->singleCoreK;
    }
    void SetOrgShape(uint32_t om, uint32_t on, uint32_t ok) { orgM=om; orgN=on; orgK=ok; }
    // singleM stays within one base row tile. singleN may cover several base
    // column tiles; Iterate yields them from N=0 toward the tail.
    void SetSingleShape(uint32_t sm, uint32_t sn, uint32_t sk) {
        if (pending || !done) throw std::runtime_error("Matmul request is still open");
        if (!sm || !sn || !sk || !baseM || !baseN || sm>baseM || sk>orgK)
            throw std::runtime_error("CPU Matmul single shape exceeds one M tile or full K");
        m=sm; n=sn; k=sk; tile=0; pending=false; done=false;
        bmmms_sim::MatmulRequests().fetch_add(1, std::memory_order_relaxed);
    }
    void SetTensorA(AscendC::GlobalTensor<typename A::T> input, bool trans) {
        if (trans && !A::isTrans) throw std::runtime_error("MatmulType must enable A transpose");
        a=input; ta=trans;
    }
    void SetTensorB(AscendC::GlobalTensor<typename B::T> input, bool trans) {
        if (trans && !B::isTrans) throw std::runtime_error("MatmulType must enable B transpose");
        b=input; tb=trans;
    }
    template <bool sync=true> bool Iterate() {
        if (pending) throw std::runtime_error("Iterate before consuming the current tile");
        if (tile * baseN >= n) { done=true; return false; }
        pending=true;
        return true;
    }
    template <bool sync=true> void GetTensorC(AscendC::LocalTensor<typename C::T> output, int atomic, bool sequential) {
        if (!pending || atomic || !sequential) throw std::runtime_error("unexpected CPU Matmul output mode");
        const uint32_t col0 = tile * baseN;
        const uint32_t tileCols = n - col0 < baseN ? n - col0 : baseN;
        for (uint32_t row=0;row<m;++row) for (uint32_t col=0;col<tileCols;++col) {
            double value=0;
            const uint32_t globalCol = col0 + col;
            for (uint32_t kk=0;kk<k;++kk) {
                const auto av=a.GetValue(ta ? kk*orgM+row : row*orgK+kk);
                const auto bv=b.GetValue(tb ? globalCol*orgK+kk : kk*orgN+globalCol);
                value+=double(float(av))*double(float(bv));
            }
            output.SetValue(row*tileCols+col,typename C::T(value));
        }
        bmmms_sim::MatmulTiles().fetch_add(1, std::memory_order_relaxed);
        ++tile;
        pending=false;
        if (tile * baseN >= n) done=true;
    }
    void End() const { if (pending || !done) throw std::runtime_error("Matmul result was not consumed"); }
    // One call covers every K panel. Each panel is a contiguous ND matrix of
    // singleCoreM x singleCoreK and singleCoreK x singleCoreN, separated by the
    // element strides. C is row-major singleCoreM x singleCoreN per panel.
    void IterateBatch(AscendC::GlobalTensor<float> output, uint32_t batchA, uint32_t batchB,
                      bool sequential, uint32_t strideA, uint32_t strideB, uint32_t strideC) {
        if (pending || !done) throw std::runtime_error("Matmul request is still open");
        if (!sequential || !batchA || batchA != batchB || !singleM || !singleN || !singleK)
            throw std::runtime_error("CPU IterateBatch expects equal batches and explicit panel shape");
        if (strideA < singleM * singleK || strideB < singleK * singleN || strideC < singleM * singleN)
            throw std::runtime_error("CPU IterateBatch stride is shorter than one panel");
        for (uint32_t panel = 0; panel < batchA; ++panel) {
            for (uint32_t row = 0; row < singleM; ++row) {
                for (uint32_t col = 0; col < singleN; ++col) {
                    double value = 0;
                    for (uint32_t kk = 0; kk < singleK; ++kk) {
                        const auto av = a.GetValue(ta ? panel * strideA + kk * singleM + row
                                                      : panel * strideA + row * singleK + kk);
                        const auto bv = b.GetValue(tb ? panel * strideB + col * singleK + kk
                                                      : panel * strideB + kk * singleN + col);
                        value += double(float(av)) * double(float(bv));
                    }
                    output.SetValue(panel * strideC + row * singleN + col, float(value));
                }
            }
        }
        bmmms_sim::MatmulRequests().fetch_add(1, std::memory_order_relaxed);
        bmmms_sim::MatmulTiles().fetch_add(batchA, std::memory_order_relaxed);
    }
};
}
#define REGIST_MATMUL_OBJ(pipe, workspace, object, tiling) object.Init(tiling)
