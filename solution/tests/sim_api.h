// Synchronous CPU model for control flow/address checks, NOT a CANN SDK shim.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#define __aicore__
#define __global__
#define __vector__
#define __mix__(a, b)
#define __schedmode__(mode)
#define __gm__
#define ASCEND_IS_AIV (true)
using half = _Float16;
struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() = default;
    bfloat16_t(float f) {
        uint32_t v;
        std::memcpy(&v, &f, 4);
        bits = uint16_t(v >> 16);
    }
    operator float() const {
        uint32_t v = uint32_t(bits) << 16;
        float f;
        std::memcpy(&f, &v, 4);
        return f;
    }
};
using GM_ADDR = uint8_t*;
using aclrtStream = void*;
using aclrtContext = void*;
using event_t = int;
struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };
enum Pipe { PIPE_ALL, PIPE_V };
namespace sim {
inline uint32_t& logicalBlocks() { static thread_local uint32_t v = 0; return v; }
inline uint32_t& block() { static thread_local uint32_t v = 0; return v; }
inline uint32_t& blocks() { static uint32_t v = 1; return v; }
inline uint32_t& launchCount() { static uint32_t v = 0; return v; }
class Collective {
    std::mutex mutex;
    std::condition_variable changed;
    uint32_t participants, arrived = 0, generation = 0;
    bool aborted = false;
public:
    explicit Collective(uint32_t n) : participants(n) {}
    void Abort() {
        std::lock_guard<std::mutex> lock(mutex);
        aborted = true;
        changed.notify_all();
    }
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (aborted) throw std::runtime_error("collective aborted");
        const auto phase = generation;
        if (++arrived == participants) {
            arrived = 0;
            ++generation;
            changed.notify_all();
        } else if (!changed.wait_for(lock, std::chrono::seconds(20), [&] {
            return generation != phase || aborted;
        })) {
            aborted = true;
            changed.notify_all();
            throw std::runtime_error("collective barrier timeout");
        }
        if (aborted) throw std::runtime_error("collective aborted");
    }
};
inline Collective*& activeCollective() { static Collective* p = nullptr; return p; }
template <typename F> void Launch(uint32_t n, F f) {
    ++launchCount();
    blocks() = n;
    Collective collective(n);
    activeCollective() = &collective;
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::vector<std::thread> workers;
    for (uint32_t i = 0; i < n; ++i) workers.emplace_back([&, i] {
        block() = i;
        logicalBlocks() = 0;
        try { f(); }
        catch (...) {
            { std::lock_guard<std::mutex> lock(failureMutex);
              if (!failure) failure = std::current_exception(); }
            collective.Abort();
        }
    });
    for (auto& worker : workers) worker.join();
    activeCollective() = nullptr;
    if (failure) std::rethrow_exception(failure);

}
template <typename F> void LaunchMixed(uint32_t n, F f) {
    Launch(n * 2, [&] { logicalBlocks() = n; f(); });
}
struct Storage {
    std::vector<uint8_t> bytes, initialized;
    bool dmaOperand = false;
    explicit Storage(size_t n) : bytes(n, 0xFF), initialized(n, 0) {}
};
}
namespace AscendC {
enum class TPosition { GM, VECIN, VECOUT, VECCALC };
enum class HardEvent { V_S, S_V, MTE2_V, MTE2_S, S_MTE2, S_MTE3, V_MTE3, MTE3_V, MTE2_MTE3, MTE3_MTE2, MTE3_S };
enum class RoundMode { CAST_NONE };
enum class ReduceOrder { ORDER_ONLY_VALUE };
inline uint32_t GetBlockIdx() { return sim::block(); }
inline uint32_t GetBlockNum() { return sim::logicalBlocks() ? sim::logicalBlocks() : sim::blocks(); }
template <bool isAIVOnly = true> inline void SyncAll() {
    static_assert(isAIVOnly, "only AIV collectives are modeled");
    if (!sim::activeCollective()) throw std::runtime_error("no collective launch");
    sim::activeCollective()->Wait();
}
template <Pipe> inline void PipeBarrier() {}
template <HardEvent> inline void SetFlag(int) {}
template <HardEvent> inline void WaitFlag(int) {}
template <typename T> class LocalTensor {
    std::shared_ptr<sim::Storage> s;
    size_t offset = 0;
public:
    LocalTensor() = default;
    explicit LocalTensor(std::shared_ptr<sim::Storage> storage, size_t at = 0) : s(storage), offset(at) {}
    template <typename U> LocalTensor<U> ReinterpretCast() const { return LocalTensor<U>(s, offset); }
    const void* Identity() const { return s.get(); }
    LocalTensor operator[](size_t n) const { return LocalTensor(s, offset + n * sizeof(T)); }
    void Check(size_t i, bool read) const {
        const size_t at = offset + i * sizeof(T);
        if (!s || at + sizeof(T) > s->bytes.size()) throw std::runtime_error("UB out of bounds");
        if (read)
            for (size_t j = at; j < at + sizeof(T); ++j)
                if (!s->initialized[j]) throw std::runtime_error("uninitialized UB read");
    }
    void Aligned() const {
        if (offset % 32) throw std::runtime_error("unaligned vector operand");
    }
    void DmaOperand() const {
        if (!s || !s->dmaOperand) throw std::runtime_error("DMA requires VECIN/VECOUT on A2");
    }
    T GetValue(size_t i) const {
        Check(i, true);
        T value;
        std::memcpy(&value, s->bytes.data() + offset + i * sizeof(T), sizeof(T));
        return value;
    }
    void SetValue(size_t i, T value) const {
        Check(i, false);
        const size_t at = offset + i * sizeof(T);
        std::memcpy(s->bytes.data() + at, &value, sizeof(T));
        std::fill(s->initialized.begin() + at, s->initialized.begin() + at + sizeof(T), 1);
    }
};
template <typename T> class GlobalTensor {
    T* p = nullptr;
    size_t length = 0;
public:
    void SetGlobalBuffer(T* ptr, size_t n = SIZE_MAX / sizeof(T)) { p = ptr; length = n; }
    GlobalTensor operator[](size_t n) const {
        if (n > length) throw std::runtime_error("GM offset out of bounds");
        GlobalTensor t;
        t.SetGlobalBuffer(p + n, length - n);
        return t;
    }
    T GetValue(size_t i) const {
        if (i >= length) throw std::runtime_error("GM read out of bounds");
        return p[i];
    }
    void SetValue(size_t i, T value) const {
        if (i >= length) throw std::runtime_error("GM write out of bounds");
        p[i] = value;
    }
};
template <TPosition P> class TBuf {
public:
    std::shared_ptr<sim::Storage> s;
    template <typename T> LocalTensor<T> Get() { return LocalTensor<T>(s); }
};
template <TPosition P, int Depth> class TQue {
    std::vector<std::shared_ptr<sim::Storage>> slots;
    std::vector<bool> busy;
    std::vector<size_t> queued;
public:
    void Init(int count, size_t bytes) {
        for (int i = 0; i < count; ++i) {
            auto storage = std::make_shared<sim::Storage>(bytes);
            storage->dmaOperand = P == TPosition::VECIN || P == TPosition::VECOUT;
            slots.push_back(storage); busy.push_back(false);
        }
    }
    template <typename T> LocalTensor<T> AllocTensor() {
        for (size_t i = 0; i < slots.size(); ++i)
            if (!busy[i]) { busy[i] = true; return LocalTensor<T>(slots[i]); }
        throw std::runtime_error("queue has no free tensor");
    }
    template <typename T> void EnQue(LocalTensor<T> tensor) {
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].get() == tensor.Identity() && busy[i]) { queued.push_back(i); return; }
        throw std::runtime_error("enqueue foreign tensor");
    }
    template <typename T> LocalTensor<T> DeQue() {
        if (queued.empty()) throw std::runtime_error("dequeue empty queue");
        const size_t i = queued.front(); queued.erase(queued.begin());
        return LocalTensor<T>(slots[i]);
    }
    template <typename T> void FreeTensor(LocalTensor<T> tensor) {
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].get() == tensor.Identity() && busy[i]) { busy[i] = false; return; }
        throw std::runtime_error("free foreign tensor");
    }
};
class TPipe {
public:
    template <TPosition P> void InitBuffer(TBuf<P>& q, size_t n) {
        q.s = std::make_shared<sim::Storage>(n);
        q.s->dmaOperand = P == TPosition::VECIN || P == TPosition::VECOUT;
        Reserve(n);
    }
    template <TPosition P, int Depth> void InitBuffer(TQue<P, Depth>& q, int count, size_t n) {
        q.Init(count, n); Reserve(size_t(count) * n);
    }
    int FetchEventID(HardEvent) { return 0; }
private:
    size_t used = 0;
    void Reserve(size_t n) {
        used += (n + 31) / 32 * 32;
        if (used > 192 * 1024) throw std::runtime_error("UB allocation exceeds conservative A2 budget");
    }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, reserved; };
template <typename T> struct DataCopyPadExtParams { bool isPad; uint8_t left, right; T value; };
template <typename T> void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    dst.Aligned(); src.Aligned();
    if (n * sizeof(T) % 32) throw std::runtime_error("nonaligned DataCopy length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, src.GetValue(i));
}
template <typename T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                     DataCopyExtParams p, DataCopyPadExtParams<T> pad) {
    dst.Aligned();
    dst.DmaOperand();
    const uint32_t count = p.blockLen / sizeof(T);
    const uint32_t left = pad.isPad ? pad.left : 0;
    const uint32_t right = pad.isPad ? pad.right : 0;
    if (left * sizeof(T) > 32 || right * sizeof(T) > 32)
        throw std::runtime_error("DataCopyPad padding exceeds 32-byte limit");
    const uint32_t dstPitch = ((left + count + right) * sizeof(T) + 31) / 32 * 32 / sizeof(T) +
                             p.dstStride * 32 / sizeof(T);
    const uint32_t srcPitch = (p.blockLen + p.srcStride) / sizeof(T);
    for (uint32_t row = 0; row < p.blockCount; ++row) {
        for (uint32_t i = 0; i < left; ++i) dst.SetValue(row * dstPitch + i, pad.value);
        for (uint32_t i = 0; i < count; ++i)
            dst.SetValue(row * dstPitch + left + i, src.GetValue(row * srcPitch + i));
        for (uint32_t i = 0; i < right; ++i)
            dst.SetValue(row * dstPitch + left + count + i, pad.value);
    }
}
template <typename T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    src.Aligned();
    src.DmaOperand();
    const uint32_t count = p.blockLen / sizeof(T);
    // UB source stride is extra 32-byte blocks after the aligned row.
    // GM destination stride is the byte gap after blockLen.
    const uint32_t srcPitch = ((count * sizeof(T) + 31) / 32) * (32 / sizeof(T)) +
                              p.srcStride * (32 / sizeof(T));
    const uint32_t dstPitch = (p.blockLen + p.dstStride) / sizeof(T);
    for (uint32_t row = 0; row < p.blockCount; ++row)
        for (uint32_t i = 0; i < count; ++i)
            dst.SetValue(row * dstPitch + i, src.GetValue(row * srcPitch + i));
}
template <typename T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    DataCopyPad(dst, src, DataCopyExtParams{1, n * uint32_t(sizeof(T)), 0, 0, 0});
}
template <typename T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t n) {
    DataCopyPad(dst, src, DataCopyExtParams{1, n * uint32_t(sizeof(T)), 0, 0, 0},
                DataCopyPadExtParams<T>{false, 0, 0, T(0)});
}
template <typename T> void Duplicate(LocalTensor<T> dst, T v, uint32_t n) {
    dst.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, v);
}
template <typename T> void Cast(LocalTensor<float> dst, LocalTensor<T> src, RoundMode, uint32_t n) {
    dst.Aligned(); src.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, float(src.GetValue(i)));
}
struct BinaryRepeatParams {
    uint8_t dstBlkStride, src0BlkStride, src1BlkStride;
    uint8_t dstRepStride, src0RepStride, src1RepStride;
};
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    if (mask < 1 || mask > 64) throw std::runtime_error("invalid FP32 Mul mask");
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < mask; ++i) {
            const uint32_t di = r * p.dstRepStride * 8 + i / 8 * p.dstBlkStride * 8 + i % 8;
            const uint32_t ai = r * p.src0RepStride * 8 + i / 8 * p.src0BlkStride * 8 + i % 8;
            const uint32_t bi = r * p.src1RepStride * 8 + i / 8 * p.src1BlkStride * 8 + i % 8;
            dst.SetValue(di, a.GetValue(ai) * b.GetValue(bi));
        }
    }
}
inline void WholeReduceSum(LocalTensor<float> dst, LocalTensor<float> src, int32_t mask,
                           int32_t repeats, int32_t dstStride, int32_t blockStride, int32_t srcStride) {
    dst.Aligned(); src.Aligned();
    if (mask < 1 || mask > 64 || repeats < 1 || repeats > 255)
        throw std::runtime_error("invalid FP32 WholeReduceSum mask/repeats");
    for (int32_t r = 0; r < repeats; ++r) {
        std::vector<float> values(64, 0.0f);
        for (int32_t i = 0; i < mask; ++i)
            values[i] = src.GetValue(r * srcStride * 8 + i / 8 * blockStride * 8 + i % 8);
        while (values.size() > 1) {
            for (size_t i = 0; i < values.size() / 2; ++i)
                values[i] = values[i * 2] + values[i * 2 + 1];
            values.resize(values.size() / 2);
        }
        dst.SetValue(r * dstStride, values[0]);
    }
}
inline float TreeSum(std::vector<float> values) {
    size_t width = 1; while (width < values.size()) width *= 2;
    values.resize(width, 0.0f);
    while (values.size() > 1) {
        for (size_t i = 0; i < values.size() / 2; ++i) values[i] = values[2*i] + values[2*i+1];
        values.resize(values.size() / 2);
    }
    return values[0];
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) * b.GetValue(i));
}
inline void ReduceSum(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float>, int32_t n) {
    dst.Aligned(); src.Aligned();
    std::vector<float> values; for (int32_t i = 0; i < n; ++i) values.push_back(src.GetValue(i));
    dst.SetValue(0, TreeSum(values));
}
inline void ReduceSum(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float>,
                      int32_t mask, int32_t repeats, int32_t stride) {
    dst.Aligned(); src.Aligned(); std::vector<float> chunks;
    for (int32_t r = 0; r < repeats; ++r) {
        std::vector<float> values;
        for (int32_t i = 0; i < mask; ++i) values.push_back(src.GetValue(r * stride * 8 + i));
        chunks.push_back(TreeSum(values));
    }
    dst.SetValue(0, TreeSum(chunks));
}
inline void Abs(LocalTensor<float> dst, LocalTensor<float> src, uint32_t n) {
    dst.Aligned(); src.Aligned();
    for (uint32_t i=0; i<n; ++i) dst.SetValue(i,std::abs(src.GetValue(i)));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) - b.GetValue(i));
}
template <typename T> void Gather(LocalTensor<T> dst, LocalTensor<T> src,
                                 LocalTensor<uint32_t> offsets, uint32_t base, uint32_t n) {
    dst.Aligned(); src.Aligned(); offsets.Aligned();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t byte = base + offsets.GetValue(i);
        if (byte % sizeof(T)) throw std::runtime_error("unaligned Gather byte offset");
        dst.SetValue(i, src.GetValue(byte / sizeof(T)));
    }
}
template <typename T> void CreateVecIndex(LocalTensor<T> dst, T first, uint32_t n) {
    dst.Aligned();
    if (!n || n * sizeof(T) % 32) throw std::runtime_error("nonaligned CreateVecIndex length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, T(first + T(i)));
}
template <typename T> void Muls(LocalTensor<T> dst, LocalTensor<T> src, T scalar, uint32_t n) {
    dst.Aligned(); src.Aligned();
    if (!n || n * sizeof(T) % 32) throw std::runtime_error("nonaligned Muls length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, T(src.GetValue(i) * scalar));
}
template <typename T> void Adds(LocalTensor<T> dst, LocalTensor<T> src, T scalar, uint32_t n) {
    dst.Aligned(); src.Aligned();
    if (!n || n * sizeof(T) % 32) throw std::runtime_error("nonaligned Adds length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, T(src.GetValue(i) + scalar));
}
template <typename T> void Max(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t n) {
    dst.Aligned(); a.Aligned(); b.Aligned();
    if (!n || n * sizeof(T) % 32) throw std::runtime_error("nonaligned Max length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, std::max(a.GetValue(i), b.GetValue(i)));
}
}

namespace AscendC {
inline void WholeReduceMax(LocalTensor<float> dst, LocalTensor<float> src, int32_t mask,
                           int32_t repeats, int32_t dstStride, int32_t blockStride, int32_t srcStride,
                           ReduceOrder order) {
    dst.Check(0, false); src.Aligned();
    if (mask < 1 || mask > 64 || repeats < 1 || repeats > 255 || order != ReduceOrder::ORDER_ONLY_VALUE)
        throw std::runtime_error("invalid FP32 WholeReduceMax parameters");
    for (int32_t r = 0; r < repeats; ++r) {
        float value = src.GetValue(r * srcStride * 8);
        for (int32_t i = 1; i < mask; ++i)
            value = std::max(value, src.GetValue(r * srcStride * 8 + i / 8 * blockStride * 8 + i % 8));
        dst.SetValue(r * dstStride, value);
    }
}
}
