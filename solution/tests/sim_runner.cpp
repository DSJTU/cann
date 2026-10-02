#include "sim_api.h"
#include "kernel_cpu.inc"
#include <iostream>
#include <array>

static bool Read(void* p, size_t n) {
    std::cin.read(static_cast<char*>(p), n);
    return bool(std::cin);
}

#ifdef SIM_FUSED_V1
static void CheckPlans() {
    const uint32_t dims[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 8191, 8192};
    for (uint32_t batch : {1u, 3u, 64u}) {
        for (uint32_t m : dims) for (uint32_t n : dims) for (uint32_t cores : {1u, 2u, 20u, 24u, 32u, 64u}) {
            auto p = bmmms::MakePlan(batch, m, n, 32, cores);
            if (!p.blocks || p.blocks > cores || !p.nGroups) throw std::runtime_error("invalid launch plan");
            const uint32_t nTiles = (n + p.tileN - 1) / p.tileN;
            uint32_t expected = 0;
            for (uint32_t ni = 0; ni < p.nGroups; ++ni) {
                const uint32_t start = ni * nTiles / p.nGroups * p.tileN;
                const uint32_t end = std::min(n, (ni + 1) * nTiles / p.nGroups * p.tileN);
                if (start != expected || start >= end) throw std::runtime_error("N partition gap/overlap");
                expected = end;
            }
            if (expected != n) throw std::runtime_error("incomplete N coverage");
            const uint64_t scratch = uint64_t(batch) * p.mTiles * p.nGroups *
                                     (p.nGroups == 1 ? bmmms::kSlot : p.tileM) * sizeof(float);
            if (scratch > bmmms::kScratchBytes) throw std::runtime_error("workspace capacity");
        }
    }
}
#endif
#if defined(SIM_VECTOR_V3) || defined(SIM_VECTOR_V5)
static void CheckPlans() {
    const uint32_t dims[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 8191, 8192};
    for (uint32_t batch : {1u, 3u, 64u}) {
        for (uint32_t m : dims) for (uint32_t n : dims) for (uint32_t cores : {1u, 2u, 20u, 24u, 32u, 64u}) {
            auto p = bmmms::MakePlan(batch, m, n, 32, cores);
            if (!p.blocks || p.blocks > cores || !p.groups || p.groups > m || p.groups > 64)
                throw std::runtime_error("invalid vector launch plan");
            uint32_t expected = 0;
            for (uint32_t group = 0; group < p.groups; ++group) {
                const uint32_t begin = group * m / p.groups, end = (group + 1) * m / p.groups;
                if (begin != expected || begin >= end) throw std::runtime_error("M partition gap/overlap");
                expected = end;
            }
            if (expected != m) throw std::runtime_error("incomplete M coverage");
            if (uint64_t(batch) * p.groups * bmmms::kSlot * sizeof(float) > bmmms::kScratchBytes)
                throw std::runtime_error("vector workspace capacity");
        }
    }
}
#endif
int main() {
    try {
#if defined(SIM_FUSED_V1) || defined(SIM_VECTOR_V3) || defined(SIM_VECTOR_V5)
        CheckPlans();
#endif
        uint32_t count = 0;
        if (!Read(&count, 4)) return 1;
        for (uint32_t ci = 0; ci < count; ++ci) {
            std::array<uint32_t, 9> h;
            if (!Read(h.data(), h.size() * 4)) return 2;
            const auto batch = h[0], m = h[1], n = h[2], k = h[3], dtype = h[4];
            const bool ta = h[5], tb = h[6];
            const auto cores = h[7];
            const aclrtStream stream = reinterpret_cast<void*>(uintptr_t(h[8] + 1));
            std::vector<uint16_t> a(size_t(batch) * m * k), b(size_t(batch) * n * k);
            if (!Read(a.data(), a.size() * 2) || !Read(b.data(), b.size() * 2)) return 3;
            const auto savedA = a, savedB = b;
            std::vector<float> y(batch + 16, -987654.0f);
            int64_t as[] = {batch, ta ? k : m, ta ? m : k};
            int64_t bs[] = {batch, tb ? n : k, tb ? k : n};
            int64_t ys[] = {batch};
            TensorInfo ai = {as, 3, int32_t(dtype)}, bi = {bs, 3, int32_t(dtype)}, yi = {ys, 1, 0};
            TensorGroupInfo ag = {&ai, 1}, bg = {&bi, 1}, yg = {&yi, 1};
            const auto invoke = [&] {
#ifdef SIM_VECTOR_V5
                const auto allocationsBefore = sim::allocCount(), syncsBefore = sim::syncCount();
#endif
                run_kernel(reinterpret_cast<uint8_t*>(a.data()), ag, reinterpret_cast<uint8_t*>(b.data()), bg,
                           reinterpret_cast<uint8_t*>(y.data()), yg, cores, stream, ta, tb);
#ifdef SIM_VECTOR_V5
                if (!sim::allocations().empty()) throw std::runtime_error("workspace leak");
                if (sim::syncCount() - syncsBefore != sim::allocCount() - allocationsBefore)
                    throw std::runtime_error("workspace released without stream synchronization");
#endif
            };
            invoke();
            const auto first = y;
#ifndef SIM_VECTOR_V5
            const auto allocs = sim::allocCount();
#endif
            invoke();
            if (std::memcmp(first.data(), y.data(), y.size() * sizeof(float)))
                throw std::runtime_error("repeatability failure");
#ifndef SIM_VECTOR_V5
            if (sim::allocCount() != allocs) throw std::runtime_error("allocation after warmup");
#endif
            if (savedA != a || savedB != b) throw std::runtime_error("input mutation");
            for (uint32_t i = batch; i < y.size(); ++i)
                if (y[i] != -987654.0f) throw std::runtime_error("output overrun");
            for (uint32_t i = 0; i < batch; ++i)
                if (!std::isfinite(y[i])) throw std::runtime_error("nonfinite output");
            std::cout.write(reinterpret_cast<const char*>(y.data()), batch * sizeof(float));
        }
        for (void* p : sim::allocations()) std::free(p);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CPU model failure: " << e.what() << '\n';
        return 4;
    }
}
