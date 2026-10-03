#include "sim_api.h"
#include "kernel_cpu.inc"
#include <iostream>
#include <array>

static bool Read(void* p, size_t n) {
    std::cin.read(static_cast<char*>(p), n);
    return bool(std::cin);
}

int main() {
    try {
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
                const auto launches = sim::launchCount();
                run_kernel(reinterpret_cast<uint8_t*>(a.data()), ag, reinterpret_cast<uint8_t*>(b.data()), bg,
                           reinterpret_cast<uint8_t*>(y.data()), yg, cores, stream, ta, tb);
                if (sim::launchCount() - launches != 1)
                    throw std::runtime_error("each invocation must launch exactly one kernel");
            };
            invoke();
            const auto first = y;
            invoke();
            if (std::memcmp(first.data(), y.data(), y.size() * sizeof(float)))
                throw std::runtime_error("repeatability failure");
            if (savedA != a || savedB != b) throw std::runtime_error("input mutation");
            for (uint32_t i = batch; i < y.size(); ++i)
                if (y[i] != -987654.0f) throw std::runtime_error("output overrun");
            for (uint32_t i = 0; i < batch; ++i)
                if (!std::isfinite(y[i])) throw std::runtime_error("nonfinite output");
            std::cout.write(reinterpret_cast<const char*>(y.data()), batch * sizeof(float));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CPU model failure: " << e.what() << '\n';
        return 4;
    }
}
