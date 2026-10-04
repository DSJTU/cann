#include "sim_api.h"
#include "kernel_cpu.inc"
#include <array>
#include <iostream>

int main() {
    try {
        uint32_t count;
        std::cin.read(reinterpret_cast<char*>(&count), sizeof(count));
        for (uint32_t ci = 0; ci < count; ++ci) {
            std::array<uint32_t, 4> h;
            std::cin.read(reinterpret_cast<char*>(h.data()), sizeof(h));
            std::vector<float> rows(size_t(h[0]) * h[1] * h[3]), y(h[0] + 16, -987654.0f);
            std::cin.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(float));
            if (!std::cin) throw std::runtime_error("truncated input");
            const auto saved = rows;
            const uint32_t paddedM = (h[1] + 15) / 16 * 16;
            std::vector<float> padded(size_t(h[0]) * h[3] * paddedM, -INFINITY);
            for (uint32_t b = 0; b < h[0]; ++b)
                for (uint32_t g = 0; g < h[3]; ++g)
                    std::copy_n(rows.data() + (size_t(b) * h[3] + g) * h[1], h[1],
                                padded.data() + (size_t(b) * h[3] + g) * paddedM);
            const auto savedPadded = padded;
            const auto invoke = [&] {
                sim::LaunchMixed(h[2], [&] {
                    bmmms::Schedule p{};
                    p.batches=h[0]; p.m=h[1]; p.n=16; p.k=32;
                    p.tileM=16; p.tileN=16; p.mTiles=paddedM/16; p.paddedM=paddedM;
                    p.nSplits=h[3]; p.workers=2*h[2];
                    AscendC::TPipe pipe;
                    bmmms::NativeCube<half, false, false> op;
                    op.Init(nullptr, nullptr, nullptr, reinterpret_cast<uint8_t*>(padded.data()),
                            reinterpret_cast<uint8_t*>(y.data()), p, {}, &pipe);
                    op.Finish();
                });
            };
            invoke();
            const auto first = y;
            invoke();
            if (rows != saved || padded != savedPadded || std::memcmp(y.data(), first.data(), y.size() * sizeof(float)))
                throw std::runtime_error("input mutation or nondeterministic Finish");
            for (uint32_t j = h[0]; j < y.size(); ++j)
                if (y[j] != -987654.0f) throw std::runtime_error("output guard changed");
            std::cout.write(reinterpret_cast<const char*>(y.data()), h[0] * sizeof(float));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
