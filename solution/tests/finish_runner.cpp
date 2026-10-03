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
            const auto invoke = [&] {
                sim::Launch(h[2], [&] {
                    bmmms_cube::Finish(reinterpret_cast<uint8_t*>(rows.data()),
                        reinterpret_cast<uint8_t*>(y.data()), h[0], h[1], h[3], h[2]);
                });
            };
            invoke();
            const auto first = y;
            invoke();
            if (rows != saved || std::memcmp(y.data(), first.data(), y.size() * sizeof(float)))
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
