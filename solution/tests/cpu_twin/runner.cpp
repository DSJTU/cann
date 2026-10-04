#include "adapter.h"
#include "kernel_twin.inc"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

struct GmBuffer {
    uint8_t* data;
    explicit GmBuffer(size_t bytes) : data(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))) {
        if (!data) throw std::bad_alloc();
    }
    ~GmBuffer() { AscendC::GmFree(data); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
};
static void Read(std::ifstream& input, void* data, size_t bytes) {
    input.read(static_cast<char*>(data), bytes);
    if (!input) throw std::runtime_error("truncated input");
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s input.bin output.bin\n", argv[0]);
        return 1;
    }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        std::ofstream output(argv[2], std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open input/output");
        uint32_t count;
        Read(input, &count, sizeof(count));
        for (uint32_t index = 0; index < count; ++index) {
            std::array<uint32_t, 9> h;
            Read(input, h.data(), h.size() * sizeof(uint32_t));
            const auto batch = h[0], m = h[1], n = h[2], k = h[3], dtype = h[4];
            const bool ta = h[5], tb = h[6];
            // Explicit low core counts keep CPU debugging practical. No silent
            // change to the requested partitioning or to the operator dispatch.
            if (!h[7]) throw std::runtime_error("CPU Twin fixtures must specify a nonzero core count");
            if (!batch || batch > 64 || !m || m > 8192 || !n || n > 8192 ||
                k < 32 || k > 8192 || k % 8 || (dtype != 1 && dtype != 2) ||
                uint64_t(batch) * m * k > (1u << 26) || uint64_t(batch) * n * k > (1u << 26))
                throw std::runtime_error("invalid dimensions");
            std::vector<uint16_t> a(size_t(batch) * m * k), b(size_t(batch) * n * k);
            Read(input, a.data(), a.size() * 2);
            Read(input, b.data(), b.size() * 2);
            GmBuffer da(a.size() * 2), db(b.size() * 2), dy((batch + 16) * sizeof(float));
            std::memcpy(da.data, a.data(), a.size() * 2);
            std::memcpy(db.data, b.data(), b.size() * 2);
            int64_t as[] = {batch, ta ? k : m, ta ? m : k};
            int64_t bs[] = {batch, tb ? n : k, tb ? k : n};
            int64_t ys[] = {batch};
            TensorInfo ai{as, 3, int32_t(dtype)}, bi{bs, 3, int32_t(dtype)}, yi{ys, 1, 0};
            TensorGroupInfo ag{&ai, 1}, bg{&bi, 1}, yg{&yi, 1};
            std::vector<float> result(batch + 16, -987654.0f), first;
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::fill(result.begin(), result.end(), -987654.0f);
                std::memcpy(dy.data, result.data(), result.size() * sizeof(float));
                const auto before = TwinLaunchCount();
                run_kernel(da.data, ag, db.data, bg, dy.data, yg, h[7], nullptr, ta, tb);
                if (TwinLaunchCount() != before + 1) throw std::runtime_error("CPU launch count differs from one");
                std::memcpy(result.data(), dy.data, result.size() * sizeof(float));
                for (uint32_t i = 0; i < batch; ++i)
                    if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite output");
                for (uint32_t i = batch; i < result.size(); ++i)
                    if (result[i] != -987654.0f) throw std::runtime_error("output guard changed");
                if (repeat == 0) first = result;
                else if (std::memcmp(first.data(), result.data(), result.size() * sizeof(float)))
                    throw std::runtime_error("nondeterministic output");
            }
            if (std::memcmp(da.data, a.data(), a.size() * 2) || std::memcmp(db.data, b.data(), b.size() * 2))
                throw std::runtime_error("input mutation");
            output.write(reinterpret_cast<const char*>(result.data()), batch * sizeof(float));
            output.flush();
            if (!output) throw std::runtime_error("output write failed");
            std::fprintf(stderr, "TWIN_CASE %u B=%u M=%u N=%u K=%u dtype=%u ta=%u tb=%u cores=%u\n",
                index, batch, m, n, k, dtype, ta, tb, h[7]);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CPU Twin failed: %s\n", error.what());
        return 2;
    }
}
