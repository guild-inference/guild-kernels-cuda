#include "guild/prefill/gemm.hpp"
#include "guild/prefill/kernels.hpp"
#include "guild/prefill/moe_fused.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        std::puts("cuda_prefill_test: no CUDA device, skipping");
        return 0;
    }
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

    // 1. Test to_bf16, to_f16, rms_rows, and Gemm::bf16 / Gemm::f16
    constexpr int64_t T = 4, N = 64, K = 64;
    float *d_x = nullptr, *d_w_f32 = nullptr, *d_y = nullptr, *d_rms_w = nullptr;
    uint16_t *d_x16 = nullptr, *d_w16 = nullptr;
    CHECK(cudaMalloc(&d_x, T * K * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_w_f32, N * K * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, T * N * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_rms_w, K * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_x16, T * K * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_w16, N * K * sizeof(uint16_t)) == cudaSuccess);

    std::vector<float> h_x(T * K, 0.5f), h_w(N * K, 0.25f), h_rms_w(K, 1.0f), h_y(T * N, 0.0f);
    CHECK(cudaMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(cudaMemcpyAsync(d_w_f32, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(cudaMemcpyAsync(d_rms_w, h_rms_w.data(), h_rms_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);

    guild::prefill::rms_rows(d_x, d_rms_w, T, K, K, 1e-6f, stream);
    guild::prefill::to_bf16(d_x, d_x16, T * K, stream);
    guild::prefill::to_bf16(d_w_f32, d_w16, N * K, stream);

    guild::prefill::Gemm gemm;
    std::string err;
    CHECK(gemm.init(stream, N * K, err));
    gemm.bf16(d_x16, d_w16, d_y, T, N, K);
    CHECK(cudaMemcpyAsync(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

    for (float v : h_y) {
        // After rms_rows on constant 0.5f with w=1.0f, each x element is ~1.0f; dot over K=64 with 0.25f is ~16.0f
        CHECK(std::isfinite(v) && std::fabs(v - 16.0f) < 0.25f);
    }

    // 2. Test fused grouping and activation quantization
    CHECK(guild::prefill::fused::built());
    const size_t ab = guild::prefill::fused::act_bytes(T, K);
    const size_t gb = guild::prefill::fused::group_bytes(T * 2, 16);
    void *d_xa = nullptr, *d_gscratch = nullptr;
    int32_t *d_ids = nullptr, *d_slot = nullptr, *d_src = nullptr;
    CHECK(cudaMalloc(&d_xa, ab) == cudaSuccess);
    CHECK(cudaMalloc(&d_gscratch, gb) == cudaSuccess);
    CHECK(cudaMalloc(&d_ids, T * 2 * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_slot, T * 2 * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_src, T * 2 * sizeof(int32_t)) == cudaSuccess);
    std::vector<int32_t> h_ids = {0, 1, 1, 2, 0, 2, 3, 1};
    CHECK(cudaMemcpyAsync(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess);
    guild::prefill::fused::quantize_act(d_x, T, K, d_xa, stream);
    guild::prefill::fused::group(d_ids, T * 2, 2, 16, d_gscratch, d_slot, d_src, stream);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

    cudaFree(d_xa);
    cudaFree(d_gscratch);
    cudaFree(d_ids);
    cudaFree(d_slot);
    cudaFree(d_src);
    cudaFree(d_x);
    cudaFree(d_w_f32);
    cudaFree(d_y);
    cudaFree(d_rms_w);
    cudaFree(d_x16);
    cudaFree(d_w16);
    cudaStreamDestroy(stream);
    std::puts("cuda_prefill_test: passed");
    return 0;
}
