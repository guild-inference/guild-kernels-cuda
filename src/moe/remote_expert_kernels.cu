#include "guild/kernels/remote_expert_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdint>

#ifndef GUILD_CUDA_MOE_MATH_NS
#define GUILD_CUDA_MOE_MATH_NS guild::kernels
#endif
namespace GUILD_CUDA_MOE_MATH_NS {
namespace {

constexpr int64_t H = 2560;
constexpr int K = 10, MAXT = 8, CAP = MAXT * K;
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

__global__ void reduce_experts(const float* parts, const ReduceMeta* meta, float* sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = meta->row[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
    }
    sum[(int64_t) t * H + c] = v;
}

__global__ void copy_cpu_rows(float4* dst, const volatile float4* src, const int32_t* mask,
                              const int32_t* hit_rows, const int32_t* count) {
    const int row = blockIdx.x;
    __shared__ int skip;
    if (threadIdx.x == 0) {
        int s = mask[row];
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    __syncthreads();
    const int64_t start = (int64_t) row * (H / 4);
    for (int i = threadIdx.x; i < H / 4; i += blockDim.x)
        dst[start + i] = skip ? make_float4(0, 0, 0, 0) : const_cast<const float4*>(src)[start + i];
}

}  // namespace

void remote_expert_reduce(const float* parts, const void* meta, float* sum, int tokens, void* stream) {
    if (tokens <= 0) return;
    reduce_experts<<<dim3((unsigned) ((H + 255) / 256), (unsigned) tokens), 256, 0, (cudaStream_t) stream>>>(
        parts, (const ReduceMeta*) meta, sum);
}

void remote_expert_copy_cpu_rows(float* dst, const float* src, const int32_t* mask,
                                 const int32_t* hit_rows, const int32_t* count, int tokens, void* stream) {
    if (tokens <= 0) return;
    copy_cpu_rows<<<(unsigned) (tokens * K), 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, mask, hit_rows, count);
}

}  // namespace GUILD_CUDA_MOE_MATH_NS
