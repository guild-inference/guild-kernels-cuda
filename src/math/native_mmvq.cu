// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/dp4a.hpp"
#include "guild/kernels/q8_1_finite.hpp"
#include "guild/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace GUILD_CUDA_DENSE_MATH_NS {
#ifndef GUILD_CUDA_MOE_MATH_NS
#define GUILD_CUDA_MOE_MATH_NS guild::kernels
#endif
using namespace guild::kernels;
using namespace GUILD_CUDA_MOE_MATH_NS;
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* __restrict__ x,
                                           Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                                   const float* __restrict__ up,
                                                   Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = make_half2(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = GUILD_DP4A(v0i, u[2 * i], GUILD_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = GUILD_DP4A(0x01010101, u[2 * i], GUILD_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = GUILD_DP4A(u, qx, sumi);
        sumi = GUILD_DP4A(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (GUILD_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
#if defined(GUILD_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (see iq_kernels.cu get_int_from_table_16): 4 v_perm_b32 per 8 values
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = GUILD_DP4A(v.x, u0, sumi);
        sumi = GUILD_DP4A(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = GUILD_DP4A(v1i, u[2 * i + 1], GUILD_DP4A(v0i, u[2 * i], 0));
        const int dot2 = GUILD_DP4A(0x01010101, u[2 * i + 1], GUILD_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (GUILD_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = GUILD_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = GUILD_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = GUILD_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = GUILD_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = GUILD_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = GUILD_DP4A(v.x, q8[i], sumi);
        sumi = GUILD_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 512;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        return dot(r, load_x(bq8, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u[4]; float d8[2]; };
    __device__ static X load_x(const Q81Block* __restrict__ bq8, int iqs) {
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + bq8_offset + i;
            v.d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            v.u[2 * i] = q8[0];
            v.u[2 * i + 1] = q8[4];
        }
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        const uint8_t sc[2] = {uint8_t(r.aux[0] & 0xffu), uint8_t(r.aux[0] >> 8)};
        const uint8_t m[2] = {uint8_t(r.aux[1] & 0xffu), uint8_t(r.aux[1] >> 8)};
        return q5_q8_dot_impl(r.vl, r.vh, v.u, sc, m, r.dm, v.d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        return dot(r, load_x(bq8, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u[4]; float d8[2]; };
    __device__ static X load_x(const Q81Block* __restrict__ bq8, int iqs) {
        const int bq8_offset = 2 * ((iqs / 2) / 4);
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + bq8_offset + i;
            v.d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            v.u[2 * i] = q8[0];
            v.u[2 * i + 1] = q8[4];
        }
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        const uint8_t sc[2] = {uint8_t(r.aux[0] & 0xffu), uint8_t(r.aux[0] >> 8)};
        const uint8_t m[2] = {uint8_t(r.aux[1] & 0xffu), uint8_t(r.aux[1] >> 8)};
        return q4_q8_dot_impl(r.v, v.u, sc, m, r.dm, v.d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = GUILD_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = GUILD_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vi[4]; int8_t sc[4]; half d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int bq8_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        const int vl = load_int_b2(w->qs, iqs);
        const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int isc = scale_offset + 2 * i;
            const int isc_low = isc % 8;
            const int sc_shift_low = 4 * (isc / 8);
            const int sc_low = (w->scales[isc_low] >> sc_shift_low) & 0xf;
            const int isc_high = isc % 4;
            const int sc_shift_high = 2 * (isc / 4);
            const int sc_high = ((w->scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
            r.sc[i] = int8_t((sc_low | sc_high) - 32);
            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;
            r.vi[i] = __vsubss4(vil, vih);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int bq8_offset = 4 * (iqs / 8);
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
            const float d8 = __low2float(x[bq8_offset + i].ds);
            sumf += d8 * (GUILD_DP4A(r.vi[i], u, 0) * int(r.sc[i]));
        }
        return float(r.d) * sumf;
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vi[2]; int8_t sc[2]; half d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = load_int_b2(w->ql, iqs);
        const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        const int8_t* scales = w->scales + scale_offset;
        r.d = w->d;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.sc[i] = scales[4 * i];
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            r.vi[i] = __vsubss4(vil | vih, 0x20202020);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u[2]; float d8[2]; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v.u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
            v.d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
        }
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) sumf += v.d8[i] * (GUILD_DP4A(r.vi[i], v.u[i], 0) * int(r.sc[i]));
        return float(r.d) * sumf;
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u0[4], u1[4]; float d8; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        X v;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v.u0[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            v.u1[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        }
        v.d8 = __low2float(x[iqs / 4].ds);
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = GUILD_DP4A(r.v[j].x, v.u0[j], sumi);
            sumi = GUILD_DP4A(r.v[j].y, v.u1[j], sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * v.d8;
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    __device__ static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    __device__ static W load(const Block* __restrict__ w, int) { return W{w}; }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int k) { return small_q8_dot(r.w, x, k); }
};
template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u0[2], u1[2]; float2 ds; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v.u0[i] = reinterpret_cast<const int*>(x->qs)[iqs + i];
            v.u1[i] = reinterpret_cast<const int*>(x->qs)[iqs + i + 4];
        }
        v.ds = __half22float2(x->ds);
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = GUILD_DP4A(r.vi0[i], v.u0[i], sumi);
            sumi = GUILD_DP4A(r.vi1[i], v.u1[i], sumi);
        }
        return r.d * (sumi * v.ds.x - 4 * v.ds.y);
    }
};
template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int vh0 = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = vh0 >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u0[2], u1[2]; float2 ds; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v.u0[i] = reinterpret_cast<const int*>(x->qs)[iqs + i];
            v.u1[i] = reinterpret_cast<const int*>(x->qs)[iqs + i + 4];
        }
        v.ds = __half22float2(x->ds);
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = GUILD_DP4A(r.vi0[i], v.u0[i], sumi);
            sumi = GUILD_DP4A(r.vi1[i], v.u1[i], sumi);
        }
        return r.d * (sumi * v.ds.x - 8 * v.ds.y);
    }
};
template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        return W{{load_int_b2(w->qs, iqs), load_int_b2(w->qs, iqs + 1)}, float(w->d)};
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int u[2]; float d1; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        const int* q = reinterpret_cast<const int*>(x->qs) + iqs;
        return X{{q[0], q[1]}, __low2float(x->ds)};
    }
    __device__ static float dot(const W& r, const X& v) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) sumi = GUILD_DP4A(r.v[i], v.u[i], sumi);
        return r.d0 * v.d1 * float(sumi);
    }
};
template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return dot(r, load_x(x, iqs));
    }
    // The activation load and the product apart, so a block of several rows loads each column once
    // (native_mmvq_multi_xr_kernel).
    struct X { int q0[2], q1[2]; float d8; };
    __device__ static X load_x(const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v.q0[i] = q8[i];
            v.q1[i] = q8[i + 4];
        }
        v.d8 = __low2float(x->ds);
        return v;
    }
    __device__ static float dot(const W& r, const X& v) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = GUILD_DP4A(r.v[i].x, v.q0[i], sumi);
            sumi = GUILD_DP4A(r.v[i].y, v.q1[i], sumi);
        }
        const float d = r.dw * v.d8;
        return d * sumi;
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
thread_local bool g_multi_exact = true; // explicit CDD1 per-call configuration; no cross-thread mutable dispatch state

template<typename F, int NCOLS, int NW, int ROWS>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<typename F, typename = void>
struct has_split_x : std::false_type {};
template<typename F>
struct has_split_x<F, std::void_t<typename F::X>> : std::true_type {};

// native_mmvq_multi_kernel for traits with load_x/dot: each column's activation values are loaded once per block
// step and used for all ROWS rows (the plain kernel loads them again for every row).  Every row keeps its thread
// mapping, its order of += and the cross-warp sum, so each output is bitwise the one-row kernel's.
template<typename F, int NCOLS, int NW, int ROWS>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_xr_kernel(const typename F::Block* __restrict__ w,
                                            const Q81Block* __restrict__ x,
                                            float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPI = F::BPI * NW / WARPS;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
        typename F::X xv[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) xv[j] = F::load_x(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::dot(wv, xv[j]);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        native_mmvq_multi_kernel<F, NCOLS, NW, 2><<<blocks, dim3(WARP, NW), 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    const dim3 threads(WARP, WARPS);
    if constexpr (has_split_x<F>::value) {
        // Rows per block, A/B in one process on a GTX 1070 (us, one-row kernel -> this): Q8_0 2560 -> 10240 at
        // 2/3/4 columns 171/203/212 -> 154/166/170 with 4 rows; the Q6_K head 2560 -> 248320 4297/5197/6137 ->
        // 4143/4669/5190 with 2 rows.  Four rows lost on every other format, and on Q8_0 beyond four columns.
        constexpr int ROWS = std::is_same<typename F::Block, Q80Block>::value && NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
        native_mmvq_multi_xr_kernel<F, NCOLS, WARPS, ROWS><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    if (n_in / F::DIV < F::BPI) {
        constexpr int ROWS = 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
        native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
}

// tmp[j][i] += row i's block (wv[i]) . column j's activations at x + j * x_stride, for the rows i < nr and the
// columns j < nc.  For traits with load_x/dot a column's activations are loaded once for every row (F::apply is
// F::dot(wv, F::load_x(...))): each tmp[j][i] gets the same += either way.
template<typename F, int ROWS, int TILE>
__device__ __forceinline__ void acc_tile(float (&tmp)[TILE][ROWS], const typename F::W (&wv)[ROWS],
                                         const Q81Block* __restrict__ x, int x_stride, int kqs, int nr, int nc) {
    if constexpr (has_split_x<F>::value && ROWS > 1) {
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
            if (j < nc) {
                const typename F::X xv = F::load_x(x + std::size_t(j) * x_stride, kqs);
#pragma unroll
                for (int i = 0; i < ROWS; ++i)
                    if (i < nr) tmp[j][i] += F::dot(wv[i], xv);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (i < nr) {
#pragma unroll
                for (int j = 0; j < TILE; ++j)
                    if (j < nc) tmp[j][i] += F::apply(wv[i], x + std::size_t(j) * x_stride, kqs);
            }
        }
    }
}

template<typename F, int NW, int ROWS, int MAX_IT>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_many_reg_kernel(const typename F::Block* __restrict__ w,
                                            const Q81Block* __restrict__ x,
                                            float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int BPI = F::BPI * NW / WARPS;
    constexpr int TILE = 8;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kbx0 = tid / F::T;
    const int kqs = F::kqs(tid);

    typename F::W wv[MAX_IT][ROWS];
#pragma unroll
    for (int it = 0; it < MAX_IT; ++it) {
        const int kbx = kbx0 + it * BPI;
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
                    const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                    wv[it][i] = F::load(w + block, kqs);
                }
            }
        }
    }

    __shared__ float partial[2][NW][TILE][ROWS][WARP];
    int buf = 0;
    for (int c0 = 0; c0 < ncols; c0 += TILE, buf ^= 1) {
        const int nc = min(TILE, ncols - c0);
        float tmp[TILE][ROWS] = {};
        const int nr = min(ROWS, n_out - row0);
#pragma unroll
        for (int it = 0; it < MAX_IT; ++it) {
            const int kbx = kbx0 + it * BPI;
            if (kbx < blocks_per_row) {
                const int kby = kbx * F::KBY;
                if (nc == TILE)
                    acc_tile<F, ROWS, TILE>(tmp, wv[it], x + std::size_t(c0) * x_stride + kby, x_stride, kqs, nr, TILE);
                else
                    acc_tile<F, ROWS, TILE>(tmp, wv[it], x + std::size_t(c0) * x_stride + kby, x_stride, kqs, nr, nc);
            }
        }
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                partial[buf][threadIdx.y][j][i][threadIdx.x] = tmp[j][i];
            }
        }
        __syncthreads();
        for (int j0 = int(threadIdx.y); j0 < TILE / 2; j0 += NW) {
            const int j1 = j0 + TILE / 2;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                float s0 = partial[buf][0][j0][i][threadIdx.x];
                float s1 = partial[buf][0][j1][i][threadIdx.x];
#pragma unroll
                for (int l = 1; l < NW; ++l) {
                    s0 += partial[buf][l][j0][i][threadIdx.x];
                    s1 += partial[buf][l][j1][i][threadIdx.x];
                }
                s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
                float s = (threadIdx.x < 16) ? s0 : s1;
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    s += __shfl_xor_sync(0xffffffffu, s, offset);
                }
                if (row0 + i < n_out) {
                    if (threadIdx.x == i && c0 + j0 < ncols) {
                        y[std::size_t(c0 + j0) * n_out + row0 + i] = s;
                    }
                    if (threadIdx.x == 16 + i && c0 + j1 < ncols) {
                        y[std::size_t(c0 + j1) * n_out + row0 + i] = s;
                    }
                }
            }
        }
    }
}

template<typename F, int NW, int ROWS>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_many_loop_kernel(const typename F::Block* __restrict__ w,
                                             const Q81Block* __restrict__ x,
                                             float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int BPI = F::BPI * NW / WARPS;
    constexpr int TILE = 8;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(tid);
    __shared__ float partial[2][NW][TILE][ROWS][WARP];
    int buf = 0;
    for (int c0 = 0; c0 < ncols; c0 += TILE, buf ^= 1) {
        const int nc = min(TILE, ncols - c0);
        float tmp[TILE][ROWS] = {};
        const int nr = min(ROWS, n_out - row0);
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
            const int kby = kbx * F::KBY;
            typename F::W wv[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (i < nr) wv[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
            acc_tile<F, ROWS, TILE>(tmp, wv, x + std::size_t(c0) * x_stride + kby, x_stride, kqs, nr, nc);
        }
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                partial[buf][threadIdx.y][j][i][threadIdx.x] = tmp[j][i];
            }
        }
        __syncthreads();
        for (int j0 = int(threadIdx.y); j0 < TILE / 2; j0 += NW) {
            const int j1 = j0 + TILE / 2;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                float s0 = partial[buf][0][j0][i][threadIdx.x];
                float s1 = partial[buf][0][j1][i][threadIdx.x];
#pragma unroll
                for (int l = 1; l < NW; ++l) {
                    s0 += partial[buf][l][j0][i][threadIdx.x];
                    s1 += partial[buf][l][j1][i][threadIdx.x];
                }
                s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
                float s = (threadIdx.x < 16) ? s0 : s1;
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    s += __shfl_xor_sync(0xffffffffu, s, offset);
                }
                if (row0 + i < n_out) {
                    if (threadIdx.x == i && c0 + j0 < ncols) {
                        y[std::size_t(c0 + j0) * n_out + row0 + i] = s;
                    }
                    if (threadIdx.x == 16 + i && c0 + j1 < ncols) {
                        y[std::size_t(c0 + j1) * n_out + row0 + i] = s;
                    }
                }
            }
        }
    }
}

template<typename F, int NW, int ROWS>
void launch_many_rows(const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    constexpr int BPI = F::BPI * NW / WARPS;
    const int blocks_per_row = n_in / F::DIV;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    const dim3 threads(WARP, NW);
    if (blocks_per_row <= BPI) {
        native_mmvq_many_reg_kernel<F, NW, ROWS, 1><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out, ncols);
    } else if (blocks_per_row <= 2 * BPI) {
        native_mmvq_many_reg_kernel<F, NW, ROWS, 2><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out, ncols);
    } else if (blocks_per_row <= 4 * BPI) {
        native_mmvq_many_reg_kernel<F, NW, ROWS, 4><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out, ncols);
    } else {
        native_mmvq_many_loop_kernel<F, NW, ROWS><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out, ncols);
    }
}

template<typename F>
void launch_many(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        launch_many_rows<F, 2, 2>(w, x, y, n_in, n_out, ncols, s);
        return;
    }
    // Two rows a block where the format loads a column's activations apart (acc_tile: once for both rows): a
    // prompt's 128 columns on a GTX 1070, one row -> two (us, the same bits): Q8_0 2048 -> 8192 3352 -> 2398,
    // 4096 -> 2048 1636 -> 1110; Q4_K 2048 -> 8192 3148 -> 2248, 4096 -> 2048 1952 -> 1297; Q6_K 2048 -> 8192
    // 3321 -> 2702.  Four rows were slower on every one.
    if (has_split_x<F>::value || n_in / F::DIV < F::BPI) {
        launch_many_rows<F, WARPS, 2>(w, x, y, n_in, n_out, ncols, s);
    } else {
        launch_many_rows<F, WARPS, 1>(w, x, y, n_in, n_out, ncols, s);
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    if (ncols > 8) {
        launch_many<F>(weights, x_q8_1, y, n_in, n_out, ncols, s);
        return;
    }
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 512");
    }
}

// ---- Staged rows (CUDA): a warp a row, read through shared memory with 16-byte loads ----
//
// The kernels above read a row with each format's own loads: Q6_K's and Q8_0's blocks (210 and 34 bytes) are only
// 2-byte aligned, so each int of weights is two 2-byte loads, and Q4_K's ints are 4-byte loads.  On a GTX 1070 that
// is 155-198 GB/s at one column, of the card's ~220, and less with more columns.  Here a warp owns a row at a time:
// it copies C blocks of it into shared memory with aligned 16-byte loads, the next C blocks (of this row or of its
// next one) arriving in registers while it computes on the last, and the block's warps read the activations of
// every column from shared memory.  Lane l stands for the threads l, 32 + l, 64 + l and 96 + l of the ncols = 1
// kernel: the same blocks in the same order into four sums, added ((0 + 1) + 2) + 3 and then across the lanes by
// warp_sum, so each output is bitwise the ncols = 1 kernel's, at every column count.  GUILD_MMVQ_STAGED=0 turns it
// off; tools: Guild's mmvq_staged_parity (bitwise) and `glm5_kernels_test shapes` (GB/s).
#if !defined(GUILD_USE_HIP)
bool g_staged_off = std::getenv("GUILD_MMVQ_STAGED") && std::string(std::getenv("GUILD_MMVQ_STAGED")) == "0";

// C: blocks a warp stages at a time.  MAX_COLS: the most columns that are faster than launch_multi's kernels on a
// GTX 1070 (Q8_0 2560 -> 10240: 161 against 151 GB/s at 5 columns but 147 against 161 for 6144 -> 2560; Q4_K at 4
// columns: 89-102 against 97-108).  ROLL: from this many columns the four sums are taken one after the other in a
// loop (rotating the accumulators) instead of unrolled, for fewer registers: Q6_K's heads at two columns 160 -> 163
// and 181 -> 193 GB/s; not Q8_0's sums, too short for it (202 -> 169 at three columns), nor Q4_K's, where the
// compiler's choices made three columns 127 -> 133 in one build and -> 121 in the next.
template<typename F> struct Staged { static constexpr int C = 0; }; // the other formats: not staged
// the rolled loop's threshold on the chunked path (rows of more than ROWC blocks): ROLL unless a format says otherwise
template<typename F> struct RollChunked { static constexpr int v = Staged<F>::ROLL; };
//
// XREC, put, get: the activations as the format's loads read them, written once per block when the columns are copied
// in, so a lane takes its ints and its scale in one 8- or 16-byte load instead of three to six 4-byte ones.  The same
// values (the scale is __low2float(ds), as load_x computes it), so get gives load_x's X bit for bit.  They take 1.8x
// (Q6_K) or 2.7x (Q4_K) the shared memory of the Q8_1 blocks, so they are used while a block's shared memory stays
// within 32 KiB (three blocks an SM before Volta): Q6_K 2048 -> 248320 at 2 / 3 / 4 columns 192 -> 208, 159 -> 176,
// 126 -> 145 GB/s on a GTX 1070, but 2560 wide at 4 columns (34 KiB) 113 -> 105.  Not for one column (the same speed)
// nor for Q8_0 (no gain measured).
//
// ROWC: rows of more than C and at most ROWC blocks are staged whole, one chunk a row, by kernels of their own (ONE in
// native_mmvq_staged_kernel), instead of as a chunk of C blocks and a short one (a 2560-wide Q6_K row is 10 blocks: 8
// and 2).  Those kernels also add each of the four sums, whole, to a running total instead of keeping four rotated
// accumulators, carry a row's address on to the warp's next row, take Q6_K's scale product in 16 bits (Staged::dot),
// read the records whenever they fit a block's 48 KiB (their registers hold an SM to two blocks from two columns
// on, so the records cost no blocks), and are built with __launch_bounds__(256, 1), whose register choices measured
// fastest.  Q6_K 2560 -> 248320 on a GTX 1070, us at 1 to 8 columns: 2505 3224 3631 4780 4961 6141 7372 7627 before,
// 2429 2439 2707 3049 3477 3923 5451 6069 after (without the 1-block bound: 2438 2493 2888 3457 4401 5530 5708 6565);
// 2304 and 3072 wide (9 and 12 blocks) at 4 columns 1064 -> 737 and 1451 -> 921 us.
// GUILD_MMVQ_ROW_CHUNK=0 turns them off.
template<> struct Staged<Q4KTraits> {                                                           // 1152 bytes
    static constexpr int C = 8, MAX_COLS = 3, ROLL = 9, ROWC = 0;
    // slot s = kqs / 2 % 4 of a Q8_1 block: its ints s and s + 4 and the scale.  6 slots of 16 bytes a block, so
    // the two blocks a quarter-warp reads (two apart) fall in different banks.
    static constexpr int XREC = 96, XSLOTS = 4;
    __device__ static void put(uint8_t* rec, const Q81Block* __restrict__ b, int s) {
        const int* q = reinterpret_cast<const int*>(b->qs);
        *reinterpret_cast<int4*>(rec + 16 * s) = make_int4(q[s], q[s + 4], __float_as_int(__low2float(b->ds)), 0);
    }
    __device__ static Q4KTraits::X get(const uint8_t* xs, int kbx, int iqs) {
        const int bq8_offset = 2 * ((iqs / 2) / 4), sl = (iqs / 2) % 4;
        Q4KTraits::X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int4 r = *reinterpret_cast<const int4*>(xs + (kbx * 8 + bq8_offset + i) * XREC + 16 * sl);
            v.u[2 * i] = r.x;
            v.u[2 * i + 1] = r.y;
            v.d8[i] = __int_as_float(r.z);
        }
        return v;
    }
};
// Q5_K: Q4_K's activation slots (the same ints and scale a thread reads), its blocks 176 bytes, so a row stays
// 16-byte aligned.  Its own kernel reads each int with a load: 163 against 214 GB/s for GLM-5.3-Flash's 4096 x 24576
// at one column on a GTX 1070 (glm5_kernels_test shapes).
template<> struct Staged<Q5KTraits> {                                                           // 1408 bytes
    static constexpr int C = 8, MAX_COLS = 3, ROLL = 9, ROWC = 0;
    static constexpr int XREC = Staged<Q4KTraits>::XREC, XSLOTS = Staged<Q4KTraits>::XSLOTS;
    __device__ static void put(uint8_t* rec, const Q81Block* __restrict__ b, int s) {
        Staged<Q4KTraits>::put(rec, b, s);
    }
    __device__ static Q5KTraits::X get(const uint8_t* xs, int kbx, int iqs) {
        const Q4KTraits::X v = Staged<Q4KTraits>::get(xs, kbx, iqs);
        Q5KTraits::X r;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.u[i] = v.u[i];
        r.d8[0] = v.d8[0];
        r.d8[1] = v.d8[1];
        return r;
    }
};
template<> struct Staged<Q6KTraits> {                                                           // 1680 bytes
    static constexpr int C = 8, MAX_COLS = 8, ROLL = 2, ROWC = 12;
    // int q = 0..7 of a Q8_1 block with the scale
    static constexpr int XREC = 64, XSLOTS = 8;
    __device__ static void put(uint8_t* rec, const Q81Block* __restrict__ b, int s) {
        *reinterpret_cast<int2*>(rec + 8 * s) =
            make_int2(reinterpret_cast<const int*>(b->qs)[s], __float_as_int(__low2float(b->ds)));
    }
    __device__ static Q6KTraits::X get(const uint8_t* xs, int kbx, int iqs) {
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        Q6KTraits::X v;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int2 r = *reinterpret_cast<const int2*>(xs + (kbx * 8 + bq8_offset + 2 * i) * XREC + 8 * (iqs % 8));
            v.u[i] = r.x;
            v.d8[i] = __int_as_float(r.y);
        }
        return v;
    }
    // Q6KTraits::dot with the scale product taken in 16 bits: the dp4a of four 6-bit weights (-32..31) and four
    // int8 activations lies in -16256..16384, so (short) keeps it whole and the one-instruction 16 x 16-bit multiply
    // gives the same int as the 32-bit one (three instructions on sm_61).  The one-chunk rows only: 2560 -> 248320
    // at 2 / 3 / 4 / 5 columns 2697 -> 2431, 3218 -> 3053, 4325 -> 4041, 4850 -> 4422 us, before the other changes.
    __device__ static float dot(const Q6KTraits::W& r, const Q6KTraits::X& v) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) sumf += v.d8[i] * (int(short(GUILD_DP4A(r.vi[i], v.u[i], 0))) * int(r.sc[i]));
        return float(r.d) * sumf;
    }
};
// Q6_K's rows of more than 12 blocks (Qwen3.8-Flash-Next's ssm_out, 24 blocks; attn_output, 16), its four sums
// unrolled up to 8 columns: at two columns ssm_out 84.9 -> 75.7 us and attn_output 57.4 -> 49.5 on a GTX 1070
// (glm5_kernels_test shapes), three and four columns faster too.  The rolled loop pays where these rows were staged
// as 8 + 2 blocks (the heads, now one chunk a row: ROWC).  (One chunk a row up to 24 blocks instead, ROWC = 24, made
// every one-chunk kernel's two columns 1.6x slower: their staging registers.)
template<> struct RollChunked<Q6KTraits> { static constexpr int v = 9; };
template<> struct Staged<SmallTraits<Q80Block, 8>> {                                            // 1088 bytes
    static constexpr int C = 32, MAX_COLS = 4, ROLL = 9, ROWC = 0, XREC = 0;
};
// the bytes of shared memory a column's activations take, as records (XR) or as Q8_1 blocks
template<typename F, bool XR>
__host__ __device__ constexpr int staged_xcol(int n_in) {
    return XR ? (n_in / Q8K) * Staged<F>::XREC : (n_in / Q8K) * int(sizeof(Q81Block));
}
constexpr int STAGED_NW = 8; // warps a block


// the bytes of shared memory a warp stages a chunk in: its 16-byte pieces, and one more where the row starts inside
// a piece.  A kernel whose rows are one chunk each (C > Staged<F>::C) takes only the row's.
template<typename F, int C>
__host__ __device__ constexpr int staged_wbytes(int n_in) {
    constexpr int CB = C * int(sizeof(typename F::Block));
    return C == Staged<F>::C ? CB + 16 : (n_in / F::DIV * int(sizeof(typename F::Block)) + 30) / 16 * 16;
}

template<typename F, int NCOLS, bool XR, int C>
__launch_bounds__(STAGED_NW * WARP, C != Staged<F>::C ? 1 : 0)
__global__ void native_mmvq_staged_kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x,
                                          float* __restrict__ y, int n_in, int n_out) {
    using Block = typename F::Block;
    // ONE: every row is one chunk (the host checked n_in / F::DIV <= C, see Staged::ROWC), so a chunk need not be
    // whole 16-byte pieces
    constexpr bool ONE = C != Staged<F>::C;
    constexpr int BS = int(sizeof(Block)), CB = C * BS;
    // 16-byte pieces a lane carries: a chunk's, and one more where the row starts inside a piece
    constexpr int KV = ((CB + 30) / 16 + WARP - 1) / WARP;
    static_assert((ONE || CB % 16 == 0) && C % F::BPI == 0 && F::BPI * F::T == WARPS * WARP && WARP % F::T == 0,
                  "a chunk must be whole 16-byte pieces and whole steps of the ncols = 1 kernel");
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb = n_in / F::DIV, xb = n_in / Q8K, xcol = staged_xcol<F, XR>(n_in), xbytes = NCOLS * xcol;
    if constexpr (XR) {
        for (int i = int(threadIdx.x); i < NCOLS * xb * Staged<F>::XSLOTS; i += STAGED_NW * WARP) {
            const int g = i / Staged<F>::XSLOTS; // column * xb + Q8_1 block
            Staged<F>::put(smem + g * Staged<F>::XREC, x + g, i - g * Staged<F>::XSLOTS);
        }
    } else {
        const uint8_t* xg = reinterpret_cast<const uint8_t*>(x);
        const int n16 = reinterpret_cast<std::uintptr_t>(xg) % 16 == 0 ? xbytes / 16 : 0;
        for (int i = int(threadIdx.x); i < n16; i += STAGED_NW * WARP)
            reinterpret_cast<int4*>(smem)[i] = __ldg(reinterpret_cast<const int4*>(xg) + i);
        for (int o = 16 * n16 + 4 * int(threadIdx.x); o < xbytes; o += 4 * STAGED_NW * WARP)
            *reinterpret_cast<int*>(smem + o) = __ldg(reinterpret_cast<const int*>(xg + o));
    }
    __syncthreads();
    const int warp = int(threadIdx.x) / WARP, lane = int(threadIdx.x) % WARP;
    int r = int(blockIdx.x) * STAGED_NW + warp; // this warp's rows: r, r + rstep, ...
    if (r >= n_out) return;
    const Q81Block* xs = reinterpret_cast<const Q81Block*>(smem);
    uint8_t* sm = smem + (xbytes + 15) / 16 * 16 + warp * staged_wbytes<F, C>(n_in);
    const std::size_t rb = std::size_t(nb) * BS;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(w), end = base + std::size_t(n_out) * rb;
    const int nchunk = ONE ? 1 : (nb + C - 1) / C, rstep = int(gridDim.x) * STAGED_NW;
    // the aligned pieces under chunk c of row rr: the host checked that w is 16-byte aligned, so only the matrix's
    // last piece can run past its end, and that one is read in 2-byte parts
    auto pieces = [&](int rr, int c, std::uintptr_t& a0) {
        const std::uintptr_t g0 = base + std::size_t(rr) * rb + std::size_t(c) * CB;
        a0 = g0 & ~std::uintptr_t(15);
        return int((g0 + min(std::size_t(CB), rb - std::size_t(c) * CB) - a0 + 15) / 16);
    };
    int4 pf[KV];
    auto fetch = [&](int rr, int c) {
        std::uintptr_t a0;
        const int n = pieces(rr, c, a0);
#pragma unroll
        for (int k = 0; k < KV; ++k) {
            const int p = lane + WARP * k;
            if (p >= n) break;
            const std::uintptr_t a = a0 + 16 * std::uintptr_t(p);
            if (a + 16 <= end) {
                pf[k] = __ldg(reinterpret_cast<const int4*>(a));
            } else {
                uint32_t v[4] = {0u, 0u, 0u, 0u};
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (a + 2 * j < end)
                        v[j / 2] |= uint32_t(__ldg(reinterpret_cast<const uint16_t*>(a) + j)) << (16 * (j % 2));
                pf[k] = make_int4(int(v[0]), int(v[1]), int(v[2]), int(v[3]));
            }
        }
    };
    // ONE: a warp's rows are rstep apart, so the next row's address is this one's plus rstep rows (no 64-bit product
    // a row), and a row's pieces are read without a test each unless they reach the matrix's last piece
    const std::uintptr_t rsb = std::uintptr_t(rstep) * rb;
    std::uintptr_t g = base + std::size_t(r) * rb; // row r's first byte (ONE)
    auto fetch1 = [&](std::uintptr_t g0, int rr) {
        const std::uintptr_t a0 = g0 & ~std::uintptr_t(15);
        const int n = int((g0 + rb - a0 + 15) / 16);
        if (a0 + 16 * std::uintptr_t(n) <= end) {
#pragma unroll
            for (int k = 0; k < KV; ++k)
                if (lane + WARP * k < n) pf[k] = __ldg(reinterpret_cast<const int4*>(a0) + lane + WARP * k);
        } else {
            fetch(rr, 0);
        }
    };
    float acc[NCOLS][WARPS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int t = 0; t < WARPS; ++t) acc[j][t] = 0.0f;
    if constexpr (ONE)
        fetch1(g, r);
    else
        fetch(r, 0);
    for (int c = 0;;) {
        std::uintptr_t a0;
        int n;
        if constexpr (ONE)
            n = int((g + rb - (g & ~std::uintptr_t(15)) + 15) / 16);
        else
            n = pieces(r, c, a0);
        __syncwarp();
#pragma unroll
        for (int k = 0; k < KV; ++k)
            if (lane + WARP * k < n) reinterpret_cast<int4*>(sm)[lane + WARP * k] = pf[k];
        __syncwarp();
        const int nr = c + 1 < nchunk ? r : r + rstep, nc = c + 1 < nchunk ? c + 1 : 0;
        if constexpr (ONE) {
            if (nr < n_out) fetch1(g + rsb, nr);
        } else {
            if (nr < n_out) fetch(nr, nc);
        }
        const uint8_t* sb = sm + int((ONE ? g : base + std::size_t(r) * rb) & 15);
        // the ncols = 1 kernel's thread WARP * t + lane: its blocks of this chunk into sum t
        auto step = [&](int t, int s) {
            const int tid = WARP * t + lane, kqs = F::kqs(tid);
#pragma unroll
            for (int i = 0; i < C / F::BPI; ++i) {
                const int b = tid / F::T + i * F::BPI, kbx = c * C + b;
                if (kbx < nb) {
                    const typename F::W wv = F::load(reinterpret_cast<const Block*>(sb + b * BS), kqs);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        if constexpr (ONE && XR)
                            acc[j][s] += Staged<F>::dot(wv, Staged<F>::get(smem + j * xcol, kbx, kqs));
                        else if constexpr (ONE)
                            acc[j][s] += Staged<F>::dot(wv, F::load_x(xs + j * xb + kbx * F::KBY, kqs));
                        else if constexpr (XR)
                            acc[j][s] += F::dot(wv, Staged<F>::get(smem + j * xcol, kbx, kqs));
                        else
                            acc[j][s] += F::apply(wv, xs + j * xb + kbx * F::KBY, kqs);
                    }
                }
            }
        };
        if constexpr (ONE) {
            // the row is this one chunk, so sum t is whole after step t: a running total adds the four sums in the
            // same order, ((0 + 1) + 2) + 3, without keeping them apart (half the accumulators, none rotated)
            float tot[NCOLS];
            auto add = [&](int t) {
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    tot[j] = t == 0 ? acc[j][0] : tot[j] + acc[j][0];
                    acc[j][0] = 0.0f;
                }
            };
            if constexpr (NCOLS >= Staged<F>::ROLL) {
#pragma unroll 1
                for (int t = 0; t < WARPS; ++t) {
                    step(t, 0);
                    add(t);
                }
            } else {
#pragma unroll
                for (int t = 0; t < WARPS; ++t) {
                    step(t, 0);
                    add(t);
                }
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const float v = warp_sum(tot[j]);
                if (lane == 0) y[std::size_t(j) * n_out + r] = v;
            }
        } else if constexpr (NCOLS >= RollChunked<F>::v) {
#pragma unroll 1
            for (int t = 0; t < WARPS; ++t) {
                step(t, 0);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) { // sum t + 1 to the front; after all four, each back in place
                    const float a = acc[j][0];
#pragma unroll
                    for (int u = 0; u + 1 < WARPS; ++u) acc[j][u] = acc[j][u + 1];
                    acc[j][WARPS - 1] = a;
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < WARPS; ++t) step(t, t);
        }
        if (!ONE && c + 1 == nchunk) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                float v = acc[j][0];
#pragma unroll
                for (int t = 1; t < WARPS; ++t) v += acc[j][t];
                v = warp_sum(v);
                if (lane == 0) y[std::size_t(j) * n_out + r] = v;
#pragma unroll
                for (int t = 0; t < WARPS; ++t) acc[j][t] = 0.0f;
            }
        }
        if (nr >= n_out) break;
        r = nr;
        c = nc;
        if constexpr (ONE) g += rsb;
    }
}

struct StagedDevice {
    int dev = -1, sms = 0;
};
thread_local StagedDevice g_staged_dev;

// As many blocks as the GPU holds at once, and every warp the same number of rows (a warp takes every
// (blocks x 8)-th row): a warp's next row loads while it finishes the last, and a block copies the activations once
// for all its rows.  The blocks a GPU holds depend on the shared memory, so they are kept per KiB of it.
template<typename F, int NCOLS, bool XR, int C>
void staged_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, std::size_t smem,
                     cudaStream_t s) {
    thread_local int dev_of = -1, resident[49] = {};
    if (g_staged_dev.dev != dev_of) {
        dev_of = g_staged_dev.dev;
        for (int& b : resident) b = 0;
    }
    int& held = resident[(smem + 1023) / 1024];
    if (held == 0) {
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, native_mmvq_staged_kernel<F, NCOLS, XR, C>,
                                                      STAGED_NW * WARP, (smem + 1023) / 1024 * 1024);
        held = per_sm > 0 ? g_staged_dev.sms * per_sm : 1;
    }
    const std::size_t per_pass = std::size_t(STAGED_NW) * held, rows = (std::size_t(n_out) + per_pass - 1) / per_pass;
    const unsigned blocks = unsigned((std::size_t(n_out) + STAGED_NW * rows - 1) / (STAGED_NW * rows));
    native_mmvq_staged_kernel<F, NCOLS, XR, C><<<blocks, STAGED_NW * WARP, smem, s>>>(
        static_cast<const uint8_t*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out);
}

// staged_launch_n<F, ncols, xr, C> for NCOLS <= ncols <= the format's MAX_COLS
template<typename F, int C, int NCOLS>
void staged_launch(int ncols, bool xr, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                   std::size_t smem, cudaStream_t s) {
    if constexpr (NCOLS <= Staged<F>::MAX_COLS) {
        if (ncols != NCOLS)
            staged_launch<F, C, NCOLS + 1>(ncols, xr, weights, x_q8_1, y, n_in, n_out, smem, s);
        else if constexpr (Staged<F>::XREC > 0 && NCOLS > 1)
            if (xr)
                staged_launch_n<F, NCOLS, true, C>(weights, x_q8_1, y, n_in, n_out, smem, s);
            else
                staged_launch_n<F, NCOLS, false, C>(weights, x_q8_1, y, n_in, n_out, smem, s);
        else
            staged_launch_n<F, NCOLS, false, C>(weights, x_q8_1, y, n_in, n_out, smem, s);
    }
}
// GUILD_MMVQ_ROW_CHUNK=0: rows of C < blocks <= ROWC in chunks of C, as before the one-chunk kernels (Staged::ROWC)
bool g_row_chunk_off = std::getenv("GUILD_MMVQ_ROW_CHUNK") && std::string(std::getenv("GUILD_MMVQ_ROW_CHUNK")) == "0";

// false: the caller takes its usual path
template<typename F>
bool staged_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if constexpr (Staged<F>::C == 0) {
        return false;
    } else {
        // the upstream multi-column layout asked for, or a row base not 16-byte aligned
        if (g_staged_off || ncols > Staged<F>::MAX_COLS || (ncols > 1 && !g_multi_exact) ||
            reinterpret_cast<std::uintptr_t>(weights) % 16 != 0)
            return false;
        const int nb = n_in / F::DIV;
        const bool one = Staged<F>::ROWC > 0 && !g_row_chunk_off && nb > Staged<F>::C && nb <= Staged<F>::ROWC;
        const int wbytes = one ? staged_wbytes<F, Staged<F>::ROWC>(n_in) : staged_wbytes<F, Staged<F>::C>(n_in);
        auto smem_of = [&](bool xr) {
            const std::size_t xbytes = std::size_t(ncols) * std::size_t(xr ? staged_xcol<F, true>(n_in)
                                                                            : staged_xcol<F, false>(n_in));
            return (xbytes + 15) / 16 * 16 + std::size_t(STAGED_NW) * wbytes;
        };
        // one column: the Q8_1 blocks (the records measured the same there, so no second kernel); the one-chunk
        // kernels take the records whenever they fit (Staged::ROWC)
        const bool xr = Staged<F>::XREC > 0 && ncols > 1 && smem_of(true) <= std::size_t(one ? 48 : 32) * 1024;
        const std::size_t smem = smem_of(xr);
        if (smem > 48 * 1024) return false; // every column's activations beside the rows, without an opt-in
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (dev != g_staged_dev.dev) {
            g_staged_dev.dev = dev;
            if (cudaDeviceGetAttribute(&g_staged_dev.sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
                g_staged_dev.sms = 0;
        }
        // too little to keep the GPU's warps busy with a row each (GTX 1070, 15 SMs: Q8_0 2560 -> 512, 1.4 MB, 135
        // against 147 GB/s; 2560 -> 640, 1.7 MB, 153 against 154; Q4_K 2048 -> 512 96 against 108)
        if (std::size_t(n_out) * (n_in / F::DIV) * sizeof(typename F::Block) <
            std::size_t(g_staged_dev.sms) * 100 * 1024)
            return false;
        if constexpr (Staged<F>::ROWC > 0)
            if (one) {
                staged_launch<F, Staged<F>::ROWC, 1>(ncols, xr, weights, x_q8_1, y, n_in, n_out, smem,
                                                     static_cast<cudaStream_t>(stream));
                return true;
            }
        staged_launch<F, Staged<F>::C, 1>(ncols, xr, weights, x_q8_1, y, n_in, n_out, smem,
                                          static_cast<cudaStream_t>(stream));
        return true;
    }
}
#else
template<typename F>
bool staged_mmvq(const void*, const void*, float*, int, int, int, void*) { return false; }
#endif

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 512");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}


#if defined(GUILD_HIP_GFX906)
// ---- AMD (wave64) layout: one wavefront per R rows, four wavefronts per block, the row's blocks strided over the
// 64 lanes exactly as the CUDA kernels stride them over a block (kbx = lane / T, stride 64 / T), and a 64-lane
// butterfly instead of the LDS partials and __syncthreads of the one-block-per-row layout: on gfx906 a block per
// 2 KB row spent most of its time being launched and joined.  The SAME kernel serves every column count 1..8, so a
// column's sums do not depend on how many columns (verify tokens) ride along.  GUILD_MMVQ_WAVE=0: the CUDA layout.
bool g_wave_off = std::getenv("GUILD_MMVQ_WAVE") && std::string(std::getenv("GUILD_MMVQ_WAVE")) == "0";
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
template<typename F, int NCOLS>
void wave_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int R = 1;
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    native_mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F>
void wave_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: wave_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: wave_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: wave_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: wave_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: wave_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: wave_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: wave_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: wave_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ (wave) requires 1 <= ncols <= 8");
    }
}
#define GUILD_WAVE_MMVQ(...) \
    if (!g_wave_off) { wave_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream); launch_check(); return; }
#else
#define GUILD_WAVE_MMVQ(...)
#endif

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(SmallTraits<Weight, Qi>)
    if (staged_mmvq<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) {
        launch_check();
        return;
    }
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

} // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                 static_cast<cudaStream_t>(stream)>>>(x, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_swiglu_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                         static_cast<cudaStream_t>(stream)>>>(
        gate, up, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(Q5KTraits)
    if (staged_mmvq<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) {
        launch_check();
        return;
    }
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(Q20Traits)
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(Q3KTraits)
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(IQ4XSTraits)
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(Q4KTraits)
    if (staged_mmvq<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) {
        launch_check();
        return;
    }
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    GUILD_WAVE_MMVQ(Q6KTraits)
    if (staged_mmvq<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) {
        launch_check();
        return;
    }
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

} // namespace guild::kernels
