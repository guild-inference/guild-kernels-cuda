#include "guild/kernels/bf16_gemv.hpp"
#include "guild/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace GUILD_CUDA_DENSE_MATH_NS {
using guild::kernels::f32_from_bf16;
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
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
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

__host__ __device__ constexpr int mmvf_log2(int m) { return m <= 1 ? 0 : 1 + mmvf_log2(m / 2); }

// M values a lane (M a power of two, at most 32) summed over the warp by mmvf_warp_sum's xor tree, M at once: at
// offset 16 a lane keeps half of the values (the upper half when its bit 4 is set) and adds its partner's copy of
// them, at offset 8 half of those, and so on; then the plain tree over the offsets left.  Every addition is one
// mmvf_warp_sum makes for that value (own + partner's, both the same partial sums, and a + b == b + a), so lane l
// returns, bit for bit, mmvf_warp_sum of value l >> (5 - log2 M), with M - 1 + 5 - log2 M shuffles instead of 5 M.
template <int M>
__device__ __forceinline__ float mmvf_warp_sum_many(float (&v)[M], int lane) {
    static_assert(M >= 1 && M <= 32 && (M & (M - 1)) == 0, "1..32 values, a power of two");
#pragma unroll
    for (int stage = 0; stage < mmvf_log2(M); ++stage) {
        const int h = M >> (stage + 1), offset = 16 >> stage;
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < M / 2; ++i) {
            if (i < h) {
                const float send = upper ? v[i] : v[i + h], keep = upper ? v[i + h] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, offset, 32);
            }
        }
    }
    float value = v[0];
#pragma unroll
    for (int offset = 16 >> mmvf_log2(M); offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) partials[t] = 0.0f;
        __syncthreads();
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT, bool Tiled = false>
__global__ void bf16_f32_mmvf_multi_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_tok) {
    if constexpr (Tiled) {
        const int first = int(blockIdx.y) * NT;
        x += size_t(first) * ldx;
        y += size_t(first) * ldy;
        n_tok = min(NT, n_tok - first);
    }
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        __syncthreads();
    }
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// R output rows a block, for row counts like a router's (n_out >= 64) and rows of exactly P pairs a thread.  The
// arithmetic of every output is the kernels' above: thread t's pairs t, t + B, ... in order with two ordered FMAs
// each, the warp's xor tree, then the B / 32 warp partials summed by the same xor tree with zeros in the other
// lanes, so each output is bitwise its bf16_f32_mmvf_kernel launch at every column count.  What changes is the
// schedule: a thread issues the loads of all R x P of its weight words before any arithmetic, and reads each
// activation pair once for R rows.  The one-row kernels run a block of 256 threads for 5 pairs and a reduction.  In
// the decode engine on a GTX 1070 (nsys, verify windows), a 512 x 2560 router took 16.1 us at one column and 22.8 at
// two with them (2.6 MB at 163 and 115 GB/s), 15.0 and 16.9 us with two rows a block; 128 x 2560 6.5 / 10.0 -> 5.6 /
// 6.1 us, 2560 x 2560 63.9 / 86.5 -> 62.5 / 67.6 us.
template <int B, int P, int R, int NT>
__global__ void __launch_bounds__(B) bf16_f32_mmvf_rows_kernel(const float* __restrict__ x, int64_t ldx,
                                                               const uint16_t* __restrict__ w, float* __restrict__ y,
                                                               int64_t ldy, int n_in, int n_out, int n_tok) {
    constexpr int NW = B / 32;
    static_assert(NW >= 2 && B % 32 == 0, "the block reduction needs at least two warps");
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int row0 = blockIdx.x * R;
    __shared__ float partials[R * NT][32];
    // lanes NW..31 of every final sum are zeros (the kernels above zero all 32 and overwrite the first NW)
    for (int i = t; i < R * NT * 32; i += B)
        if ((i & 31) >= NW) (&partials[0][0])[i] = 0.0f;
    uint32_t wv[R][P];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // a partial last block re-reads its last row; those sums are not stored
        const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w + (size_t) min(row0 + r, n_out - 1) * n_in);
#pragma unroll
        for (int p = 0; p < P; ++p) wv[r][p] = w2[t + p * B];
    }
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[t + p * B];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r][k] = __fmaf_rn(f32_from_bf16((uint16_t) wv[r][p]), input.x, acc[r][k]);
                    acc[r][k] = __fmaf_rn(f32_from_bf16((uint16_t) (wv[r][p] >> 16)), input.y, acc[r][k]);
                }
            }
        }
    }
    // the warp sums, M = R x NT at once (in chunks of 32), then lane 0 of each value's lanes stores it
    constexpr int M = R * NT, MC = M < 32 ? M : 32, LC = mmvf_log2(MC);
#pragma unroll
    for (int c0 = 0; c0 < M; c0 += MC) {
        float v[MC];
#pragma unroll
        for (int j = 0; j < MC; ++j) v[j] = acc[(c0 + j) / NT][(c0 + j) % NT];
        const float sum = mmvf_warp_sum_many<MC>(v, lane);
        if ((lane & ((32 >> LC) - 1)) == 0) partials[c0 + (lane >> (5 - LC))][warp] = sum;
    }
    __syncthreads();
    // the final sums: C values a warp, the same tree with the zero lanes
    constexpr int C = M > NW ? M / NW : 1, LF = mmvf_log2(C);
    if (warp * C < M) {
        float u[C];
#pragma unroll
        for (int c = 0; c < C; ++c) u[c] = partials[warp * C + c][lane];
        const float sum = mmvf_warp_sum_many<C>(u, lane);
        const int j = warp * C + (lane >> (5 - LF)), r = j / NT, k = j % NT;
        if ((lane & ((32 >> LF) - 1)) == 0 && k < n_tok && row0 + r < n_out) y[(size_t) k * ldy + row0 + r] = sum;
    }
}

// GUILD_MMVF_ROWS: rows a block for the kernel above (1, 2, 4 or 8; 0 = the one-row kernels).  2 by default on
// CUDA: on a GTX 1070, 512 x 2560 at one, two and four columns took 14.2, 15.6 and 21.7 us a call against 15.1, 21.7
// and 23.6 for the one-row kernels (mmvf_rows_parity --bench); 4 and 8 rows were slower at two columns.  HIP: opt-in.
int mmvf_rows() {
    static const int rows = [] {
        const char* v = std::getenv("GUILD_MMVF_ROWS");
#if defined(__HIPCC__)
        const int r = v ? std::atoi(v) : 0;
#else
        const int r = v ? std::atoi(v) : 2;
#endif
        return r == 1 || r == 2 || r == 4 || r == 8 ? r : 0;
    }();
    return rows;
}

template <int P, int R>
void mmvf_rows_launch(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in, int n_out,
                      int n_tok, cudaStream_t st) {
    const unsigned grid = (unsigned) ((n_out + R - 1) / R);
    if (n_tok == 1)
        bf16_f32_mmvf_rows_kernel<256, P, R, 1><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, 1);
    else if (n_tok == 2)
        bf16_f32_mmvf_rows_kernel<256, P, R, 2><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, 2);
    else if (n_tok <= 4)
        bf16_f32_mmvf_rows_kernel<256, P, R, 4><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
    else
        bf16_f32_mmvf_rows_kernel<256, P, R, 8><<<grid, 256, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
}

template <int P>
void mmvf_rows_launch(int rows, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in,
                      int n_out, int n_tok, cudaStream_t st) {
    switch (rows) {
        case 1: mmvf_rows_launch<P, 1>(x, ldx, w, y, ldy, n_in, n_out, n_tok, st); break;
        case 2: mmvf_rows_launch<P, 2>(x, ldx, w, y, ldy, n_in, n_out, n_tok, st); break;
        case 4: mmvf_rows_launch<P, 4>(x, ldx, w, y, ldy, n_in, n_out, n_tok, st); break;
        default: mmvf_rows_launch<P, 8>(x, ldx, w, y, ldy, n_in, n_out, n_tok, st); break;
    }
}

// the rows kernel when it applies: n_in 2048, 2560 or 4096 (256-thread blocks, the one-row kernels' choice for these
// widths, and exactly 4, 5 or 8 pairs a thread) and at least 64 rows; false: the caller runs the one-row kernels
bool mmvf_rows_try(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
                   int64_t n_out, int n_tok, cudaStream_t st) {
    const int rows = mmvf_rows();
    if (rows == 0 || n_out < 64 || n_out > std::numeric_limits<int>::max() || n_tok < 1 || n_tok > 8 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0)
        return false;
    switch (n_in) {
        case 2048: mmvf_rows_launch<4>(rows, x, ldx, w, y, ldy, 2048, (int) n_out, n_tok, st); return true;
        case 2560: mmvf_rows_launch<5>(rows, x, ldx, w, y, ldy, 2560, (int) n_out, n_tok, st); return true;
        case 4096: mmvf_rows_launch<8>(rows, x, ldx, w, y, ldy, 4096, (int) n_out, n_tok, st); return true;
        default: return false;
    }
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
    if (mmvf_rows_try(x, ldx, w, y, ldy, n_in, n_out, n_tok, st)) {
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
        return;
    }
#define GUILD_MMVF_M(N) case N: \
    if (n_tok <= 4) bf16_f32_mmvf_multi_kernel<N, 4><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); \
    else bf16_f32_mmvf_multi_kernel<N, 8><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); break
    switch (mmvf_block_size(n_in)) {
        GUILD_MMVF_M(32); GUILD_MMVF_M(64); GUILD_MMVF_M(96); GUILD_MMVF_M(128);
        GUILD_MMVF_M(160); GUILD_MMVF_M(192); GUILD_MMVF_M(224); GUILD_MMVF_M(256);
    }
#undef GUILD_MMVF_M
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
    if (mmvf_rows_try(x, n_in, w, y, n_out, n_in, n_out, 1, st)) {
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
        return;
    }
#define GUILD_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        GUILD_MMVF_CASE(32);
        GUILD_MMVF_CASE(64);
        GUILD_MMVF_CASE(96);
        GUILD_MMVF_CASE(128);
        GUILD_MMVF_CASE(160);
        GUILD_MMVF_CASE(192);
        GUILD_MMVF_CASE(224);
        GUILD_MMVF_CASE(256);
    }
#undef GUILD_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                               void* stream) {
    static const bool tiled = [] {
        const char* v = std::getenv("GUILD_PREFILL_BATCH_PROJECTIONS");
        return !v || std::atoi(v) != 0;
    }();
    if (ncols > 8 && tiled) {
        if (ncols > 512 || n_in <= 0 || n_in > std::numeric_limits<int>::max() || (n_in & 1) ||
            n_out <= 0 || n_out > std::numeric_limits<int>::max() || !x || !w || !y ||
            (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
            throw std::invalid_argument("bf16_gemv_fp32_mmvf_cols: invalid batched projection");
        const dim3 grid(unsigned(n_out), unsigned((ncols + 7) / 8));
#define GUILD_MMVF_TILE(N) case N: \
        bf16_f32_mmvf_multi_kernel<N, 8, true><<<grid, N, 0, (cudaStream_t) stream>>>(x, n_in, w, y, n_out, int(n_in), ncols); break
        switch (mmvf_block_size(n_in)) {
            GUILD_MMVF_TILE(32); GUILD_MMVF_TILE(64); GUILD_MMVF_TILE(96); GUILD_MMVF_TILE(128);
            GUILD_MMVF_TILE(160); GUILD_MMVF_TILE(192); GUILD_MMVF_TILE(224); GUILD_MMVF_TILE(256);
        }
#undef GUILD_MMVF_TILE
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return;
    }
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}

}  // namespace guild::kernels
