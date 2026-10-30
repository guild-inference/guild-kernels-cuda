// src/attention/gdn_multi.cu - multi-token GDN convolution, A/B projection, and recurrence kernels.
#include "guild/kernels/gdn_multi.hpp"
#include "guild/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifndef GUILD_CUDA_ATTN_MATH_NS
#define GUILD_CUDA_ATTN_MATH_NS guild::kernels
#endif
namespace GUILD_CUDA_ATTN_MATH_NS {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template <int MAX_T = kVerifyMaxT, bool Tiled = false>
__global__ void __launch_bounds__(64) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                          const uint16_t* __restrict__ wb,
                                                          const float* __restrict__ dt,
                                                          const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                          float* __restrict__ beta, int n, int h_v, int T) {
    if constexpr (Tiled) {
        const int first = int(blockIdx.y) * MAX_T;
        x += size_t(first) * n;
        gate += size_t(first) * h_v;
        beta += size_t(first) * h_v;
        T = min(MAX_T, T - first);
    }
    const int row = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
__global__ void __launch_bounds__(32 * RG) gdn_step_commit_kernel(float* __restrict__ state,
                                                                  const float* __restrict__ hbuf, int C,
                                                                  const float* __restrict__ gate,
                                                                  const float* __restrict__ beta,
                                                                  int h_k, int h_v,
                                                                  const int32_t* __restrict__ n_keep) {
    const int n = *n_keep;
    if (n <= 0) return;
    __shared__ float sk[S];
    __shared__ float red[RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin,
                                                                     bool gate_silu) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = (t >= t_out_begin);
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) {
            sk[tid] = ht[qk + qh * S + tid];
            if (need_out) sq[tid] = ht[qh * S + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float gz = gate_silu ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * gz;
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// The verify windows' recurrence and the commit with a head's 128 state columns split over four blocks of 32: 48 heads
// x 4 = 192 blocks of 128 threads at 64 registers a thread, eight blocks an SM, where gdn_step_norm_multi_kernel runs
// 48 blocks of 512 threads, one an SM.  gdn_step_commit_kernel's layout: a warp per row group, a thread per column,
// the four row groups' partial sums added through shared memory in gdn_step_norm_multi_kernel's order (red[0] +
// red[1] + red[2] + red[3]); every product, fma chain and sum is the one that kernel computes, so the outputs and the
// state are bitwise its own.  Verify (COMMIT false): the state is read and not written, and each output token's
// column value before the norm (oc) goes to y; gdn_out_norm_kernel then applies the gated RMSNorm, which needs the
// head's 128 columns.  Commit (COMMIT true): the first *n_keep tokens are run, there are no outputs, and the state is
// written.  HV > 0 is the value-head count as a constant, so a thread's 32 state rows are one base address plus
// immediate offsets.  The store addresses are formed again after the token loop rather than kept live from the loads
// (gdn_step_commit_kernel keeps them: 110 registers against 68).  Measured on the GTX 1070 (gdn_step_parity --bench,
// 36 layers' states in a graph, us a call): verify 1 token 22.9 -> 18.8, 2 tokens 30.5 -> 21.9, 4 tokens with 2
// outputs 38.5 -> 26.1; commit n_keep 1 / 2 / 4 32.1 / 33.4 / 38.6 -> 31.7 / 32.5 / 35.5.  In the engine (nsys of the
// short-context decode, medians a layer, the norm pass included): a 1-token row group 26.7 -> 21.8 us, a 2-token one
// 38.6 -> 26.9 us; the commit 40.5 -> 40.1 us.
template <bool COMMIT, int HV>
__global__ void __launch_bounds__(32 * RG) gdn_step_cols_kernel(float* __restrict__ state,
                                                                const float* __restrict__ hbuf, int C,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ y, int h_k, int h_v_arg, int T,
                                                                int t_out_begin,
                                                                const int32_t* __restrict__ n_keep) {
    const int h_v = HV > 0 ? HV : h_v_arg;
    int n = T;
    if (COMMIT) {
        n = *n_keep;
        if (n <= 0) return;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = !COMMIT && t >= t_out_begin;
        __syncthreads();                    // the previous token is done with sk/sq/red
        sk[tid] = ht[qk + qh * S + tid];
        if (need_out) sq[tid] = ht[qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col =
            __fadd_rn(__fadd_rn(__fadd_rn(red[0][col_local], red[1][col_local]), red[2][col_local]), red[3][col_local]);
        const float delta =
            __fmul_rn(__fmaf_rn(-g, kv_col, ht[2 * qk + head * S + col]), beta[(size_t) t * h_v + head]);
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], __fmul_rn(sk[rg * RPG + r], delta));
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], __fmul_rn(sk[rg * RPG + r], delta));
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();                    // every row group has read red[][] (kv)
        red[rg][col_local] = o;
        __syncthreads();
        if (rg == 0)
            y[(size_t) t * value_dim + head * S + col] = __fmul_rn(
                __fadd_rn(__fadd_rn(__fadd_rn(red[0][col_local], red[1][col_local]), red[2][col_local]),
                          red[3][col_local]),
                rsqrtf((float) S));
    }
    if (COMMIT) {
#if !defined(__HIPCC__)
        asm volatile("" : "+l"(base));
#endif
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// The gated RMSNorm of gdn_step_cols_kernel's output tokens, in place in y: a block per (head, token), the thread
// of column c in lane c % 32 of warp c / 32 as in gdn_step_norm_multi_kernel's row group 0, the same warp sums
// added as wsum[0] + wsum[1] + wsum[2] + wsum[3].
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, int h_v, int t_out_begin,
                                                         bool gate_silu) {
    __shared__ float wsum[S / 32];
    const int head = blockIdx.x;
    const int t = t_out_begin + (int) blockIdx.y;
    const int col = threadIdx.x;
    const size_t i = (size_t) t * S * h_v + head * S + col;
    const float oc = y[i];
    float sq_part = __fmul_rn(oc, oc);
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part = __fadd_rn(sq_part, __shfl_xor_sync(0xffffffffu, sq_part, o2));
    if ((col & 31) == 0) wsum[col >> 5] = sq_part;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float scale = rsqrtf(ss / (float) S + eps);
    const float zz = z[i];
    const float gz = gate_silu ? (zz / (1.0f + __expf(-zz))) : (1.0f / (1.0f + __expf(-zz)));
    y[i] = oc * scale * gamma[col] * gz;
}

template <bool COMMIT>
void launch_step_cols(float* state, const float* h, int C, const float* gate, const float* beta, float* y, int h_k,
                      int h_v, int T, int t_out_begin, const int32_t* n_keep, cudaStream_t stream) {
    const dim3 grid((unsigned) h_v, (unsigned) (S / 32)), block(32, RG);
    if (h_v == 48)
        gdn_step_cols_kernel<COMMIT, 48><<<grid, block, 0, stream>>>(state, h, C, gate, beta, y, h_k, h_v, T,
                                                                      t_out_begin, n_keep);
    else
        gdn_step_cols_kernel<COMMIT, 0><<<grid, block, 0, stream>>>(state, h, C, gate, beta, y, h_k, h_v, T,
                                                                     t_out_begin, n_keep);
}

}  // namespace

void gdn_prefill_conv(const float* history, const float* qkv, const float* conv_w, float* h,
                      int channels, int qk_heads, float eps, int n_tok, void* stream) {
    if (!history || !qkv || !conv_w || !h || !stream || channels % S || n_tok < 1 || n_tok > 512)
        throw std::invalid_argument("GDN prefill: invalid convolution buffers or batch");
    gdn_conv_l2_multi_kernel<<<dim3(unsigned(channels / S), unsigned(n_tok)), S, 0,
        static_cast<cudaStream_t>(stream)>>>(history, qkv, conv_w, h, channels, qk_heads, eps, 0);
    check("gdn_prefill_conv");
}

void gdn_prefill_recurrence(float* state, const float* h, int channels, const float* gate,
                            const float* beta, const float* z, const float* gamma, float eps,
                            float* y, int h_k, int h_v, int n_tok, const int32_t* keep,
                            bool gate_silu, void* stream) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || !keep || !stream ||
        h_k < 1 || h_v % h_k || n_tok < 1 || n_tok > 512)
        throw std::invalid_argument("GDN prefill: invalid recurrence buffers or batch");
    gdn_step_norm_multi_kernel<<<unsigned(h_v), dim3(S, RG), 0, static_cast<cudaStream_t>(stream)>>>(
        state, h, channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, keep, 0, gate_silu);
    check("gdn_prefill_recurrence");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    if (n_tok <= 4) {
        gdn_ab_multi_kernel<4><<<blocks, 64, 0, (cudaStream_t) stream>>>(
            x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    } else {
        gdn_ab_multi_kernel<kVerifyMaxT><<<blocks, 64, 0, (cudaStream_t) stream>>>(
            x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    }
    check("gdn_ab_multi");
}

void gdn_ab_prefill(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                    float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd <= 0 || n_embd % 8 || h_v <= 0 || n_tok < 1 || n_tok > 512)
        throw std::invalid_argument("gdn_ab_prefill: invalid projection shape");
    if (n_tok <= kVerifyMaxT) {
        gdn_ab_multi(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok, stream);
        return;
    }
    gdn_ab_multi_kernel<kVerifyMaxT, true><<<dim3(unsigned(h_v), unsigned((n_tok + kVerifyMaxT - 1) / kVerifyMaxT)), 64, 0, (cudaStream_t) stream>>>(
        x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    check("gdn_ab_prefill");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, bool gate_silu) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("GUILD_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    // GUILD_GDN_STEP_COLS=0: the verify recurrence in gdn_step_norm_multi_kernel and the commit in
    // gdn_step_commit_kernel, the kernels before gdn_step_cols_kernel (the same bits).  Measured on CUDA only, so
    // HIP builds take the column kernels with GUILD_GDN_STEP_COLS=1.
    static const bool cols = [] {
        const char* e = std::getenv("GUILD_GDN_STEP_COLS");
#if defined(__HIPCC__)
        return e != nullptr && e[0] != '0';
#else
        return !e || e[0] != '0';
#endif
    }();
    const auto cs = (cudaStream_t) stream;
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        if (cols)
            launch_step_cols<true>(state, h, conv_channels, gate, beta, y, h_k, h_v, n_tok, n_tok, n_keep, cs);
        else
            gdn_step_commit_kernel<<<dim3((unsigned) h_v, 4u), dim3(32, RG), 0, cs>>>(state, h, conv_channels, gate,
                                                                                      beta, h_k, h_v, n_keep);
    } else if (cols && n_keep == nullptr) {
        launch_step_cols<false>(state, h, conv_channels, gate, beta, y, h_k, h_v, n_tok, t_out_begin, nullptr, cs);
        if (t_out_begin < n_tok)
            gdn_out_norm_kernel<<<dim3((unsigned) h_v, (unsigned) (n_tok - t_out_begin)), S, 0, cs>>>(
                z, gamma, eps, y, h_v, t_out_begin, gate_silu);
    } else {
        gdn_step_norm_multi_kernel<<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin,
            gate_silu);
    }
    check("gdn_step_norm_multi");
}

}  // namespace GUILD_CUDA_ATTN_MATH_NS
