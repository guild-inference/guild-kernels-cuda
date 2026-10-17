// include/guild/kernels/gdn_multi.hpp - multi-token GDN convolution, A/B projection, and recurrence kernels.
#pragma once

#include <cstdint>

#ifndef GUILD_CUDA_ATTN_MATH_NS
#define GUILD_CUDA_ATTN_MATH_NS guild::kernels
#endif
namespace GUILD_CUDA_ATTN_MATH_NS {

inline constexpr int kVerifyMaxT = 8;

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written. Bitwise `fused_gdn_conv_l2` per token.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v). Bitwise `fused_gdn_ab` per column.
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// Prefill: independent eight-token tiles in grid.y, preserving the verify reduction tree.
void gdn_ab_prefill(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                    float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)). With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch). Bitwise `fused_gdn_step_norm`.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0, bool gate_silu = false);
/// Known prompt rows, up to 512: the SAME ordered recurrence/reduction as verification.
/// Writes all outputs and commits state once. No associative scan or precision change.
/// keep points to n_tok in device memory; history is committed separately after conv.
void gdn_prefill_conv(const float* history, const float* qkv, const float* conv_w, float* h,
                      int channels, int qk_heads, float eps, int n_tok, void* stream);
void gdn_prefill_recurrence(float* state, const float* h, int channels, const float* gate,
                            const float* beta, const float* z, const float* gamma, float eps,
                            float* y, int h_k, int h_v, int n_tok, const int32_t* keep,
                            bool gate_silu, void* stream);

}  // namespace GUILD_CUDA_ATTN_MATH_NS
