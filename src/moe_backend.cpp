#include "guild/cuda/moe_backend.h"
#include "guild/kernels/cvec.hpp"
#include "guild/kernels/dequant_bf16.hpp"
#include "guild/kernels/dequant_s2.hpp"
#include "guild/kernels/elementwise.hpp"
#include "guild/kernels/fused_gr.hpp"
#include "guild/kernels/gr.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/native_gr_norm.hpp"
#include "guild/kernels/native_gr_postops.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_ple_postops.hpp"
#include "guild/kernels/native_router.hpp"
#include "guild/kernels/ple.hpp"
#include "guild/kernels/quantize_act.hpp"
#include "guild/kernels/remote_expert_kernels.hpp"
#include "guild/kernels/router_top10.hpp"
#include "guild/kernels/s2_expert_grouped.hpp"
#include "guild/kernels/s2_gemv.hpp"
#include "guild/kernels/s2_gemv_q8.hpp"
#include "guild/kernels/s_gemv.hpp"
#include "guild/kernels/sampler.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "guild/kernels/verify_kernels.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace {
namespace math = guild::cuda_moe_math;

struct AbiMismatch : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

std::atomic<uint64_t> calls[8]{};

void count(unsigned category) {
    static const bool enabled = [] {
        const char* v = std::getenv("GUILD_CUDA_INSTRUMENT");
        return v && std::atoi(v) != 0;
    }();
    if (enabled && category < 8) {
        calls[category].fetch_add(1, std::memory_order_relaxed);
    }
}

guild_status_t fail(guild_error_info_t* error, guild_status_t code, const char* message) {
    if (error) {
        error->struct_size = sizeof(*error);
        error->code = code;
        error->source_component = "cuda.moe.v1";
        std::snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return code;
}

template <class Function>
guild_status_t guarded(guild_error_info_t* error, Function function) noexcept {
    try {
        function();
        return GUILD_STATUS_OK;
    } catch (const std::bad_alloc& e) {
        return fail(error, GUILD_STATUS_ERR_OUT_OF_MEMORY, e.what());
    } catch (const AbiMismatch& e) {
        return fail(error, GUILD_STATUS_ERR_ABI_MISMATCH, e.what());
    } catch (const std::invalid_argument& e) {
        return fail(error, GUILD_STATUS_ERR_INVALID_ARGUMENT, e.what());
    } catch (const std::exception& e) {
        return fail(error, GUILD_STATUS_ERR_EXECUTION_FAILED, e.what());
    } catch (...) {
        return fail(error, GUILD_STATUS_ERR_INTERNAL, "unknown CUDA MoE backend exception");
    }
}

void* checked_stream(const guild_exec_context_v1_t* ctx, const char* name) {
    if (!ctx) throw std::invalid_argument(std::string(name) + ": null exec context");
    if (ctx->struct_size < sizeof(*ctx)) {
        throw AbiMismatch(std::string(name) + ": exec context struct_size is too small");
    }
    if (GUILD_KERNEL_ABI_VERSION_MAJOR_OF(ctx->abi_version) != GUILD_KERNEL_ABI_VERSION_MAJOR) {
        throw AbiMismatch(std::string(name) + ": incompatible exec context ABI");
    }
    if (ctx->backend != GUILD_BACKEND_CUDA || ctx->device_id < 0) {
        throw std::invalid_argument(std::string(name) + ": expected valid CUDA exec context and non-negative device_id");
    }
    return ctx->device_stream;
}

template <class Command>
const Command& checked(const Command* cmd, const char* name) {
    if (!cmd) throw std::invalid_argument(std::string(name) + ": null command");
    if (cmd->struct_size < sizeof(Command)) {
        throw AbiMismatch(std::string(name) + ": command struct_size is too small");
    }
    return *cmd;
}

template <class Nested>
void check_nested(const Nested& s, const char* name, bool allow_zero = false) {
    if (allow_zero && s.struct_size == 0) return;
    if (s.struct_size < sizeof(Nested)) {
        throw AbiMismatch(std::string(name) + ": nested struct_size is too small");
    }
}

math::SForm make_sform(const guild_cuda_sform_v1_t& f) {
    check_nested(f, "sform", true);
    math::SForm out{};
    out.code_bits = f.code_bits;
    out.code_bias = f.code_bias;
    out.group_elems = f.group_elems;
    out.codebook = (f.codebook == 1) ? math::Codebook::Iq4Nl : math::Codebook::Affine;
    out.has_offset = (f.has_offset != 0);
    out.act_kind = f.act_kind;
    return out;
}

math::NativeSharedWeights make_native_shared(const guild_cuda_native_shared_weights_v1_t& ns) {
    check_nested(ns, "native_shared");
    math::NativeSharedWeights nw{};
    nw.gate_type = ns.gate_type;
    nw.up_type = ns.up_type;
    nw.down_type = ns.down_type;
    nw.gate_data = ns.gate_data;
    nw.up_data = ns.up_data;
    nw.down_data = ns.down_data;
    nw.q8_1 = ns.q8_1;
    nw.x_q8_1 = ns.x_q8_1;
    return nw;
}

math::PleWeights make_ple_weights(const guild_cuda_ple_weights_v1_t& pw) {
    check_nested(pw, "ple_weights");
    math::PleWeights w{};
    w.key_codes = pw.key_codes;
    w.key_scales = pw.key_scales;
    w.value_bf16 = pw.value_bf16;
    w.norm_key = pw.norm_key;
    w.norm_query = pw.norm_query;
    w.norm_conv = pw.norm_conv;
    w.conv1d_f16 = pw.conv1d_f16;
    w.key_native_data = pw.key_native_data;
    w.key_native_type = pw.key_native_type;
    w.key_native_q8_1 = pw.key_native_q8_1;
    w.key_bf16 = pw.key_bf16;
    return w;
}

math::PleOut make_ple_out(const guild_cuda_ple_out_v1_t& po) {
    check_nested(po, "ple_out");
    math::PleOut out{};
    out.key = po.key;
    out.value = po.value;
    out.gate = po.gate;
    out.gated = po.gated;
    out.normalized = po.normalized;
    out.conv = po.conv;
    out.result = po.result;
    return out;
}

math::FusedGrArgs make_fused_gr_args(const guild_cuda_fused_gr_args_v1_t& fa) {
    check_nested(fa, "fused_gr_args");
    math::FusedGrArgs a{};
    a.R = fa.R;
    a.R_out = fa.R_out;
    a.apply = (fa.apply != 0);
    a.bo_prev = fa.bo_prev;
    a.inj_prev = fa.inj_prev;
    a.w_norm = fa.w_norm;
    a.w_down = fa.w_down;
    a.w_up = fa.w_up;
    a.w_inject = fa.w_inject;
    a.eps = fa.eps;
    a.lo = fa.lo;
    a.rs = fa.rs;
    a.inject_out = fa.inject_out;
    a.mixed = fa.mixed;
    return a;
}

guild_status_t configure_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_moe_config_v1_t* raw,
                            guild_error_info_t* error) {
    return guarded(error, [&] {
        (void) checked_stream(ctx, "configure");
        const auto& c = checked(raw, "configure");
        switch (c.mode) {
            case GUILD_CUDA_MOE_CFG_SET_FLAGS_V1:
                math::shared_expert_set_native_bf16((c.flags & GUILD_CUDA_MOE_FLAG_SHARED_NATIVE_BF16_V1) != 0);
                math::native_router_set_enabled((c.flags & GUILD_CUDA_MOE_FLAG_NATIVE_ROUTER_V1) != 0);
                math::native_moe_combine_set_enabled((c.flags & GUILD_CUDA_MOE_FLAG_NATIVE_MOE_COMBINE_V1) != 0);
                math::gr_set_fp32_activations((c.flags & GUILD_CUDA_MOE_FLAG_GR_FP32_ACT_V1) != 0);
                math::gr_set_native_mmvf((c.flags & GUILD_CUDA_MOE_FLAG_GR_NATIVE_MMVF_V1) != 0);
                math::ple_set_native_bf16((c.flags & GUILD_CUDA_MOE_FLAG_PLE_NATIVE_BF16_V1) != 0);
                math::ple_set_native_postops((c.flags & GUILD_CUDA_MOE_FLAG_PLE_NATIVE_POSTOPS_V1) != 0);
                math::native_grouped_set_v1((c.flags & GUILD_CUDA_MOE_FLAG_NATIVE_GROUPED_V1_V1) != 0);
                math::iq_set_old_kernels((c.flags & GUILD_CUDA_MOE_FLAG_IQ_OLD_KERNELS_V1) != 0);
                if (c.grouped_select_old >= -1) {
                    math::moe_grouped_select_old(c.grouped_select_old);
                }
                break;
            case GUILD_CUDA_MOE_CFG_QUERY_GROUPED_LAST_PATH_V1:
                if (!c.out_value) throw std::invalid_argument("configure: null out_value");
                *c.out_value = math::moe_grouped_last_path();
                break;
            case GUILD_CUDA_MOE_CFG_DEQUANT_BF16_SUPPORTED_V1:
                if (!c.out_value) throw std::invalid_argument("configure: null out_value");
                *c.out_value = math::dequant_bf16_supported(c.value) ? 1 : 0;
                break;
            case GUILD_CUDA_MOE_CFG_COUPLED_SCRATCH_BYTES_V1:
                if (!c.out_bytes) throw std::invalid_argument("configure: null out_bytes");
                *c.out_bytes = (uint64_t) math::coupled_draft_scratch_bytes(c.value);
                break;
            case GUILD_CUDA_MOE_CFG_FUSED_GR_SUPPORTED_V1:
                if (!c.out_value) throw std::invalid_argument("configure: null out_value");
                *c.out_value = math::fused_gr_supported(c.n_embd, c.hc, c.hc_lr) ? 1 : 0;
                break;
            case GUILD_CUDA_MOE_CFG_FUSED_GR_SET_FAST_V1:
                math::fused_gr_set_fast(c.value);
                break;
            case GUILD_CUDA_MOE_CFG_FUSED_GR_CHECK_V1:
                math::fused_gr_check();
                break;
            case GUILD_CUDA_MOE_CFG_FUSED_GR_VARIANT_V1:
                if (!c.out_value) throw std::invalid_argument("configure: null out_value");
                *c.out_value = math::fused_gr_variant();
                break;
            case GUILD_CUDA_MOE_CFG_PLE_AVAILABLE_V1:
                if (!c.out_value) throw std::invalid_argument("configure: null out_value");
                *c.out_value = math::ple_block_available() ? 1 : 0;
                break;
            case GUILD_CUDA_MOE_CFG_NATIVE_EXPERT_SET_MODE_V1:
                math::native_expert_set_mode(c.value, c.value2);
                break;
            default:
                throw std::invalid_argument("configure: unsupported mode");
        }
        count(0);
    });
}

guild_status_t quantize_dequant_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_quant_dequant_op_v1_t* raw,
                                   guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "quantize_dequant");
        const auto& c = checked(raw, "quantize_dequant");
        switch (c.mode) {
            case GUILD_CUDA_QD_DEQUANT_BF16_V1:
                math::dequant_bf16(c.ggml_type, c.src, c.row0, c.rows, c.cols, (uint16_t*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_DEQUANT_F16_V1:
                math::dequant_f16(c.ggml_type, c.src, c.row0, c.rows, c.cols, (uint16_t*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_DEQUANT_F32_V1:
                math::dequant_f32(c.ggml_type, c.src, c.row0, c.rows, c.cols, (float*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_DEQUANT_S2_V1:
                math::dequant_s2((const uint8_t*) c.src, c.scales, (float*) c.dst, c.n);
                break;
            case GUILD_CUDA_QD_QUANTIZE_Q8_0_V1:
                math::quantize_q8_0((const float*) c.src, (uint8_t*) c.dst, c.n, stream);
                break;
            case GUILD_CUDA_QD_QUANTIZE_Q8_0_SCALED_V1:
                math::quantize_q8_0_scaled((const float*) c.src, (uint8_t*) c.dst, c.out_scales, c.n, stream);
                break;
            case GUILD_CUDA_QD_DEQUANT_Q8_0_V1:
                math::dequant_q8_0((const uint8_t*) c.src, (float*) c.dst, c.n, stream);
                break;
            case GUILD_CUDA_QD_QUANTIZE_Q8_K_V1:
                math::quantize_q8_K((const float*) c.src, (uint8_t*) c.dst, c.n, stream);
                break;
            case GUILD_CUDA_QD_DEQUANT_Q8_K_V1:
                math::dequant_q8_K((const uint8_t*) c.src, (float*) c.dst, c.n, stream);
                break;
            case GUILD_CUDA_QD_QUANTIZE_Q8_1_ROWS_V1:
                math::quantize_q8_1_rows((const float*) c.src, c.rows, c.cols, c.dst, stream);
                break;
            case GUILD_CUDA_QD_IQ_DEQUANT_F16_V1:
                math::iq_dequant_f16(c.ggml_type, c.src, c.n, (uint16_t*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_IQ_DEQUANT_F32_V1:
                math::iq_dequant_f32(c.ggml_type, c.src, c.n, (float*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_IQ_EMBED_ROWS_V1:
                math::iq_embed_rows(c.ggml_type, c.src, (size_t) c.row_bytes, c.tokens, c.rows, c.cols, (float*) c.dst,
                                    stream);
                break;
            case GUILD_CUDA_QD_IQ_DEQUANT_GU_F16_V1:
                math::iq_dequant_gu_f16(c.ggml_type, c.src, c.src2, c.rows, c.cols, (uint16_t*) c.dst, stream);
                break;
            case GUILD_CUDA_QD_IQ_MMVQ_V1:
                math::iq_mmvq(c.ggml_type, c.src, c.src2, (float*) c.dst, (int) c.cols, (int) c.rows, c.ncols, stream);
                break;
            default:
                throw std::invalid_argument("quantize_dequant: unsupported mode");
        }
        count(1);
    });
}

guild_status_t s_gemv_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_s_gemv_op_v1_t* raw,
                         guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "s_gemv");
        const auto& c = checked(raw, "s_gemv");
        const auto form = make_sform(c.form);
        switch (c.mode) {
            case GUILD_CUDA_SGEMV_S2_V1:
                math::s2_gemv((const uint16_t*) c.x, c.codes, c.scales, c.y, c.n_in, c.n_out);
                break;
            case GUILD_CUDA_SGEMV_S2_QUADS_V1:
                math::s2_gemv_quads((const uint16_t*) c.x, c.codes, c.scales, c.y, c.n_in, c.n_out, c.threads_per_row);
                break;
            case GUILD_CUDA_SGEMV_S2_FAST_V1:
                math::s2_gemv_fast((const uint16_t*) c.x, c.codes, c.scales, c.y, c.n_in, c.n_out, c.threads_per_row,
                                   c.stage_x != 0);
                break;
            case GUILD_CUDA_SGEMV_S2_Q8_V1:
                math::s2_gemv_q8((const uint8_t*) c.x, c.codes, c.scales, c.y, c.n_in, c.n_out, c.threads_per_row,
                                 stream);
                break;
            case GUILD_CUDA_SGEMV_S_V1:
                math::s_gemv((const uint16_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form);
                break;
            case GUILD_CUDA_SGEMV_S_SPLIT_V1:
                math::s_gemv_split((const uint16_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form,
                                   c.threads_per_row);
                break;
            case GUILD_CUDA_SGEMV_S_SPLIT_ASYNC_V1:
                math::s_gemv_split_async((const uint16_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form,
                                         c.threads_per_row, stream);
                break;
            case GUILD_CUDA_SGEMV_S_Q8K_V1:
                math::s_gemv_q8k((const uint8_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form, stream);
                break;
            case GUILD_CUDA_SGEMV_S_Q8K_SPLIT_V1:
                math::s_gemv_q8k_split((const uint8_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form,
                                       stream);
                break;
            case GUILD_CUDA_SGEMV_S_Q8_0_SPLIT_V1:
                math::s_gemv_q8_0_split((const uint8_t*) c.x, c.codes, c.scales, c.offset, c.y, c.n_in, c.n_out, form,
                                        stream);
                break;
            default:
                throw std::invalid_argument("s_gemv: unsupported mode");
        }
        count(2);
    });
}

guild_status_t moe_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_moe_op_v1_t* raw,
                      guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "moe");
        const auto& c = checked(raw, "moe");
        switch (c.mode) {
            case GUILD_CUDA_MOE_ROUTER_TOP10_V1:
                math::router_top10(c.logits, c.n_tokens, c.n_expert, (int) c.k, c.ids, (float*) c.weights, stream);
                break;
            case GUILD_CUDA_MOE_ROUTER_TOP10_VARIANT_V1: {
                const bool ok = math::router_top10_variant(c.logits, c.n_tokens, c.n_expert, (int) c.k, c.ids,
                                                           (float*) c.weights, stream, c.variant);
                if (c.out_supported) *c.out_supported = ok ? 1u : 0u;
                break;
            }
            case GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_V1:
                math::native_router_top10(c.logits, c.ids, (float*) c.weights, stream);
                break;
            case GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_MULTI_V1:
                math::native_router_top10_multi(c.logits, c.ids, (float*) c.weights, c.n_tokens, stream);
                break;
            case GUILD_CUDA_MOE_COMBINE_V1:
                math::moe_combine(c.parts, c.weights, c.shared, c.out, c.n_embd, c.k, stream);
                break;
            case GUILD_CUDA_MOE_NATIVE_COMBINE_V1:
                math::native_moe_combine(c.parts, c.weights, c.shared, c.out, c.n_embd, c.k, stream);
                break;
            case GUILD_CUDA_MOE_NATIVE_COMBINE_MULTI_V1:
                math::native_moe_combine_multi(c.parts, c.weights, c.shared, c.out, c.n_embd, c.k, c.n_tokens, stream);
                break;
            case GUILD_CUDA_MOE_SHARED_EXPERT_V1: {
                const auto gf = make_sform(c.gate_form);
                const auto uf = make_sform(c.up_form);
                const auto df = make_sform(c.down_form);
                math::NativeSharedWeights nw{};
                const math::NativeSharedWeights* nwp = nullptr;
                if (c.has_native_shared) {
                    nw = make_native_shared(c.native_shared);
                    nwp = &nw;
                }
                math::shared_expert(c.x_q8_0, c.x_q8k, c.x_bf16, gf, c.gate_codes, c.gate_scales, c.gate_off, uf,
                                    c.up_codes, c.up_scales, c.up_off, df, c.down_codes, c.down_scales, c.down_off,
                                    c.gate_inp_bf16, (float*) c.scratch, c.out, c.n_embd, c.n_ff, c.tpr, stream,
                                    c.x_f32, nwp);
                break;
            }
            case GUILD_CUDA_MOE_SHARED_EXPERT_MULTI_V1: {
                const math::NativeSharedWeights nw = make_native_shared(c.native_shared);
                math::shared_expert_multi(c.n_tokens, c.x_f32, c.x_bf16, nw, c.gate_inp_bf16, c.gate_buf, c.up_buf,
                                          c.g_buf, c.out, c.n_embd, c.n_ff, stream);
                break;
            }
            case GUILD_CUDA_MOE_HIT_SELECT_V1:
                math::moe_hit_select(c.ids, c.res_row, (int) c.k, c.n_expert, c.slot_out, c.dst_out, c.count_out,
                                     stream);
                break;
            case GUILD_CUDA_MOE_HIT_SELECT_MULTI_V1:
                math::moe_hit_select_multi(c.ids, c.res_row, c.n_tokens, c.n_expert, c.slot_out, c.dst_out, c.count_out,
                                           stream);
                break;
            case GUILD_CUDA_MOE_HIT_ADD_V1:
                math::moe_hit_add(c.out, c.parts, c.dst_index, c.d_count, c.cap, c.n_embd, stream);
                break;
            case GUILD_CUDA_MOE_HIT_GROUPED_S2_V1:
                math::moe_hit_grouped_s2(c.blob_base, c.slot_index, c.dst_index, c.n_hits, c.blob_bytes, c.x_q8_0,
                                         c.scratch, c.out, stream, c.x_scales);
                break;
            case GUILD_CUDA_MOE_HIT_GROUPED_S2_DEV_V1:
                math::moe_hit_grouped_s2_dev(c.blob_base, c.slot_index, c.dst_index, c.d_count, c.cap, c.blob_bytes,
                                             c.x_q8_0, c.scratch, c.out, stream, c.x_scales);
                break;
            case GUILD_CUDA_MOE_HIT_GROUPED_S2_MULTI_V1:
                math::moe_hit_grouped_s2_multi(c.blob_base, c.slot_index, c.dst_index, c.d_count, c.cap, c.blob_bytes,
                                               c.x_q8_0, c.x_scales, c.k_per_token, c.scratch, c.out, stream);
                break;
            case GUILD_CUDA_MOE_HIT_GROUPED_S2_CPU_ORDER_V1:
                math::moe_hit_grouped_s2_cpu_order(c.blob_base, c.slot_index, c.dst_index, c.n_hits, c.blob_bytes,
                                                   c.x_q8_0, c.scratch, c.out, stream, c.x_scales, c.gate_up_trace);
                break;
            case GUILD_CUDA_MOE_GROUPED_S2_V1:
                math::moe_grouped_s2(c.grp_ptr, c.grp_start, c.n_groups, c.ent_dst, c.ent_tok, c.cap_groups,
                                     c.cap_entries, c.x_q8_0, c.x_scales, c.scratch, c.out, stream);
                break;
            case GUILD_CUDA_MOE_GROUP_RESIDENT_V1:
                math::moe_group_resident(c.ids, c.n_tokens, c.k_per_token, c.blob_base, c.blob_bytes, c.grp_ptr_mut,
                                         c.grp_start_mut, c.count_out, c.ent_dst_mut, c.ent_tok_mut, stream);
                break;
            case GUILD_CUDA_MOE_REMOTE_REDUCE_V1:
                math::remote_expert_reduce(c.parts, c.reduce_meta, c.out, c.n_tokens, stream);
                break;
            case GUILD_CUDA_MOE_REMOTE_COPY_CPU_ROWS_V1:
                math::remote_expert_copy_cpu_rows(c.out, c.parts, c.mask, c.dst_index, c.d_count, c.n_tokens, stream);
                break;
            case GUILD_CUDA_MOE_NATIVE_EXPERT_GROUPED_V1: {
                const auto L = math::native_expert_layout(c.gu_type, c.d_type, c.n_embd, c.n_ff);
                math::native_expert_grouped(L, c.grp_ptr, c.grp_start, c.n_groups, c.ent_dst, c.ent_tok, c.cap_groups,
                                            c.cap_entries, c.x_q8_1, c.scratch, c.out, stream, c.grid_groups);
                break;
            }
            default:
                throw std::invalid_argument("moe: unsupported mode");
        }
        count(3);
    });
}

guild_status_t elementwise_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_elementwise_op_v1_t* raw,
                              guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "elementwise");
        const auto& c = checked(raw, "elementwise");
        switch (c.mode) {
            case GUILD_CUDA_ELEM_EMBEDDING_GATHER_V1:
                math::embedding_gather(c.codes, c.scales, c.offsets, c.n, c.code_bits, c.code_bias, c.group_elems,
                                       c.dst, stream);
                break;
            case GUILD_CUDA_ELEM_GDN_GATE_V1:
                math::gdn_gate(c.src, c.dt, c.ssm_a, c.dst, c.rows, c.cols, stream);
                break;
            case GUILD_CUDA_ELEM_SCALE_INPLACE_V1:
                math::scale_inplace(c.dst, c.n, c.scale, stream);
                break;
            case GUILD_CUDA_ELEM_ADD_INPLACE_V1:
                math::add_inplace(c.dst, c.src, c.n, stream);
                break;
            case GUILD_CUDA_ELEM_F32_TO_F16_V1:
                math::f32_to_f16_bulk(c.src, c.dst_u16, c.n, stream);
                break;
            case GUILD_CUDA_ELEM_F32_TO_BF16_V1:
                math::f32_to_bf16_bulk(c.src, c.dst_u16, c.n, stream);
                break;
            case GUILD_CUDA_ELEM_SILU_INPLACE_V1:
                math::silu_inplace(c.dst, c.n, stream);
                break;
            case GUILD_CUDA_ELEM_RMS_NORM_WEIGHTED_V1:
                math::rms_norm_weighted(c.dst, c.scales, c.rows, c.cols, c.eps, stream);
                break;
            case GUILD_CUDA_ELEM_DOORBELL_RING_V1:
                math::doorbell_ring(c.d_seq, stream);
                break;
            case GUILD_CUDA_ELEM_DOORBELL_WAIT_V1:
                math::doorbell_wait(c.d_flag, c.d_seq, stream);
                break;
            case GUILD_CUDA_ELEM_COPY_FROM_MAPPED_V1:
                math::copy_from_mapped(c.dst, c.src, c.n, stream);
                break;
            case GUILD_CUDA_ELEM_SCATTER_ROWS_F32_V1:
                math::scatter_rows_f32(c.src, c.dst, c.rows_idx, c.rows, c.cols, stream);
                break;
            case GUILD_CUDA_ELEM_COPY_ROWS_FROM_MAPPED_V1:
                math::copy_rows_from_mapped(c.dst, c.src, c.rows, c.cols, c.hit_rows, c.count, stream);
                break;
            case GUILD_CUDA_ELEM_DOORBELL_PUBLISH_V1:
                math::doorbell_publish(c.src, c.ids, c.scales, c.n, c.cols, c.dst, c.ids_out, c.weights_out, c.d_seq,
                                       stream);
                break;
            case GUILD_CUDA_ELEM_DOORBELL_PUBLISH_VALUE_V1:
                math::doorbell_publish_value(c.src, c.ids, c.scales, c.n, c.cols, c.dst, c.ids_out, c.weights_out,
                                             c.d_seq, c.value, stream);
                break;
            case GUILD_CUDA_ELEM_DOORBELL_PUBLISH_RES_V1:
                math::doorbell_publish_res(c.src, c.ids, c.d_res, c.n_expert, c.n, c.cols, c.dst, c.ids_out, c.d_seq,
                                           stream);
                break;
            case GUILD_CUDA_ELEM_COPY_I32_FROM_MAPPED_V1:
                math::copy_i32_from_mapped(c.ids_out, c.ids, c.n, stream);
                break;
            default:
                throw std::invalid_argument("elementwise: unsupported mode");
        }
        count(4);
    });
}

guild_status_t sampler_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_sampler_op_v1_t* raw,
                          guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "sampler");
        const auto& c = checked(raw, "sampler");
        switch (c.mode) {
            case GUILD_CUDA_SAMPLER_SAMPLE_TOKENS_V1: {
                check_nested(c.params, "sampler_params");
                math::SamplerParams p{};
                p.top_k = c.params.top_k;
                p.top_p = c.params.top_p;
                p.min_p = c.params.min_p;
                p.temperature = c.params.temperature;
                p.min_keep = c.params.min_keep;
                p.penalty_last_n = c.params.penalty_last_n;
                p.penalty_repeat = c.params.penalty_repeat;
                p.penalty_freq = c.params.penalty_freq;
                p.penalty_present = c.params.penalty_present;
                p.seed = c.params.seed;
                p.counter = c.params.counter;
                p.greedy = (c.params.greedy != 0);
                math::sample_tokens(c.logits, c.n_tokens, c.n_vocab, c.history, c.history_len, p, c.out, stream);
                break;
            }
            case GUILD_CUDA_SAMPLER_GREEDY_CLUSTER_V1: {
                const bool ok = math::sample_greedy_cluster(c.logits, c.n_tokens, c.n_vocab, c.out, stream);
                if (c.out_launched) *c.out_launched = ok ? 1u : 0u;
                break;
            }
            case GUILD_CUDA_SAMPLER_COUPLED_STAGE_V1:
                math::coupled_draft_stage((const math::SamplerParams*) c.mapped_params, c.mapped_hist,
                                          (math::SamplerParams*) c.dev_params, c.ring, c.cap, stream);
                break;
            case GUILD_CUDA_SAMPLER_COUPLED_SAMPLE_V1:
                math::coupled_draft_sample(c.logits_mut, c.n_vocab, c.sub_to_id, c.id_to_sub, c.id_vocab,
                                           (const math::SamplerParams*) c.dev_params, c.ring, c.cap, c.j, c.step_rec,
                                           c.scratch, c.out, c.out_prob, stream);
                break;
            default:
                throw std::invalid_argument("sampler: unsupported mode");
        }
        count(5);
    });
}

guild_status_t verify_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_verify_op_v1_t* raw,
                         guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "verify");
        const auto& c = checked(raw, "verify");
        switch (c.mode) {
            case GUILD_CUDA_VERIFY_FINITE_FIRST_BAD_V1:
                math::finite_first_bad(c.values, c.count, c.u64_dst, stream);
                break;
            case GUILD_CUDA_VERIFY_WAIT_FLAG_GE_V1:
                math::wait_flag_ge(c.u32_flag, c.u32_val, stream);
                break;
            case GUILD_CUDA_VERIFY_GPU_STAMP_V1:
                math::gpu_stamp(c.u64_dst, c.idx, stream);
                break;
            case GUILD_CUDA_VERIFY_RESIDENT_PLAN_V1:
                math::resident_plan(c.i32_src, c.n_tok, c.k, c.i32_src2, c.n_expert, c.bytes_src, c.u64_src,
                                    (long long) c.blob_bytes, c.i32_dst, (long long) c.n, c.u32_skip_mut, c.u32_val,
                                    stream);
                break;
            case GUILD_CUDA_VERIFY_WAIT_FLAG_GE_OR_V1:
                math::wait_flag_ge_or(c.u32_flag, c.u32_val, c.u32_skip, stream);
                break;
            case GUILD_CUDA_VERIFY_COPY_I32_UNLESS_V1:
                math::copy_i32_from_mapped_unless(c.i32_dst, c.i32_src, (long long) c.n, c.u32_skip, c.u32_val, stream);
                break;
            case GUILD_CUDA_VERIFY_COPY_OR_ZERO_V1:
                math::copy_or_zero_from_mapped(c.dst, c.values, (long long) c.n, c.u32_skip, c.u32_val, stream);
                break;
            case GUILD_CUDA_VERIFY_EMBEDDING_GATHER_DEV_V1:
                math::embedding_gather_dev(c.bytes_src, c.scales, c.offsets, c.i32_src, c.n_tok, c.n, c.code_bits,
                                           c.code_bias, c.group_elems, c.row_codes, c.row_groups, c.dst, stream);
                break;
            case GUILD_CUDA_VERIFY_BROADCAST_STREAMS_V1:
                math::broadcast_streams(c.values, c.dst, c.n, c.hc, c.n_tok, stream);
                break;
            case GUILD_CUDA_VERIFY_COPY_INDEXED_V1:
                math::copy_indexed(c.dst, c.values, c.stride, c.i32_src, c.n, stream);
                break;
            case GUILD_CUDA_VERIFY_FETCH_BLOBS_V1:
                math::fetch_blobs(c.u64_src, c.i32_src, c.bytes_dst, c.blob_bytes, c.cap, stream);
                break;
            case GUILD_CUDA_VERIFY_REBASE_PTRS_V1:
                math::rebase_ptrs(c.u64_dst, c.i32_src, c.bytes_dst, c.blob_bytes, stream);
                break;
            case GUILD_CUDA_VERIFY_ADD_STREAMS_BROADCAST_V1:
                math::add_streams_broadcast(c.values, c.src2, c.dst, c.n, c.hc, c.n_tok, stream);
                break;
            case GUILD_CUDA_VERIFY_IDENT_HITS_V1:
                math::ident_hits(c.i32_src, c.n_tok, c.i32_dst, c.i32_dst2, c.i32_dst3, stream);
                break;
            case GUILD_CUDA_VERIFY_MTP_SELECT_V1:
                math::mtp_select(c.values, c.stride, c.i32_src, c.i32_src2, c.dst, c.i32_dst, c.i32_dst2, c.idx, stream,
                                 c.src2, c.out_p);
                break;
            case GUILD_CUDA_VERIFY_GATHER_ROWS_V1:
                math::gather_rows(c.bytes_src, c.stride, c.i32_src, c.n, c.bytes_dst, stream);
                break;
            case GUILD_CUDA_VERIFY_MAP_IDS_V1:
                math::map_ids(c.i32_dst, c.i32_src, c.n_tok, stream);
                break;
            case GUILD_CUDA_VERIFY_ROW_TOP_PROB_V1:
                math::row_top_prob(c.values, c.n_tok, c.cap, c.i32_src, c.dst, stream);
                break;
            case GUILD_CUDA_VERIFY_DENSE_STEPS_V1:
                math::dense_steps(c.i32_src, c.n_tok, c.i32_dst, stream);
                break;
            case GUILD_CUDA_VERIFY_WINDOW_IDS_V1:
                math::window_ids(c.i32_dst, c.n_tok, c.cap, c.i32_dst2, c.stride, stream);
                break;
            default:
                throw std::invalid_argument("verify: unsupported mode");
        }
        count(6);
    });
}

guild_status_t qwen_v1(const guild_exec_context_v1_t* ctx, const guild_cuda_qwen_op_v1_t* raw,
                       guild_error_info_t* error) {
    return guarded(error, [&] {
        void* stream = checked_stream(ctx, "qwen");
        const auto& c = checked(raw, "qwen");
        switch (c.mode) {
            case GUILD_CUDA_QWEN_GR_READ_V1: {
                check_nested(c.gr_shapes, "gr_shapes");
                check_nested(c.gr_ws, "gr_ws");
                const math::GrShapes s{c.gr_shapes.n_embd, c.gr_shapes.hc, c.gr_shapes.hc_lr};
                const math::GrWorkspace ws{c.gr_ws.xn, c.gr_ws.xq, c.gr_ws.lq, c.gr_ws.gated, c.gr_ws.lo,
                                           (size_t) c.gr_ws.bytes};
                math::gr_read(c.R, c.w_norm, c.w_down, c.w_up, c.w_inject, c.eps, s, ws, c.mixed, c.inject_out, stream);
                break;
            }
            case GUILD_CUDA_QWEN_GR_WRITE_V1: {
                check_nested(c.gr_shapes, "gr_shapes");
                const math::GrShapes s{c.gr_shapes.n_embd, c.gr_shapes.hc, c.gr_shapes.hc_lr};
                math::gr_write(c.R, c.block_out, c.inject, s, c.R_out, stream);
                break;
            }
            case GUILD_CUDA_QWEN_FUSED_GR_READ_V1: {
                const auto a = make_fused_gr_args(c.fused_gr);
                math::fused_gr_read(a, stream);
                break;
            }
            case GUILD_CUDA_QWEN_FUSED_GR_READ_MULTI_V1: {
                if (c.n_tok < 0 || c.n_tok > math::kFusedGrMaxT || (c.n_tok && !c.fused_gr_multi)) {
                    throw std::invalid_argument("qwen: fused_gr_read_multi n_tok out of range");
                }
                math::FusedGrArgs args[math::kFusedGrMaxT]{};
                for (int t = 0; t < c.n_tok; ++t) {
                    args[t] = make_fused_gr_args(c.fused_gr_multi[t]);
                }
                math::fused_gr_read_multi(args, c.n_tok, c.xn_scratch, stream, c.stamp_buf, c.stamp_i0);
                break;
            }
            case GUILD_CUDA_QWEN_NATIVE_GR_RMS_NORM_V1:
                math::native_gr_rms_norm_weighted(c.R, c.w_norm, c.R_out, c.n_cols, c.n_rows, c.eps, stream);
                break;
            case GUILD_CUDA_QWEN_NATIVE_GR_DOWN_SILU_V1:
                check_nested(c.gr_shapes, "gr_shapes");
                math::native_gr_down_silu(c.R_out, (int) c.gr_shapes.hc_lr, (int) c.gr_shapes.hc, stream);
                break;
            case GUILD_CUDA_QWEN_NATIVE_GR_PRE_GATED_V1:
                check_nested(c.gr_shapes, "gr_shapes");
                math::native_gr_pre_gated(c.R, c.gated, c.mixed, (int) c.gr_shapes.n_embd, (int) c.gr_shapes.hc,
                                          c.fused_layer != 0, stream);
                break;
            case GUILD_CUDA_QWEN_NATIVE_GR_POST_V1:
                check_nested(c.gr_shapes, "gr_shapes");
                math::native_gr_post(c.R, c.block_out, c.inject, c.R_out, (int) c.gr_shapes.n_embd,
                                     (int) c.gr_shapes.hc, stream);
                break;
            case GUILD_CUDA_QWEN_PLE_BLOCK_V1: {
                const auto w = make_ple_weights(c.ple_weights);
                auto out = make_ple_out(c.ple_out);
                math::ple_block(c.emb, c.hidden, c.hist_rows, w, out, c.scratch, stream);
                break;
            }
            case GUILD_CUDA_QWEN_PLE_BLOCK_PROJECTED_V1: {
                const auto w = make_ple_weights(c.ple_weights);
                auto out = make_ple_out(c.ple_out);
                math::ple_block_projected(c.projected_key, c.projected_value, c.hidden, c.hist_rows, w, out, c.scratch,
                                          stream);
                break;
            }
            case GUILD_CUDA_QWEN_PLE_HISTORY_ADVANCE_V1:
                math::ple_history_advance(c.hist_mut, c.normalized, stream);
                break;
            case GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_V1: {
                const auto w = make_ple_weights(c.ple_weights);
                check_nested(c.ple_buffers, "ple_buffers");
                const math::NativePlePostopsBuffers b{c.ple_buffers.key,        c.ple_buffers.query,
                                                      c.ple_buffers.gate,       c.ple_buffers.gated,
                                                      c.ple_buffers.normalized, c.ple_buffers.conv,
                                                      c.ple_buffers.result};
                math::native_ple_postops(c.projected_key, c.hidden, c.projected_value, c.hist_rows, w, b, stream);
                break;
            }
            case GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_BATCH_V1: {
                const auto w = make_ple_weights(c.ple_weights);
                math::native_ple_postops_batch(c.key_mut, c.hidden_mut, c.projected_value, c.hist_mut, w, c.query_norm,
                                               c.gated, c.gate, c.n_tok, stream);
                break;
            }
            case GUILD_CUDA_QWEN_CVEC_APPLY_V1:
                check_nested(c.gr_shapes, "gr_shapes");
                math::cvec_launch_kernel(c.R_out, c.cvec_dir, c.cvec_s, c.cvec_on, c.cvec_mode, c.layer, c.n_tok,
                                         c.gr_shapes.n_embd, c.gr_shapes.hc, c.r_ld, c.block_out, c.bo_ld, c.inject,
                                         c.inj_ld, c.write != 0, stream);
                break;
            default:
                throw std::invalid_argument("qwen: unsupported mode");
        }
        count(7);
    });
}

const guild_cuda_moe_ext_v1_t table = {
    sizeof(guild_cuda_moe_ext_v1_t),
    GUILD_CUDA_MOE_TAG_V1,
    GUILD_CUDA_MOE_VERSION_V1,
    0u,
    configure_v1,
    quantize_dequant_v1,
    s_gemv_v1,
    moe_v1,
    elementwise_v1,
    sampler_v1,
    verify_v1,
    qwen_v1,
};

}  // namespace

extern "C" const guild_cuda_moe_ext_v1_t* guild_cuda_moe_table_v1(void) {
    return &table;
}

extern "C" uint64_t guild_cuda_moe_call_count_v1(uint32_t category) {
    return category < 8 ? calls[category].load(std::memory_order_relaxed) : 0;
}
