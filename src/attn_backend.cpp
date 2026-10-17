#include "guild/cuda/attn_backend.h"
#include "guild/kernels/rope_scaling.hpp"
#include "guild/kernels/mrope.hpp"
#include "guild/kernels/rope.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/gdn.hpp"
#include "guild/kernels/fused_gdn.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_gdn_preprocess.hpp"
#include "guild/kernels/gdn_multi.hpp"
#include "guild/kernels/qsa.hpp"
#include "guild/kernels/qsa_decode_attn.hpp"
#include "guild/kernels/qsa_prompt_attn.hpp"
#include "guild/kernels/qsa_select.hpp"
#include "guild/kernels/native_flash_attn.hpp"
#include "guild/kernels/native_qsa_indexer.hpp"
#include "guild/kernels/native_qsa_score.hpp"
#include "guild/kernels/kv_q8.hpp"
#include "guild/kernels/kv_q4.hpp"
#include "guild/kernels/kv_stream.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
#include <stdexcept>

namespace {
namespace math = guild::cuda_attn_math;

struct AbiMismatch : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

std::atomic<uint64_t> calls[7]{};

void count(unsigned category) {
    static const bool enabled = [] {
        const char* v = std::getenv("GUILD_CUDA_INSTRUMENT");
        return v && std::atoi(v) != 0;
    }();
    if (enabled && category < 7) {
        calls[category].fetch_add(1, std::memory_order_relaxed);
    }
}

guild_status_t fail(guild_error_info_t* error, guild_status_t code, const char* message) {
    if (error) {
        error->struct_size = sizeof(*error);
        error->code = code;
        error->source_component = "cuda.attn.v1";
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
        return fail(error, GUILD_STATUS_ERR_INTERNAL, "unknown CUDA attention backend exception");
    }
}

void context(const guild_exec_context_v1_t* c, bool ordered = false) {
    if (!c || c->struct_size < sizeof(*c) || GUILD_ABI_VERSION_MAJOR(c->abi_version) != 1)
        throw AbiMismatch("CUDA attention execution context/ABI mismatch");
    if (c->backend != GUILD_BACKEND_CUDA || c->device_id < 0 || (ordered && !c->device_stream))
        throw std::invalid_argument("CUDA attention operation requires a valid CUDA execution context");
}

math::RopeScaling to_scaling(const guild_cuda_rope_scaling_v1_t& s) {
    if (s.struct_size == 0) return {};
    if (s.struct_size < sizeof(s)) throw AbiMismatch("CUDA RoPE scaling struct_size mismatch");
    math::RopeScaling out{};
    switch (s.scaling_type) {
    case GUILD_CUDA_ROPE_SCALING_NONE_V1: out.type = math::RopeScalingType::None; break;
    case GUILD_CUDA_ROPE_SCALING_LINEAR_V1: out.type = math::RopeScalingType::Linear; break;
    case GUILD_CUDA_ROPE_SCALING_YARN_V1: out.type = math::RopeScalingType::YaRN; break;
    default: throw std::invalid_argument("unknown CUDA RoPE scaling type");
    }
    out.freq_base = s.freq_base;
    out.factor = s.factor;
    out.freq_scale_in = s.freq_scale_in;
    out.orig_ctx = s.orig_ctx;
    out.ext_factor = s.ext_factor;
    out.attn_factor = s.attn_factor;
    out.beta_fast = s.beta_fast;
    out.beta_slow = s.beta_slow;
    return out;
}

math::QsaShapes to_qsa_shapes(const guild_cuda_qsa_shapes_v1_t& s) {
    if (s.struct_size < sizeof(s)) throw AbiMismatch("CUDA QSA shapes struct_size mismatch");
    math::QsaShapes out{};
    out.n_head = s.n_head;
    out.n_head_kv = s.n_head_kv;
    out.head_dim = s.head_dim;
    out.n_rot = s.n_rot;
    out.idx_n_head = s.idx_n_head;
    out.idx_dim = s.idx_dim;
    out.idx_block = s.idx_block;
    out.idx_top_k = s.idx_top_k;
    out.page_size = s.page_size;
    return out;
}

math::GdnShapes to_gdn_shapes(const guild_cuda_gdn_shapes_v1_t& s) {
    if (s.struct_size < sizeof(s)) throw AbiMismatch("CUDA GDN shapes struct_size mismatch");
    math::GdnShapes out{};
    out.S = s.state_size;
    out.h_k = s.h_k;
    out.h_v = s.h_v;
    return out;
}

math::KvHostPools to_host_pools(const guild_cuda_kv_host_pools_v1_t& p) {
    if (p.struct_size < sizeof(p)) throw AbiMismatch("CUDA KV host pools struct_size mismatch");
    math::KvHostPools out{};
    out.k_pool = p.k_pool;
    out.v_pool = p.v_pool;
    out.k_q = p.k_q;
    out.v_q = p.v_q;
    out.k_scale = p.k_scale;
    out.v_scale = p.v_scale;
    out.k_q4 = p.k_q4;
    out.v_q4 = p.v_q4;
    return out;
}

math::QsaAttnPools to_attn_pools(const guild_cuda_qsa_attn_pools_v1_t& p) {
    if (p.struct_size < sizeof(p)) throw AbiMismatch("CUDA QSA attention pools struct_size mismatch");
    math::QsaAttnPools out{};
    out.k_pool = p.k_pool;
    out.v_pool = p.v_pool;
    out.k_q = p.k_q;
    out.v_q = p.v_q;
    out.k_scale = p.k_scale;
    out.v_scale = p.v_scale;
    out.k_q4 = p.k_q4;
    out.v_q4 = p.v_q4;
    out.page_table = p.page_table;
    return out;
}

math::KvStreamMap to_stream_map(const guild_cuda_kv_stream_map_v1_t& m) {
    if (m.struct_size < sizeof(m)) throw AbiMismatch("CUDA KV stream map struct_size mismatch");
    math::KvStreamMap out{};
    out.page_table = m.page_table;
    out.slot_block = m.slot_block;
    out.slot_stamp = m.slot_stamp;
    out.slot_ref = m.slot_ref;
    out.ctl = m.ctl;
    out.miss_block = m.miss_block;
    out.miss_slot = m.miss_slot;
    out.n_blocks = m.n_blocks;
    out.n_slots = m.n_slots;
    return out;
}

math::QsaIndexerBuffers to_indexer_buffers(const guild_cuda_qsa_indexer_buffers_v1_t& b) {
    if (b.struct_size < sizeof(b)) throw AbiMismatch("CUDA QSA indexer buffers struct_size mismatch");
    math::QsaIndexerBuffers out{};
    out.tail = b.tail;
    out.dead = b.dead;
    out.pooled = b.pooled;
    out.block_pos = b.block_pos;
    return out;
}

guild_status_t configure_op(const guild_exec_context_v1_t* c, const guild_cuda_attn_config_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA attention config parameter size mismatch");
        if (p->actions & GUILD_CUDA_ATTN_CFG_SET_TOGGLES_V1) {
            if (p->toggle_mask & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1)
                math::native_qsa_set_enabled((p->toggle_values & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1) != 0);
            if (p->toggle_mask & GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1)
                math::native_rope_set_enabled((p->toggle_values & GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1) != 0);
            if (p->toggle_mask & GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1)
                math::native_gdn_set_enabled((p->toggle_values & GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1) != 0);
            if (p->toggle_mask & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1)
                math::native_qsa_indexer_set_enabled((p->toggle_values & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1) != 0);
            if (p->toggle_mask & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1)
                math::native_qsa_score_set_enabled((p->toggle_values & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1) != 0);
        }
        if (p->actions & GUILD_CUDA_ATTN_CFG_SET_ROPE_SCALING_V1) {
            math::rope_scaling_set(to_scaling(p->rope_scaling));
        }
        if (p->actions & GUILD_CUDA_ATTN_CFG_SET_MROPE_TABLE_V1) {
            math::mrope_table_set(p->mrope_table);
        }
        if (p->actions & GUILD_CUDA_ATTN_CFG_SET_ROPE_TABLE_V1) {
            math::rope_table_set(p->rope_cos_tab, p->rope_sin_tab, p->rope_max_pos, to_scaling(p->rope_scaling));
        }
        if (p->actions & GUILD_CUDA_ATTN_CFG_RELEASE_ROPE_TABLE_V1) {
            math::rope_table_release(p->rope_cos_tab);
        }
        if (p->out_toggle_values) {
            uint32_t v = 0;
            if (math::native_qsa_enabled()) v |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1;
            if (math::native_rope_enabled()) v |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1;
            if (math::native_gdn_enabled()) v |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1;
            if (math::native_qsa_indexer_enabled()) v |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1;
            if (math::native_qsa_score_enabled()) v |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1;
            *p->out_toggle_values = v;
        }
        count(0);
    });
}

guild_status_t rope_op(const guild_exec_context_v1_t* c, const guild_cuda_rope_op_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA RoPE parameter size mismatch");
        switch (p->mode) {
        case GUILD_CUDA_ROPE_NEOX_APPLY_V1:
            math::rope_neox_apply(p->x, p->out, p->rows, p->head_dim, p->n_rot, p->cos_tab, p->sin_tab, p->positions, c->device_stream);
            break;
        case GUILD_CUDA_ROPE_NATIVE_APPLY_V1:
            math::native_rope_apply(p->x, p->out, static_cast<int>(p->rows), p->head_dim, p->n_rot, to_scaling(p->scaling), p->positions, c->device_stream);
            break;
        default:
            throw std::invalid_argument("unknown CUDA RoPE mode");
        }
        count(1);
    });
}

guild_status_t qsa_norm_gate_op(const guild_exec_context_v1_t* c, const guild_cuda_qsa_norm_gate_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA QSA norm/gate parameter size mismatch");
        switch (p->mode) {
        case GUILD_CUDA_QSA_RMS_NORM_WEIGHTED_V1:
            math::native_qsa_rms_norm_weighted(p->x, p->aux, static_cast<float*>(p->out),
                                               static_cast<int>(p->cols_or_head_dim), static_cast<int>(p->rows_or_n_head),
                                               p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_QSA_GATE_APPLY_NATIVE_V1:
            math::native_qsa_gate_apply(p->x, p->aux, static_cast<float*>(p->out),
                                        static_cast<int>(p->rows_or_n_head), static_cast<int>(p->cols_or_head_dim),
                                        c->device_stream);
            break;
        case GUILD_CUDA_QSA_GATE_APPLY_F16_V1:
            math::qsa_gate_apply(p->x, p->aux, to_qsa_shapes(p->shapes), static_cast<uint16_t*>(p->out), c->device_stream);
            break;
        case GUILD_CUDA_QSA_GATE_APPLY_F32_V1:
            math::qsa_gate_apply_f32(p->x, p->aux, to_qsa_shapes(p->shapes), static_cast<float*>(p->out), c->device_stream);
            break;
        default:
            throw std::invalid_argument("unknown CUDA QSA norm/gate mode");
        }
        count(2);
    });
}

guild_status_t gdn_op(const guild_exec_context_v1_t* c, const guild_cuda_gdn_op_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA GDN parameter size mismatch");
        switch (p->mode) {
        case GUILD_CUDA_GDN_STEP_V1:
            math::gdn_step(p->state, p->x, p->k, p->v, p->gate, p->beta, p->out, to_gdn_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_GDN_CONV_STEP_V1:
            math::gdn_conv_step(p->state, p->x, p->k, p->out, p->channels_or_rows, p->cols_or_d_conv, c->device_stream);
            break;
        case GUILD_CUDA_GDN_L2_NORM_V1:
            math::gdn_l2_norm(p->state, p->channels_or_rows, p->cols_or_d_conv, p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_GDN_BETA_GATE_V1:
            math::gdn_beta_gate(p->state, p->channels_or_rows, c->device_stream);
            break;
        case GUILD_CUDA_GDN_OUT_NORM_V1:
            math::gdn_out_norm(p->x, p->z, p->gamma, p->out, p->channels_or_rows, p->cols_or_d_conv, p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_GDN_FUSED_CONV_L2_V1:
            math::fused_gdn_conv_l2(p->state, p->x, p->k, p->out, static_cast<int>(p->channels_or_rows), p->qk_heads_or_h_k, p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_GDN_FUSED_AB_V1:
            math::fused_gdn_ab(p->x, p->w_alpha, p->w_beta, p->k, p->v, p->out, p->aux_out, static_cast<int>(p->channels_or_rows), p->h_v, c->device_stream);
            break;
        case GUILD_CUDA_GDN_FUSED_STEP_NORM_V1:
            math::fused_gdn_step_norm(p->state, p->x, p->k, p->v, p->gate, p->beta, p->z, p->gamma, p->epsilon, p->out,
                                      p->qk_heads_or_h_k, p->h_v, c->device_stream, p->gate_silu != 0);
            break;
        case GUILD_CUDA_GDN_NATIVE_STEP_V1:
            math::native_gdn_step(p->state, p->x, p->k, p->v, p->gate, p->beta, p->out, to_gdn_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_GDN_NATIVE_CONV_SILU_V1:
            math::native_gdn_conv_silu(p->state, p->x, p->k, p->aux_out, p->out, p->channels_or_rows, p->cols_or_d_conv, c->device_stream);
            break;
        case GUILD_CUDA_GDN_NATIVE_L2_NORM_V1:
            math::native_gdn_l2_norm(p->state, p->channels_or_rows, p->cols_or_d_conv, p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_GDN_NATIVE_BETA_GATE_V1:
            math::native_gdn_beta_gate(p->state, p->channels_or_rows, c->device_stream);
            break;
        case GUILD_CUDA_GDN_NATIVE_GATE_V1:
            math::native_gdn_gate(p->x, p->k, p->v, p->out, p->channels_or_rows, c->device_stream);
            break;
        case GUILD_CUDA_GDN_NATIVE_OUT_NORM_V1:
            math::native_gdn_out_norm(p->x, p->z, p->gamma, p->out, p->channels_or_rows, p->cols_or_d_conv, p->epsilon, c->device_stream);
            break;
        case GUILD_CUDA_GDN_PREFILL_CONV_V1:
            math::gdn_prefill_conv(p->x, p->v, p->k, p->out, static_cast<int>(p->channels_or_rows), p->qk_heads_or_h_k, p->epsilon, p->n_tok, c->device_stream);
            break;
        case GUILD_CUDA_GDN_PREFILL_RECURRENCE_V1:
            math::gdn_prefill_recurrence(p->state, p->x, static_cast<int>(p->channels_or_rows), p->gate, p->beta, p->z, p->gamma,
                                         p->epsilon, p->out, p->qk_heads_or_h_k, p->h_v, p->n_tok, p->n_keep, p->gate_silu != 0, c->device_stream);
            break;
        case GUILD_CUDA_GDN_CONV_L2_MULTI_V1:
            math::gdn_conv_l2_multi(p->x, p->v, p->k, p->out, static_cast<int>(p->channels_or_rows), p->qk_heads_or_h_k, p->epsilon, p->n_tok, c->device_stream, p->t_begin);
            break;
        case GUILD_CUDA_GDN_CONV_COMMIT_V1:
            math::gdn_conv_commit(p->state, p->x, static_cast<int>(p->channels_or_rows), p->n_keep, c->device_stream);
            break;
        case GUILD_CUDA_GDN_AB_MULTI_V1:
            math::gdn_ab_multi(p->x, p->w_alpha, p->w_beta, p->k, p->v, p->out, p->aux_out, static_cast<int>(p->channels_or_rows), p->h_v, p->n_tok, c->device_stream);
            break;
        case GUILD_CUDA_GDN_AB_PREFILL_V1:
            math::gdn_ab_prefill(p->x, p->w_alpha, p->w_beta, p->k, p->v, p->out, p->aux_out, static_cast<int>(p->channels_or_rows), p->h_v, p->n_tok, c->device_stream);
            break;
        case GUILD_CUDA_GDN_STEP_NORM_MULTI_V1:
            math::gdn_step_norm_multi(p->state, p->x, static_cast<int>(p->channels_or_rows), p->gate, p->beta, p->z, p->gamma,
                                      p->epsilon, p->out, p->qk_heads_or_h_k, p->h_v, p->n_tok, p->n_keep, c->device_stream, p->t_begin, p->gate_silu != 0);
            break;
        default:
            throw std::invalid_argument("unknown CUDA GDN mode");
        }
        count(3);
    });
}

guild_status_t kv_op(const guild_exec_context_v1_t* c, const guild_cuda_kv_op_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA KV parameter size mismatch");
        math::KvHostPools hp{};
        const math::KvHostPools* hp_ptr = nullptr;
        if (p->has_host_pools) {
            hp = to_host_pools(p->host_pools);
            hp_ptr = &hp;
        }
        math::KvHostPools sp{};
        const math::KvHostPools* sp_ptr = nullptr;
        if (p->has_stage_pools) {
            sp = to_host_pools(p->stage_pools);
            sp_ptr = &sp;
        }
        switch (p->mode) {
        case GUILD_CUDA_KV_APPEND_SYNC_V1:
            math::kv_append(static_cast<uint16_t*>(p->k_pool), static_cast<uint16_t*>(p->v_pool),
                            p->page_table, p->pos_or_b0, p->kcur, p->vcur, to_qsa_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_KV_APPEND_STEP_F16_V1:
            math::kv_append_step(static_cast<uint16_t*>(p->k_pool), static_cast<uint16_t*>(p->v_pool),
                                 p->page_table, p->step, p->kcur, p->vcur, to_qsa_shapes(p->shapes), c->device_stream, hp_ptr);
            break;
        case GUILD_CUDA_KV_GATHER_SYNC_V1:
            math::kv_gather(static_cast<const uint16_t*>(p->k_pool_ro), static_cast<const uint16_t*>(p->v_pool_ro),
                            p->page_table, p->ids, p->count_or_b1, to_qsa_shapes(p->shapes), p->k_scratch, p->v_scratch, c->device_stream);
            break;
        case GUILD_CUDA_KV_GATHER_STEP_F16_V1:
            math::kv_gather_step(static_cast<const uint16_t*>(p->k_pool_ro), static_cast<const uint16_t*>(p->v_pool_ro),
                                 p->page_table, p->ids, p->step, p->cap_or_n_slots, to_qsa_shapes(p->shapes),
                                 p->k_scratch, p->v_scratch, c->device_stream);
            break;
        case GUILD_CUDA_KV_APPEND_STEP_Q8_V1:
            math::kv_append_q8_step(static_cast<int8_t*>(p->k_pool), static_cast<int8_t*>(p->v_pool),
                                    p->k_scale, p->v_scale, p->page_table, p->step, p->kcur, p->vcur,
                                    to_qsa_shapes(p->shapes), c->device_stream, hp_ptr);
            break;
        case GUILD_CUDA_KV_GATHER_STEP_Q8_V1:
            math::kv_gather_q8_step(static_cast<const int8_t*>(p->k_pool_ro), static_cast<const int8_t*>(p->v_pool_ro),
                                    p->k_scale_ro, p->v_scale_ro, p->page_table, p->ids, p->step, p->cap_or_n_slots,
                                    to_qsa_shapes(p->shapes), p->k_scratch, p->v_scratch, c->device_stream);
            break;
        case GUILD_CUDA_KV_FWHT256_V1:
            math::fwht256_cuda(p->kcur, p->fwht_dst, p->count_or_b1, c->device_stream);
            break;
        case GUILD_CUDA_KV_APPEND_STEP_Q4_V1:
            math::kv_append_q4_step(static_cast<uint8_t*>(p->k_pool), static_cast<uint8_t*>(p->v_pool),
                                    p->page_table, p->step, p->kcur, p->vcur, to_qsa_shapes(p->shapes), c->device_stream, hp_ptr);
            break;
        case GUILD_CUDA_KV_APPEND_BATCH_Q4_V1:
            math::kv_append_q4(static_cast<uint8_t*>(p->k_pool), static_cast<uint8_t*>(p->v_pool),
                               p->page_table, p->pos_or_b0, p->count_or_b1, p->kcur, p->vcur,
                               to_qsa_shapes(p->shapes), c->device_stream, hp_ptr, sp_ptr);
            break;
        case GUILD_CUDA_KV_GATHER_STEP_Q4_V1:
            math::kv_gather_q4_step(static_cast<const uint8_t*>(p->k_pool_ro), static_cast<const uint8_t*>(p->v_pool_ro),
                                    p->page_table, p->ids, p->step, p->cap_or_n_slots, to_qsa_shapes(p->shapes),
                                    p->k_scratch, p->v_scratch, c->device_stream);
            break;
        case GUILD_CUDA_KV_STREAM_RESET_V1:
            math::kv_stream_reset(to_stream_map(p->stream_map), c->device_stream);
            break;
        case GUILD_CUDA_KV_STREAM_INVALIDATE_V1:
            math::kv_stream_invalidate(to_stream_map(p->stream_map), c->device_stream);
            break;
        case GUILD_CUDA_KV_STREAM_RESOLVE_V1:
            math::kv_stream_resolve(to_stream_map(p->stream_map), to_attn_pools(p->attn_pools), to_host_pools(p->host_pools),
                                    p->kv_format, p->ids, p->step, p->count_or_b1, p->cap_or_n_slots,
                                    to_qsa_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_KV_RING_TABLE_V1:
            math::kv_ring_table(p->page_table_mut, p->count_or_b1, p->cap_or_n_slots, c->device_stream);
            break;
        case GUILD_CUDA_KV_RING_RESTORE_V1:
            math::kv_ring_restore(to_attn_pools(p->attn_pools), to_host_pools(p->host_pools), p->kv_format,
                                  p->pos_or_b0, p->count_or_b1, p->cap_or_n_slots, to_qsa_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_KV_STAGE_FROM_HOST_V1:
            math::kv_stage_from_host(to_attn_pools(p->attn_pools), to_host_pools(p->host_pools), p->kv_format,
                                     p->count_or_b1, to_qsa_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_KV_UNSTAGE_TO_HOST_V1:
            math::kv_unstage_to_host(to_attn_pools(p->attn_pools), to_host_pools(p->host_pools), p->kv_format,
                                     p->pos_or_b0, p->count_or_b1, to_qsa_shapes(p->shapes), c->device_stream);
            break;
        case GUILD_CUDA_KV_STREAM_COUNTERS_V1: {
            if (!p->out_counters || p->out_counters->struct_size < sizeof(*p->out_counters))
                throw AbiMismatch("CUDA KV stream counters output struct_size mismatch");
            const auto cnt = math::kv_stream_counters(to_stream_map(p->stream_map));
            p->out_counters->overflow = cnt.overflow ? 1u : 0u;
            p->out_counters->misses = cnt.misses;
            p->out_counters->lookups = cnt.lookups;
            p->out_counters->calls = cnt.calls;
            break;
        }
        default:
            throw std::invalid_argument("unknown CUDA KV mode");
        }
        count(4);
    });
}

guild_status_t qsa_index_select_op(const guild_exec_context_v1_t* c, const guild_cuda_qsa_index_select_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA QSA index/select parameter size mismatch");
        switch (p->mode) {
        case GUILD_CUDA_QSA_INDEXER_APPEND_V1:
            math::indexer_key_append(p->raw_or_pooled, p->step_or_pos_dev, p->pos_base, p->q_idx_or_gamma,
                                     p->epsilon, to_indexer_buffers(p->buffers), to_qsa_shapes(p->shapes),
                                     p->bias_or_cos, p->sin_tab, c->device_stream);
            break;
        case GUILD_CUDA_QSA_INDEXER_NATIVE_APPEND_V1:
            math::native_qsa_indexer_append(p->raw_or_pooled, p->step_or_pos_dev, p->pos_base, p->q_idx_or_gamma,
                                            p->epsilon, to_indexer_buffers(p->buffers), to_qsa_shapes(p->shapes),
                                            p->max_cells_or_cap, to_scaling(p->scaling), c->device_stream);
            break;
        case GUILD_CUDA_QSA_INDEXER_NATIVE_BATCH_V1:
            math::native_qsa_indexer_append_batch(p->raw_or_pooled, p->n_or_nq, p->p0_or_n_bid, p->pos_base,
                                                  p->q_idx_or_gamma, p->epsilon, to_indexer_buffers(p->buffers),
                                                  to_qsa_shapes(p->shapes), p->max_cells_or_cap, to_scaling(p->scaling),
                                                  c->device_stream);
            break;
        case GUILD_CUDA_QSA_INDEX_SYNC_V1:
            math::qsa_index(p->raw_or_pooled, p->p0_or_n_bid, p->q_idx_or_gamma, p->bias_or_cos,
                            to_qsa_shapes(p->shapes), p->n_or_nq, p->scores_out, c->device_stream);
            break;
        case GUILD_CUDA_QSA_INDEX_STEP_V1:
            math::qsa_index_step(p->raw_or_pooled, p->q_idx_or_gamma, p->bias_or_cos, to_qsa_shapes(p->shapes),
                                 p->step_or_pos_dev, p->max_blocks, p->scores_out, c->device_stream);
            break;
        case GUILD_CUDA_QSA_NATIVE_SCORE_V1:
            math::native_qsa_score(p->raw_or_pooled, p->dead_or_query, p->bias_or_cos, to_qsa_shapes(p->shapes),
                                   p->step_or_pos_dev, p->max_blocks, p->max_cells_or_cap, p->scores_out, c->device_stream);
            break;
        case GUILD_CUDA_QSA_TOPK_512_SYNC_V1:
            math::topk_512(p->raw_or_pooled, p->n_or_nq, to_qsa_shapes(p->shapes), p->max_cells_or_cap, p->ids_out, c->device_stream);
            break;
        case GUILD_CUDA_QSA_TOPK_512_STEP_V1:
            math::topk_512_step(p->raw_or_pooled, to_qsa_shapes(p->shapes), p->max_cells_or_cap, p->step_or_pos_dev, p->ids_out, c->device_stream);
            break;
        case GUILD_CUDA_QSA_BLOCK_SCORES_V1:
            math::qsa_block_scores(p->raw_or_pooled, p->dead_or_query, p->q_idx_or_gamma, p->step_or_pos_dev,
                                   p->n_or_nq, p->max_blocks, to_qsa_shapes(p->shapes), p->scores_out, c->device_stream, p->active_blocks);
            break;
        case GUILD_CUDA_QSA_BLOCK_SCORES_TC_V1: {
            const bool ok = math::qsa_block_scores_tc(p->raw_or_pooled, p->dead_or_query, p->q_idx_or_gamma, p->step_or_pos_dev,
                                                      p->n_or_nq, p->max_blocks, to_qsa_shapes(p->shapes), p->scores_out,
                                                      c->device_stream, p->active_blocks);
            if (p->out_launched) *p->out_launched = ok ? 1u : 0u;
            break;
        }
        case GUILD_CUDA_QSA_BLOCK_TOPK_V1:
            math::qsa_block_topk(p->raw_or_pooled, p->step_or_pos_dev, p->n_or_nq, p->max_blocks, p->max_cells_or_cap,
                                 to_qsa_shapes(p->shapes), p->ids_out, c->device_stream, p->active_blocks);
            break;
        case GUILD_CUDA_QSA_BLOCK_TOPK_CLUSTER_V1: {
            const bool ok = math::qsa_block_topk_cluster(p->raw_or_pooled, p->step_or_pos_dev, p->n_or_nq, p->max_blocks,
                                                         p->max_cells_or_cap, to_qsa_shapes(p->shapes), p->ids_out, c->device_stream);
            if (p->out_launched) *p->out_launched = ok ? 1u : 0u;
            break;
        }
        case GUILD_CUDA_QSA_BLOCK_TOPK_REF_V1:
            math::qsa_block_topk_ref(p->raw_or_pooled, p->step_or_pos_dev, p->n_or_nq, p->max_blocks, p->max_cells_or_cap,
                                     to_qsa_shapes(p->shapes), p->ids_out, c->device_stream);
            break;
        default:
            throw std::invalid_argument("unknown CUDA QSA index/select mode");
        }
        count(5);
    });
}

guild_status_t attention_op(const guild_exec_context_v1_t* c, const guild_cuda_attention_op_v1_t* p, guild_error_info_t* e) {
    return guarded(e, [&] {
        context(c);
        if (!p || p->struct_size < sizeof(*p)) throw AbiMismatch("CUDA attention parameter size mismatch");
        switch (p->mode) {
        case GUILD_CUDA_ATTN_QSA_SYNC_V1:
            math::qsa_attend(p->q, p->k_scratch, p->v_scratch, p->cap_or_n_ids, to_qsa_shapes(p->shapes), p->attn_out, p->weights_out, c->device_stream);
            break;
        case GUILD_CUDA_ATTN_QSA_STEP_V1:
            math::qsa_attend_step(p->q, p->k_scratch, p->v_scratch, p->steps, p->cap_or_n_ids, to_qsa_shapes(p->shapes), p->attn_out, p->weights_out, c->device_stream);
            break;
        case GUILD_CUDA_ATTN_DECODE_STEP_V1:
            math::qsa_decode_attn_step(p->q, to_attn_pools(p->pools), p->ids, p->steps, p->cap_or_n_ids,
                                       to_qsa_shapes(p->shapes), p->scratch, p->attn_out, c->device_stream);
            break;
        case GUILD_CUDA_ATTN_DECODE_BATCH_V1:
            math::qsa_decode_attn_batch(p->q, to_attn_pools(p->pools), p->ids, p->steps, p->cap_or_n_ids,
                                        to_qsa_shapes(p->shapes), p->scratch, p->attn_out, p->n_q, c->device_stream, p->id_stride);
            break;
        case GUILD_CUDA_ATTN_PROMPT_BATCH_V1: {
            const bool ok = math::qsa_prompt_attn_batch(p->q, to_attn_pools(p->pools), p->ids, p->steps, p->cap_or_n_ids,
                                                        to_qsa_shapes(p->shapes), p->attn_out, p->n_q, c->device_stream);
            if (p->out_launched) *p->out_launched = ok ? 1u : 0u;
            break;
        }
        case GUILD_CUDA_ATTN_NATIVE_FLASH_SHORT_V1:
            math::native_flash_attn_short_step(p->q, p->k_scratch, p->v_scratch, p->steps, p->cap_or_n_ids, p->max_context,
                                               to_qsa_shapes(p->shapes), p->attn_out, p->status_out, p->mask, c->device_stream);
            break;
        default:
            throw std::invalid_argument("unknown CUDA attention mode");
        }
        count(6);
    });
}

const guild_cuda_attn_ext_v1_t table = {
    sizeof(table),
    GUILD_CUDA_ATTN_TAG_V1,
    GUILD_CUDA_ATTN_VERSION_V1,
    0,
    configure_op,
    rope_op,
    qsa_norm_gate_op,
    gdn_op,
    kv_op,
    qsa_index_select_op,
    attention_op
};

}  // namespace

extern "C" GUILD_PLUGIN_EXPORT const guild_cuda_attn_ext_v1_t* guild_cuda_attn_table_v1(void) {
    return &table;
}

extern "C" GUILD_PLUGIN_EXPORT uint64_t guild_cuda_attn_call_count_v1(uint32_t category) {
    return category < 7 ? calls[category].load(std::memory_order_relaxed) : 0;
}
