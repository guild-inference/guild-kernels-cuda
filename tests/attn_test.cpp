// Independent scalar oracles for the CDA1 CUDA attention, recurrent (GDN),
// RoPE, and KV C ABI extension.
#include "guild/cuda/attn_backend.h"
#include "guild/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"attn check line %d: %s\n",__LINE__,#x);std::abort();}}while(0)

int main() {
    const auto& table = *guild_cuda_attn_table_v1();
    CHECK(table.struct_size == sizeof(table) && table.tag == GUILD_CUDA_ATTN_TAG_V1 && table.version == 1);

    cudaStream_t stream{};
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);

    guild_exec_context_v1_t ctx{};
    ctx.struct_size = sizeof(ctx);
    ctx.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    ctx.backend = GUILD_BACKEND_CUDA;
    ctx.device_id = 0;
    ctx.device_stream = stream;
    ctx.sync_mode = GUILD_SYNC_STREAM_ORDERED;
    guild_error_info_t error{};

    // 1. Configure toggles round-trip
    uint32_t toggles = 0;
    guild_cuda_attn_config_v1_t cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.actions = GUILD_CUDA_ATTN_CFG_SET_TOGGLES_V1;
    cfg.toggle_mask = GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1 | GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1;
    cfg.toggle_values = GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1;
    cfg.out_toggle_values = &toggles;
    CHECK(table.configure(&ctx, &cfg, &error) == GUILD_STATUS_OK);
    CHECK((toggles & GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1) != 0);
    CHECK((toggles & GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1) == 0);

    // 2. Weighted RMSNorm + Native RoPE (position 0 exact identity) + Gate apply + CUDA Graph capture/replay
    constexpr int COLS = 128, ROWS = 4;
    std::vector<float> hx(COLS * ROWS, 2.0f), hw(COLS, 1.5f), hy(COLS * ROWS, 0.0f);
    std::vector<int32_t> hpos(ROWS, 0);
    float *dx = nullptr, *dw = nullptr, *dy = nullptr, *dr = nullptr;
    int32_t* dpos = nullptr;
    CHECK(cudaMalloc(&dx, hx.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dw, hw.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dy, hy.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dr, hy.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dpos, hpos.size() * 4) == cudaSuccess);
    CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dpos, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_qsa_norm_gate_v1_t norm{};
    norm.struct_size = sizeof(norm);
    norm.mode = GUILD_CUDA_QSA_RMS_NORM_WEIGHTED_V1;
    norm.x = dx;
    norm.aux = dw;
    norm.out = dy;
    norm.cols_or_head_dim = COLS;
    norm.rows_or_n_head = ROWS;
    norm.epsilon = 0.0f;

    guild_cuda_rope_op_v1_t rope{};
    rope.struct_size = sizeof(rope);
    rope.mode = GUILD_CUDA_ROPE_NATIVE_APPLY_V1;
    rope.x = dy;
    rope.out = dr;
    rope.positions = dpos;
    rope.rows = ROWS;
    rope.head_dim = COLS;
    rope.n_rot = 64;
    rope.scaling.struct_size = sizeof(rope.scaling);
    rope.scaling.scaling_type = GUILD_CUDA_ROPE_SCALING_NONE_V1;
    rope.scaling.freq_base = 1e7;
    rope.scaling.factor = 1.0;
    rope.scaling.orig_ctx = 262144.0;
    rope.scaling.attn_factor = 1.0;
    rope.scaling.beta_fast = 32.0;
    rope.scaling.beta_slow = 1.0;

    CHECK(table.qsa_norm_gate(&ctx, &norm, &error) == GUILD_STATUS_OK);
    CHECK(table.rope(&ctx, &rope, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hy.data(), dr, hy.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hy) CHECK(v == 1.5f);

    // Graph capture/replay of norm + rope
    CHECK(cudaMemsetAsync(dr, 0, hy.size() * 4, stream) == cudaSuccess);
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(table.qsa_norm_gate(&ctx, &norm, &error) == GUILD_STATUS_OK);
    CHECK(table.rope(&ctx, &rope, &error) == GUILD_STATUS_OK);
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
    for (int i = 0; i < 3; ++i) CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hy.data(), dr, hy.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hy) CHECK(v == 1.5f);
    CHECK(cudaGraphExecDestroy(exec) == cudaSuccess && cudaGraphDestroy(graph) == cudaSuccess);

    // 3. GDN prefill conv + commit scalar oracle
    constexpr int C = 128;
    std::vector<float> h_hist(C * 3, 0.0f), h_qkv(C * 2, 1.0f), h_convw(C * 4, 0.0f), h_out(C * 2, 0.0f);
    for (int c = 0; c < C; ++c) h_convw[c * 4 + 3] = 0.0f; // tap 3 = 0 -> sum = 0 -> silu(0) = 0
    float *d_hist = nullptr, *d_qkv = nullptr, *d_convw = nullptr, *d_out = nullptr;
    int32_t* d_keep = nullptr;
    int32_t h_keep = 2;
    CHECK(cudaMalloc(&d_hist, h_hist.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_qkv, h_qkv.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_convw, h_convw.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_out, h_out.size() * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_keep, 4) == cudaSuccess);
    CHECK(cudaMemcpy(d_hist, h_hist.data(), h_hist.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_convw, h_convw.data(), h_convw.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_keep, &h_keep, 4, cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_gdn_op_v1_t gdn{};
    gdn.struct_size = sizeof(gdn);
    gdn.mode = GUILD_CUDA_GDN_PREFILL_CONV_V1;
    gdn.x = d_hist;
    gdn.v = d_qkv;
    gdn.k = d_convw;
    gdn.out = d_out;
    gdn.channels_or_rows = C;
    gdn.qk_heads_or_h_k = 0;
    gdn.epsilon = 1e-6f;
    gdn.n_tok = 2;
    CHECK(table.gdn(&ctx, &gdn, &error) == GUILD_STATUS_OK);

    gdn.mode = GUILD_CUDA_GDN_CONV_COMMIT_V1;
    gdn.state = d_hist;
    gdn.x = d_qkv;
    gdn.n_keep = d_keep;
    CHECK(table.gdn(&ctx, &gdn, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(h_hist.data(), d_hist, h_hist.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : h_out) CHECK(v == 0.0f);
    for (int c = 0; c < C; ++c) {
        CHECK(h_hist[c * 3 + 0] == 0.0f);
        CHECK(h_hist[c * 3 + 1] == 1.0f);
        CHECK(h_hist[c * 3 + 2] == 1.0f);
    }

    // 4. KV append + decode attention scalar oracle (1 cell, uniform value = 3.25f)
    guild_cuda_qsa_shapes_v1_t shapes{};
    shapes.struct_size = sizeof(shapes);
    shapes.n_head = 24;
    shapes.n_head_kv = 2;
    shapes.head_dim = 256;
    shapes.n_rot = 64;
    shapes.idx_n_head = 4;
    shapes.idx_dim = 128;
    shapes.idx_block = 4;
    shapes.idx_top_k = 2048;
    shapes.page_size = 4;

    constexpr int KV_ELEMS = 2 * 4 * 256;
    constexpr int Q_ELEMS = 24 * 256;
    uint16_t *dk_pool = nullptr, *dv_pool = nullptr;
    int32_t *d_pt = nullptr, *d_step = nullptr, *d_ids = nullptr;
    float *dq = nullptr, *dkcur = nullptr, *dvcur = nullptr, *d_scratch = nullptr, *d_attn = nullptr;
    CHECK(cudaMalloc(&dk_pool, KV_ELEMS * 2) == cudaSuccess);
    CHECK(cudaMalloc(&dv_pool, KV_ELEMS * 2) == cudaSuccess);
    CHECK(cudaMalloc(&d_pt, 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_step, 16) == cudaSuccess);
    CHECK(cudaMalloc(&d_ids, 4) == cudaSuccess);
    CHECK(cudaMalloc(&dq, Q_ELEMS * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dkcur, 2 * 256 * 4) == cudaSuccess);
    CHECK(cudaMalloc(&dvcur, 2 * 256 * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_scratch, 65536 * 4) == cudaSuccess);
    CHECK(cudaMalloc(&d_attn, Q_ELEMS * 4) == cudaSuccess);

    int32_t h_pt = 0, h_step[4] = {0, 1, 0, 1}, h_ids = 0;
    std::vector<float> h_q(Q_ELEMS, 1.0f), h_kcur(2 * 256, 1.0f), h_vcur(2 * 256, 3.25f), h_attn(Q_ELEMS, 0.0f);
    CHECK(cudaMemcpy(d_pt, &h_pt, 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_step, h_step, 16, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_ids, &h_ids, 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dq, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dkcur, h_kcur.data(), h_kcur.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dvcur, h_vcur.data(), h_vcur.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_kv_op_v1_t kv{};
    kv.struct_size = sizeof(kv);
    kv.mode = GUILD_CUDA_KV_APPEND_STEP_F16_V1;
    kv.k_pool = dk_pool;
    kv.v_pool = dv_pool;
    kv.page_table = d_pt;
    kv.step = d_step;
    kv.kcur = dkcur;
    kv.vcur = dvcur;
    kv.shapes = shapes;
    CHECK(table.kv(&ctx, &kv, &error) == GUILD_STATUS_OK);

    guild_cuda_attention_op_v1_t attn{};
    attn.struct_size = sizeof(attn);
    attn.mode = GUILD_CUDA_ATTN_DECODE_BATCH_V1;
    attn.q = dq;
    attn.ids = d_ids;
    attn.steps = d_step;
    attn.scratch = d_scratch;
    attn.attn_out = d_attn;
    attn.cap_or_n_ids = 1;
    attn.n_q = 1;
    attn.id_stride = -1;
    attn.pools.struct_size = sizeof(attn.pools);
    attn.pools.k_pool = dk_pool;
    attn.pools.v_pool = dv_pool;
    attn.pools.page_table = d_pt;
    attn.shapes = shapes;
    CHECK(table.attention(&ctx, &attn, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(h_attn.data(), d_attn, h_attn.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : h_attn) CHECK(v == 3.25f);

    // 5. Negative / malformed parameter checks
    norm.struct_size = 4;
    CHECK(table.qsa_norm_gate(&ctx, &norm, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);
    norm.struct_size = sizeof(norm);
    norm.mode = 999;
    CHECK(table.qsa_norm_gate(&ctx, &norm, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    ctx.abi_version = GUILD_MAKE_ABI_VERSION(2, 0, 0);
    CHECK(table.configure(&ctx, &cfg, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);
    CHECK(cudaGetLastError() == cudaSuccess);

    CHECK(cudaFree(dx) == cudaSuccess && cudaFree(dw) == cudaSuccess && cudaFree(dy) == cudaSuccess);
    CHECK(cudaFree(dr) == cudaSuccess && cudaFree(dpos) == cudaSuccess);
    CHECK(cudaFree(d_hist) == cudaSuccess && cudaFree(d_qkv) == cudaSuccess);
    CHECK(cudaFree(d_convw) == cudaSuccess && cudaFree(d_out) == cudaSuccess && cudaFree(d_keep) == cudaSuccess);
    CHECK(cudaFree(dk_pool) == cudaSuccess && cudaFree(dv_pool) == cudaSuccess && cudaFree(d_pt) == cudaSuccess);
    CHECK(cudaFree(d_step) == cudaSuccess && cudaFree(d_ids) == cudaSuccess && cudaFree(dq) == cudaSuccess);
    CHECK(cudaFree(dkcur) == cudaSuccess && cudaFree(dvcur) == cudaSuccess);
    CHECK(cudaFree(d_scratch) == cudaSuccess && cudaFree(d_attn) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("CUDA attention/GDN/RoPE/KV scalar oracles, capture/replay and malformed requests passed");
}
