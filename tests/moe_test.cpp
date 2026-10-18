// Independent scalar oracles for the CDM1 CUDA MoE, S-family GEMV/dequant,
// elementwise, sampler, verify, and Qwen specialization C ABI extension.
#include "guild/cuda/moe_backend.h"
#include "guild/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"moe check line %d: %s\n",__LINE__,#x);std::abort();}}while(0)

int main() {
    const auto& table = *guild_cuda_moe_table_v1();
    CHECK(table.struct_size == sizeof(table) && table.tag == GUILD_CUDA_MOE_TAG_V1 && table.version == 1);

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

    // Reject malformed ABI envelopes before touching device buffers.
    guild_cuda_elementwise_op_v1_t invalid{};
    invalid.struct_size = sizeof(invalid);
    invalid.mode = UINT32_MAX;
    CHECK(table.elementwise(&ctx, &invalid, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    invalid.struct_size = sizeof(uint32_t);
    CHECK(table.elementwise(&ctx, &invalid, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);
    CHECK(table.elementwise(&ctx, nullptr, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    CHECK(table.elementwise(nullptr, &invalid, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    auto bad_ctx = ctx;
    bad_ctx.struct_size = sizeof(uint32_t);
    CHECK(table.elementwise(&bad_ctx, &invalid, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);
    bad_ctx = ctx; bad_ctx.abi_version = GUILD_KERNEL_ABI_MAKE_VERSION(2, 0, 0);
    CHECK(table.elementwise(&bad_ctx, &invalid, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);
    bad_ctx = ctx; bad_ctx.backend = GUILD_BACKEND_CPU;
    CHECK(table.elementwise(&bad_ctx, &invalid, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    bad_ctx = ctx; bad_ctx.device_id = -1;
    CHECK(table.elementwise(&bad_ctx, &invalid, &error) == GUILD_STATUS_ERR_INVALID_ARGUMENT);

    guild_cuda_sampler_op_v1_t bad_samp{};
    bad_samp.struct_size = sizeof(bad_samp);
    bad_samp.mode = GUILD_CUDA_SAMPLER_SAMPLE_TOKENS_V1;
    bad_samp.params.struct_size = sizeof(uint32_t);
    CHECK(table.sampler(&ctx, &bad_samp, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);

    guild_cuda_qwen_op_v1_t bad_qwen{};
    bad_qwen.struct_size = sizeof(bad_qwen);
    bad_qwen.mode = GUILD_CUDA_QWEN_GR_WRITE_V1;
    bad_qwen.gr_shapes.struct_size = sizeof(uint32_t);
    CHECK(table.qwen(&ctx, &bad_qwen, &error) == GUILD_STATUS_ERR_ABI_MISMATCH);

    // 1. Configure flags + queries
    int32_t val = -1;
    guild_cuda_moe_config_v1_t cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.mode = GUILD_CUDA_MOE_CFG_SET_FLAGS_V1;
    cfg.flags = GUILD_CUDA_MOE_FLAG_SHARED_NATIVE_BF16_V1 | GUILD_CUDA_MOE_FLAG_NATIVE_ROUTER_V1;
    cfg.grouped_select_old = -2;
    CHECK(table.configure(&ctx, &cfg, &error) == GUILD_STATUS_OK);

    cfg.mode = GUILD_CUDA_MOE_CFG_DEQUANT_BF16_SUPPORTED_V1;
    cfg.value = 12;
    cfg.out_value = &val;
    CHECK(table.configure(&ctx, &cfg, &error) == GUILD_STATUS_OK);
    CHECK(val == 1);

    cfg.mode = GUILD_CUDA_MOE_CFG_PLE_AVAILABLE_V1;
    val = -1;
    CHECK(table.configure(&ctx, &cfg, &error) == GUILD_STATUS_OK);
    CHECK(val == 1);

    // 2. Quantize Q8_0 + Dequant Q8_0 round-trip
    constexpr int NQ = 64;
    std::vector<float> hx(NQ, 1.0f), hy(NQ, 0.0f);
    float *dx = nullptr, *dy = nullptr;
    uint8_t* dq = nullptr;
    CHECK(cudaMalloc(&dx, NQ * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dy, NQ * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dq, (NQ / 32) * 34) == cudaSuccess);
    CHECK(cudaMemcpy(dx, hx.data(), NQ * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_quant_dequant_op_v1_t qcmd{};
    qcmd.struct_size = sizeof(qcmd);
    qcmd.mode = GUILD_CUDA_QD_QUANTIZE_Q8_0_V1;
    qcmd.src = dx;
    qcmd.dst = dq;
    qcmd.n = NQ;
    CHECK(table.quantize_dequant(&ctx, &qcmd, &error) == GUILD_STATUS_OK);

    qcmd.mode = GUILD_CUDA_QD_DEQUANT_Q8_0_V1;
    qcmd.src = dq;
    qcmd.dst = dy;
    CHECK(table.quantize_dequant(&ctx, &qcmd, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hy.data(), dy, NQ * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hy) CHECK(std::fabs(v - 1.0f) < 0.02f);

    // 3. Router top-10 + MoE combine
    constexpr int NE = 512, K = 10, H = 2560;
    std::vector<float> hlogits(NE, -10.0f);
    for (int i = 0; i < K; ++i) hlogits[i] = 10.0f;
    float *dlogits = nullptr, *dw = nullptr, *dexp = nullptr, *dsh = nullptr, *dmout = nullptr;
    int32_t* dids = nullptr;
    CHECK(cudaMalloc(&dlogits, NE * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dids, K * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&dw, K * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dexp, K * H * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dsh, H * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dmout, H * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(dlogits, hlogits.data(), NE * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    std::vector<float> hexp(K * H, 2.0f), hsh(H, 3.0f), hmout(H, 0.0f);
    CHECK(cudaMemcpy(dexp, hexp.data(), hexp.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dsh, hsh.data(), hsh.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_moe_op_v1_t mcmd{};
    mcmd.struct_size = sizeof(mcmd);
    mcmd.mode = GUILD_CUDA_MOE_ROUTER_TOP10_V1;
    mcmd.logits = dlogits;
    mcmd.ids = dids;
    mcmd.weights = dw;
    mcmd.n_tokens = 1;
    mcmd.n_expert = NE;
    mcmd.k = K;
    CHECK(table.moe(&ctx, &mcmd, &error) == GUILD_STATUS_OK);

    mcmd.mode = GUILD_CUDA_MOE_COMBINE_V1;
    mcmd.parts = dexp;
    mcmd.shared = dsh;
    mcmd.out = dmout;
    mcmd.n_embd = H;
    mcmd.k = K;
    CHECK(table.moe(&ctx, &mcmd, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hmout.data(), dmout, H * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hmout) CHECK(std::fabs(v - 5.0f) < 1e-4f);

    // 4. Elementwise scale + CUDA Graph capture/replay
    guild_cuda_elementwise_op_v1_t ecmd{};
    ecmd.struct_size = sizeof(ecmd);
    ecmd.mode = GUILD_CUDA_ELEM_SCALE_INPLACE_V1;
    ecmd.dst = dmout;
    ecmd.scale = 2.0f;
    ecmd.n = H;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    CHECK(table.elementwise(&ctx, &ecmd, &error) == GUILD_STATUS_OK);
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
    CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hmout.data(), dmout, H * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hmout) CHECK(std::fabs(v - 10.0f) < 1e-3f);
    CHECK(cudaGraphExecDestroy(exec) == cudaSuccess && cudaGraphDestroy(graph) == cudaSuccess);

    // 5. Verify finite_first_bad + dense_steps
    unsigned long long* dbad = nullptr;
    unsigned long long hbad = 0;
    int32_t *dcell = nullptr, *dsteps = nullptr;
    int32_t hcell = 10, hsteps[4] = {0, 0, 0, 0};
    CHECK(cudaMalloc(&dbad, sizeof(unsigned long long)) == cudaSuccess);
    CHECK(cudaMalloc(&dcell, sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&dsteps, 4 * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMemcpy(dcell, &hcell, sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);

    guild_cuda_verify_op_v1_t vcmd{};
    vcmd.struct_size = sizeof(vcmd);
    vcmd.mode = GUILD_CUDA_VERIFY_FINITE_FIRST_BAD_V1;
    vcmd.values = dmout;
    vcmd.count = H;
    vcmd.u64_dst = dbad;
    CHECK(table.verify(&ctx, &vcmd, &error) == GUILD_STATUS_OK);

    vcmd.mode = GUILD_CUDA_VERIFY_DENSE_STEPS_V1;
    vcmd.i32_src = dcell;
    vcmd.n_tok = 1;
    vcmd.i32_dst = dsteps;
    CHECK(table.verify(&ctx, &vcmd, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(&hbad, dbad, sizeof(unsigned long long), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(hsteps, dsteps, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(hbad == ~0ull);
    CHECK(hsteps[0] == 10 && hsteps[1] == 11);

    // 6. Qwen native_gr_rms_norm_weighted
    std::vector<float> hgrw(H, 1.5f), hgrout(H, 0.0f);
    float *dgrw = nullptr, *dgrout = nullptr;
    CHECK(cudaMalloc(&dgrw, H * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&dgrout, H * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(dgrw, hgrw.data(), H * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    guild_cuda_qwen_op_v1_t qwen{};
    qwen.struct_size = sizeof(qwen);
    qwen.mode = GUILD_CUDA_QWEN_NATIVE_GR_RMS_NORM_V1;
    qwen.R = dsh;
    qwen.w_norm = dgrw;
    qwen.R_out = dgrout;
    qwen.n_rows = 1;
    qwen.n_cols = H;
    qwen.eps = 0.0f;
    CHECK(table.qwen(&ctx, &qwen, &error) == GUILD_STATUS_OK);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaMemcpy(hgrout.data(), dgrout, H * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : hgrout) CHECK(std::fabs(v - 1.5f) < 1e-4f);

    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    for (void* ptr : {static_cast<void*>(dx), static_cast<void*>(dy), static_cast<void*>(dq),
                      static_cast<void*>(dlogits), static_cast<void*>(dw), static_cast<void*>(dexp),
                      static_cast<void*>(dsh), static_cast<void*>(dmout), static_cast<void*>(dids),
                      static_cast<void*>(dbad), static_cast<void*>(dcell), static_cast<void*>(dsteps),
                      static_cast<void*>(dgrw), static_cast<void*>(dgrout)}) CHECK(cudaFree(ptr) == cudaSuccess);
    return 0;
}
