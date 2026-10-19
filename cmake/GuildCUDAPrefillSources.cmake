# Canonical CUDA prefill GEMM, elementwise/stateful prompt kernels, fused MoE prompt kernels, and MMQ context sources.
get_filename_component(GUILD_CUDA_PREFILL_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GUILD_CUDA_PREFILL_SOURCES
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/gemm.cu"
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/kernels.cu")
set(GUILD_CUDA_PREFILL_MMQ_CUDA_SOURCES
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/moe_mmq.cu"
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/moe_fused.cu"
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/moe_fused_iq.cu"
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/ggml_cuda_host.cu")
set(GUILD_CUDA_PREFILL_MMQ_HIP_SOURCES
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/moe_mmq.cu"
    "${GUILD_CUDA_PREFILL_ROOT}/src/prefill/ggml_cuda_host.cu")
function(guild_cuda_attach_prefill target)
  target_sources(${target} PRIVATE ${GUILD_CUDA_PREFILL_SOURCES})
  target_include_directories(${target} PUBLIC "${GUILD_CUDA_PREFILL_ROOT}/include" PRIVATE "${GUILD_CUDA_PREFILL_ROOT}/src/prefill")
endfunction()
function(guild_cuda_attach_prefill_mmq target)
  target_sources(${target} PRIVATE ${GUILD_CUDA_PREFILL_MMQ_CUDA_SOURCES})
  target_include_directories(${target} PUBLIC "${GUILD_CUDA_PREFILL_ROOT}/include")
endfunction()
