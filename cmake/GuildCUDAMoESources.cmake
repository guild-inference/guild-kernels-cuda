# Canonical MoE, S-family GEMV/dequant, elementwise, sampler, verify, and Qwen specialization CUDA sources.
get_filename_component(GUILD_CUDA_MOE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GUILD_CUDA_MOE_SOURCES
    "${GUILD_CUDA_MOE_ROOT}/src/moe_backend.cpp"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/dequant_bf16.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/dequant_s2.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/elementwise.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/native_moe.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/native_router.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/quantize_act.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/remote_expert_kernels.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/router_top10.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s2_expert_grouped.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s2_gemv.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s2_gemv_fast.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s2_gemv_q8.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s2_gemv_quads.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/s_gemv.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/sampler.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/shared_expert.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/verify_kernels.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/cvec.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/fused_gr.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/gr.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_gr_norm.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_gr_postops.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_ple_postops.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/ple.cu")
function(guild_cuda_attach_moe target)
  target_sources(${target} PRIVATE ${GUILD_CUDA_MOE_SOURCES})
  set(_dense_math_ns guild::cuda_dense_math)
  if(ARGC GREATER 1)
    set(_dense_math_ns "${ARGV1}")
  endif()
  target_include_directories(${target} PUBLIC "${GUILD_CUDA_MOE_ROOT}/include")
  target_compile_definitions(${target} PRIVATE
    GUILD_CUDA_MOE_MATH_NS=guild::cuda_moe_math
    GUILD_CUDA_DENSE_MATH_NS=${_dense_math_ns}
    GUILD_CUDA_ATTN_MATH_NS=guild::cuda_attn_math)
  set(_has_pre75 FALSE)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_arch}")
    if(_base MATCHES "^[0-9]+$" AND _base LESS 75)
      set(_has_pre75 TRUE)
    endif()
  endforeach()
  if(GUILD_EXPERIMENTAL_SM60 OR _has_pre75)
    target_compile_definitions(${target} PRIVATE GUILD_EXPERIMENTAL_SM60=1)
  endif()
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(
    "${GUILD_CUDA_MOE_ROOT}/src/moe/native_moe.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/moe/native_router.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_gr_norm.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_gr_postops.cu"
    "${GUILD_CUDA_MOE_ROOT}/src/qwen/native_ple_postops.cu"
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
endfunction()
