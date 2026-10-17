# Canonical attention, recurrent (GDN), RoPE, KV and selection CUDA sources.
get_filename_component(GUILD_CUDA_ATTN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
function(guild_cuda_attach_attn target)
  target_sources(${target} PRIVATE
    "${GUILD_CUDA_ATTN_ROOT}/src/attn_backend.cpp"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/fused_gdn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/gdn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/gdn_multi.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/kv_q4.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/kv_q8.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/kv_stream.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_flash_attn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_gdn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_gdn_preprocess.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa_indexer.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa_score.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_rope.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/qsa.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/qsa_decode_attn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/qsa_prompt_attn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/qsa_select.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/rope.cu")
  target_include_directories(${target} PUBLIC "${GUILD_CUDA_ATTN_ROOT}/include")
  target_compile_definitions(${target} PRIVATE GUILD_CUDA_ATTN_MATH_NS=guild::cuda_attn_math)
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
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_flash_attn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_gdn.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_gdn_preprocess.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa_indexer.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_qsa_score.cu"
    "${GUILD_CUDA_ATTN_ROOT}/src/attention/native_rope.cu"
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
endfunction()
