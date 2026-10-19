# Canonical source list and arithmetic flags, shared by bundled and DSO builds.
get_filename_component(GUILD_CUDA_DENSE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
function(guild_cuda_attach_dense target)
  target_sources(${target} PRIVATE
    "${GUILD_CUDA_DENSE_ROOT}/src/dense_backend.cpp"
    "${GUILD_CUDA_DENSE_ROOT}/src/math/bf16_gemv.cu"
    "${GUILD_CUDA_DENSE_ROOT}/src/math/native_bf16.cu"
    "${GUILD_CUDA_DENSE_ROOT}/src/math/native_mmvq.cu")
  target_include_directories(${target} PUBLIC "${GUILD_CUDA_DENSE_ROOT}/include")
  target_compile_definitions(${target} PRIVATE
    GUILD_CUDA_DENSE_MATH_NS=guild::cuda_dense_math
    GUILD_CUDA_MOE_MATH_NS=guild::cuda_moe_math)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(
    "${GUILD_CUDA_DENSE_ROOT}/src/math/native_bf16.cu"
    "${GUILD_CUDA_DENSE_ROOT}/src/math/native_mmvq.cu"
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
endfunction()
include("${CMAKE_CURRENT_LIST_DIR}/GuildCUDAAttnSources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/GuildCUDAMoESources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/GuildCUDAPrefillSources.cmake")
