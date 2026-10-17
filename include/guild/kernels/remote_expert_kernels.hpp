#pragma once

#include <cstdint>

#ifndef GUILD_CUDA_MOE_MATH_NS
#define GUILD_CUDA_MOE_MATH_NS guild::kernels
#endif
namespace GUILD_CUDA_MOE_MATH_NS {

void remote_expert_reduce(const float* parts, const void* meta, float* sum, int tokens, void* stream);
void remote_expert_copy_cpu_rows(float* dst, const float* src, const int32_t* mask,
                                 const int32_t* hit_rows, const int32_t* count, int tokens, void* stream);

}  // namespace GUILD_CUDA_MOE_MATH_NS
