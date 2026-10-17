#pragma once
#include "guild/abi/cuda_moe_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
// Pure table discovery; initializes no CUDA context or mutable device tables.
GUILD_PLUGIN_EXPORT const guild_cuda_moe_ext_v1_t* guild_cuda_moe_table_v1(void);
// Successful callback launches; enabled explicitly with GUILD_CUDA_INSTRUMENT=1.
// 0=configure, 1=quantize_dequant, 2=s_gemv, 3=moe, 4=elementwise, 5=sampler, 6=verify, 7=qwen.
GUILD_PLUGIN_EXPORT uint64_t guild_cuda_moe_call_count_v1(uint32_t category);
#ifdef __cplusplus
}
#endif
