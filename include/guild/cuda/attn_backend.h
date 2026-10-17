#pragma once
#include "guild/abi/cuda_attn_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
// Pure table discovery; initializes no CUDA context or mutable device tables.
GUILD_PLUGIN_EXPORT const guild_cuda_attn_ext_v1_t* guild_cuda_attn_table_v1(void);
// Successful callback launches; enabled explicitly with GUILD_CUDA_INSTRUMENT=1.
// 0=configure, 1=rope, 2=qsa_norm_gate, 3=gdn, 4=kv, 5=qsa_index_select, 6=attention.
GUILD_PLUGIN_EXPORT uint64_t guild_cuda_attn_call_count_v1(uint32_t category);
#ifdef __cplusplus
}
#endif
