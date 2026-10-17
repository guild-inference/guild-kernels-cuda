#pragma once
#include <guild/abi/cuda_dense_abi.h>
#ifdef __cplusplus
extern "C" {
#endif
// Pure table discovery; initializes no CUDA context or mutable device tables.
GUILD_PLUGIN_EXPORT const guild_cuda_dense_ext_v1_t* guild_cuda_dense_table_v1(void);
// Successful callback launches; enabled explicitly with GUILD_CUDA_INSTRUMENT=1.
// 0=BF16, 1=Q8_1 quantization, 2=MMVQ; numerical tests remain necessary.
GUILD_PLUGIN_EXPORT uint64_t guild_cuda_dense_call_count_v1(uint32_t operation);
#ifdef __cplusplus
}
#endif
