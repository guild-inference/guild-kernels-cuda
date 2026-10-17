#include <guild/abi/cuda_dense_abi.h>
#include <guild/abi/cuda_attn_abi.h>
#include <guild/abi/cuda_moe_abi.h>
typedef char fixed_prefix[(offsetof(guild_cuda_dense_ext_v1_t,bf16)==16)?1:-1];
typedef char fixed_attn_prefix[(offsetof(guild_cuda_attn_ext_v1_t,configure)==16)?1:-1];
typedef char fixed_moe_prefix[(offsetof(guild_cuda_moe_ext_v1_t,configure)==16)?1:-1];
typedef char fixed_status[(sizeof(guild_status_t)==4)?1:-1];
guild_cuda_dense_ext_v1_t cuda_c_contract(void) {
    guild_cuda_dense_ext_v1_t table={0};
    table.struct_size=sizeof(table);table.tag=GUILD_CUDA_DENSE_TAG_V1;
    table.version=GUILD_CUDA_DENSE_VERSION_V1;return table;
}
guild_cuda_attn_ext_v1_t cuda_attn_c_contract(void) {
    guild_cuda_attn_ext_v1_t table={0};
    table.struct_size=sizeof(table);table.tag=GUILD_CUDA_ATTN_TAG_V1;
    table.version=GUILD_CUDA_ATTN_VERSION_V1;return table;
}
guild_cuda_moe_ext_v1_t cuda_moe_c_contract(void) {
    guild_cuda_moe_ext_v1_t table={0};
    table.struct_size=sizeof(table);table.tag=GUILD_CUDA_MOE_TAG_V1;
    table.version=GUILD_CUDA_MOE_VERSION_V1;return table;
}
