#include <guild/abi/cuda_dense_abi.h>
typedef char fixed_prefix[(offsetof(guild_cuda_dense_ext_v1_t,bf16)==16)?1:-1];
typedef char fixed_status[(sizeof(guild_status_t)==4)?1:-1];
guild_cuda_dense_ext_v1_t cuda_c_contract(void) {
    guild_cuda_dense_ext_v1_t table={0};
    table.struct_size=sizeof(table);table.tag=GUILD_CUDA_DENSE_TAG_V1;
    table.version=GUILD_CUDA_DENSE_VERSION_V1;return table;
}
