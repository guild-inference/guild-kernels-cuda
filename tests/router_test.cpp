#include <guild/abi/kernel_abi.h>
#include <guild/abi/cuda_dense_abi.h>
#include <guild/abi/cuda_attn_abi.h>
#include <guild/abi/cuda_moe_abi.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
extern "C" guild_status_t guild_plugin_entry_v1(uint32_t,guild_plugin_descriptor_v1_t*,guild_error_info_t*);
static guild_prefill_kernel_ext_v1_t table{};
int main(){
    guild_plugin_descriptor_v1_t p{};guild_host_api_v1_t host{};host.struct_size=sizeof(host);
    host.negotiate_abi=+[](uint32_t a,uint32_t* b)->guild_status_t{*b=a;return GUILD_STATUS_OK;};
    host.register_kernel_op=+[](guild_core_host_t,const guild_kernel_op_desc_v1_t* d,guild_error_info_t*)->guild_status_t {
        if(d->op_kind==GUILD_OP_QUANTIZED_GEMV) {
            const auto* dense=static_cast<const guild_cuda_dense_ext_v1_t*>(d->next_ext);
            CHECK(dense && dense->struct_size==sizeof(*dense) && dense->tag==GUILD_CUDA_DENSE_TAG_V1 && dense->bf16 && dense->mmvq);
            return GUILD_STATUS_OK;
        }
        if(d->op_kind==GUILD_OP_ATTENTION_DECODE) {
            const auto* attn=static_cast<const guild_cuda_attn_ext_v1_t*>(d->next_ext);
            CHECK(attn && attn->struct_size==sizeof(*attn) && attn->tag==GUILD_CUDA_ATTN_TAG_V1 &&
                  attn->configure && attn->rope && attn->qsa_norm_gate && attn->gdn &&
                  attn->kv && attn->qsa_index_select && attn->attention);
            return GUILD_STATUS_OK;
        }
        if(d->op_kind==GUILD_OP_MOE_COMBINE) {
            const auto* moe=static_cast<const guild_cuda_moe_ext_v1_t*>(d->next_ext);
            CHECK(moe && moe->struct_size==sizeof(*moe) && moe->tag==GUILD_CUDA_MOE_TAG_V1 &&
                  moe->configure && moe->quantize_dequant && moe->s_gemv && moe->moe &&
                  moe->elementwise && moe->sampler && moe->verify && moe->qwen);
            return GUILD_STATUS_OK;
        }
        CHECK(d->op_kind==GUILD_OP_MOE_ROUTER_TOPK && d->next_ext);
        std::memcpy(&table,d->next_ext,sizeof(table));return GUILD_STATUS_OK;
    };
    CHECK(guild_plugin_entry_v1(GUILD_KERNEL_ABI_VERSION_V1,&p,nullptr)==GUILD_STATUS_OK);
    CHECK(p.init(reinterpret_cast<guild_core_host_t>(1),&host,nullptr)==GUILD_STATUS_OK);
    CHECK(table.router_projection);
    std::vector<float> x(256*3,1),got(16*3);
    std::vector<uint16_t> weights(256*16,0x3f80);
    float *dx=nullptr,*dy=nullptr;uint16_t* dw=nullptr;cudaStream_t s{};
    CHECK(cudaStreamCreate(&s)==cudaSuccess);
    CHECK(cudaMalloc(&dx,x.size()*4)==cudaSuccess && cudaMalloc(&dy,got.size()*4)==cudaSuccess);
    CHECK(cudaMalloc(&dw,weights.size()*2)==cudaSuccess);
    CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(dw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice)==cudaSuccess);
    table.router_projection(dx,dw,dy,256,16,3,s);
    CHECK(cudaStreamSynchronize(s)==cudaSuccess);
    CHECK(cudaMemcpy(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    for(float v:got)CHECK(v==256);
    CHECK(cudaFree(dx)==cudaSuccess && cudaFree(dy)==cudaSuccess && cudaFree(dw)==cudaSuccess);
    CHECK(cudaStreamDestroy(s)==cudaSuccess);
}
