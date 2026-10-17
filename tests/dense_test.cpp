// Independent scalar oracles use exactly representable dyadic weights and
// integer activations: no tolerance can hide a block-layout or reduction error.
#include "guild/cuda/dense_backend.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"dense check line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
int main() {
    const auto& table=*guild_cuda_dense_table_v1();
    CHECK(table.struct_size==sizeof(table) && table.tag==GUILD_CUDA_DENSE_TAG_V1 && table.version==1);
    constexpr int K=256,N=16,MAX_ROWS=256;
    float *x=nullptr,*y=nullptr,*by=nullptr;void* q=nullptr;uint16_t* bw=nullptr;uint8_t* qw=nullptr;
    cudaStream_t stream{};CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(cudaMalloc(&x,K*MAX_ROWS*4)==cudaSuccess && cudaMalloc(&y,N*MAX_ROWS*4)==cudaSuccess);
    CHECK(cudaMalloc(&by,N*MAX_ROWS*4)==cudaSuccess && cudaMalloc(&bw,K*N*2)==cudaSuccess);
    CHECK(cudaMalloc(&q,K/32*36*MAX_ROWS)==cudaSuccess && cudaMalloc(&qw,K/32*18*N)==cudaSuccess);
    guild_exec_context_v1_t ctx{};ctx.struct_size=sizeof(ctx);ctx.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    ctx.backend=GUILD_BACKEND_CUDA;ctx.device_id=0;ctx.device_stream=stream;ctx.sync_mode=GUILD_SYNC_STREAM_ORDERED;
    guild_error_info_t error{};
    std::vector<float> host(K*MAX_ROWS),expected(N*MAX_ROWS),bf_expected(N*MAX_ROWS),got(N*MAX_ROWS);
    for(int r=0;r<MAX_ROWS;++r)for(int i=0;i<K;++i)
        host[r*K+i]=(i%32==31)?127.0f:float((i%32)*(((i+r)&1)?-1:1));
    std::vector<uint16_t> weights(K*N,0x3f80); // BF16 one
    std::vector<uint8_t> quantized(K/32*18*N);
    for(int o=0;o<N;++o)for(int b=0;b<K/32;++b) {
        uint8_t* block=quantized.data()+(o*K/32+b)*18;
        const uint16_t scale=uint16_t(0x3400+(o%4)*0x400);std::memcpy(block,&scale,2);
        for(int j=0;j<16;++j)block[2+j]=uint8_t(((j+o)%16)<<4)|uint8_t((15-j+o)%16);
    }
    for(int r=0;r<MAX_ROWS;++r)for(int o=0;o<N;++o) {
        double sum=0,bf_sum=0;
        for(int i=0;i<K;++i) {
            const auto* block=quantized.data()+(o*K/32+i/32)*18;
            const unsigned codes=block[2+i%16];const int code=int((codes>>((i%32>=16)?4:0))&15)-8;
            sum+=double(code)*std::ldexp(0.25,o%4)*host[r*K+i];bf_sum+=host[r*K+i];
        }
        expected[r*N+o]=float(sum);bf_expected[r*N+o]=float(bf_sum);
    }
    CHECK(cudaMemcpy(x,host.data(),host.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(bw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice)==cudaSuccess);
    CHECK(cudaMemcpy(qw,quantized.data(),quantized.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    guild_cuda_bf16_projection_v1_t bf{};bf.struct_size=sizeof(bf);bf.mode=GUILD_CUDA_BF16_F32_COLS_V1;
    bf.x=x;bf.w=bw;bf.y=by;bf.n_in=K;bf.n_out=N;bf.ldx=K;bf.ldy=N;
    guild_cuda_quantize_q8_1_v1_t quant{};quant.struct_size=sizeof(quant);quant.x=x;quant.q=q;quant.n_in=K;
    quant.q_bytes=K/32*36*MAX_ROWS;
    guild_cuda_mmvq_v1_t product{};product.struct_size=sizeof(product);product.exact_multi=1;
    product.weight_dtype=GUILD_DTYPE_Q4_0;product.n_in=K;product.n_out=N;product.weights=qw;product.q=q;product.y=y;
    for(int rows:{1,2,4,8,17,256}) {
        bf.rows=quant.rows=product.rows=rows;
        CHECK(table.bf16(&ctx,&bf,&error)==GUILD_STATUS_OK);
        CHECK(table.quantize_q8_1(&ctx,&quant,&error)==GUILD_STATUS_OK);
        CHECK(table.mmvq(&ctx,&product,&error)==GUILD_STATUS_OK);
        CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
        CHECK(cudaMemcpy(got.data(),by,N*rows*4,cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(std::memcmp(got.data(),bf_expected.data(),N*rows*4)==0);
        CHECK(cudaMemcpy(got.data(),y,N*rows*4,cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(std::memcmp(got.data(),expected.data(),N*rows*4)==0);
    }
    // Capture/replay the three operations with fixed caller-owned addresses.
    bf.rows=quant.rows=product.rows=17;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)==cudaSuccess);
    CHECK(table.bf16(&ctx,&bf,&error)==GUILD_STATUS_OK);
    CHECK(table.quantize_q8_1(&ctx,&quant,&error)==GUILD_STATUS_OK);
    CHECK(table.mmvq(&ctx,&product,&error)==GUILD_STATUS_OK);
    cudaGraph_t graph{};cudaGraphExec_t executable{};
    CHECK(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
    CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)==cudaSuccess);
    for(int repeat=0;repeat<3;++repeat)CHECK(cudaGraphLaunch(executable,stream)==cudaSuccess);
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
    CHECK(cudaMemcpy(got.data(),y,N*17*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    CHECK(std::memcmp(got.data(),expected.data(),N*17*4)==0);
    CHECK(cudaGraphExecDestroy(executable)==cudaSuccess && cudaGraphDestroy(graph)==cudaSuccess);
    // All malformed requests fail before a device launch; no exception escapes C.
    product.weight_dtype=999;CHECK(table.mmvq(&ctx,&product,&error)!=GUILD_STATUS_OK && error.message[0]);
    bf.struct_size=4;CHECK(table.bf16(&ctx,&bf,&error)!=GUILD_STATUS_OK);
    quant.q_bytes=1;CHECK(table.quantize_q8_1(&ctx,&quant,&error)!=GUILD_STATUS_OK);
    ctx.abi_version=GUILD_MAKE_ABI_VERSION(2,0,0);CHECK(table.mmvq(&ctx,&product,&error)!=GUILD_STATUS_OK);
    CHECK(cudaGetLastError()==cudaSuccess);
    CHECK(cudaFree(x)==cudaSuccess && cudaFree(y)==cudaSuccess && cudaFree(by)==cudaSuccess);
    CHECK(cudaFree(bw)==cudaSuccess && cudaFree(qw)==cudaSuccess && cudaFree(q)==cudaSuccess);
    CHECK(cudaStreamDestroy(stream)==cudaSuccess);
    std::puts("CUDA dense dyadic scalar oracles, 1..256 rows, capture/replay and malformed requests passed");
}
