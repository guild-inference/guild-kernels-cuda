#include "guild/cuda/dense_backend.h"
#include "guild/kernels/bf16_gemv.hpp"
#include "guild/kernels/native_mmvq.hpp"
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
#include <stdexcept>

namespace {
namespace math = guild::cuda_dense_math;
struct AbiMismatch : std::invalid_argument { using std::invalid_argument::invalid_argument; };
std::atomic<uint64_t> calls[3]{};
void count(unsigned operation) {
    static const bool enabled = [] { const char* v=std::getenv("GUILD_CUDA_INSTRUMENT");return v && std::atoi(v)!=0; }();
    if(enabled)calls[operation].fetch_add(1,std::memory_order_relaxed);
}
guild_status_t fail(guild_error_info_t* error,guild_status_t code,const char* message) {
    if(error) {
        error->struct_size=sizeof(*error);error->code=code;
        error->source_component="cuda.dense.v1";
        std::snprintf(error->message,sizeof(error->message),"%s",message);
    }
    return code;
}
template<class Function> guild_status_t guarded(guild_error_info_t* error,Function function) noexcept {
    try { function();return GUILD_STATUS_OK; }
    catch(const std::bad_alloc& e){return fail(error,GUILD_STATUS_ERR_OUT_OF_MEMORY,e.what());}
    catch(const AbiMismatch& e){return fail(error,GUILD_STATUS_ERR_ABI_MISMATCH,e.what());}
    catch(const std::invalid_argument& e){return fail(error,GUILD_STATUS_ERR_INVALID_ARGUMENT,e.what());}
    catch(const std::exception& e){return fail(error,GUILD_STATUS_ERR_EXECUTION_FAILED,e.what());}
    catch(...){return fail(error,GUILD_STATUS_ERR_INTERNAL,"unknown CUDA backend exception");}
}
void context(const guild_exec_context_v1_t* c,bool ordered=true) {
    if(!c || c->struct_size<sizeof(*c) || GUILD_ABI_VERSION_MAJOR(c->abi_version)!=1)
        throw AbiMismatch("CUDA dense execution context/ABI mismatch");
    if(c->backend!=GUILD_BACKEND_CUDA || c->device_id<0 || (ordered && !c->device_stream))
        throw std::invalid_argument("CUDA dense operation requires an explicit CUDA device and ordered stream");
}
bool aligned(const void* p,unsigned alignment=4){return p && !(reinterpret_cast<uintptr_t>(p)&(alignment-1));}
guild_status_t bf16(const guild_exec_context_v1_t* c,const guild_cuda_bf16_projection_v1_t* p,guild_error_info_t* e) {
    return guarded(e,[&] {
        if(!p || p->struct_size<sizeof(*p))throw AbiMismatch("CUDA BF16 parameter size mismatch");
        const bool legacy=p->mode<=GUILD_CUDA_BF16_ACT_SPLIT_V1;
        context(c,false); // F32 MMVF also permits CUDA's nonblocking default-stream form.
        if(p->n_in<=0 || p->n_in>INT_MAX || p->n_out<=0 || p->n_out>INT_MAX || p->rows<1 || p->rows>512 ||
           !aligned(p->x,legacy?2:8) || !aligned(p->w) || !aligned(p->y) || p->ldy<p->n_out || p->ldx<p->n_in)
            throw std::invalid_argument("CUDA BF16 invalid geometry, stride or pointer alignment");
        if(!legacy && ((p->n_in&1) || (p->ldx&1)))throw std::invalid_argument("CUDA BF16 F32 requires even input width/stride");
        switch(p->mode) {
        case GUILD_CUDA_BF16_ACT_AUTO_V1:
            if(p->rows!=1)throw std::invalid_argument("BF16-activation GEMV requires one row");
            math::bf16_gemv(static_cast<const uint16_t*>(p->x),p->w,p->y,p->n_in,p->n_out,c->device_stream);break;
        case GUILD_CUDA_BF16_ACT_SPLIT_V1:
            if(p->rows!=1 || p->threads_per_row<1 || p->threads_per_row>1024 || (p->threads_per_row&(p->threads_per_row-1)))
                throw std::invalid_argument("BF16 split requires one row and power-of-two threads <=1024");
            math::bf16_gemv_split(static_cast<const uint16_t*>(p->x),p->w,p->y,p->n_in,p->n_out,p->threads_per_row,c->device_stream);break;
        case GUILD_CUDA_BF16_F32_SINGLE_V1:
            if(p->rows!=1)throw std::invalid_argument("single-row BF16 projection requires one row");
            math::bf16_gemv_fp32_mmvf(static_cast<const float*>(p->x),p->w,p->y,p->n_in,p->n_out,c->device_stream);break;
        case GUILD_CUDA_BF16_F32_MULTI_V1:
            if(p->rows>8)throw std::invalid_argument("strided BF16 projection supports at most eight rows");
            math::bf16_gemv_fp32_mmvf_multi(static_cast<const float*>(p->x),p->ldx,p->w,p->y,p->ldy,p->n_in,p->n_out,p->rows,c->device_stream);break;
        case GUILD_CUDA_BF16_F32_COLS_V1:
            if(p->ldx!=p->n_in || p->ldy!=p->n_out)throw std::invalid_argument("batched BF16 columns require contiguous rows");
            math::bf16_gemv_fp32_mmvf_cols(static_cast<const float*>(p->x),p->w,p->y,p->n_in,p->n_out,p->rows,c->device_stream);break;
        default:throw std::invalid_argument("unknown BF16 execution mode");
        }
        count(0);
    });
}
guild_status_t quantize(const guild_exec_context_v1_t* c,const guild_cuda_quantize_q8_1_v1_t* p,guild_error_info_t* e) {
    return guarded(e,[&] {
        context(c);
        if(!p || p->struct_size<sizeof(*p) || p->reserved || p->n_in<=0 || p->n_in%32 || p->rows<1 || p->rows>512 ||
           p->n_in>INT_MAX/p->rows || !aligned(p->x) || !aligned(p->q) || (p->up && !aligned(p->up)) ||
           p->q_bytes<uint64_t(p->n_in/32)*36*uint64_t(p->rows))
            throw std::invalid_argument("CUDA Q8_1 invalid geometry, scratch size or pointer");
        if(p->up)math::native_swiglu_quantize_q8_1(p->x,p->up,p->q,p->n_in,p->rows,c->device_stream);
        else math::native_quantize_q8_1(p->x,p->q,p->n_in,p->rows,c->device_stream);
        count(1);
    });
}
guild_status_t mmvq(const guild_exec_context_v1_t* c,const guild_cuda_mmvq_v1_t* p,guild_error_info_t* e) {
    return guarded(e,[&] {
        context(c);
        if(!p || p->struct_size<sizeof(*p) || p->exact_multi>1 || p->rows<1 || p->rows>512 ||
           !aligned(p->weights) || !aligned(p->q) || !aligned(p->y))throw std::invalid_argument("CUDA MMVQ invalid request");
        (void)math::native_mmvq_weight_bytes(p->weight_dtype,p->n_in,p->n_out);
        math::native_mmvq_set_multi_exact(p->exact_multi!=0);
        math::native_mmvq(p->weight_dtype,p->weights,p->q,p->y,p->n_in,p->n_out,p->rows,c->device_stream);
        count(2);
    });
}
const guild_cuda_dense_ext_v1_t table={sizeof(table),GUILD_CUDA_DENSE_TAG_V1,GUILD_CUDA_DENSE_VERSION_V1,0,bf16,quantize,mmvq};
}
extern "C" const guild_cuda_dense_ext_v1_t* guild_cuda_dense_table_v1(){return &table;}
extern "C" uint64_t guild_cuda_dense_call_count_v1(uint32_t operation){return operation<3?calls[operation].load(std::memory_order_relaxed):0;}
