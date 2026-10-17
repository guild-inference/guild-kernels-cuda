# guild-kernels-cuda

Independent CUDA arithmetic, package `org.guild.kernels.cuda` version 1.2.0,
Kernel ABI v1, installed SDK 1.4. Core software versions are independent.

## Implemented source ownership

- BF16 activation GEMV and row-split GEMV.
- BF16-weight/F32-activation single-row, strided 1..8-row and tiled 1..512-row
  projection paths, preserving the existing batched projection topology.
- Native Q8_1 activation quantization and fused SwiGLU/quantization.
- Native MMVQ for Q4_0/Q5_0/Q5_1/Q8_0, Q3_K/Q4_K/Q5_K/Q6_K, IQ4_NL/IQ4_XS,
  Q2_0 and the existing IQ fallbacks, including exact multi-row dispatch.
- IQ codebooks, quantization, dequantization, embedding and grouped expert source.
- Attention, recurrent (GDN), RoPE, KV cache/streaming and QSA selection kernels:
  - QSA weighted RMSNorm, gate application (F16, F32, native), indexer key
    append/pooling, cell/block scoring (FP32, tensor-core, native SM120),
    top-k selection (bit-lifting, 4-pass radix, cluster), split-K decode
    attention, prompt attention, and short-context FlashAttention adapter.
  - Gated Delta-Net (GDN) single-token, fused, native, and multi-token
    prefill/verify convolution, A/B projection, state recurrence, commit, and
    output normalization kernels.
  - NEOX partial RoPE, analytic native IMRoPE (none/Linear/YaRN), and
    device-local M-RoPE / float64 angle table registration.
  - Paged KV append/gather (FP16, INT8, Hadamard-rotated Q4_0) and KV streaming
    CLOCK residency resolution, ring restore, and host staging.
- Shared BF16/FP16/DP4A/Q8_1 finite conversion and CC emulation helpers.
- The existing PFKE router projection remains compatible.

`cmake/GuildCUDADenseSources.cmake` and `cmake/GuildCUDAAttnSources.cmake` own
the bundled/DSO source lists and per-file arithmetic flags. `native_bf16.cu`,
`native_mmvq.cu`, and the seven `native_*.cu` attention/GDN/RoPE adapters retain
fast math; the IQ, legacy BF16, and canonical QSA/GDN/KV sources retain their
previous flags. Backend implementation symbols are private; CDD1 and CDA1
callbacks expose only public C records/status codes. No private Guild
headers/libraries or source-tree paths are build inputs.

`external.cuda.dense.v1` exposes CDD1 (`guild_cuda_dense_ext_v1_t`) on
`GUILD_OP_QUANTIZED_GEMV`, and `external.cuda.attn.v1` exposes CDA1
(`guild_cuda_attn_ext_v1_t`) on `GUILD_OP_ATTENTION_DECODE`, alongside the
frozen PFKE `external.cuda.router.bf16.v1` operation. Discovery performs no CUDA
initialization. Calls use caller-owned buffers and an explicit stream, preserve
capture/replay launch geometry, and translate C++ exceptions to C errors. Normal
ordered-stream calls allocate nothing and add no wait. Optional
successful-callback instrumentation uses `GUILD_CUDA_INSTRUMENT=1`.

The IQ data is compiled from pinned GGML
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Device codebooks are module constants;
these operations do not call or depend on another DSO's GGML CPU initialization.
MMQ context initialization, cuBLAS GEMM, MoE grouped dispatch, elementwise,
sampling, verification helpers, and Qwen specializations still require their own
contracts/migration. Full external-only CUDA inference remains an open
integration gate; independent kernel tests do not establish that gate.

## Independent build and device tests

Needs installed SDK1.4, CMake3.24+, CUDA and a compatible host compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j
ctest --test-dir build --output-on-failure --parallel 1
cmake --install build --prefix "$PWD/build/package"
```

CUDA12.0/GCC12 nvcc host, GCC13.3 C++, GTX1070 SM61: the router test, dense
test, and attention/GDN/RoPE/KV test pass. Dense and attention tests use exact
scalar oracles, capture/replay, short records, wrong ABI, invalid modes and
undersized scratch. Checks remain active in Release. Hosted CI compiles
CUDA12/SM61 and CUDA13/SM75, including C99 headers and all three GPU test
programs; hosted compilation is not counted as actual GPU test execution.

Metadata records the explicitly compiled architecture range. Only SM61 has
real-device validation here; other architectures require device gates. Use
separate packages for disjoint device targets rather than assuming every SM
between two compiled SASS targets has a compatible image.

## Sign and install

```sh
guild kernel sign build/package --key-id YOUR-KEY-ID --seed-file /secure/release.seed --bundle build/cuda-1.2.0.gkernel
guild kernel trust-key --publisher org.guild-inference --key-id YOUR-KEY-ID --pubkey YOUR-PUBLIC-KEY
guild kernel install build/cuda-1.2.0.gkernel
guild kernel active --json
```

Seeds stay outside the repository and package. CI artifacts are unsigned staging
directories. Independently supplied production signing is a separate release
step. Core verifies signatures/hashes before loading native code; signing is
provenance, not sandboxing.

The source migration started from Guild `a5cd236`/`73905fa`. MIT licenses and
original Guild/Strata/GGML attribution are preserved. Standalone installation
includes Guild/Strata and pinned GGML license files.
