# guild-kernels-cuda

Independent CUDA arithmetic, package `org.guild.kernels.cuda` version1.1.0,
Kernel ABI v1, installed SDK1.4. Core software versions are independent.

## Implemented source ownership

- BF16 activation GEMV and row-split GEMV.
- BF16-weight/F32-activation single-row, strided 1..8-row and tiled 1..512-row
  projection paths, preserving the existing batched projection topology.
- Native Q8_1 activation quantization and fused SwiGLU/quantization.
- Native MMVQ for Q4_0/Q5_0/Q5_1/Q8_0, Q3_K/Q4_K/Q5_K/Q6_K, IQ4_NL/IQ4_XS,
  Q2_0 and the existing IQ fallbacks, including exact multi-row dispatch.
- IQ codebooks, quantization, dequantization, embedding and grouped expert source.
- Shared BF16/DP4A/Q8_1 finite conversion helpers.
- The existing PFKE router projection remains compatible.

`cmake/GuildCUDADenseSources.cmake` owns the bundled/DSO source list and per-file
arithmetic flags. `native_bf16.cu` and `native_mmvq.cu` retain fast math; the IQ
and legacy BF16 sources retain their previous flags. Backend implementation
symbols are private; CDD1 callbacks expose only public C records/status codes.
No private Guild headers/libraries or source-tree paths are build inputs.

The new `external.cuda.dense.v1` operation exposes CDD1 in `next_ext`; it is
separate from the frozen PFKE `external.cuda.router.bf16.v1` operation. Discovery
performs no CUDA initialization. Calls use caller-owned buffers and an explicit
stream, preserve capture/replay launch geometry, and translate C++ exceptions to
C errors. Legacy BF16-activation default-stream calls retain their blocking
semantics. Normal ordered-stream calls allocate nothing and add no wait.
Exact/nonexact MMVQ selection is explicit per call and thread-local internally.
Optional successful-callback instrumentation uses `GUILD_CUDA_INSTRUMENT=1`.

The IQ data is compiled from pinned GGML
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Device codebooks are module constants;
these operations do not call or depend on another DSO's GGML CPU initialization.
MMQ context initialization, cuBLAS GEMM, attention, recurrent, KV and other
remaining operations still require their own contracts/migration. Core's direct
IQ calls are not yet CDD1-bound. Full external-only CUDA inference remains an
open integration gate; independent kernel tests do not establish that gate.

## Independent build and device tests

Needs installed SDK1.4, CMake3.24+, CUDA and a compatible host compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j
ctest --test-dir build --output-on-failure --parallel 1
cmake --install build --prefix "$PWD/build/package"
```

CUDA12.0/GCC12 nvcc host, GCC13.3 C++, GTX1070 SM61: the router test and the new
dense test pass. Dense tests use exact scalar dyadic BF16/Q4_0 oracles at
1/2/4/8/17/256 rows, capture/replay, short records, wrong ABI, unsupported dtype
and undersized scratch. Checks remain active in Release. Hosted CI compiles
CUDA12/SM61 and CUDA13/SM75, including C99 headers and both GPU test programs;
hosted compilation is not counted as actual GPU test execution.

Metadata records the explicitly compiled architecture range. Only SM61 has
real-device validation here; other architectures require device gates. Use
separate packages for disjoint device targets rather than assuming every SM
between two compiled SASS targets has a compatible image.

## Sign and install

```sh
guild kernel sign build/package --key-id YOUR-KEY-ID --seed-file /secure/release.seed --bundle build/cuda-1.1.0.gkernel
guild kernel trust-key --publisher org.guild-inference --key-id YOUR-KEY-ID --pubkey YOUR-PUBLIC-KEY
guild kernel install build/cuda-1.1.0.gkernel
guild kernel active --json
```

Seeds stay outside the repository and package. CI artifacts are unsigned staging
directories. Independently supplied production signing is a separate release
step. Core verifies signatures/hashes before loading native code; signing is
provenance, not sandboxing.

The source migration started from Guild `a5cd236`/`73905fa`. MIT licenses and
original Guild/Strata/GGML attribution are preserved. Standalone installation
includes Guild/Strata and pinned GGML license files.
