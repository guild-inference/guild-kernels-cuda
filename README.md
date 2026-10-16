# guild-kernels-cuda (Phase 2 migration preview)

First independently buildable CUDA migration slice: the validated BF16-to-FP32
MoE router projection, package `org.guild.kernels.cuda`, implementation
`external.cuda.router.bf16.v1`. It uses Kernel ABI v1's C prefill projection
extension, binds once at model load and launches on the supplied stream.

**This is not the full CUDA backend extraction.** Dense projections, attention,
GDN, MMQ, shared experts and decode still execute Core's built-in kernels.
External router execution has been verified alongside independent CPU expert and
Qwen-runner packages during real Ornith inference with bitwise logits/state.

## Build and test independently

Needs an installed Guild Kernel SDK 1.1, CMake 3.24+, CUDA toolkit and a supported
host compiler. The current package explicitly targets GTX 1070 / SM 6.1 and its
manifest restricts compatibility to SM 6.1. CUDA 12.0 with GCC 12 was tested.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/package"
```

The CUDA test requires a real GPU; it is not counted as passing on CPU-only CI.
`CUDA_RUNTIME_LIBRARY Shared` uses the host's CUDA runtime and permits unloading
after stream completion. No Guild private headers/classes are build inputs.

## Sign/install

```sh
guild kernel sign build/package --key-id YOUR-KEY-ID --seed-file /secure/release.seed --bundle build/cuda-1.0.0.gkernel
guild kernel trust-key --publisher org.guild-inference --key-id YOUR-KEY-ID --pubkey YOUR-PUBLIC-KEY
guild kernel install build/cuda-1.0.0.gkernel
guild kernel active --json
```

Package version is `1.0.0`. Keep seed files out of the repository. Core verifies
signatures and artifact hashes before `dlopen`. Disabling this package restores
the built-in router for future models; bound model leases prevent unsafe unload.

Source: Guild `42018db`, `plugins/ornith_prefill_ext/ornith_prefill_plugin.cu`,
with CPU-operation registration removed and the package identity changed. MIT;
`LICENSE` retains the original Guild/Strata notices. Full backend ownership,
external-only CUDA builds and broader architecture releases remain future
migration gates, not supported claims of this preview.
