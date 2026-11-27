# SmRast-VK v1.0 — Pure-Compute Vulkan 1.3 (+ Emulated 1.4) Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** with **emulated Vulkan 1.4 (`VK_KHR_maintenance5` / `VK_KHR_maintenance6`)** that runs real-time 3D Vulkan graphics, **DirectX 12 (`vkd3d-proton`, SM 6.6)**, and **DirectX 11 (`DXVK`)** workloads **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]` with up to **8 independent hardware queues + native sparse binding**), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, hardware GPU timestamps, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (`vkQueueSubmit` on graphics queues hangs DMA on H20), forcing workloads onto CPU Lavapipe (`llvmpipe`, 0.1–10 FPS on complex 3D scenes). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **8 independent pure-compute hardware queues with sparse binding (`qf[2]`)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)**.

**SmRast-VK v1.0.0** breaks **1.29 billion triangles/sec (`1,299.0 FPS` / `0.77 ms` on 1M-tri 1080p Cook-Torrance PBR — `125×` faster than 48-Core CPU Lavapipe)**, reaches **`85–108.6 FPS` (`9.2–11.6 ms`)** on live commercial 1M-tri AAA 3D deferred DX11 scenes (**`710×–905×` faster than 48-Core CPU Lavapipe**, zero flicker, zero white blocks), renders multiple **commercial SM 6.6 DirectX 12 engines live at `30–60 FPS` (VSync-capped) and `250–297 FPS` (DX11)** with **0 crashes and 0 shader lowering failures across 8,000+ production graphics pipelines**, and runs **1,000-draw 1080p cross-material Uber-Batches at `325–426 FPS`** (`2.35–3.07 ms`) — fusing four classic GPU software rasterization architectures with a validated SPIR-V lowering compiler, zero-copy linear-aliased G-Buffers, and GPU-driven cross-material uber-batching:

| Classic Reference | Technique Integrated in SmRast-VK v1.0 |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | **7-Plane Homogeneous Guard-Band Sutherland-Hodgman Clipping** (`w>=1e-4`, `z>=0`, `w-z>=0`, `±4w` XY guard-band) + **`VK_EXT_depth_clip_enable`**, **Exact Triangle-vs-Tile Edge Trivial-Reject**, **64-Way Warp-Cooperative Large-Triangle Binning**, **Pre-Binning Coarse Hi-Z Culling**, GPU **Active-Tile Compaction**, Depth Bias, `0x7FC0DEAD` NaN Primitive Restart, and Lines/Points binning (`shaders/smrast_bin.comp`, `shaders/smrast_prim_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path expanded to **`bbox ≤ 128 px`** with **branchless precomputed canonical edges**, deterministic **Vulkan Painter's Order** `(uint64_t(~tid) << 32) \| tid`, self-clearing `VisBuf`, and zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | **Precomputed Branchless Fine Rasterizer** with **Loader-Time Hi-Z Compaction**, **Warp-Uniform Y-Band Early-Out**, and **Intra-Tile Early-Z + Inter-Batch Live Hi-Z Tightening** (`shaders/smrast_fine.comp`), **Zero-Bin Inline-Setup Fused Quad FS** (with inline 7-plane clip + barycentric remap, skipping `smrast_bin.comp` completely!), **Cross-Material Per-Draw Push-Constant & Dynamic-UBO SSBO Uber-Batching**, **Zero-Copy Linear-Aliased Color G-Buffers**, watertight Top-Left Fill-Rule & 4-bit subpixel snapping, and quad-swizzled subgroup compute fragment shading |
| **In-Process SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Thread-local `SPIRV-Cross` + `libshaderc` + **SPIR-V Binary `OpSampledImage` Basic-Block Sinker + Post-Sink `spirv-opt -O`** (`861/861` `spirv-val` clean), **Dynamic Compact Per-Pipeline `vert_stride_vec4` (`2..16` vec4)**, bare `gl_ClipDistance[8]`/`gl_CullDistance[8]`/`sampler*ArrayShadow` lowering, automatic Push-Constant & Dynamic-UBO SSBO lowering, `flat` & `noperspective` varyings, `#extension` hoisting (`vkd3d-proton` bindless), 8 VBs + `VK_EXT_vertex_attribute_divisor`, `UINT8/16/32` indices, 8-MRT × 21 RT formats with per-RT `rt_stride_px`, 8-bit Stencil Test/Op + Sampled Stencil Aspect SRVs, spec-exact `gl_FragCoord.w = 1/w_clip`, and branchless quad derivatives |

---

## What's New in `v1.0.0` (Multi-Engine DX11/DX12 Hardening Release)

Stress-testing SmRast-VK across **10+ distinct commercial DirectX 12 & DirectX 11 engines** uncovered and resolved **10 production ICD, synchronization, memory-management, and SPIR-V compiler bugs**:

- **SPIR-V `OpSampledImage` Basic-Block Sinker + Post-Sink `spirv-opt -O` (`src/spirv_lower.cpp`)**:
  - When DXIL-derived fragment shaders place conditional/ternary coordinate expressions inside separate `texture2D` + `sampler` calls, `glslang` can emit `OpSampledImage` in an earlier basic block than its consuming `OpImageSample*` (violating `VUID-StandaloneSpirv-OpSampledImage-04651` and crashing NVIDIA 535's `libnvidia-glvkspirv.so` on un-inlined `-O0` shaders).
  - Added a zero-dependency binary SPIR-V pass `sm_sink_op_sampled_image()` that sinks every cross-block `OpSampledImage` immediately before its consumer(s) (cloning IDs + `OpDecorate` annotations inside the SPIR-V Annotations section when multi-used), followed by a thread-safe `posix_spawnp("spirv-opt", "-O", ...)` pass on the ~1% `-O0` fallback shaders — achieving **0/861 `spirv-val` errors** and eliminating driver SPIR-V compiler crashes across 5 distinct DX12 engines.
- **Bare Built-In Stage I/O & Shadow-Array Lowering (`src/spirv_lower.cpp`)**:
  - Strips bare built-in stage redeclarations (`out float gl_ClipDistance[8];`, `in float gl_CullDistance[N];`), injects lowered FS/VS built-ins (`sm_gl_ClipDistance`, `sm_gl_CullDistance`, `gl_PointCoord`, `gl_ViewIndex`, `gl_VertexID`, `gl_InstanceID`, `gl_BaseVertexARB`, `gl_BaseInstanceARB`, `gl_DrawIDARB`), and lowers `sampler2DArrayShadow` / `sampler1DArrayShadow` to `textureLod(..., 0.0)` — driving **0 shader lowering failures across 8,000+ compiled commercial pipelines**.
- **8-Queue Hardware Pool + Native `vkQueueBindSparse` & Deadlock-Free Cross-Queue Timeline Semaphores (`src/smrast_icd.cpp`, `src/smrast_wsi.cpp`)**:
  - Exposes up to **8 independent hardware pure-compute queues** (`qf[2]` on H20) alongside `VK_QUEUE_SPARSE_BINDING_BIT` (strictly synchronized with `features.sparseBinding`), preventing single-queue cross-queue Timeline Semaphore deadlocks and `vkd3d-proton` null sparse-queue dereferences (`UpdateTileMappings`).
  - Forwards `vkQueueBindSparse`, `vkGetImageSparseMemoryRequirements(2)`, and `vkGetDeviceImageSparseMemoryRequirements` natively to `qf[2]` with proper `SmBuffer`/`SmImage` handle unwrapping and per-queue mutex locking, and submits WSI present DMA on the caller's exact `VkQueue`.
- **64-Bit `VkAccessFlags2` & Vulkan 1.2/1.3 Layout Sanitization (`src/smrast_icd.cpp`)**:
  - Fixed 32-bit truncation in `sm_CmdPipelineBarrier2` that previously dropped upper-32-bit Sync2 flags (`VK_ACCESS_2_SHADER_SAMPLED_READ_BIT`, `SHADER_STORAGE_READ_BIT`, `SHADER_STORAGE_WRITE_BIT`), and added full Vulkan 1.2/1.3 layout mapping (`ATTACHMENT_OPTIMAL`, `READ_ONLY_OPTIMAL`, `DEPTH/STENCIL_*`, `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`).
- **`VK_KHR_dedicated_allocation` Unwrapping, `linear_aliased` Req Parity & `VkImageViewUsageCreateInfo` Sanitization (`src/smrast_icd.cpp`)**:
  - Unwraps `VkMemoryDedicatedAllocateInfo` (`SmImage*`/`SmBuffer*` → real backend handles) in `sm_AllocateMemory`, omits image-dedicated allocation on `linear_aliased` RTs shared between `real_img` and `attach_buf`, aligns `vkGetDeviceImageMemoryRequirements` & `vkGetImageMemoryRequirements2` byte-for-byte (with `VkMemoryDedicatedRequirements`), sanitizes `VkImageViewUsageCreateInfo` attachment bits, unwraps `basePipelineHandle` in `sm_CreateComputePipelines`, and preserves `RequiredSubgroupSize` with maintenance5 inline shaders.
- **Hardware `SmQueryPool` GPU Timestamps + Deterministic `vkCmdCopyQueryPoolResults` (`src/smrast_icd.cpp`)**:
  - Routes `VK_QUERY_TYPE_TIMESTAMP` (`vkCmdResetQueryPool`, `vkCmdWriteTimestamp/2`, `vkCmdCopyQueryPoolResults`, `vkGetQueryPoolResults`) directly to hardware compute queue `qf[2]` and emits deterministic non-zero/available readbacks for `OCCLUSION` and `PIPELINE_STATISTICS` queries.
- **Multi-Threaded DX12 Engine Hardening (`src/smrast_raster_engine.cpp`, `src/spirv_lower.cpp`, `src/smrast_icd.cpp`)**:
  - Protects `sm_ensure_vis_size` / `sm_ensure_image_attach_buf` / `sm_ensure_image_stencil_buf` with `dev->res_mtx` (atomic swap + pre-sized `2048×2048` VisBuf), wraps `spirv-cross`/`shaderc` lowering in `pthread_setcancelstate(PTHREAD_CANCEL_DISABLE)` RAII guards (eliminating `FATAL: exception not rethrown` during Wine thread teardown), adds 23 safe H20-backed DX12/compute extensions + promoted feature merging, and upgrades `SMRAST_CRASH_BT=1` to 32-frame `backtrace_symbols_fd()`.

<details>
<summary>Highlights from v0.9.0, v0.8.0, v0.7.0, v0.6.0, v0.5.0 & v0.4.0</summary>

- **`v0.9.0`**: Precomputed branchless `smrast_fine.comp` (Loader-Time Hi-Z Compaction + Warp-Uniform Y-Skip + Intra-Tile Early-Z + Inter-Batch Live Hi-Z Tightening), Exact 3-Edge Tile Trivial-Reject + `128 px` Branchless CuRast Fast-Path, Dynamic `vert_stride_vec4` (`2..16`), Zero-Bin Inline-Setup Fused Quad FS (7-plane clip + bary remap), Zero-DMA `smrast_indirect_prep.comp`, Empty-Clear `SyncA2I` Skip, and Runtime AVX2 Overlapped MIT-SHM (`1,299 FPS` 1M-tri PBR, `85.5–108.6 FPS` live 3D).
- **`v0.8.0`**: 7-Plane CUDARaster Guard-Band Clipping (`±4w`) + `VK_EXT_depth_clip_enable`, Painter's Order (Fine + Fused Quad), `flat`/`noperspective` varyings, Per-Draw Push-Constant & Dynamic-UBO SSBO Cross-Material Uber-Batching (`14.6×` speedup), Zero-Copy Linear-Aliased Color G-Buffers (`14` H20 formats, `rt_stride_px[8]`) + exact bound-set lazy sync, and Double-Buffered MIT-SHM Ping-Pong Ring.
- **`v0.7.0`**: Direct (`≤256`) & GPU-Driven Indirect (`≤128`) Multi-Draw Uber-Batching (`21.5×` speedup), 64-Way Warp-Cooperative Large-Triangle Binning, Fused Quad FS, Hardware DMA `vkCmdFillBuffer` Fast-Clear, Lazy Attachment Sync + `stn_can_write`, 1-Frame Pipelined MIT-SHM Present Ring, `gl_FragCoord.w = 1/w_clip`, `0x7FC0DEAD` NaN Primitive Restart, and `vkCmdWriteTimestamp` GPU X-Ray profiler (`SMRAST_XRAY=1`).
- **`v0.6.0`**: Hybrid Active-Tile Indirect Dispatch + `discard;`-safe Self-Clearing `VisBuf`/`tile_counts`, Zero-Update `STORAGE_BUFFER_DYNAMIC` DS Cache, Persistent Pre-Binning Coarse Hi-Z, Primitive Restart (`UINT8/16/32`), and Lines/Points rasterization.
- **`v0.5.0`**: Sampled Stencil Aspect SRVs (`R32_UINT` companion shadow `VkImage`), watertight Top-Left Fill-Rule + 4-bit snapping (`2304/2304` exact-once pixels), Depth Bias + `depthClampEnable` + `rasterizerDiscardEnable`, 2-Level Tile Hi-Z, and `VK_EXT_vertex_attribute_divisor`.
- **`v0.4.0`**: End-to-end DirectX 12 (`vkd3d-proton 3.1.0`, SM 6.6) & DirectX 11 (`DXVK 3.1.1`), `VK_KHR_push_descriptor`, 8-MRT × 21 RT formats, 8-bit Stencil Test/Op, separate RGB/Alpha blend, `gl_FrontFacing`, `vkCmdDrawIndirectCount`, and portable backend ICD auto-probe.

</details>

---

## Architecture (`v1.0.0`)

> Deep-dive architecture, GPU timestamp profiling breakdown, multi-engine DX11/DX12 bug post-mortems, engineering lessons, and the `v1.1+` roadmap are documented in **[docs/ARCHITECTURE_AND_NOTES.md](docs/ARCHITECTURE_AND_NOTES.md)**.

```
Unmodified Vulkan 1.3 App / DX12 Engine (vkd3d-proton 3.1.0) / DX11 Engine (DXVK 3.1.1)
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               SmRast-VK v1.0 Wrapper ICD (libvulkan_smrast.so)             │
│  • Exposes virtual Queue Family 0: GRAPHICS|COMPUTE|TRANSFER|SPARSE_BINDING│
│  • Maps up to 8 independent hardware queues on H20 pure Compute qf[2]      │
│  • Native vkQueueBindSparse + HW Timestamp QueryPool + DedicatedAlloc Unwrap│
│  • Cross-Material Direct (<=256) & Indirect (<=128) Multi-Draw Uber-Batch  │
│  • Zero-Copy Linear-Aliased Color G-Buffers (14 formats) + Empty-Clear Skip│
│  • Full 64-bit VkAccessFlags2 & VK1.3 layout sanitization + Sync2 barriers │
│  • Full DX12/VK1.4 emulation: push_descriptor, mutable_descriptor_type,    │
│    depth_clip_enable, maintenance5/6, index_type_uint8, vertex_divisor     │
│  • Sampled Stencil Aspect SRVs + Zero-Update DS Cache + 49,152 UBO slots   │
└───────────────┬──────────────────────────────────────────┬─────────────────┘
                │ Graphics Pipelines                       │ Compute / DMA
                ▼                                          ▼
  ┌──────────────────────────────────────────────┐   ┌───────────────────┐
  │ In-Process SPIR-V -> Compute Lowerer         │   │ Native H20 qf[2]  │
  │  • Thread-local libshaderc + atomic SPV cache│   │ 8x HW Queues,     │
  │  • OpSampledImage BB Sinker + spirv-opt -O   │   │ SparseBind, Sync2,│
  │  • Dynamic compact vert_stride_vec4 (2..16!) │   │ Timestamps & DMA  │
  │  • Bare gl_ClipDistance/CullDistance & Shadow│   └───────────────────┘
  │  • Zero-Bin Inline-Setup Fused Quad FS       │
  │  • Branchless subgroupQuadSwap dFdx/dFdy     │
  │  • Hardware H20 TMU textureGrad() + 8x MRT   │
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 8-Shader GPU Compute Rasterization Engine (H20 78 SMs / 96 GB HBM3)      │
  │  0. DMA vkCmdFillBuffer Fast-Clear (fallback: smrast_clear.comp)         │
  │     + Zero-DMA smrast_indirect_prep.comp (GPU atomic slice + header init)│
  │  1. K Barrier-Free Parallel Compute VS Dispatches -> Compact ClipVertBuf │
  │  2. K Parallel smrast_bin.comp (SKIPPED completely on Inline Fused Quad!)│
  │     (7-Plane Guard-Band Clip + Exact Tile-Edge Trivial-Reject + 128px    │
  │      Branchless CuRast + 64-Way Warp-Coop Big + Pre-Bin Hi-Z + Painter)  │
  │  3. 1x smrast_fine.comp (Loader-Time Hi-Z Compaction + Precomp Branchless│
  │     Edges + Warp-Uniform Y-Skip + Intra-Tile Early-Z + Live Hi-Z Tighten)│
  │  4. 1x Lowered Quad-Subgroup 8-MRT/Stencil FS OR smrast_depth_only.comp  │
  │  5. 1x smrast_hiz_update.comp (Active-Tile Incremental Coarse Hi-Z)      │
  │  6. smrast_blit.comp (Pure-Compute Scaled/Bilinear vkCmdBlitImage)       │
  └──────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
   Fused-Sem Async DMA Ring + HOST_CACHED + AVX2 Swizzle + Overlapped MIT-SHM
```

---

## Live Empirical Benchmarks & Verification on NVIDIA H20

Tested live on **NVIDIA H20 (78 SMs, 96 GB HBM3, Driver `535.161.08`)** running 100% on pure Compute Queue `qf[2]` (with zero `SMRAST_BACKEND_ICD` override):

### 1. Real-Time Commercial AAA Engine & Compute Rasterizer Throughput (`v0.6.0 → v0.7.0 → v0.8.0 → v0.9.0`)

Measured on a shared H20 pod (under 40–92% concurrent tenant load) against **48-Core CPU Mesa Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Workload / Benchmark | 48-Core CPU Lavapipe | `v0.6.0` | `v0.7.0` | `v0.8.0` | **`SmRast-VK v0.9.0` (H20 SMs)** | **Speedup vs CPU / `v0.6`** |
|---|---:|---:|---:|---:|---:|---:|
| **1M-Tri 1080p Cook-Torrance PBR** (`smrast_stress_bench`) | 9.7–10.4 FPS (`96–103 ms`) | 278–682 FPS | 885–934 FPS | 556–934 FPS | **`1,299.0 FPS`** (`0.77 ms`, **`1.293 Gtris/s`**) | **125×–134×** vs CPU *(+39% vs v0.7)* |
| **1,000-Draw 1080p Per-Draw Dynamic-UBO** (`test_v08_features`) | — | — | — | 2.54 ms (`393.6 FPS`) | **`2.35 ms`** (**`426.1 FPS`**) | **+8.3%** vs v0.8 |
| **1,000-Draw 1080p Per-Draw Push-Constant** (`test_v06_features`) | — | 56.72 ms (`17.6 FPS`) | 42.53 ms (`23.5 FPS`) | 2.92 ms (`342.3 FPS`) | **`2.75 ms`** (`GPU 1.25 ms`, **`364.1 FPS`**) | **20.6×** vs v0.6 *(15.5× vs v0.7)* |
| **1,000-Draw 1080p Direct Uber-Batch** (`test_v07_features`) | — | 61.07 ms (`16.4 FPS`) | 2.84 ms (`352.7 FPS`) | 2.50 ms (`399.9 FPS`) | **`2.61 ms`** (`GPU 1.02 ms`, **`382.5 FPS`**) | **23.4×** vs v0.6 |
| **512-Layer Reverse-Z Overdraw Stress** (`test_v09_features`) | — | — | — | — | **`1.014 ms`** (`1,024` tris/tile w/ Live Hi-Z) | — |
| **Commercial AAA DX11 Engine — DoF Intro** (`1280×720`) | 27.1 FPS (`36.9 ms`) | 8.90 FPS (`112.4 ms`) | 275–283 FPS | 226–254.5 FPS | **`306–324.7 FPS`** (`3.1–3.3 ms`) | **11.3×–12.0×** vs CPU *(36.5× vs v0.6)* |
| **Commercial AAA DX11 Engine — 3D 1M-Tri Deferred Scene** (`1280×720`, 5-MRT + PBR + Volumetric Fog + DoF + 43 AppCS) | 0.12 FPS (`8,400 ms`) | 3.71 FPS (`269.5 ms`) | 58.8–70.1 FPS | 76.0–81.3 FPS | **`85.5–108.6 FPS`** (`9.2–11.7 ms`, **`0` flicker spikes**) | **712×–905×** vs CPU *(29.3× vs v0.6)* |
| **Commercial AAA DX11 Engine — Interactive 3D Menu** (`87` draws, `36` indirect) | 0.12–0.13 FPS (`7,834 ms`) | — | 57.2 FPS (`17.5 ms`) | 67.6–71.1 FPS | **`76.0–80.0 FPS`** (`12.5–13.1 ms`, `diff_max=0.40`) | **585×–667×** vs CPU *(+40% vs v0.7)* |

### 2. DirectX 12 (`vkd3d-proton 3.1.0`) & DirectX 11 (`DXVK 3.1.1`) End-to-End Rendering
- **DirectX 12 3D Lit Textured Cube + 2x MRT (`RGBA8` + `RGBA16F`) + Bindless `Texture2DArray` + Push-Descriptor Root CBV** (`bench/dx12_smrast_test.c` → Proton 11 **`vkd3d-proton 3.1.0`** → `SmRast-VK v0.9.0` → H20 78 SMs):
  - **`DX12_VKD3D_TEST PASS`**: `lit_px=125114/262144`, `BG=(13,20,46)`, `CENTER=(137,106,36)` (`screenshots/dx12_vkd3d_h20.png`).
- **DirectX 11 3-RT Deferred HDR PBR + `DrawIndexedInstancedIndirect`** (`bench/dx11_3d_mrt_test.c` → Proton 11 **`DXVK 3.1.1`** → `SmRast-VK v0.9.0` → H20 78 SMs):
  - **`DEFERRED_MRT_PIXEL PASS`**: `BG=(8,13,31)`, `CENTER=(228,228,209)` — 3-RT G-Buffer (`RGBA8` + `RGBA16F` + `R11G11B10F` + `D32F`) + deferred ACES lighting resolve (`screenshots/dx11_deferred_mrt_h20.png`).
- **Real-World Commercial 3A Engine Compatibility (Metrics Only)**:
  - **Commercial AAA DX11 Open-World Engines (Proton 11 + DXVK 3.1.1)**: **230+ production graphics pipelines compiled with 0 failures (100% pass rate)** across multiple titles; **13,800+ continuous frames** reaching **`85–108.6 FPS`** (1M-tri 5-MRT deferred 3D scene, `0` temporal flicker spikes) and **`258–297 FPS`** (8,552+ frames in 40s on a second DX11 open-world title).
  - **Commercial AAA DX12 Engines (Proton 11 + vkd3d-proton 3.1.0, SM 6.6 / FL 12_2)**: **8,000+ production DX12 graphics pipelines compiled across 8 distinct modern engines with 0 shader lowering failures and 0 crashes (`861/861` `spirv-val` clean)**, achieving live pixel-exact presentation at **`60.0 FPS`** (VSync-capped next-gen deferred DX12 open-world RPG, 1,532+ frames), **`30.0 FPS`** (VSync-capped 2025 DX12 action RPG, 1,184+ frames with volumetric mist/particles/UI), **`60–100 FPS`** (multi-queue sparse-bound DX12 RPG, 1,040+ pipelines), and **`287 FPS`** (high-throughput DX12 action title, 1,570+ pipelines).

| DirectX 12 (`vkd3d-proton 3.1.0`) on H20 SMs | DirectX 11 Deferred HDR PBR (`DXVK 3.1.1`) | 1080p 1M-Tri PBR (`1,299 FPS` H20) | `vkcube` (`AVX2 MIT-SHM` WSI on H20) |
|---|---|---|---|
| ![DX12 vkd3d-proton H20](screenshots/dx12_vkd3d_h20.png) | ![DX11 Deferred MRT H20](screenshots/dx11_deferred_mrt_h20.png) | ![1M-Tri PBR H20](screenshots/pbr_1m_h20.png) | ![vkcube H20](screenshots/vkcube_h20.png) |

### 3. Feature & Regression Suites (12/12 Passing on H20 `qf[2]`, `0` Compiler Warnings)
- **`bench/test_v10_features`**: **`V10_FEATURE_TEST PASS`** — Bare `gl_ClipDistance[8]` I/O + Cross-Block `OpSampledImage` Sinker + Post-Sink `spirv-opt` (`L=(255,0,0) R=(0,255,0)`), 8-Queue Hardware Pool + Cross-Queue Timeline Semaphore Wait-Before-Signal + Native `vkQueueBindSparse` (`qcount=8, SparseBind=1`), 64-bit `VkAccessFlags2` + Vulkan 1.3 Layouts (`mid=(0,128,255)`), `VK_KHR_dedicated_allocation` Unwrapping + `linear_aliased` Req Parity (`dev_sz=img_sz=3574592, dedReq=0`) + `VkImageViewUsageCreateInfo` Sanitization, Hardware Timestamp + Emulated Occlusion `vkCmdCopyQueryPoolResults` (`dt=2400 ns, occ0=1`), and 4-Thread Concurrent PSO Compile & `>2K` VisBuf Resize Stress (`4/4 OK`).
- **`bench/test_v09_features`**: **`V09_FEATURE_TEST PASS`** — Dynamic `vert_stride_vec4` (`2/5/16` vec4 incl. `gl_PointSize`), Zero-Bin Inline Fused Quad FS (7-plane negative-`w` clip + Painter's Order), 512-Layer Reverse-Z Overdraw Fine + Live Hi-Z (**`0.536–1.014 ms`**), `128 px` CuRast Watertight 80×80 Grid (**`6400/6400` exact, `0` cracks, `0` double-hits**), and Empty-Clear `SyncA2I` Skip + Invalidation → Hardware TMU (`bg=191 mid=64`).
- **`bench/test_v08_features`**: **`V08_FEATURE_TEST PASS`** — 7-Plane Guard-Band Clipping & `VK_EXT_depth_clip_enable`, Painter's Order + `flat`/`noperspective`, Per-Draw Push-Constant & Dynamic-UBO Uber-Batching (`2.35–3.07 ms` / `326–426 FPS`), and Odd-Width (`1257×707`) Linear-Aliased 3-MRT → TMU Deferred Resolve.
- **`bench/test_v07_features`**: **`V07_FEATURE_TEST PASS`** — 2-Plane Reverse-Z & `w < -1.5` clipping (`1776`/`692` px), Direct/Indirect Uber-Batching (`2.61–2.90 ms` Direct), and Lazy Attachment Sync.
- **`bench/test_v06_features`**: **`V06_FEATURE_TEST PASS`** — Active-Tile Indirect Dispatch, `discard;`-safe Self-Clearing `VisBuf`, Pre-Binning Hi-Z, Primitive Restart, `LINE_LIST`/`POINT_LIST`, and 1,000-Draw Push-Constant (`2.39–2.85 ms`).
- **`bench/test_v05_features`**: **`V05_FEATURE_TEST PASS`** — Sampled Stencil Aspect SRV, Watertight Top-Left Fill-Rule (`2304/2304`), Depth Bias, Depth Clamp + Discard, Vertex Divisor, Reverse-Z Tile Hi-Z.
- **`bench/test_v04_features`**: **`V04_FEATURE_TEST PASS`** — 8 simultaneous MRTs across 8 formats (incl. 1B/2B atomics), Stencil Test/Op, `colorWriteMask`, separate RGB/Alpha blend, `gl_FrontFacing`, `DrawIndexedIndirectCount`, `UINT8` indices, `VK_KHR_push_descriptor`.
- **`bench/smrast_stress_bench` / `bench/test_vk13_game_features` / `bench/dx12_smrast_test.c` / `bench/dx11_3d_mrt_test.c` / `vkcube`**: **ALL PASS**.

---

## Build & Run

### Dependencies (Ubuntu 24.04 / Debian)
```bash
apt-get install -y g++ glslang-tools spirv-tools libspirv-cross-c-shared-dev libshaderc-dev libvulkan-dev libxcb1-dev libxcb-shm0-dev libx11-dev libxext-dev
```

### Build
```bash
bash build.sh
# or: make -j8
```

### Run Any Vulkan App, DX11 (DXVK), or DX12 (vkd3d-proton) Workload on Pure Compute Queue
```bash
export VK_ICD_FILENAMES="$PWD/smrast_icd.json"
export VK_DRIVER_FILES="$PWD/smrast_icd.json"
export SMRAST_SHADER_DIR="$PWD/build"
# Optional: explicitly point to your GPU's native Vulkan ICD if not in standard ld.so paths
# export SMRAST_BACKEND_ICD=/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
# export SMRAST_BACKEND_ICD=/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so
# Optional: enable Wine-safe 1-frame pass + vkCmdWriteTimestamp GPU X-Ray profiler (trigger: touch /tmp/smrast_xray.trigger)
# export SMRAST_XRAY=1

DISPLAY=:99 vkcube --width 800 --height 600 --c 60
./build/test_v10_features
./build/test_v09_features
./build/test_v08_features
./build/test_v07_features
./build/test_v06_features
./build/test_v05_features
./build/test_v04_features
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap (`v1.1+`)

1. **Advanced DX12 Stages & Descriptor Buffer**: `VK_EXT_descriptor_buffer` / BDA lowering, compute-emulated Geometry, Hull/Domain Tessellation & Mesh Shader stages, and 4× MSAA resolve.
2. **Vertex-Reuse Indexed Shading Cache (`VS` Deduplication)**: Shade unique index-buffer vertices once per meshlet/draw instead of per-index expansion, cutting `VS` work `3×–5×` on high-poly instanced meshes.
3. **Zero-Copy Streaming Output & Cross-Vendor CDNA3 Parity**: Direct HBM3-to-NVENC / DMA-BUF hardware video encoding bridge and Wave64 verification on AMD MI300X (`RADV`/`AMDVLK`).
