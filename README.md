# SmRast-VK v0.4 — Pure-Compute Vulkan 1.3 (+ Emulated 1.4) Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** with **emulated Vulkan 1.4 (`VK_KHR_maintenance5` / `VK_KHR_maintenance6`)** that runs real-time 3D Vulkan graphics, **DirectX 12 (`vkd3d-proton`, SM 6.6)**, and **DirectX 11 (`DXVK`)** workloads **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]`), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (`vkQueueSubmit` on graphics queues hangs DMA on H20), forcing workloads to fall back to CPU Lavapipe (`llvmpipe`, 5–10 FPS at 1080p). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)** on its pure Compute Queue (`qf[2]`).

**SmRast-VK** fuses four classic GPU software rasterization architectures with an automated in-process SPIR-V compiler into a drop-in Vulkan graphics driver capable of running modern DX11/DX12/Vulkan 3D engines:

| Classic Reference | Technique Integrated in SmRast-VK v0.4 |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning for macro triangles + Sutherland-Hodgman near-plane clipping (`shaders/smrast_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`), zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | `16×16` cooperative shared-memory batch fine rasterizer (`shaders/smrast_fine.comp`) + quad-swizzled subgroup compute fragment shading |
| **In-Process SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Thread-local `SPIRV-Cross` + `libshaderc` (zero fork/exec, atomic disk + memory cache) lowering arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders with `#extension` hoisting (`vkd3d-proton` bindless), 8 VBs, `UINT8/16/32` indices, 8-MRT × 21 RT formats (incl. race-free bit-disjoint 1B/2B atomics), 8-bit Stencil Test/Op + null-FS stencil, separate RGB/Alpha blend + `blendConstants` + `colorWriteMask`, exact `gl_FrontFacing`, branchless quad derivatives, and SSBO param blocks |

---

## What's New in `v0.4.0`

- **DirectX 12 (`vkd3d-proton 3.1.0`, SM 6.6) End-to-End Rendering**:
  - Full `VK_KHR_push_descriptor` (`vkCmdPushDescriptorSetKHR`/`2`, `WithTemplate`, captured & replayed across VS, FS, and Compute restore).
  - Bindless & DX12 feature set: `VK_EXT_mutable_descriptor_type`, `VK_EXT_descriptor_indexing`, `VK_EXT_dynamic_rendering_unused_attachments`, `VK_EXT_index_type_uint8`, `VK_EXT_border_color_swizzle`, `VK_EXT_conditional_rendering`, `VK_KHR_shader_float16_int8`, `VK_KHR_16bit_storage`, `VK_KHR_8bit_storage`, `VK_KHR_format_feature_flags2`, plus Vulkan 1.1–1.3 core device procs.
  - `#extension` hoisting in `src/spirv_lower.cpp` for `vkd3d-proton`'s `GL_EXT_nonuniform_qualifier`, `GL_EXT_scalar_block_layout`, `GL_EXT_buffer_reference`, and explicit arithmetic types.
  - **SSBO Parameter Blocks (`std430 readonly buffer`)** for `SmVsParams` & `SmFsParams`, completely eliminating NVIDIA 535 compute constant-bank aliasing at `DescriptorSet >= 5` under `vkd3d-proton`.
  - Deadlock-free WSI (`present_fence` waited outside `main_queue->mtx`) for `vkd3d-proton` multi-queue timeline-semaphore submissions.
- **Full 8-MRT × 21 Render-Target Formats (incl. Atomic 1B/2B Targets)**:
  - `SM_MAX_MRT = 8` across clear, rasterization, and lowered fragment shaders.
  - 21 formats: `RGBA8`/`BGRA8` (UNORM/SRGB), `RGBA8_SNORM`, `RGB10A2_UNORM`, `B10G11R11_UFLOAT` (`NaN`-free max finite `0x7BF`/`0x3DF`), `RGBA16F`/`RG16F`/`R16F`, `RGBA16`/`RG16` (UNORM/SNORM), `RGBA32F`/`RG32F`/`R32F`, `R32_UINT`/`SINT`, plus race-free bit-disjoint `atomicAnd`/`atomicOr` 1B (`R8_UNORM`/`UINT`) and 2B (`RG8_UNORM`/`SNORM`, `R16_SFLOAT`, `R16_UNORM`/`UINT`) targets.
- **8-bit Stencil Test/Op + Separate Alpha Blend + Per-Channel WriteMask + Exact `gl_FrontFacing`**:
  - Full front/back Stencil Test/Op (`compareOp`, `failOp`, `passOp`, `depthFailOp`, `compareMask`, `writeMask`, `reference`), dynamic stencil state, and dedicated null-FS stencil compute shader (`sm_build_null_fs_stencil_spv`).
  - Separate RGB + Alpha blending, `vkCmdSetBlendConstants`, `SRC_ALPHA_SATURATE`, spec-exact `MIN`/`MAX`, per-channel `colorWriteMask` (zero-RMW fast path when opaque & `0xF`), and exact `gl_FrontFacing` packed in bit 31 of `SmTriSetup.idx.w`.
- **Large Open-World Scene Scaling & GPU-Driven Pipeline**:
  - True GPU `vkCmdDrawIndirectCount` / `vkCmdDrawIndexedIndirectCount` (zero CPU readback, count-buffer gating in `shaders/smrast_indirect_prep.comp`).
  - **`49,152` in-flight draw slots** (+ `16,384` indirect + `4,096` clear/blit, batch-allocated `vkAllocateDescriptorSets`) supporting 15K+ draw/frame open-world scenes with zero slot wraparound.
  - Per-mip & per-layer (`attach_mip`/`attach_layer`) subresource sync for 10-level HDR bloom / Hi-Z pyramids and CSM shadow-cascade arrays.
  - Wine-safe file-triggered 1-frame pass tracer (`SMRAST_XRAY=1` + `touch /tmp/smrast_xray.trigger`).

---

## Architecture (`v0.4.0`)

```
Unmodified Vulkan 1.3 App / DX12 Engine (vkd3d-proton 3.1.0) / DX11 Engine (DXVK 3.1.1)
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               SmRast-VK v0.4 Wrapper ICD (libvulkan_smrast.so)             │
│  • Exposes virtual Queue Family 0: GRAPHICS | COMPUTE | TRANSFER           │
│  • Maps ALL work strictly to H20 pure Compute Queue (qf[2]) w/ mutex sync  │
│  • Full DX12/VK1.4 emulation: push_descriptor, mutable_descriptor_type,    │
│    maintenance5/6, index_type_uint8, dynamic_rendering_unused_attachments  │
│  • 49,152-slot zero-collision draw ring + per-mip/layer attach_buf sync    │
│  • Masks H20 BAR1/HVV HOST_VISIBLE off DEVICE_LOCAL (prevents Wine faults) │
│  • Native pass-through: vkCmdDispatch, HBM3, TMUs, Timeline Semaphores     │
└───────────────┬──────────────────────────────────────────┬─────────────────┘
                │ Graphics Pipelines                       │ Compute / DMA
                ▼                                          ▼
  ┌──────────────────────────────────────────────┐   ┌───────────────────┐
  │ In-Process SPIR-V -> Compute Lowerer         │   │ Native H20 qf[2]  │
  │  • Thread-local libshaderc + atomic SPV cache│   │ Compute, Events,  │
  │  • #extension hoisting for vkd3d-proton      │   │ Sync2 & Transfers │
  │  • VS -> Compute (8 VBs, UINT8/16/32, inst.) │   └───────────────────┘
  │  • FS -> Quad-Swizzled Compute (64 quads)    │
  │    - Branchless subgroupQuadSwap dFdx/dFdy   │
  │    - Hardware H20 TMU textureGrad()          │
  │    - 8x MRT + 21 RT formats (1B/2B atomics)  │
  │    - 8-bit Stencil + SepAlpha + WriteMask    │
  │    - SSBO param blocks (zero NV UBO aliasing)│
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 6-Shader GPU Compute Rasterization Engine (H20 78 SMs / 96 GB HBM3)      │
  │  0. smrast_clear.comp / smrast_indirect_prep.comp (GPU DrawIndirectCount)│
  │  1. Lowered Compute Vertex Shader -> Pre-alloc HBM3 ClipVertBuf          │
  │  2. smrast_bin.comp (Sutherland-Hodgman clip + CuRast + 16x16 binning)   │
  │  3. smrast_fine.comp (16x16 shared-mem batch fine rasterizer)            │
  │  4. Lowered Quad-Subgroup 8-MRT/Stencil FS OR smrast_depth_only.comp     │
  │  5. smrast_blit.comp (Pure-Compute Scaled/Bilinear vkCmdBlitImage)       │
  └──────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
   Deadlock-Free Zero-Copy MIT-SHM XCB / Xlib WSI Bridge (smrast_wsi.cpp)
```

---

## Live Empirical Benchmarks & Verification on NVIDIA H20

Tested live on **NVIDIA H20 (78 SMs, 96 GB HBM3, Driver `535.161.08`)** running 100% on pure Compute Queue `qf[2]`:

### 1. DirectX 12 (`vkd3d-proton 3.1.0`) & DirectX 11 (`DXVK 3.1.1`) End-to-End Rendering
- **DirectX 12 3D Lit Textured Cube + 2x MRT (`RGBA8` + `RGBA16F`) + Bindless `Texture2DArray` + Push-Descriptor Root CBV** (`bench/dx12_smrast_test.c` → Proton 11 **`vkd3d-proton 3.1.0`** → `SmRast-VK v0.4.0` → H20 78 SMs):
  - **`DX12_VKD3D_TEST PASS`**: `lit_px=125126/262144`, `BG=(13,20,46)`, `CENTER=(137,106,36)` — perspective-correct textured 3D cube with specular/diffuse shading and hardware TMU sampling (`screenshots/dx12_vkd3d_h20.png`).
- **DirectX 11 3-RT Deferred HDR PBR + `DrawIndexedInstancedIndirect`** (`bench/dx11_3d_mrt_test.c` → Proton 11 **`DXVK 3.1.1`** → `SmRast-VK v0.4.0` → H20 78 SMs):
  - **`DEFERRED_MRT_PIXEL PASS`**: 3-RT G-Buffer (`RGBA8` + `RGBA16F` + `R11G11B10F` + `D32F`) + deferred ACES lighting resolve (`screenshots/dx11_deferred_mrt_h20.png`).
- **Real-World Commercial 3A Engine Compatibility (Metrics Only)**:
  - **Commercial AAA DX11 Open-World Engine (Proton 11 + DXVK 3.1.1)**: **230+ production graphics pipelines compiled with 0 lowering/compile failures (100% pass rate)**, **1,440+ continuous frames presented** at `1280×707` (~18–20 FPS on shared H20), exercising 5-MRT deferred G-Buffers (`RGBA8_SRGB + RGBA8_UNORM + B10G11R11_UFLOAT + D32_SFLOAT_S8_UINT`), 9× `2048×2048` shadow cascades, `512×256 → 2×1` Hi-Z pyramids, 10-level HDR bloom, and `~14,616` draw calls / `~215` render passes per frame.
  - **Commercial AAA DX12 Open-World Engine (Proton 11 + vkd3d-proton 3.1.0, SM 6.6 / FL 12_2)**: **20+ production DX12 pipelines (`QuadFS` + `DepthOnly`) compiled with 0 failures (100% pass rate)** through bindless `VK_EXT_mutable_descriptor_type` and swapchain initialization.

| DirectX 12 (`vkd3d-proton 3.1.0`) on H20 SMs | DirectX 11 Deferred HDR PBR (`DXVK 3.1.1`) | 1080p 1M-Tri PBR (`SmRast-VK` H20) | `vkcube` (`MIT-SHM` WSI on H20) |
|---|---|---|---|
| ![DX12 vkd3d-proton H20](screenshots/dx12_vkd3d_h20.png) | ![DX11 Deferred MRT H20](screenshots/dx11_deferred_mrt_h20.png) | ![1M-Tri PBR H20](screenshots/pbr_1m_h20.png) | ![vkcube H20](screenshots/vkcube_h20.png) |

### 2. 1920×1080 Cook-Torrance PBR + Hardware `sampler2D` + Alpha-Blended Glass + Near-Plane Clipping
Exact same unmodified Vulkan 1.3 binary (`bench/smrast_stress_bench`) on **NVIDIA H20 (`SmRast-VK v0.4.0`)** vs **Mesa CPU Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Scene Scale (`1920×1080`) | CPU Lavapipe (`llvmpipe`) | **H20 `SmRast-VK` (78 SMs)** | Throughput (H20) | **Speedup** |
|---|---:|---:|---:|---:|
| **98,576 triangles** (~100K) | 54.6 FPS (`18.32 ms`) | **737.5 FPS** (`1.36 ms`) | 72.70 Mtris/s | **13.5×** |
| **498,086 triangles** (~500K) | 19.6 FPS (`51.13 ms`) | **638.9 FPS** (`1.57 ms`) | 318.25 Mtris/s | **32.6×** |
| **995,366 triangles** (~1M) | 10.2 FPS (`98.14 ms`) | **494.1–552.8 FPS** (`1.81–2.02 ms`) | **491.8–550.2 Mtris/s** | **48.4–54.2×** |
| **1,994,218 triangles** (~2M) | 5.2 FPS (`192.85 ms`) | **225.0–361.6 FPS** (`2.77–4.45 ms`) | **448.6–721.1 Mtris/s** | **43.3–69.6×** |

*(Lower bound measured on shared H20 node with `v0.4.0` 49K-slot build alongside concurrent workloads; upper bound measured dedicated on H20.)*
- **Pixel Accuracy vs Mesa Lavapipe (`1920×1080`, 995,366 tris)**: **`MAE = 0.079 / 255` (`0.03%`)**, **`PSNR = 58.70 dB`**.

### 3. Feature & Regression Suites (All Passing on H20 `qf[2]`)
- **`bench/test_v04_features`**: **`V04_FEATURE_TEST PASS`** — bit-exact verification across all 8 simultaneous MRTs (`RGBA8`, `RGBA8_SNORM`, 1B atomic `R8_UNORM`, 2B atomic `RG8_UNORM` / `R16_SFLOAT` / `R16_UNORM`, 8B `RGBA16_UNORM` / `RGBA16_SFLOAT`), front/back 8-bit Stencil Test/Op, `colorWriteMask`, separate RGB/Alpha blend + `vkCmdSetBlendConstants`, exact `gl_FrontFacing`, GPU `vkCmdDrawIndexedIndirectCount`, `VK_INDEX_TYPE_UINT8_EXT`, and `VK_KHR_push_descriptor`.
- **`bench/dx12_smrast_test.c` (DX12 `vkd3d-proton 3.1.0`)**: **`DX12_VKD3D_TEST PASS`**.
- **`bench/dx11_3d_mrt_test.c` (DX11 `DXVK 3.1.1`)**: **`DEFERRED_MRT_PIXEL PASS`**.
- **`bench/dx11_smrast_test.c` (DX11 `DXVK`)**: **PASS** (`screenshots/dx11_dxvk_h20.png`).
- **`bench/test_vk13_game_features`**: **`VK13_GAME_TEST PASS`** (`dynamic_rendering`, `extended_dynamic_state`, `synchronization2`, 256-instance GPU instancing, multi-pass RTT → `TRIANGLE_STRIP`).
- **`/usr/bin/vkcube --width 800 --height 600 --c 60`**: **PASS** (`60/60`, `shm=1`; native `nvidia_icd.json` hangs permanently on H20).

---

## Build & Run

### Dependencies (Ubuntu 24.04 / Debian)
```bash
apt-get install -y g++ glslang-tools spirv-tools libspirv-cross-c-shared-dev libshaderc-dev libvulkan-dev libxcb1-dev libxcb-shm0-dev libx11-dev
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
# Optional: enable Wine-safe 1-frame pass X-Ray tracer (trigger with: touch /tmp/smrast_xray.trigger)
# export SMRAST_XRAY=1

DISPLAY=:99 vkcube --width 800 --height 600 --c 60
./build/test_v04_features
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap (`v0.5+`)

1. **Sampled Stencil Aspect SRVs (`R8_UINT` Shadow Backing)**: Dedicated `R8_UINT` `VkImage` synced from `stencil_buf` for `VK_IMAGE_ASPECT_STENCIL_BIT` shader reads (`usampler2D`), enabling full stencil-masked deferred sky/lighting passes in modern engines.
2. **Full Commercial DX12 Engine Live Presentation**: Resolve `vkd3d-proton` pre-present staggered timeline synchronization and add `VK_EXT_descriptor_buffer` + compute-emulated Geometry/Tessellation stages.
3. **Hopper Performance Scaling**: Hi-Z early-reject culling inside `smrast_bin.comp`, persistent work-stealing tile scheduler, sub-pixel top-left fill-rule tie-breaking, and zero-copy DMA-BUF / NVENC frame encoding.
