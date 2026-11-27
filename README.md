# SmRast-VK v0.3 — Pure-Compute Vulkan 1.3 (+ Emulated 1.4) Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** with **emulated Vulkan 1.4 (`VK_KHR_maintenance5` / `VK_KHR_maintenance6`)** that runs real-time 3D Vulkan graphics & commercial DirectX 11 3A games (via **Proton 11 / DXVK 3.1.1**) **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]`), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or break their hardware graphics/ROP engines (`vkQueueSubmit` on `qf[0]` graphics hangs DMA with `Failed to idle DMA` on H20), forcing workloads to fall back to CPU Lavapipe (`llvmpipe`, 5–10 FPS at 1080p). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)** on its pure Compute Queue (`qf[2]`).

**SmRast-VK** fuses four classic GPU software rasterization architectures with an automated in-process SPIR-V compiler into a real, drop-in Vulkan graphics driver capable of running commercial 3A game engines:

| Classic Reference | Technique Integrated in SmRast-VK v0.3 |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning for macro triangles (`shaders/smrast_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`), zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | `16×16` cooperative shared-memory batch fine rasterizer (`shaders/smrast_fine.comp`) + quad-swizzled subgroup compute fragment shading |
| **In-Process SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Thread-local `SPIRV-Cross` + `libshaderc` (zero fork/exec, atomic disk + memory cache) lowering arbitrary VS/FS SPIR-V into Hopper Compute Shaders with 8 VBs, flat int/uint varyings, `demote`/`helperInvocationEXT`, branchless quad derivatives, `sampler_heap[]`, compile-time RT format/blend specialization, and automatic `-O0` fallback |

---

## Architecture (`v0.3.0`)

```
Unmodified Vulkan 1.3 App / Commercial 3A DX11 Game (Proton 11 + DXVK 3.1.1)
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               SmRast-VK v0.3 Wrapper ICD (libvulkan_smrast.so)             │
│  • Exposes virtual Queue Family 0: GRAPHICS | COMPUTE | TRANSFER           │
│  • Maps ALL work strictly to H20 pure Compute Queue (qf[2]) w/ mutex sync  │
│  • Emulates VK_KHR_maintenance5 (inline shaders, BufferUsageFlags2) &      │
│    VK_KHR_maintenance6 (BindDescriptorSets2/PushConstants2/IndexBuffer2)   │
│  • Masks H20 BAR1/HVV HOST_VISIBLE off DEVICE_LOCAL (prevents Wine faults) │
│  • Preserves app COMPUTE state (Set 0 sampler_heap[]) across raster passes │
│  • Native pass-through: vkCmdDispatch, HBM3, TMUs, Timeline Semaphores     │
└───────────────┬──────────────────────────────────────────┬─────────────────┘
                │ Graphics Pipelines                       │ Compute / DMA
                ▼                                          ▼
  ┌──────────────────────────────────────────────┐   ┌───────────────────┐
  │ In-Process SPIR-V -> Compute Lowerer         │   │ Native H20 qf[2]  │
  │  • Thread-local libshaderc + atomic SPV cache│   │ Compute, Events,  │
  │  • VS -> Compute (8 VBs, unaligned, inst.)   │   │ Sync2 & Transfers │
  │  • FS -> Quad-Swizzled Compute (64 quads)    │   └───────────────────┘
  │    - Branchless subgroupQuadSwap dFdx/dFdy   │
  │    - Hardware H20 TMU textureGrad()          │
  │    - 4x MRT + 12 HDR/Float/Uint RT formats   │
  │    - Per-RT independent blending + demote    │
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 6-Shader GPU Compute Rasterization Engine (H20 78 SMs / 96 GB HBM3)      │
  │  0. smrast_clear.comp / smrast_indirect_prep.comp (GPU DrawIndirect)     │
  │  1. Lowered Compute Vertex Shader -> Pre-alloc HBM3 ClipVertBuf          │
  │  2. smrast_bin.comp (Sutherland-Hodgman clip + CuRast + 16x16 binning)   │
  │  3. smrast_fine.comp (16x16 shared-mem batch fine rasterizer)            │
  │  4. Lowered Quad-Subgroup MRT FS OR smrast_depth_only.comp (Shadow/Z)    │
  │  5. smrast_blit.comp (Pure-Compute Scaled/Bilinear vkCmdBlitImage)       │
  └──────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
   Zero-Copy MIT-SHM XCB / Xlib WSI Bridge (smrast_wsi.cpp -> Xvfb)
   + Mip-Level / Array-Layer / View-Format RTT Sync (attach_buf <-> VkImage)
```

---

## Live Empirical Benchmarks & Commercial 3A Verification on NVIDIA H20

Tested live on **NVIDIA H20 (78 SMs, 96 GB HBM3, Driver `535.161.08`)** running 100% on pure Compute Queue `qf[2]`:

### 1. Real-World Commercial 3A DX11 Engine Compatibility (Proton 11 + DXVK 3.1.1)
- **Commercial AAA DX11 Open-World Engine (`D3D_FEATURE_LEVEL_11_0`)** running purely on H20 `qf[2]` SMs via `SmRast-VK v0.3` (zero hardware rasterizer, zero DLL overrides):
  - **60+ production graphics pipelines compiled** with **`Failed_VS = 0`, `Failed_FS = 0`, `CreateCompFail = 0` (100% shader lowering pass rate)**.
  - **1,260+ continuous real-time frames presented** at `1280×707` (~18–20 FPS on shared H20) across multi-pass `2048×2048` D32 shadow maps, `1232×693` sRGB scene + D32S8, `512×256` 9-level mip chain, 5-pass bloom, GPU-driven `DrawIndirect`, `R32_UINT`, and native `CmdDispatch` compute shaders.
  - *(Commercial game screenshots omitted from public repository.)*

### 2. 1920×1080 Cook-Torrance PBR + Hardware `sampler2D` + Alpha-Blended Glass + Near-Plane Clipping
Exact same unmodified Vulkan 1.3 binary (`bench/smrast_stress_bench`) on **NVIDIA H20 (`SmRast-VK`)** vs **Mesa CPU Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Scene Scale (`1920×1080`) | CPU Lavapipe (`llvmpipe`) | **H20 `SmRast-VK` (78 SMs)** | Throughput (H20) | **Speedup** |
|---|---:|---:|---:|---:|
| **98,576 triangles** (~100K) | 54.6 FPS (`18.32 ms`) | **737.5 FPS** (`1.36 ms`) | 72.70 Mtris/s | **13.5×** |
| **498,086 triangles** (~500K) | 19.6 FPS (`51.13 ms`) | **638.9 FPS** (`1.57 ms`) | 318.25 Mtris/s | **32.6×** |
| **995,366 triangles** (~1M) | 10.2 FPS (`98.14 ms`) | **495.8–552.8 FPS** (`1.81–2.02 ms`) | **493.5–550.2 Mtris/s** | **48.6–54.2×** |
| **1,994,218 triangles** (~2M) | 5.2 FPS (`192.85 ms`) | **255.2–361.6 FPS** (`2.77–3.92 ms`) | **509.0–721.1 Mtris/s** | **49.1–69.6×** |

*(Lower bound measured on shared H20 pod alongside active concurrent workloads; upper bound measured dedicated on H20.)*
- **Pixel Accuracy vs Mesa Lavapipe (`1920×1080`, 995,366 tris)**: **`MAE = 0.079 / 255` (`0.03%`)**, **`PSNR = 58.70 dB`**.

### 3. DirectX 11 Multi-Pass Deferred HDR PBR (DXVK 3.1.1), Vulkan 1.3 Features & Standard Vulkan Apps
- **DirectX 11 3-RT HDR Deferred PBR + `DrawIndexedInstancedIndirect`** (`bench/dx11_3d_mrt_test.c` → Proton 11 **DXVK 3.1.1** → `SmRast-VK v0.3` → H20 78 SMs): **PASS (`EXIT=0`)**, pixel-exact 3-RT G-Buffer (`RGBA8` + `RGBA16F` + `R11G11B10F` + `D32F`) + deferred ACES lighting resolve (`screenshots/dx11_deferred_mrt_h20.png`).
- **DirectX 11 Shaded Triangle** (`bench/dx11_smrast_test.c`): **PASS** (`screenshots/dx11_dxvk_h20.png`).
- **Vulkan 1.3 Game Engine Suite** (`bench/test_vk13_game_features`): **PASS** (`dynamic_rendering`, `extended_dynamic_state`, `synchronization2`, 256-instance GPU instancing, multi-pass RTT → `TRIANGLE_STRIP`).
- **`/usr/bin/vkcube --width 800 --height 600`**: **PASS** (120/120 frames, zero hangs; native `nvidia_icd.json` hangs permanently on H20).

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

### Run Any Vulkan App or Proton/DXVK Game on H20 Pure Compute Queue
```bash
export VK_ICD_FILENAMES=$PWD/smrast_icd.json
export VK_DRIVER_FILES=$PWD/smrast_icd.json
export SMRAST_SHADER_DIR=./build
# Optional: override backend ICD path (defaults to libGLX_nvidia.so.0)
# export SMRAST_BACKEND_ICD=/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so

DISPLAY=:99 vkcube --width 800 --height 600
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap (`v0.4+`)

1. **DirectX 12 (`vkd3d-proton`) Enablement**: `VK_KHR_push_descriptor`, `VK_EXT_descriptor_buffer` / `VK_KHR_buffer_device_address` bindless lowering, and 8-RT MRT (`SM_MAX_MRT = 8`) for UE5 / RE Engine / Anvil DX12 titles (modern DX12 open-world engines).
2. **Geometry / Tessellation & Stencil / Depth-Bias Compute Stages**: Compute-emulated GS/TS expansion and full 8-bit stencil buffer test/op for shadow volumes & decals.
3. **Async Multi-Tile & Persistent-Kernel Hopper Optimizations**: Sub-pixel top-left fill-rule tie-breaking, hierarchical Z (Hi-Z) culling in `smrast_bin.comp`, and zero-copy DMA-BUF / NVENC video stream capture.
