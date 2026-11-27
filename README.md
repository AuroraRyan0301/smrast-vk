# SmRast-VK — Pure-Compute Vulkan 1.3 Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** that runs real-time 3D Vulkan graphics pipelines **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]`), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or break their hardware graphics/ROP engines (`vkQueueSubmit` on `qf[0]` graphics hangs DMA with `Failed to idle DMA` on H20), forcing workloads to fall back to CPU Lavapipe (`llvmpipe`, 5–10 FPS at 1080p). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)** on its pure Compute Queue (`qf[2]`).

**SmRast-VK** synthesizes four classic GPU software rasterization architectures into a real, drop-in Vulkan 1.3 graphics driver:

| Classic Reference | Technique Integrated in SmRast-VK |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning for macro triangles (`shaders/smrast_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`), zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | `16×16` cooperative shared-memory batch fine rasterizer (`shaders/smrast_fine.comp`) + quad-swizzled subgroup compute fragment shading |
| **SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Automatic `SPIRV-Cross` reflection & GLSL 460 lowering of arbitrary app Vertex & Fragment SPIR-V shaders into H20 Compute Shaders preserving all UBOs, SSBOs, Push Constants, and Hardware `sampler2D`/`samplerCube` |

---

## Architecture

```
Unmodified Vulkan 1.3 App / Game / DXVK
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                SmRast-VK Wrapper ICD (libvulkan_smrast.so)              │
│  • Exposes virtual Queue Family 0: GRAPHICS | COMPUTE | TRANSFER        │
│  • Maps ALL backend work strictly to H20 pure Compute Queue (qf[2])     │
│  • Sanitizes graphics barriers/layouts -> COMPUTE_SHADER / GENERAL      │
│  • Native pass-through: vkCmdDispatch, HBM3 Memory, Buffers, TMUs, DMA  │
└───────────────┬─────────────────────────────────────────┬───────────────┘
                │ Graphics Pipelines                      │ Compute / DMA
                ▼                                         ▼
  ┌────────────────────────────────────────────┐    ┌──────────────────┐
  │ SPIR-V -> Compute Lowerer (SPIRV-Cross)    │    │ Native H20 qf[2] │
  │  • VS -> Compute (single-dispatch inst.)   │    │ Compute & Copy   │
  │  • FS -> Quad-Swizzled Compute (64 quads)  │    └──────────────────┘
  │    - 1-cycle subgroupQuadSwap dFdx/dFdy    │
  │    - Hardware H20 TMU textureGrad()        │
  │    - Perspective-correct varyings + blend  │
  └─────────────────────┬──────────────────────┘
                        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 4-Pass GPU Compute Rasterization Pipeline (H20 78 SMs / 96 GB HBM3) │
  │  Pass 1: Lowered Compute Vertex Shader -> ClipVertBuf (6M verts)    │
  │  Pass 2: smrast_bin.comp (Sutherland-Hodgman near-plane clip +      │
  │          CuRast <=64px 64-bit atomicMin + CUDARaster 16x16 binning) │
  │  Pass 3: smrast_fine.comp (16x16 shared-mem batch fine rasterizer)  │
  │  Pass 4: Lowered Quad-Subgroup Compute FS (0 overdraw + blend/Z)    │
  └──────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
     XCB / Xlib Headless WSI Bridge (smrast_wsi.cpp -> Xvfb)
     or Render-To-Texture Sync (attach_buf -> real VkImage for TMU read)
```

---

## Live Empirical Benchmarks on NVIDIA H20 (78 SMs, Driver 535.161.08)

### 1. 1920×1080 Cook-Torrance PBR + Hardware `sampler2D` + Alpha-Blended Glass + Near-Plane Clipping
Exact same unmodified Vulkan 1.3 binary (`bench/smrast_stress_bench`) run on **NVIDIA H20 (`SmRast-VK v0.2`)** vs **Mesa CPU Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Scene Scale (`1920×1080`) | CPU Lavapipe (`llvmpipe`) | **H20 `SmRast-VK v0.2` (78 SMs)** | Throughput (H20) | **Speedup** |
|---|---:|---:|---:|---:|
| **98,576 triangles** (~100K) | 54.6 FPS (`18.32 ms`) | **737.5 FPS** (`1.36 ms`) | 72.70 Mtris/s | **13.5×** |
| **498,086 triangles** (~500K) | 19.6 FPS (`51.13 ms`) | **638.9 FPS** (`1.57 ms`) | 318.25 Mtris/s | **32.6×** |
| **995,366 triangles** (~1M) | 10.2 FPS (`98.14 ms`) | **552.8 FPS** (`1.81 ms`) | 550.22 Mtris/s | **54.2×** |
| **1,994,218 triangles** (~2M) | 5.2 FPS (`192.85 ms`) | **361.6 FPS** (`2.77 ms`) | **721.11 Mtris/s** | **69.6×** |

- **Pixel Accuracy vs Mesa Lavapipe (`1920×1080`, 995,366 tris)**: **`MAE = 0.079 / 255` (`0.03%`)**, **`PSNR = 58.70 dB`** (indistinguishable from ground truth).

### 2. Unmodified Standard Vulkan Apps, Vulkan 1.3 Game Engine Features & DirectX 11 via DXVK 2.4.1
- **`/usr/bin/vkcube --width 800 --height 600`** (unmodified distro binary, XCB WSI → `Xvfb :99`): **73.8 FPS** warm, crisp perspective-correct LunarG texture, zero cracks, zero DMA hangs (native `nvidia_icd.json` hangs permanently).
- **`/usr/bin/vkcubepp`**: **59.6 FPS**, exit code 0.
- **`bench/test_vk13_game_features`**: **PASS** — `VK_KHR_dynamic_rendering` (`vkCmdBeginRendering`/`EndRendering`), `VK_EXT_extended_dynamic_state`, `VK_KHR_synchronization2` (`vkCmdPipelineBarrier2`/`vkQueueSubmit2`), single-dispatch 256-instance GPU instancing, and multi-pass Render-To-Texture sampled via hardware TMU in a `TRIANGLE_STRIP` post-pass.
- **Unmodified DirectX 11 (`HLSL vs_5_0 / ps_5_0`) → Unmodified DXVK 2.4.1 (`d3d11.dll` + `dxgi.dll`) → `SmRast-VK v0.2` → H20 78 SMs** (`bench/dx11_smrast_test.c`): **PASS (`EXIT=0`)**!
  - DXGI enumerates `NVIDIA H20 (SmRast-VK Compute Rasterizer) (Vendor=0x10de Device=0x2329 VRAM=97871 MiB)`.
  - DXVK 2.4.1 accepts `Maximum supported feature level: D3D_FEATURE_LEVEL_12_1`, creates `D3D_FEATURE_LEVEL_11_0` device, compiles HLSL `vs_5_0`/`ps_5_0` → DXBC → SPIR-V, `SmRast-VK` lowers DXVK's SPIR-V into H20 Compute Shaders (`Compiled Graphics Pipeline #0 -> H20 Compute VS+Bin+Fine+QuadFS OK!`), handles DX11 negative-height viewport (`VK_KHR_maintenance1`) & dynamic vertex strides (`vkCmdBindVertexBuffers2`), and rasterizes pixel-exact shaded output (`screenshots/dx11_dxvk_h20.png`).

---

## Build & Run

### Dependencies (Ubuntu 24.04 / Debian)
```bash
apt-get install -y g++ glslang-tools spirv-tools libspirv-cross-c-shared-dev libvulkan-dev libxcb1-dev libx11-dev
```

### Build
```bash
bash build.sh
# or: make -j8
```

### Run Any Vulkan Graphics App on H20 Pure Compute Queue
```bash
export VK_ICD_FILENAMES=$PWD/smrast_icd.json
export SMRAST_SHADER_DIR=./build
# Optional: override backend ICD path (defaults to libGLX_nvidia.so.0)
# export SMRAST_BACKEND_ICD=/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so

DISPLAY=:99 vkcube --width 800 --height 600
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap to Full 3A DX11/DX12 Games (DXVK / vkd3d-proton)

`SmRast-VK v0.2` already implements the core compute rasterization engine, automated SPIR-V VS/FS lowering, near-plane clipping, single-dispatch instancing, hardware TMU quad derivatives, dynamic rendering, extended dynamic state, and synchronization2. Key engineering milestones to reach full unmodified 3A DX11/DX12 game playability:

1. **MRT & HDR Render Target Formats** (`R16G16B16A16_SFLOAT`, `R11G11B10_UFLOAT`, `R32_SFLOAT`, up to 8 color attachments for deferred G-Buffers).
2. **Direct SPIR-V IR Lowering** (bypassing `SPIRV-Cross -> glslangValidator` subprocesses via `SPIRV-Tools` / in-process `glslang` C++ API for <1 ms PSO compilation and zero stutter on 10,000+ game shaders, plus `Bindless` / `VK_EXT_descriptor_indexing` / `VK_KHR_buffer_device_address`).
3. **Indirect Draw & Multi-Draw Indirect** (`vkCmdDrawIndirect` / `vkCmdDrawIndexedIndirect` / `Count` via GPU-generated `VkDispatchIndirectCommand` — critical for Unreal Engine 5 / Anvil / RE Engine GPU-driven culling).
4. **Geometry / Tessellation & Stencil / Depth-Bias / Conservative Raster** compute emulation, plus **Zero-Copy WSI** (`MIT-SHM` / `DMA-BUF` / headless NVENC capture instead of `xcb_put_image`).
5. **DXVK / vkd3d-proton Extension Surface** (`VK_EXT_transform_feedback`, `VK_EXT_robustness2`, `VK_KHR_push_descriptor`, `VK_EXT_graphics_pipeline_library` filtering & `VkBufferView` texel buffers).
