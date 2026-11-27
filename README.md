# SmRast-VK v0.5 — Pure-Compute Vulkan 1.3 (+ Emulated 1.4) Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** with **emulated Vulkan 1.4 (`VK_KHR_maintenance5` / `VK_KHR_maintenance6`)** that runs real-time 3D Vulkan graphics, **DirectX 12 (`vkd3d-proton`, SM 6.6)**, and **DirectX 11 (`DXVK`)** workloads **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]`), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (`vkQueueSubmit` on graphics queues hangs DMA on H20), forcing workloads to fall back to CPU Lavapipe (`llvmpipe`, 5–10 FPS at 1080p). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)** on its pure Compute Queue (`qf[2]`).

**SmRast-VK** fuses four classic GPU software rasterization architectures with an automated in-process SPIR-V compiler into a drop-in Vulkan graphics driver capable of running modern DX11/DX12/Vulkan 3D engines:

| Classic Reference | Technique Integrated in SmRast-VK v0.5 |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning for macro triangles + Sutherland-Hodgman near-plane clipping + hardware-accurate Depth Bias & `depthClampEnable` (`shaders/smrast_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`), zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | `16×16` cooperative shared-memory batch fine rasterizer with **2-Level Tile Hi-Z Early-Reject** (subgroup + shared reduction, Standard-Z & Reverse-Z) + **watertight Top-Left Fill-Rule & 4-bit subpixel snapping** (`shaders/smrast_fine.comp`) + quad-swizzled subgroup compute fragment shading |
| **In-Process SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Thread-local `SPIRV-Cross` + `libshaderc` (zero fork/exec, atomic disk + memory cache) lowering arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders with `#extension` hoisting (`vkd3d-proton` bindless), 8 VBs + `VK_EXT_vertex_attribute_divisor` (`0` / `>1`), `UINT8/16/32` indices, 8-MRT × 21 RT formats (incl. race-free bit-disjoint 1B/2B atomics), 8-bit Stencil Test/Op + Sampled Stencil Aspect SRVs, separate RGB/Alpha blend + `blendConstants` + `colorWriteMask`, exact `gl_FrontFacing`, branchless quad derivatives, and SSBO param blocks |

---

## What's New in `v0.5.0`

- **Sampled Stencil Aspect SRVs (`VK_FORMAT_R32_UINT` Companion Shadow `VkImage`)**:
  - Automatically allocates a device-local `VK_FORMAT_R32_UINT` companion image (`real_stencil_img`) alongside `stencil_buf` for combined depth-stencil (`D24_UNORM_S8_UINT` / `D32_SFLOAT_S8_UINT` / `S8_UINT`) images.
  - Routes `VK_IMAGE_ASPECT_STENCIL_BIT`-only `VkImageView`s to `real_stencil_img` while preserving full `VkComponentMapping` swizzles (e.g., DX11/DX12 `{ZERO, R, ZERO, ZERO}` for `X24_TYPELESS_G8_UINT` / `X32_TYPELESS_G8X24_UINT` `usampler2D` deferred sky/lighting masks).
  - Full bidirectional `stencil_buf ↔ real_stencil_img` subresource sync across `vkCmdBeginRendering`/`vkCmdEndRendering`, `vkCmdBeginRenderPass`/`vkCmdEndRenderPass`, `vkCmdClearDepthStencilImage`, `vkCmdCopyImage`, and `vkCmdPipelineBarrier`/`2`.
- **Hardware-Exact Watertight Top-Left Fill-Rule + 4-bit (`1/16 px`) Subpixel Snapping**:
  - Snaps projected triangle vertices to `1/16 px` subpixel grid and evaluates shared edges with canonical endpoint ordering (`edge_eval(a,b) == -edge_eval(b,a)` bit-for-bit in FP32) plus Y-down Top-Left tie-breaking across **both** the CuRast micro-triangle fast-path (`smrast_bin.comp`) and the `16×16` fine rasterizer (`smrast_fine.comp`).
  - Eliminates shared-edge cracks and double-shading under additive/translucent blending (verified `2304/2304` exact-once pixels across 32 shared-edge triangles, `0` cracks, `0` double-hits).
- **Hardware-Accurate Depth Bias + `depthClampEnable` + `rasterizerDiscardEnable`**:
  - Spec-accurate polygon depth offset (`depthBiasConstantFactor * 2^-24 + depthBiasSlopeFactor * max(|dz/dx|, |dz/dy|)` with `depthBiasClamp`), supporting both static pipeline state and dynamic `vkCmdSetDepthBias` / `vkCmdSetDepthBiasEnable` for shadow cascades and decals.
  - Full `depthClampEnable` (shadow pancake mode: skips far-plane reject and clips only against `w > 1e-5`, clamping fragment depth into `[0, 1]`).
  - Static & dynamic `rasterizerDiscardEnable` (`vkCmdSetRasterizerDiscardEnable`).
- **2-Level Tile Hi-Z Early-Reject (`GL_KHR_shader_subgroup_arithmetic`)**:
  - Single-pass subgroup + shared-memory `tile_zmin` / `tile_zmax` reduction across every `16×16` tile in `smrast_fine.comp`, rejecting occluded macro triangles in 1 uniform instruction for both Standard-Z (`LESS`/`LEQUAL`) and Reverse-Z (`GREATER`/`GEQUAL`).
- **Exact `VK_EXT_vertex_attribute_divisor` + Extended Vertex Formats + Lifecycle Hardening**:
  - Full per-binding instance divisors (`divisor == 0` constant instance, `divisor > 1` stepped instance `firstInstance + inst / divisor`) lowered directly in `src/spirv_lower.cpp`.
  - Added signed/normalized vertex unpackers: `R8_SNORM`, `R8G8_SNORM`, `R8_SINT`, `R8G8_SINT`, `R16_SINT`.
  - Hardened command-buffer reset (`vtx_bufs` / `idx_buf` cleared on `vkBeginCommandBuffer`) and `used_vb_mask` binding guard so destroyed streaming vertex buffers can never dangle across reused command buffers.

<details>
<summary>Highlights from v0.4.0</summary>

- **DirectX 12 (`vkd3d-proton 3.1.0`, SM 6.6) End-to-End Rendering**: Full `VK_KHR_push_descriptor`, `VK_EXT_mutable_descriptor_type`, `VK_EXT_descriptor_indexing`, `VK_EXT_index_type_uint8`, `#extension` hoisting, SSBO param blocks (zero NV535 constant-bank aliasing at `set >= 5`), and deadlock-free WSI for timeline-semaphore queues.
- **Full 8-MRT × 21 Render-Target Formats**: Including race-free bit-disjoint `atomicAnd`/`atomicOr` 1B/2B targets (`R8`, `RG8`, `R16F`, `R16`) and `NaN`-free `B10G11R11_UFLOAT` (`0x7BF`/`0x3DF`).
- **8-bit Stencil Test/Op + Separate Alpha Blend + Per-Channel WriteMask + Exact `gl_FrontFacing`**: Plus dedicated null-FS stencil compute shader.
- **Open-World Scaling**: `49,152` in-flight draw slots (+ `16,384` indirect + `4,096` clear/blit), GPU `vkCmdDrawIndirectCount`, per-mip/layer subresource sync, and portable backend ICD auto-probe.

</details>

---

## Architecture (`v0.5.0`)

```
Unmodified Vulkan 1.3 App / DX12 Engine (vkd3d-proton 3.1.0) / DX11 Engine (DXVK 3.1.1)
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               SmRast-VK v0.5 Wrapper ICD (libvulkan_smrast.so)             │
│  • Exposes virtual Queue Family 0: GRAPHICS | COMPUTE | TRANSFER           │
│  • Maps ALL work strictly to H20 pure Compute Queue (qf[2]) w/ mutex sync  │
│  • Full DX12/VK1.4 emulation: push_descriptor, mutable_descriptor_type,    │
│    maintenance5/6, index_type_uint8, vertex_attribute_divisor, depthBias   │
│  • Sampled Stencil Aspect SRVs (R32_UINT companion VkImage w/ swizzle)     │
│  • 49,152-slot zero-collision draw ring + per-mip/layer attach_buf sync    │
│  • Native pass-through: vkCmdDispatch, HBM3, TMUs, Timeline Semaphores     │
└───────────────┬──────────────────────────────────────────┬─────────────────┘
                │ Graphics Pipelines                       │ Compute / DMA
                ▼                                          ▼
  ┌──────────────────────────────────────────────┐   ┌───────────────────┐
  │ In-Process SPIR-V -> Compute Lowerer         │   │ Native H20 qf[2]  │
  │  • Thread-local libshaderc + atomic SPV cache│   │ Compute, Events,  │
  │  • #extension hoisting for vkd3d-proton      │   │ Sync2 & Transfers │
  │  • VS -> Compute (8 VBs, divisor 0/N, u8/16) │   └───────────────────┘
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
  │  2. smrast_bin.comp (Clip + DepthBias/Clamp + 1/16-px Top-Left + CuRast) │
  │  3. smrast_fine.comp (16x16 Shared-Mem + 2-Level Tile Hi-Z + Top-Left)   │
  │  4. Lowered Quad-Subgroup 8-MRT/Stencil FS OR smrast_depth_only.comp     │
  │  5. smrast_blit.comp (Pure-Compute Scaled/Bilinear vkCmdBlitImage)       │
  └──────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
   Deadlock-Free Zero-Copy MIT-SHM XCB / Xlib WSI Bridge (smrast_wsi.cpp)
```

---

## Live Empirical Benchmarks & Verification on NVIDIA H20

Tested live on **NVIDIA H20 (78 SMs, 96 GB HBM3, Driver `535.161.08`)** running 100% on pure Compute Queue `qf[2]` (with zero `SMRAST_BACKEND_ICD` override):

### 1. DirectX 12 (`vkd3d-proton 3.1.0`) & DirectX 11 (`DXVK 3.1.1`) End-to-End Rendering
- **DirectX 12 3D Lit Textured Cube + 2x MRT (`RGBA8` + `RGBA16F`) + Bindless `Texture2DArray` + Push-Descriptor Root CBV** (`bench/dx12_smrast_test.c` → Proton 11 **`vkd3d-proton 3.1.0`** → `SmRast-VK v0.5.0` → H20 78 SMs):
  - **`DX12_VKD3D_TEST PASS`**: `lit_px=125114/262144`, `BG=(13,20,46)`, `CENTER=(137,106,36)` — perspective-correct textured 3D cube with specular/diffuse shading and hardware TMU sampling (`screenshots/dx12_vkd3d_h20.png`).
- **DirectX 11 3-RT Deferred HDR PBR + `DrawIndexedInstancedIndirect`** (`bench/dx11_3d_mrt_test.c` → Proton 11 **`DXVK 3.1.1`** → `SmRast-VK v0.5.0` → H20 78 SMs):
  - **`DEFERRED_MRT_PIXEL PASS`**: 3-RT G-Buffer (`RGBA8` + `RGBA16F` + `R11G11B10F` + `D32F`) + deferred ACES lighting resolve (`screenshots/dx11_deferred_mrt_h20.png`).
- **Real-World Commercial 3A Engine Compatibility (Metrics Only)**:
  - **Commercial AAA DX11 Open-World Engine (Proton 11 + DXVK 3.1.1)**: **230+ production graphics pipelines compiled with 0 lowering/compile failures (100% pass rate)**, **1,440+ continuous frames presented** at `1280×707`, exercising 5-MRT deferred G-Buffers (`RGBA8_SRGB + RGBA8_UNORM + B10G11R11_UFLOAT + D32_SFLOAT_S8_UINT`), sampled stencil aspect SRVs, 9× `2048×2048` shadow cascades with depth bias & depth clamp, `512×256 → 2×1` Hi-Z pyramids, 10-level HDR bloom, and `~14,616` draw calls / `~215` render passes per frame.
  - **Commercial AAA DX12 Open-World Engine (Proton 11 + vkd3d-proton 3.1.0, SM 6.6 / FL 12_2)**: **20+ production DX12 pipelines (`QuadFS` + `DepthOnly`) compiled with 0 failures (100% pass rate)** through bindless `VK_EXT_mutable_descriptor_type` and swapchain initialization.

| DirectX 12 (`vkd3d-proton 3.1.0`) on H20 SMs | DirectX 11 Deferred HDR PBR (`DXVK 3.1.1`) | 1080p 1M-Tri PBR (`SmRast-VK` H20) | `vkcube` (`MIT-SHM` WSI on H20) |
|---|---|---|---|
| ![DX12 vkd3d-proton H20](screenshots/dx12_vkd3d_h20.png) | ![DX11 Deferred MRT H20](screenshots/dx11_deferred_mrt_h20.png) | ![1M-Tri PBR H20](screenshots/pbr_1m_h20.png) | ![vkcube H20](screenshots/vkcube_h20.png) |

### 2. 1920×1080 Cook-Torrance PBR + Hardware `sampler2D` + Alpha-Blended Glass + Near-Plane Clipping
Exact same unmodified Vulkan 1.3 binary (`bench/smrast_stress_bench`) on **NVIDIA H20 (`SmRast-VK v0.5.0`)** vs **Mesa CPU Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Scene Scale (`1920×1080`) | CPU Lavapipe (`llvmpipe`) | **H20 `SmRast-VK v0.5` (78 SMs)** | Throughput (H20) | **Speedup** |
|---|---:|---:|---:|---:|
| **98,576 triangles** (~100K) | 54.6 FPS (`18.32 ms`) | **737.5 FPS** (`1.36 ms`) | 72.70 Mtris/s | **13.5×** |
| **498,086 triangles** (~500K) | 19.6 FPS (`51.13 ms`) | **638.9 FPS** (`1.57 ms`) | 318.25 Mtris/s | **32.6×** |
| **995,366 triangles** (~1M) | 9.7–10.2 FPS (`98.1–103.4 ms`) | **477.2–681.6 FPS** (`1.47–2.10 ms`) | **475.0–678.5 Mtris/s** | **48.4–70.5×** |
| **1,994,218 triangles** (~2M) | 5.2 FPS (`192.85 ms`) | **226.9–361.6 FPS** (`2.77–4.41 ms`) | **452.4–721.1 Mtris/s** | **43.6–69.6×** |

*(Lower bound measured on shared H20 pod alongside concurrent workloads; upper bound measured with low node contention. Tile Hi-Z + Top-Left + 4-bit snapping sustains up to `681.6 FPS` / `1.47 ms` at 1M triangles.)*
- **Pixel Accuracy vs Mesa Lavapipe (`1920×1080`, 995,366 tris, `v0.5.0`)**: **`MSE = 1.443`**, **`PSNR = 46.54 dB`** (with 4-bit subpixel vertex snapping & watertight Top-Left edge ownership across 2.07M pixels).

### 3. Feature & Regression Suites (All Passing on H20 `qf[2]`)
- **`bench/test_v05_features`**: **`V05_FEATURE_TEST PASS`** — verifies:
  1. Sampled Stencil Aspect SRV (`D32_SFLOAT_S8_UINT` → `usampler2D` with `{ZERO, R, ZERO, ZERO}` swizzle: exact `(0, 173, 255)` / `(0, 37, 0)`),
  2. Watertight Top-Left Fill-Rule across 32 shared-edge triangles under additive blending (`2304/2304` exact-once interior pixels, `0` cracks, `0` double-hits),
  3. Hardware-accurate Depth Bias (`vkCmdSetDepthBias` + `vkCmdSetDepthBiasEnable`),
  4. `depthClampEnable` (far-plane shadow pancake survival at `z = 1.5`) + dynamic `vkCmdSetRasterizerDiscardEnable`,
  5. `VK_EXT_vertex_attribute_divisor` (`divisor=2` and `divisor=0`) + `R8G8_SNORM` & `R16_SINT` vertex formats,
  6. 2-Level Tile Hi-Z Early-Reject under Reverse-Z (`VK_COMPARE_OP_GREATER`).
- **`bench/test_v04_features`**: **`V04_FEATURE_TEST PASS`** — 8 simultaneous MRTs across 8 formats (incl. 1B/2B atomics), Stencil Test/Op, `colorWriteMask`, separate RGB/Alpha blend + `blendConstants`, exact `gl_FrontFacing`, GPU `vkCmdDrawIndexedIndirectCount`, `VK_INDEX_TYPE_UINT8_EXT`, and `VK_KHR_push_descriptor`.
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
./build/test_v05_features
./build/test_v04_features
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap (`v0.6+`)

1. **Coarse-Level Hi-Z Pyramid & Persistent Work-Stealing Tile Scheduler**: Propagate `16×16` tile Hi-Z bounds back into `smrast_bin.comp` for pre-binning occlusion culling and add a persistent warp-level tile queue for high-overdraw 15K-draw open-world frames.
2. **Full Commercial DX12 Engine Live Presentation**: Resolve `vkd3d-proton` pre-present staggered timeline synchronization and add `VK_EXT_descriptor_buffer` + compute-emulated Geometry/Tessellation stages.
3. **Zero-Copy Streaming Output**: Direct HBM3-to-NVENC / DMA-BUF hardware video encoding bridge for remote cloud rendering without X11 readback overhead.
