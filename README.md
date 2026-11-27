# SmRast-VK v0.6 — Pure-Compute Vulkan 1.3 (+ Emulated 1.4) Software Rasterizer ICD for H20 / H100 / MI300X

**SmRast-VK** is a drop-in **Khronos Vulkan 1.3 Wrapper ICD (`libvulkan_smrast.so`)** with **emulated Vulkan 1.4 (`VK_KHR_maintenance5` / `VK_KHR_maintenance6`)** that runs real-time 3D Vulkan graphics, **DirectX 12 (`vkd3d-proton`, SM 6.6)**, and **DirectX 11 (`DXVK`)** workloads **100% on compute-only datacenter GPU SMs** (`VK_QUEUE_COMPUTE_BIT`, e.g., NVIDIA H20/H100 `qf[2]`), touching **zero hardware rasterizer, zero ROPs, and zero RT Cores** — while passing native GPU Compute Shaders, 96 GB HBM3 memory, DMA transfers, and hardware Texture Mapping Units (TMUs) straight through to silicon.

---

## Why SmRast-VK?

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (`vkQueueSubmit` on graphics queues hangs DMA on H20), forcing workloads to fall back to CPU Lavapipe (`llvmpipe`, 5–10 FPS at 1080p). Yet an H20 still has **78 Hopper SMs**, **96 GB HBM3 (4 TB/s)**, **64-bit buffer atomics (`VK_KHR_shader_atomic_int64`)**, **1-cycle quad subgroup shuffles**, and **working hardware TMUs (`textureGrad`)** on its pure Compute Queue (`qf[2]`).

**SmRast-VK** fuses four classic GPU software rasterization architectures with an automated in-process SPIR-V compiler into a drop-in Vulkan graphics driver capable of running modern DX11/DX12/Vulkan 3D engines:

| Classic Reference | Technique Integrated in SmRast-VK v0.6 |
|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning with **Coarse Pre-Binning Hi-Z Culling**, GPU **Active-Tile Compaction**, Sutherland-Hodgman near-plane clipping, Depth Bias & `depthClampEnable`, Primitive Restart, and dedicated Lines/Points binning (`shaders/smrast_bin.comp`, `shaders/smrast_prim_bin.comp`) |
| **CuRast / compute_rasterizer** (Schütz / Nanite-style) | Lock-free 64-bit `atomicMin` (`depth << 32 \| tri_id`) direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`) with self-clearing `VisBuf`, zero binning traffic |
| **cuRE / RetroWarp** (Kenzel / Themaister) | Hybrid Active-Tile Indirect / Coherent-2D `16×16` shared-memory fine rasterizer (`shaders/smrast_fine.comp`) with self-clearing `tile_counts`, 1-read Persistent Tile Hi-Z + live subgroup fallback, watertight Top-Left Fill-Rule & 4-bit subpixel snapping, and quad-swizzled subgroup compute fragment shading |
| **In-Process SPIR-V → Compute Lowerer** (`src/spirv_lower.cpp`) | Thread-local `SPIRV-Cross` + `libshaderc` (zero fork/exec, atomic disk + memory cache) lowering arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders with `#extension` hoisting (`vkd3d-proton` bindless), 8 VBs + `VK_EXT_vertex_attribute_divisor` (`0` / `>1`), `UINT8/16/32` indices + Primitive Restart, 8-MRT × 21 RT formats (incl. race-free 1B/2B atomics), 8-bit Stencil Test/Op + Sampled Stencil Aspect SRVs, separate RGB/Alpha blend + `blendConstants` + `colorWriteMask`, exact `gl_FrontFacing`, branchless quad derivatives, and dynamic SSBO param blocks |

---

## What's New in `v0.6.0`

- **GPU-Driven Hybrid Active-Tile Indirect Dispatch + Self-Clearing `VisBuf` & `tile_counts`**:
  - **Eliminates per-draw `vkCmdFillBuffer(vis_buf)` (16 MB/draw at 1080p) and `vkCmdFillBuffer(tile_count_buf)` (64 KB/draw)** completely — `smrast_clear.comp` initializes once per RenderPass, `smrast_fine.comp` self-clears touched `tile_counts` & active flags, and lowered FS / `smrast_depth_only.comp` self-clear `vis_px[pidx]` immediately on read (100% immune to `discard;` / `demote;`).
  - For normal/small draws (`<64K` tris & indirect), `smrast_bin.comp` compacts touched tiles into `active_tiles[]` + `VkDispatchIndirectCommand` in-flight so Pass 3 (`Fine`), Pass 4 (`FS`/`DepthOnly`), and Pass 5 (`Hi-Z Update`) launch **only touched `16×16` tiles** via `vkCmdDispatchIndirect`.
  - For giant meshes (`≥64K` tris), automatically switches to **Coherent-2D Direct Grid mode** to preserve 100% L2/TMU 2D spatial locality with zero active-tile atomic contention.
- **Zero-Update Cached Descriptor Sets (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC`)**:
  - Converts internal `SmVsParams` & `SmFsParams` bindings to dynamic SSBOs (`slot * stride` passed via `vkCmdBindDescriptorSets`) and caches all 6 internal descriptor-set families (`VS`, `FS`, `Bin`, `Fine`, `DepthOnly`, `HiZUpdate`) by exact `VkBuffer` handle hash (auto-invalidated on buffer/image destroy with zero in-flight hazards).
  - Cuts per-draw `vkUpdateDescriptorSets` from **45 writes/draw → 0 writes/draw** on steady-state render passes (CPU recording for **1,000 draws at 1080p takes only `5.17–5.44 ms`**, >183 Kdraws/s single-threaded) and trims pure compute barriers (`COMPUTE → COMPUTE`) separately from `DRAW_INDIRECT` transitions.
- **Persistent Coarse Pre-Binning Hi-Z Culling (`shaders/smrast_hiz_update.comp`)**:
  - Each depth image maintains a 64 KB L2-resident `tile_hiz` (`(tile_zmin, tile_zmax)`) buffer initialized in `smrast_clear.comp` and incrementally refreshed ONLY on touched tiles after depth-writing draws (`hiz_has_occluders` skips useless Hi-Z reads on Draw #1).
  - `smrast_bin.comp` rejects occluded triangles **per-tile BEFORE binning into `tile_bins` and BEFORE CuRast rasterization** for both Standard-Z (`LESS`/`LEQUAL`) and Reverse-Z (`GREATER`/`GEQUAL`), while `smrast_fine.comp` reads persistent Tile Hi-Z in 1 scalar load (with automatic live subgroup fallback after `LOAD_OP_LOAD`).
- **Hardware-Exact Primitive Restart + Dedicated Lines & Points Rasterization (`shaders/smrast_prim_bin.comp`)**:
  - Full `primitiveRestartEnable` (static & dynamic `vkCmdSetPrimitiveRestartEnable`) for `UINT8` (`0xFF`), `UINT16` (`0xFFFF`), and `UINT32` (`0xFFFFFFFF`) indices — cleanly discards degenerate strip connectors before clipping.
  - Full `VK_PRIMITIVE_TOPOLOGY_LINE_LIST`, `LINE_STRIP` (honoring static & dynamic `vkCmdSetLineWidth`), and `POINT_LIST` (honoring VS `gl_PointSize`) screen-space quad expansion in a dedicated compute kernel so `smrast_bin.comp` preserves peak Hopper SM register occupancy.

<details>
<summary>Highlights from v0.5.0 & v0.4.0</summary>

- **`v0.5.0`**: Sampled Stencil Aspect SRVs (`R32_UINT` companion shadow `VkImage` preserving `VkComponentMapping` swizzle across all passes & barriers), watertight Top-Left Fill-Rule + 4-bit (`1/16 px`) subpixel snapping (`2304/2304` exact-once pixels, `0` cracks, `0` double-hits), hardware-accurate Depth Bias (`vkCmdSetDepthBias`/`Enable`) + `depthClampEnable` + `rasterizerDiscardEnable`, 2-Level Tile Hi-Z, `VK_EXT_vertex_attribute_divisor` (`0` / `>1`), and `R8/R8G8_SNORM` + `R8/R8G8/R16_SINT` vertex unpackers.
- **`v0.4.0`**: End-to-end DirectX 12 (`vkd3d-proton 3.1.0`, SM 6.6) & DirectX 11 (`DXVK 3.1.1`), full `VK_KHR_push_descriptor`, 8-MRT × 21 RT formats (incl. atomic 1B/2B targets & `NaN`-free `B10G11R11_UFLOAT`), 8-bit Stencil Test/Op + null-FS stencil, separate RGB/Alpha blend + `blendConstants` + `colorWriteMask`, exact `gl_FrontFacing`, GPU `vkCmdDrawIndirectCount`, `49,152` draw slots, per-mip/layer subresource sync, and portable backend ICD auto-probe.

</details>

---

## Architecture (`v0.6.0`)

> Deep-dive architecture, engineering lessons, driver/SPIR-V pitfalls, and the `v0.7+` Multi-Draw Uber-Binning blueprint are documented in **[docs/ARCHITECTURE_AND_NOTES.md](docs/ARCHITECTURE_AND_NOTES.md)**.

```
Unmodified Vulkan 1.3 App / DX12 Engine (vkd3d-proton 3.1.0) / DX11 Engine (DXVK 3.1.1)
        │  VK_ICD_FILENAMES=smrast_icd.json
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               SmRast-VK v0.6 Wrapper ICD (libvulkan_smrast.so)             │
│  • Exposes virtual Queue Family 0: GRAPHICS | COMPUTE | TRANSFER           │
│  • Maps ALL work strictly to H20 pure Compute Queue (qf[2]) w/ mutex sync  │
│  • Full DX12/VK1.4 emulation: push_descriptor, mutable_descriptor_type,    │
│    maintenance5/6, index_type_uint8, vertex_attribute_divisor, depthBias   │
│  • Sampled Stencil Aspect SRVs (R32_UINT companion VkImage w/ swizzle)     │
│  • Zero-Update DS Cache (STORAGE_BUFFER_DYNAMIC) + 49,152-slot UBO ring    │
│  • Native pass-through: vkCmdDispatch, HBM3, TMUs, Timeline Semaphores     │
└───────────────┬──────────────────────────────────────────┬─────────────────┘
                │ Graphics Pipelines                       │ Compute / DMA
                ▼                                          ▼
  ┌──────────────────────────────────────────────┐   ┌───────────────────┐
  │ In-Process SPIR-V -> Compute Lowerer         │   │ Native H20 qf[2]  │
  │  • Thread-local libshaderc + atomic SPV cache│   │ Compute, Events,  │
  │  • #extension hoisting for vkd3d-proton      │   │ Sync2 & Transfers │
  │  • VS -> Compute (8 VBs, div 0/N, PrimRst)   │   └───────────────────┘
  │  • FS -> Quad-Swizzled Compute (64 quads)    │
  │    - Self-clearing VisBuf (discard-safe!)    │
  │    - Branchless subgroupQuadSwap dFdx/dFdy   │
  │    - Hardware H20 TMU textureGrad()          │
  │    - 8x MRT + 21 RT formats (1B/2B atomics)  │
  │    - 8-bit Stencil + SepAlpha + WriteMask    │
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 8-Shader GPU Compute Rasterization Engine (H20 78 SMs / 96 GB HBM3)      │
  │  0. smrast_clear.comp (init RT+Depth+Vis+Stencil+HiZ+TileCounts)         │
  │     + smrast_indirect_prep.comp (GPU DrawIndirectCount)                  │
  │  1. Lowered Compute Vertex Shader -> Pre-alloc HBM3 ClipVertBuf          │
  │  2. smrast_bin.comp (Pure-Tri Pre-Bin Hi-Z + Active-Tile + CuRast + Clip)│
  │     OR smrast_prim_bin.comp (LINE_LIST / LINE_STRIP / POINT_LIST)        │
  │  3. smrast_fine.comp (Active-Tile/2D Fine + 1-Read Hi-Z + Self-Clr Tiles)│
  │  4. Lowered Quad-Subgroup 8-MRT/Stencil FS OR smrast_depth_only.comp     │
  │  5. smrast_hiz_update.comp (Active-Tile Incremental Coarse Hi-Z Refresh) │
  │  6. smrast_blit.comp (Pure-Compute Scaled/Bilinear vkCmdBlitImage)       │
  └──────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
   Deadlock-Free Zero-Copy MIT-SHM XCB / Xlib WSI Bridge (smrast_wsi.cpp)
```

---

## Live Empirical Benchmarks & Verification on NVIDIA H20

Tested live on **NVIDIA H20 (78 SMs, 96 GB HBM3, Driver `535.161.08`)** running 100% on pure Compute Queue `qf[2]` (with zero `SMRAST_BACKEND_ICD` override):

### 1. DirectX 12 (`vkd3d-proton 3.1.0`) & DirectX 11 (`DXVK 3.1.1`) End-to-End Rendering
- **DirectX 12 3D Lit Textured Cube + 2x MRT (`RGBA8` + `RGBA16F`) + Bindless `Texture2DArray` + Push-Descriptor Root CBV** (`bench/dx12_smrast_test.c` → Proton 11 **`vkd3d-proton 3.1.0`** → `SmRast-VK v0.6.0` → H20 78 SMs):
  - **`DX12_VKD3D_TEST PASS`**: `lit_px=125114/262144`, `BG=(13,20,46)`, `CENTER=(137,106,36)` — perspective-correct textured 3D cube with specular/diffuse shading and hardware TMU sampling (`screenshots/dx12_vkd3d_h20.png`).
- **DirectX 11 3-RT Deferred HDR PBR + `DrawIndexedInstancedIndirect`** (`bench/dx11_3d_mrt_test.c` → Proton 11 **`DXVK 3.1.1`** → `SmRast-VK v0.6.0` → H20 78 SMs):
  - **`DEFERRED_MRT_PIXEL PASS`**: 3-RT G-Buffer (`RGBA8` + `RGBA16F` + `R11G11B10F` + `D32F`) + deferred ACES lighting resolve (`screenshots/dx11_deferred_mrt_h20.png`).
- **Real-World Commercial 3A Engine Compatibility (Metrics Only)**:
  - **Commercial AAA DX11 Open-World Engine (Proton 11 + DXVK 3.1.1)**: **230+ production graphics pipelines compiled with 0 lowering/compile failures (100% pass rate)**, **1,440+ continuous frames presented** at `1280×707`, exercising 5-MRT deferred G-Buffers (`RGBA8_SRGB + RGBA8_UNORM + B10G11R11_UFLOAT + D32_SFLOAT_S8_UINT`), sampled stencil aspect SRVs, 9× `2048×2048` shadow cascades with depth bias & depth clamp, `512×256 → 2×1` Hi-Z pyramids, 10-level HDR bloom, and `~14,616` draw calls / `~215` render passes per frame.
  - **Commercial AAA DX12 Open-World Engine (Proton 11 + vkd3d-proton 3.1.0, SM 6.6 / FL 12_2)**: **20+ production DX12 pipelines (`QuadFS` + `DepthOnly`) compiled with 0 failures (100% pass rate)** through bindless `VK_EXT_mutable_descriptor_type` and swapchain initialization.

| DirectX 12 (`vkd3d-proton 3.1.0`) on H20 SMs | DirectX 11 Deferred HDR PBR (`DXVK 3.1.1`) | 1080p 1M-Tri PBR (`SmRast-VK` H20) | `vkcube` (`MIT-SHM` WSI on H20) |
|---|---|---|---|
| ![DX12 vkd3d-proton H20](screenshots/dx12_vkd3d_h20.png) | ![DX11 Deferred MRT H20](screenshots/dx11_deferred_mrt_h20.png) | ![1M-Tri PBR H20](screenshots/pbr_1m_h20.png) | ![vkcube H20](screenshots/vkcube_h20.png) |

### 2. Multi-Draw & High-Geometry Benchmarks (`1920×1080` on NVIDIA H20)
- **1,000-Draw 1080p Scattered-Prop Throughput (`bench/test_v06_features`)**:
  - **`56.72 ms/frame`** (**`CPU = 5.17–5.44 ms`**, **`GPU = 51.28 ms`**, **`17.6 FPS`**, **`17.6 Kdraws/s`**) on shared H20 pod (`33%` concurrent load) — zero `CmdFillBuffer(vis_buf)` (saving 16 GB/frame of HBM3 clear traffic) and zero per-draw `vkUpdateDescriptorSets`.
- **1920×1080 Cook-Torrance PBR + Hardware `sampler2D` + Alpha-Blended Glass + Near-Plane Clipping (`bench/smrast_stress_bench`)** vs **Mesa CPU Lavapipe (`llvmpipe`, LLVM 20.1.2 AVX2)**:

| Scene Scale (`1920×1080`) | CPU Lavapipe (`llvmpipe`) | **H20 `SmRast-VK` (78 SMs)** | Throughput (H20) | **Speedup** |
|---|---:|---:|---:|---:|
| **98,576 triangles** (~100K) | 54.6 FPS (`18.32 ms`) | **737.5 FPS** (`1.36 ms`) | 72.70 Mtris/s | **13.5×** |
| **498,086 triangles** (~500K) | 19.6 FPS (`51.13 ms`) | **638.9 FPS** (`1.57 ms`) | 318.25 Mtris/s | **32.6×** |
| **995,366 triangles** (~1M) | 9.7–10.4 FPS (`96.1–103.4 ms`) | **277.6–681.6 FPS** (`1.47–3.60 ms`) | **276.3–678.5 Mtris/s** | **26.7–70.5×** |
| **1,994,218 triangles** (~2M) | 5.2 FPS (`192.85 ms`) | **171.7–361.6 FPS** (`2.77–5.83 ms`) | **342.3–721.1 Mtris/s** | **33.0–69.6×** |

*(Lower bound measured under 33% concurrent GPU tenant load on shared H20 pod; upper bound measured dedicated on H20.)*
- **Pixel Accuracy vs Mesa Lavapipe (`1920×1080`, 995,366 tris, `v0.6.0`)**: **`MSE = 1.4430`**, **`PSNR = 46.54 dB`** (bit-identical to `v0.5.0` across all 2.07M pixels).

### 3. Feature & Regression Suites (All Passing on H20 `qf[2]`, `0` Compiler Warnings)
- **`bench/test_v06_features`**: **`V06_FEATURE_TEST PASS`** — Active-Tile Indirect Dispatch + `discard;`-safe Self-Clearing `VisBuf` + Depth-Only Pre-Binning Hi-Z Early-Reject, `UINT16 0xFFFF` Primitive Restart in `TRIANGLE_STRIP` (zero bridge triangles), `LINE_LIST` (`lineWidth=4`) & `POINT_LIST` (`gl_PointSize=6` exact `36 px`), and 1,000-Draw 1080p throughput benchmark.
- **`bench/test_v05_features`**: **`V05_FEATURE_TEST PASS`** — Sampled Stencil Aspect SRV with `{ZERO, R, ZERO, ZERO}` swizzle, Watertight Top-Left Fill-Rule (`2304/2304` exact-once pixels, `0` cracks, `0` double-hits), Depth Bias, `depthClampEnable` + `rasterizerDiscardEnable`, `VK_EXT_vertex_attribute_divisor` (`0` / `2`) + `R8G8_SNORM`/`R16_SINT`, and Reverse-Z Tile Hi-Z.
- **`bench/test_v04_features`**: **`V04_FEATURE_TEST PASS`** — 8 simultaneous MRTs across 8 formats (incl. 1B/2B atomics), Stencil Test/Op, `colorWriteMask`, separate RGB/Alpha blend + `blendConstants`, exact `gl_FrontFacing`, GPU `vkCmdDrawIndexedIndirectCount`, `VK_INDEX_TYPE_UINT8_EXT`, and `VK_KHR_push_descriptor`.
- **`bench/dx12_smrast_test.c` (DX12 `vkd3d-proton 3.1.0`)**: **`DX12_VKD3D_TEST PASS`**.
- **`bench/dx11_3d_mrt_test.c` (DX11 `DXVK 3.1.1`)**: **`DEFERRED_MRT_PIXEL PASS`**.
- **`bench/dx11_smrast_test.c` (DX11 `DXVK`)**: **PASS** (`screenshots/dx11_dxvk_h20.png`).
- **`bench/test_vk13_game_features`**: **`VK13_GAME_TEST PASS`**.
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
./build/test_v06_features
./build/test_v05_features
./build/test_v04_features
./build/smrast_stress_bench --tris 1000000 --warmup 5 --frames 30 --out /tmp/out.ppm
```

---

## Roadmap (`v0.7+`)

1. **Multi-Draw Batching / Uber-Binning**: Fuse consecutive same-pipeline small draws (e.g., 1,000+ scattered props) into a single GPU-driven mega-dispatch to eliminate inter-draw Hopper compute grid barriers (`51 ms → <5 ms` for 1,000 draws).
2. **Full Commercial DX12 Engine Live Presentation**: Resolve `vkd3d-proton` pre-present staggered timeline synchronization and add `VK_EXT_descriptor_buffer` + compute-emulated Geometry/Tessellation stages.
3. **Zero-Copy Streaming Output**: Direct HBM3-to-NVENC / DMA-BUF hardware video encoding bridge for remote cloud rendering without X11 readback overhead.
