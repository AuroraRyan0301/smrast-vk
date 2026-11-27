# SmRast-VK: Architecture, Engineering Lessons & Technical Roadmap

> **Living technical memory** for **SmRast-VK** — a pure-Compute **Vulkan 1.3 (+ emulated Vulkan 1.4)** Software Rasterizer Wrapper ICD (`libvulkan_smrast.so`) designed for compute-only datacenter GPUs (**NVIDIA H20 / H100**, **AMD MI300X**).
>
> *Last updated: `v0.6.0` (commit `b7b2f07`+).*

---

## 1. Core Thesis & Hardware Motivation

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (graphics queue submissions hang DMA on H20), forcing legacy graphics stacks onto CPU software rasterization (Mesa Lavapipe / `llvmpipe`, typically 5–10 FPS at 1080p while saturating dozens of CPU cores).

However, a compute-only H20 retains massive graphics-capable silicon on its pure Compute Queue (`VK_QUEUE_COMPUTE_BIT`, `qf[2]`):
- **78 Hopper SMs** (`FP32` / `INT32` / `FP16`)
- **96 GB HBM3** (~4 TB/s bandwidth)
- **Hardware 64-bit buffer atomics** (`VK_KHR_shader_atomic_int64`)
- **1-cycle Subgroup Quad Shuffles** (`VK_KHR_shader_subgroup_quad`)
- **Functional Hardware Texture Mapping Units (TMUs)** supporting `textureGrad()` anisotropic / mipmapped sampling inside Compute Shaders

**SmRast-VK's Hybrid Principle**:
1. **Pass through everything silicon already does natively**: Compute Shaders (`vkCmdDispatch`), HBM3 allocations, DMA transfers, hardware TMU sampling, Timeline Semaphores, and Sync2 events run natively on `qf[2]`.
2. **Replace ONLY the missing graphics rasterizer & ROP fixed-function pipeline**: Intercept `VkPipeline` (graphics) creation and `vkCmdDraw*`, lower arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders in-process, and execute a hybrid 4-stage Compute Rasterizer directly on the GPU SMs.

---

## 2. Architecture & Reference Lineage (`v0.6.0`)

SmRast-VK fuses four classic GPU software rasterization architectures with a production Vulkan 1.3/1.4 ICD and live SPIR-V lowering compiler:

| Classic Reference | Technique Adapted | SmRast-VK Implementation |
|---|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning, Sutherland-Hodgman near-plane clipping, watertight Top-Left fill rule | `shaders/smrast_bin.comp` + `v0.6` **Pre-Binning Coarse Hi-Z Culling** (`shaders/smrast_hiz_update.comp`) & **GPU Active-Tile Compaction** |
| **CuRast / compute_rasterizer** (Schütz / UE5 Nanite) | Lock-free 64-bit `atomicMin(depth << 32 \| tri_id)` direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`), completely bypassing binning & fine stages | `shaders/smrast_bin.comp` micro-triangle fast-path (key driver of `360–720 Mtris/s` throughput); upgraded to `discard;`-safe self-clearing `VisBuf` in `v0.6` |
| **cuRE & RetroWarp** (Kenzel / Themaister) | 100% pure Vulkan Compute graphics pipeline (zero ROPs), `16×16` shared-memory Tile Fine rasterizer, subgroup quad derivatives | `shaders/smrast_fine.comp` (2-level Tile Hi-Z + self-clearing `tile_counts`) + `src/spirv_lower.cpp` (`subgroupQuadSwapHorizontal/Vertical` branchless `dFdx`/`dFdy` + native TMU `textureGrad`) |
| **Original ICD & Compiler Layer** | Drop-in Khronos ICD + in-process SPIR-V → Compute compiler (missing from fixed-shader research demos) | `src/smrast_icd.cpp`, `src/spirv_lower.cpp`, `src/smrast_raster_engine.cpp`, `src/smrast_wsi.cpp` |

### 2.1 End-to-End 8-Shader Compute Pipeline

```
RenderPass Begin (LOAD_OP_CLEAR / LOAD_OP_LOAD)
  └─ [0] smrast_clear.comp
         Initializes Color MRTs (up to 8), Depth (D32F), Stencil (8-bit packed),
         VisBuf (64-bit sentinel 0xFFFFFFFFFFFFFFFFul), Tile Hi-Z (64 KB @ 1080p),
         Tile Counts & Tile Active Flags — ONCE per RenderPass.

Per Draw Call (vkCmdDraw / vkCmdDrawIndexed / vkCmdDraw[Indexed]Indirect[Count]):
  ├─ [0b] smrast_indirect_prep.comp  (indirect draws only)
  │       Translates GPU VkDraw[Indexed]IndirectCommand + drawCount buffer
  │       into VS / Binning VkDispatchIndirectCommand structs.
  ├─ [1] Lowered Compute Vertex Shader  (src/spirv_lower.cpp)
  │       • Up to 8 Vertex Buffers + VK_EXT_vertex_attribute_divisor (0 / >1)
  │       • Full vertex format unpackers (incl. R8/R8G8_SNORM, R8/R8G8/R16_SINT, 10_10_10_2, 16F)
  │       • UINT8 / UINT16 / UINT32 indices + Primitive Restart (0xFF/0xFFFF/0xFFFFFFFF -> w = -2.0)
  │       • Writes Clip-Space position + interpolated varyings into HBM3 ClipVertBuf (2.2M verts)
  ├─ [2] smrast_bin.comp  (pure-triangle hot path)  OR  smrast_prim_bin.comp  (LINE_LIST/STRIP, POINT_LIST)
  │       • Discards primitive-restart sentinels before clipping
  │       • Sutherland-Hodgman near-plane clipping (z >= 0 / z >= -w)
  │       • Hardware-accurate Depth Bias (constant + slope-scaled + clamp) & depthClampEnable
  │       • Pre-Binning Persistent Coarse Hi-Z early-reject per 16×16 tile (Standard-Z & Reverse-Z)
  │       • Micro-tri fast-path (bbox <= 64 px): direct 64-bit atomicMin into VisBuf
  │       • Large/medium tris: appends tri_id to 16×16 tile_bins[]
  │       • Hybrid dispatch mode:
  │           - Small/medium draws (<64K tris & indirect): compacts touched tiles into active_tiles[]
  │             and atomically increments VkDispatchIndirectCommand (offset 32).
  │           - Giant meshes (>=64K tris): Coherent-2D Direct Grid (flag bit 15) to preserve L2/TMU locality.
  ├─ [3] smrast_fine.comp  (vkCmdDispatchIndirect OR Coherent-2D direct grid)
  │       • 1 workgroup = 1 active 16×16 tile (256 threads)
  │       • 1-read Persistent Tile Hi-Z (with automatic live subgroup fallback after LOAD_OP_LOAD)
  │       • Watertight Top-Left Fill-Rule + 4-bit (1/16 px) subpixel snapping (0 cracks, 0 double-hits)
  │       • Writes winning (depth << 32 | tri_id) into VisBuf; self-clears tile_counts & active flags on exit.
  ├─ [4] Lowered Quad-Swizzled Compute FS  OR  smrast_depth_only.comp  (Indirect / 2D)
  │       • 256 threads organized as 64 swizzled 2×2 quads per 16×16 tile
  │       • Reads VisBuf[pidx] and IMMEDIATELY writes back 0xFFFFFFFFFFFFFFFFul (100% discard/demote-safe!)
  │       • Perspective-correct barycentric attribute interpolation + exact gl_FrontFacing
  │       • Branchless subgroupQuadSwap dFdx/dFdy -> native hardware TMU textureGrad()
  │       • 8-bit Stencil Test/Op (front/back, ref/compareMask/writeMask, null-FS stencil pass supported)
  │       • 8 simultaneous MRTs × 21 RT formats (incl. race-free CAS/atomic 1B/2B targets & NaN-free B10G11R11F)
  │       • Separate RGB/Alpha blend equations + blendConstants + per-RT colorWriteMask
  └─ [5] smrast_hiz_update.comp  (depth-writing draws only, Indirect / 2D)
          Incrementally updates persistent 16×16 Coarse Hi-Z (tile_zmin, tile_zmax) ONLY for touched tiles.

Blit / Present:
  ├─ [6] smrast_blit.comp — pure-compute scaled & bilinear vkCmdBlitImage
  └─ src/smrast_wsi.cpp — lock-free present_fence + zero-copy MIT-SHM XCB/Xlib WSI bridge
```

### 2.2 Key Internal Memory & Slot Capacities (`include/smrast_internal.h`)

| Resource | Capacity (`v0.6.0`) | Purpose |
|---|---:|---|
| Draw parameter UBO/SSBO ring | **49,152 slots** | Supports full 3A open-world frames (`~14.6K` draws / `~215` render passes without mid-frame wrap) |
| Indirect dispatch slots | **16,384 slots** | `vkCmdDraw[Indexed]Indirect[Count]` + Active-Tile `VkDispatchIndirectCommand` |
| Clear / Blit slots | **4,096 slots** | High-pass-count deferred/bloom/shadow cascades |
| `ClipVertBuf` (HBM3) | **2.2M vertices** | Post-VS clip-space positions + varyings |
| `tile_hiz` (per Depth Image) | **64 KB @ 1080p** | `120 × 68` tiles × 8 bytes (`(zmin, zmax)`), L2-resident |
| Descriptor-set cache pools | **16K sets / ring** | Zero-update `STORAGE_BUFFER_DYNAMIC` caching across all 6 internal DS layouts |

---

## 3. Why DX11 (`DXVK`) and DX12 (`vkd3d-proton`) Behave Differently on a Unified Vulkan ICD

SmRast-VK implements **one single Vulkan ICD** — there are no separate DX11 or DX12 rasterizers. The stark difference in bring-up difficulty comes from the **Vulkan API surface and SPIR-V complexity** emitted by the two Windows translation layers:

| Dimension | DX11 (`DXVK`) | DX12 (`vkd3d-proton`) | Engineering Impact on SmRast-VK |
|---|---|---|---|
| **Resource Binding** | Classic fixed-slot Descriptor Sets (bounded CBV/SRV/UAV tables) | Massive **Bindless arrays** (`nonuniformEXT`), `VK_EXT_mutable_descriptor_type`, `VK_KHR_push_descriptor`, and `VK_EXT_descriptor_buffer` (GPU pointers) | Required full emulation of mutable descriptors, descriptor indexing, push descriptors (incl. `WithTemplate`/`2KHR`), and dynamic SSBOs |
| **SPIR-V IR** | Translated from DXBC (SM 5.0); clean, regular GLSL-like IR | Translated from DXIL (SM 6.0–6.6); heavy `GL_EXT_buffer_reference` (raw 64-bit pointers), `scalar_block_layout`, 16-bit types, Wave/Subgroup ops, Root Constants | Required `#extension` hoisting above injected preambles in `spirv_lower.cpp` and careful SSBO lowering |
| **Queues & Sync** | Single immediate context; simple binary semaphores & fences | Multi-queue (Direct / Async Compute / Copy) + **out-of-order cross-thread Timeline Semaphore Wait/Signal** (`vkQueueSubmit2`) | Virtualizing multiple queues onto one physical `qf[2]` requires careful deadlock-free Timeline/Present scheduling |
| **Pipeline State** | Predominantly traditional `VkRenderPass` + `VkFramebuffer` | Heavy `VK_KHR_dynamic_rendering`, `VK_EXT_extended_dynamic_state1/2/3`, `VK_KHR_maintenance5/6`, `VK_EXT_graphics_pipeline_library`, `DrawIndirectCount` | Every missing dynamic state or maintenance feature blocks commercial DX12 startup |

**Key takeaway**: Solving the modern Vulkan 1.3/1.4 subset needed by `vkd3d-proton` automatically lifts compatibility for **DX11, DX12, and native modern Vulkan engines** simultaneously.

---

## 4. Hard-Won Engineering Lessons & Pitfall Log

Record of every major bug, silicon/driver quirk, and architectural bottleneck encountered from `v0.1` to `v0.6.0` (so future iterations never re-learn these the hard way):

### 4.1 Driver & Silicon Quirks (NVIDIA Hopper / Driver `535.x`)
1. **Compute UBO Constant-Bank Aliasing at `DescriptorSet >= 5`**:
   - *Symptom*: Passing `SmVsParams`/`SmFsParams` as `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` on high set indices (`>= 5`, typical in `vkd3d-proton` layouts) silently aliased constant banks on NVIDIA 535 and produced garbage transforms/colors.
   - *Fix*: Permanently switched internal parameter blocks to `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` (`std430 readonly buffer` with dynamic byte offsets). Never use `UNIFORM_BUFFER` for internal SmRast descriptor sets.
2. **H20 Graphics Queue DMA Hang (`Xid 69 / 109`)**:
   - *Symptom*: Submitting any graphics work to `qf[0]` permanently stalls the GPU DMA engine on H20.
   - *Fix*: SmRast-VK strictly binds all internal and pass-through submissions to pure Compute `qf[2]` (`VK_QUEUE_COMPUTE_BIT`), while advertising virtual `qf[0]` (`GRAPHICS | COMPUTE | TRANSFER`) to client apps.
3. **Background Thread Teardown `SIGSEGV` in `libnvidia-eglcore.so.535`**:
   - *Symptom*: Standalone Vulkan test binaries that returned from `main()` without calling `vkDeviceWaitIdle` + `vkDestroyDevice` + `vkDestroyInstance` occasionally crashed inside NVIDIA's background worker thread during `exit()`.
   - *Fix*: Always perform explicit `vkDeviceWaitIdle` → `vkDestroyDevice` → `vkDestroyInstance` teardown in test binaries and WSI shutdown paths.
4. **Portable Backend ICD Probing**:
   - *Lesson*: Never hardcode distro/container-specific driver paths. `smrast_icd.cpp` probes `SMRAST_BACKEND_ICD` first, then standard `ld.so` names (`libGLX_nvidia.so.0`, system paths, `libvulkan_radeon.so`).

### 4.2 Correctness & Compatibility Traps
5. **Ring Buffer Wrap Inside Real Open-World Frames**:
   - *Discovery*: Built-in `SMRAST_XRAY=1` one-frame pass tracer revealed a real 3A DX11 open-world frame executes **`~14,616` draw calls across `~215` render passes** (`~11K` shadow-cascade draws + `1,078` Z-prepass draws + 5-MRT deferred G-Buffer/lighting/bloom). The old `2,048`-slot parameter ring wrapped 7× mid-frame before submission!
   - *Fix*: Scaled draw parameter rings to **`49,152` slots**, indirect slots to **`16,384`**, clear/blit slots to **`4,096`**.
6. **Discard/Demote Immunity for Self-Clearing `VisBuf`**:
   - *Trap*: If `sm_vis_px[pidx] = SENTINEL` is written at the *end* of the lowered Fragment Shader, any pixel executing `discard;` or `demote;` exits early and leaks stale `(depth | tri_id)` into the next draw!
   - *Fix*: Clear `sm_vis_px[pidx] = 0xFFFFFFFFFFFFFFFFul` **immediately after reading `vis_val`** at the top of lowered FS and `smrast_depth_only.comp`, *before* running any user fragment code.
7. **Command-Buffer Dangling Vertex/Index Buffer Handles**:
   - *Trap*: Reusing a command buffer across tests/frames where an earlier VB was destroyed left stale `VkBuffer` pointers in `scb->vtx_bufs[]` / `scb->idx_buf`, causing segfaults when building DS tuples.
   - *Fix*: Zero `vtx_bufs` and `idx_buf` in `sm_BeginCommandBuffer`, mask descriptor binding strictly by the pipeline's `used_vb_mask`, and clear DS lookup maps on `vkDestroyBuffer` / `vkDestroyImage` (leaving underlying pools alive until frame reset so in-flight CBs stay hazard-free).
8. **Sampled Stencil Aspect SRVs (`VK_IMAGE_ASPECT_STENCIL_BIT`)**:
   - *Trap*: Deferred engines (shadow volumes, lighting masks, foliage/skin masks) sample the stencil aspect of `D32_SFLOAT_S8_UINT` / `D24_UNORM_S8_UINT` directly in FS/CS, often with non-identity `VkComponentMapping` swizzles (e.g. `{ZERO, R, ZERO, ZERO}`).
   - *Fix*: Maintain a companion `R32_UINT` shadow `VkImage` per depth-stencil image, synchronized across passes and layout barriers, and construct the sampled `VkImageView` preserving the app's exact `VkComponentMapping`.
9. **Watertight Rasterization & Exact `gl_FrontFacing`**:
   - *Fix*: 4-bit (`1/16 px`) fixed-point subpixel vertex snapping + strict Top-Left Fill-Rule edge bias eliminates shared-edge cracks and double-hits (`2304/2304` exact-once pixels verified in `test_v05_features`). `gl_FrontFacing` is computed consistently from signed screen-space area + `frontFace` + viewport Y-flip sign.
10. **SPIR-V Lowering Stability (`spirv_lower.cpp`)**:
    - Always run thread-local `libshaderc` instances with an atomic write-and-rename disk + memory cache (`/tmp/smrast_v6_cache_`) — zero `fork/exec` overhead, safe under multi-threaded DXVK/vkd3d pipeline compilation.
    - Hoist `#extension` directives above all injected macros/preambles so `vkd3d-proton` extensions (`GL_EXT_nonuniform_qualifier`, `GL_EXT_scalar_block_layout`, `GL_EXT_buffer_reference`) compile cleanly.
    - Keep `-O0` fallback when `shaderc` optimizer hits pathological game shaders.

### 4.3 Performance Optimization Lessons (`v0.5 → v0.6`)
11. **Eliminating Per-Draw `CmdFillBuffer` & `vkUpdateDescriptorSets`**:
    - At 1080p, clearing a 64-bit `VisBuf` every draw costs `16 MB × N_draws` of HBM3 traffic (**16 GB/frame** at 1,000 draws!). Self-clearing `VisBuf` + self-clearing `tile_counts` eliminated this completely.
    - Converting `SmVsParams`/`SmFsParams` to `STORAGE_BUFFER_DYNAMIC` + caching all 6 internal DS families by exact `VkBuffer` handle tuple cut steady-state per-draw `vkUpdateDescriptorSets` from **45 → 0**, dropping 1,000-draw single-threaded CPU recording time to **`5.17–5.66 ms` (`>180 Kdraws/s`)**.
12. **Hopper Barrier Tightening & Register-Pressure Isolation**:
    - Splitting pure `COMPUTE → COMPUTE` barriers from `DRAW_INDIRECT` transitions and Pass-0 `TRANSFER` (`CmdUpdateBuffer` WAR) barriers reduced 1,000-draw 1080p GPU time by **26%** (`69.13 ms → 51.28 ms`).
    - Splitting `LINE_LIST`/`LINE_STRIP`/`POINT_LIST` screen-space quad expansion out of `smrast_bin.comp` into `smrast_prim_bin.comp` prevented 5× `process_tri()` inlining register bloat in the hot triangle binning kernel.
13. **Hybrid Active-Tile vs Coherent-2D Heuristic**:
    - Active-Tile Indirect Dispatch saves huge work on small/localized draws (UI, props, decals, shadow casters), whereas giant full-screen meshes (`≥64K` tris) suffer atomic compaction contention and lose 2D L2/TMU cache coherence. Switching automatically at `64K` tris gets the best of both worlds.

---

## 5. Version History & Verified H20 Baseline

| Version | Major Milestone | Verified H20 Metrics (`qf[2]`, 78 SMs) |
|---|---|---|
| `v0.1 – v0.2` | Initial Vulkan Compute Rasterizer ICD + SPIR-V lowerer + CuRast micro-tri + CUDARaster tile binning | 1M-tri 1080p PBR **470–680 FPS** vs CPU Lavapipe **10 FPS** (`PSNR = 46.54 dB`); `vkcube` 60/60 via MIT-SHM WSI |
| `v0.3.0` | Commercial DX11 3A bring-up (DXVK 3.1.1), `smrast_depth_only.comp`, `smrast_blit.comp`, Vulkan 1.4 emulation | 60+ commercial DX11 pipelines 0 failures; 1,260+ continuous frames (`~18–20 FPS` on 3D title/menu) |
| `v0.4.0` | DX12 (`vkd3d-proton 3.1.0`, SM 6.6) end-to-end, 8-MRT × 21 formats, 8-bit Stencil, `VK_KHR_push_descriptor`, `DrawIndirectCount`, 49K draw slots | `DX12_VKD3D_TEST PASS`, `DEFERRED_MRT_PIXEL PASS`; 230+ commercial DX11 pipelines & 20+ commercial DX12 pipelines 0 failures |
| `v0.5.0` | Sampled Stencil Aspect SRVs, watertight Top-Left + 4-bit snapping, Depth Bias / Depth Clamp / Rasterizer Discard, 2-Level Tile Hi-Z, `vertex_attribute_divisor` | `V05_FEATURE_TEST PASS` (`2304/2304` watertight pixels); 1M PBR `477–681 FPS` |
| **`v0.6.0`** | **Hybrid Active-Tile Indirect Dispatch, Self-Clearing `VisBuf`/`tile_counts`, Zero-Update Dynamic SSBO DS Cache, Persistent Pre-Binning Coarse Hi-Z, Primitive Restart, Dedicated Lines/Points** | **`V06_FEATURE_TEST PASS`**; **1,000-Draw 1080p: `56.72 ms` (`CPU 5.17 ms`, `GPU 51.28 ms`, `17.6 Kdraws/s`)**; 1M PBR `277.6–681.6 FPS` (`PSNR = 46.54 dB`); 2M PBR `171.7–361.6 FPS` (`721 Mtris/s` peak); all DX11/DX12/Vulkan regression tests PASS |

### Anatomy of the Remaining Bottleneck (`v0.6.0`)
- **Geometry-bound workloads (100K – 2M tris, few draws)**: **Already 27×–70× faster than CPU Lavapipe** (`1.4–5.8 ms` at 1080p).
- **Draw-call-bound workloads (1,000 – 15,000 small draws/frame)**:
  - CPU recording is now fast (`5.2 ms` per 1,000 draws).
  - GPU execution is `51.3 ms` per 1,000 draws (**`~51 µs` per tiny draw**), dominated by **4–5 serialized Compute Grid launches + global inter-pass Hopper barriers per draw** (`VS → barrier → Bin → barrier → Fine → barrier → FS → barrier → Hi-Z`).
  - On a real open-world 3A scene with `~14,616` draws/frame, `14.6K × 51 µs ≈ 745 ms` (~1.3 FPS) unless draws are coalesced on the GPU.

---

## 6. Technical Roadmap (`v0.7.0+`)

### Phase 1 — `v0.7.0`: Multi-Draw Batching / Uber-Binning (Highest ROI for Real 3A FPS)
**Goal**: Collapse `51.3 ms / 1,000 draws` down to **`< 5 ms / 1,000 draws`** (`>200 Kdraws/s` GPU throughput), unlocking real-time 20–60 FPS in 10K+ draw open-world gameplay.

**Design Blueprint**:
1. **Command-Stream Draw Coalescing (`src/smrast_icd.cpp`)**:
   - Buffer consecutive draw calls inside a RenderPass whenever they share `(VkPipeline, Framebuffer/Attachments, Depth/Blend/Stencil dynamic state, App DescriptorSet bindings)` — or even across differing Push Constants / Dynamic UBO offsets (stored per-draw in a `SmDrawDesc[]` SSBO!).
   - Flush the open batch automatically on pipeline change, render-pass end, app barrier, or resource write hazard.
   - Shadow cascades (`~11K` draws/frame) and Z-prepass (`~1,078` draws/frame) typically share 1–3 depth-only pipelines and collapse into **< 10 mega-batches**!
2. **Mega-VS + Mega-Bin (`draw_id`-Indexed Geometry)**:
   - Allocate contiguous slices of `ClipVertBuf` per batched draw using prefix-summed vertex counts (`base_clip_vert[draw_id]`) and emit a flat triangle-to-draw lookup (binary search over `tri_prefix[]` or 1-thread-per-tri `draw_id` array).
   - **One** VS dispatch (or small batched VS dispatches) + **one** `smrast_bin.comp` dispatch processes all triangles from `K` draws simultaneously, tagging `VisBuf` entries with `(depth << 32) | global_tri_id` where `global_tri_id` maps directly to `(draw_id, local_tri_id)`!
   - **Depth-only shadow/Z-prepass batches** become literally **1 Bin + 1 Fine + 1 DepthOnly + 1 Hi-Z** for hundreds of meshes at once — eliminating >95% of Hopper inter-draw compute barriers!
3. **Mega-FS & Draw-Order Correctness**:
   - For opaque depth-tested draws (`depthWriteEnable=1`, `blendEnable=0`), 64-bit `atomicMin` resolves exact closest triangle across the entire batch in a single FS pass (`draw_id` fetched per winning pixel indexes per-draw Push Constants / instance metadata).
   - For alpha-blended draws (`blendEnable=1`) or overlapping stencil-increment draws, automatically break batch boundaries (or flush per overlapping draw) to preserve exact API submission order.

### Phase 2 — `v0.7.x`: Full Commercial DX12 3A Presentation & Modern Vulkan Coverage
1. **Out-of-Order Timeline Semaphore Scheduler**: Handle `vkd3d-proton`'s cross-thread wait-before-signal `vkQueueSubmit2` patterns cleanly on multiplexed `qf[2]` without blocking the Present thread.
2. **`VK_EXT_descriptor_buffer` & BDA Support**: Lower buffer-device-address descriptor reads or safely advertise optimal fallback profile for `vkd3d-proton`.
3. **Compute-Emulated Geometry & Tessellation Stages + 4× MSAA Resolve**: Unblock engines relying on GS particle expansion / hull-domain terrain tessellation and native MSAA render targets (`AlphaToCoverage`, `SampleMask`).

### Phase 3 — `v0.8.0`: Zero-Copy Streaming & Cross-Vendor Compute GPUs
1. **Zero-Copy HBM3 → NVENC / DMA-BUF Output**: Bypass X11 MIT-SHM CPU readback for low-latency cloud game streaming and high-throughput RGB data capture.
2. **AMD MI300X (`RADV` / `AMDVLK`, Wave64) Validation**: Audit subgroup quad shuffle masks and LDS tile layouts for Wave64 parity on CDNA3 silicon.
