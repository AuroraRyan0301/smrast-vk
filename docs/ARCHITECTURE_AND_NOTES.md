# SmRast-VK: Architecture, Engineering Lessons & Technical Roadmap

> **Living technical memory** for **SmRast-VK** — a pure-Compute **Vulkan 1.3 (+ emulated Vulkan 1.4)** Software Rasterizer Wrapper ICD (`libvulkan_smrast.so`) designed for compute-only datacenter GPUs (**NVIDIA H20 / H100**, **AMD MI300X**).
>
> *Last updated: `v0.8.0`.*

---

## 1. Core Thesis & Hardware Motivation

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (graphics queue submissions hang DMA on H20), forcing legacy graphics stacks onto CPU software rasterization (Mesa Lavapipe / `llvmpipe`, typically 0.1–10 FPS on complex 3D scenes while saturating dozens of CPU cores).

However, a compute-only H20 retains massive graphics-capable silicon on its pure Compute Queue (`VK_QUEUE_COMPUTE_BIT`, `qf[2]`):
- **78 Hopper SMs** (`FP32` / `INT32` / `FP16`)
- **96 GB HBM3** (~4 TB/s bandwidth)
- **Hardware 64-bit buffer atomics** (`VK_KHR_shader_atomic_int64`)
- **1-cycle Subgroup Quad Shuffles** (`VK_KHR_shader_subgroup_quad`)
- **Functional Hardware Texture Mapping Units (TMUs)** supporting `textureGrad()` anisotropic / mipmapped sampling inside Compute Shaders — including native `VK_IMAGE_TILING_LINEAR | VK_IMAGE_CREATE_ALIAS_BIT` sampling across all 14 standard color formats (`offset=0`, 32B-aligned `rowPitch`).

**SmRast-VK's Hybrid Principle**:
1. **Pass through everything silicon already does natively**: Compute Shaders (`vkCmdDispatch`), HBM3 allocations, DMA transfers (`vkCmdFillBuffer` / `vkCmdCopyBuffer`), hardware TMU sampling (zero-copy over linear-aliased G-Buffers), Timeline Semaphores, and Sync2 events run natively on `qf[2]`.
2. **Replace ONLY the missing graphics rasterizer & ROP fixed-function pipeline**: Intercept `VkPipeline` (graphics) creation and `vkCmdDraw*`, lower arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders in-process, and execute a cross-material multi-draw uber-batched Compute Rasterizer directly on the GPU SMs.

---

## 2. Architecture & Reference Lineage (`v0.8.0`)

SmRast-VK fuses four classic GPU software rasterization architectures with a production Vulkan 1.3/1.4 ICD, live SPIR-V lowering compiler, zero-copy linear-aliased G-Buffers, and cross-material GPU multi-draw uber-batching:

| Classic Reference | Technique Adapted | SmRast-VK v0.8 Implementation |
|---|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Homogeneous guard-band clipping, coarse `16×16` tile binning, watertight Top-Left fill rule | `shaders/smrast_bin.comp` + `v0.8` **7-Plane Homogeneous Guard-Band Clipping** (`w>=1e-4`, `z>=0`, `w-z>=0`, `±4w` XY) + **`VK_EXT_depth_clip_enable`**, **64-Way Warp-Cooperative Big-Tri Binning**, **Zero-Atomic Coop Small-Tri Binner**, **Pre-Binning Coarse Hi-Z** (`shaders/smrast_hiz_update.comp`), & **GPU Active-Tile Compaction** |
| **CuRast / compute_rasterizer** (Schütz / UE5 Nanite) | Lock-free 64-bit `atomicMin(depth << 32 \| tri_id)` direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`) | `shaders/smrast_bin.comp` micro-triangle fast-path with deterministic **Vulkan Painter's Order** `(uint64_t(~tid) << 32) \| tid`, driving **`556–934 FPS`** on 1M-tri 1080p PBR, with `discard;`-safe self-clearing `VisBuf` |
| **cuRE & RetroWarp** (Kenzel / Themaister) | 100% pure Vulkan Compute graphics pipeline (zero ROPs), `16×16` shared-memory Tile Fine rasterizer, subgroup quad derivatives | `shaders/smrast_fine.comp` (2-level Tile Hi-Z + Painter's Order + self-clearing `tile_counts`) + `src/spirv_lower.cpp` (**Fused Quad FS** with Painter's order, **Per-Draw Push-Constant / Dynamic-UBO SSBO Unpacking**, `flat`/`noperspective` varyings, branchless `subgroupQuadSwap` `dFdx`/`dFdy`, native TMU `textureGrad` over linear-aliased G-Buffers) |
| **Original ICD, Uber-Batcher & Compiler Layer** | Drop-in Khronos ICD + in-process SPIR-V → Compute compiler + Cross-Material Direct/Indirect Uber-Batching + Zero-Copy Linear-Aliased G-Buffers + Double-Buffered MIT-SHM WSI | `src/smrast_icd.cpp`, `src/spirv_lower.cpp`, `src/smrast_raster_engine.cpp`, `src/smrast_wsi.cpp` |

### 2.1 End-to-End Cross-Material Multi-Draw Uber-Batched Compute Pipeline (`v0.8.0`)

```
RenderPass Begin (LOAD_OP_CLEAR / LOAD_OP_LOAD / LOAD_OP_DONT_CARE)
  ├─ LOAD_OP_CLEAR / DONT_CARE immediately cancels stale attach_pending_sync (0 wasted copies!)
  └─ [0] Hardware DMA Fast-Clear (vkCmdFillBuffer)  [fallback: smrast_clear.comp w/ rt_stride_px]
         • Uniform 4B/8B/16B color, D32 depth + Hi-Z, and packed stencil cleared via DMA
           (0 descriptor updates, 0 compute shader launch; 15 clears = 0.13 ms total)
         • VisBuf & tile_counts remain invariant-clean via self-clearing (skips ~75% of RP clears)

Cross-Material Multi-Draw Uber-Batcher (src/smrast_icd.cpp):
  • Direct Batch:   coalesces up to 256 consecutive draws — INCLUDING draws with differing
                    FS Push Constants or Dynamic UBO offsets! Each sub-draw writes its 128B
                    push block into batch_push_buf[base+draw_id] and dynamic UBO offset into
                    batch_dyn_off_buf[base+draw_id], packing draw_id (0..255) into TriSetup.idx.w[16..23].
  • Indirect Batch: coalesces up to 128 consecutive vkCmdDraw[Indexed]Indirect[Count] draws
                    inside 32 KB ind_disp_buf using GPU atomic vert/tri slice allocators.
  • Fused Quad FS:  for <=2-tri draws without depth test/write, skips Fine & VisBuf completely
                    while preserving exact Painter's Order via TriSetup.idx.w[24..30]!

Per Flushed Batch (1..256 Direct OR 1..128 Indirect Draws):
  ├─ [0b] smrast_indirect_prep.comp  (indirect batch only, cached via ind_prep_ds_cache)
  │       Allocates contiguous ClipVertBuf / TriVerts slices on-GPU via atomicAdd(disp_cmds[12..13])
  │       and populates per-draw VS & Bin VkDispatchIndirectCommand structs.
  ├─ [1] K Barrier-Free Parallel Lowered Compute Vertex Shaders  (src/spirv_lower.cpp)
  │       • Up to 8 Vertex Buffers + VK_EXT_vertex_attribute_divisor (0 / >1)
  │       • Full vertex format unpackers (incl. unaligned R16_* load_u(0), USCALED/SSCALED/A2R10G10B10)
  │       • UINT8 / UINT16 / UINT32 indices + exact 0x7FC0DEAD NaN Primitive Restart sentinel
  │       • Writes Clip-Space position + varyings into 4.0M-vert HBM3 ClipVertBuf
  ├─ [2] K Parallel smrast_bin.comp Dispatches  (OR smrast_prim_bin.comp for Lines/Points)
  │       • Discards 0x7FC0DEAD restart sentinels & NaN/Inf vertices before clipping
  │       • 7-Plane CUDARaster Homogeneous Guard-Band Clipping:
  │           w >= 1e-4, z >= 0 (if depth_clip), w - z >= 0 (if depth_clip),
  │           and ±4w XY guard-band (eliminates 4-bit fixed-point integer overflow!)
  │       • VK_EXT_depth_clip_enable decoupled from depthClampEnable + Hardware Depth Bias
  │       • Pre-Binning Persistent Coarse Hi-Z early-reject per 16×16 tile (Standard-Z & Reverse-Z)
  │       • 3-Tier Work Distribution:
  │           1) Micro-tris (bbox <= 64 px): direct 64-bit atomicMin into VisBuf (CuRast)
  │              using Painter's Order (uint64_t(~tid)<<32 | tid) when depth test disabled
  │           2) Normal tris (<= 32 tiles): per-thread 16×16 tile_bins[] append
  │           3) Giant tris (> 32 tiles): 64-Way Warp-Cooperative shared-memory binning
  │              (s_big_tid/bb/z[64] -> all 64 threads cooperate row-major, 64× speedup!)
  │       • Packs draw_id into TriSetup.idx.w[16..23] and fused_quad order into bits [24..30]
  ├─ [3] 1× smrast_fine.comp  (SKIPPED completely in Fused Quad FS mode!)
  │       • 1 workgroup = 1 active 16×16 tile (256 threads), Indirect OR Coherent-2D grid
  │       • 1-read Persistent Tile Hi-Z + Watertight Top-Left Fill-Rule & 4-bit snapping
  │       • Deterministic Painter's Order tie-breaking; self-clears tile_counts/active
  ├─ [4] 1× Lowered Quad-Swizzled Compute FS  OR  smrast_depth_only.comp
  │       • Fused Quad FS (direct_2d == 2u): uniform tile AABB early-out + max-order Painter's
  │         Order inline edge testing directly from ClipVertBuf — zero Fine pass, zero VisBuf R/W!
  │       • Cross-Material SSBO Unpacking: extracts sm_draw_id = (ts.idx.w >> 16) & 0xFF and reads
  │         per-draw push constants (binding 15) and dynamic UBO fields (bindings 16/17)!
  │       • flat (provoking-vertex) & noperspective (linear screen-space) varying interpolation
  │       • Spec-exact gl_FragCoord.w = 1 / w_clip, exact gl_FrontFacing, subgroupQuadSwap dFdx/dFdy
  │       • Native hardware TMU textureGrad() directly on Zero-Copy Linear-Aliased G-Buffers!
  │       • 8 MRTs × 21 RT formats w/ per-RT rt_stride_px[8] linear-alias addressing + 8-bit Stencil
  └─ [5] 1× smrast_hiz_update.comp  (depth-writing batches only)
          Incrementally refreshes persistent 16×16 Coarse Hi-Z (tile_zmin, tile_zmax).

Zero-Copy Linear-Aliased Sync & Double-Buffered MIT-SHM Present:
  ├─ Eligible 2D 1-mip 1-layer color RTs (14 H20 formats): real_img (LINEAR | ALIAS_BIT) and
  │  attach_buf share the EXACT same VkDeviceMemory + offset -> ZERO SyncA2I / SyncI2A copies!
  ├─ Non-aliased images (mipmapped bloom/depth): exact ds_bound_imgs / ever_bound_in_ds lazy flush
  ├─ [6] smrast_blit.comp — pure-compute scaled & bilinear vkCmdBlitImage (rt_stride_px aware)
  └─ src/smrast_wsi.cpp — 2-slot Async DMA Ring (present_cmds[2]/fences[2]) + HOST_CACHED
                          + 32-bit R<->B swizzle + Double-Buffered XCB/Xlib MIT-SHM Ping-Pong!
```

### 2.2 Key Internal Memory & Slot Capacities (`include/smrast_internal.h`)

| Resource | Capacity (`v0.8.0`) | Purpose |
|---|---:|---|
| Draw parameter UBO/SSBO ring | **49,152 slots** | Supports full 3A open-world frames (`~14.6K` draws / `~215` render passes without wrap) |
| Direct / Indirect Uber-Batch width | **256 direct / 128 indirect** | Collapses consecutive draws (even across differing FS Push Constants / Dynamic UBOs!) |
| Per-Draw Metadata SSBO Rings | **65,536 × 128B Push (`8 MB`) + 65,536 × 4B DynOff (`256 KB`)** | Host-coherent per-draw FS Push-Constant & Dynamic-UBO offset rings indexed by `batch_push_base + sm_draw_id` |
| `ClipVertBuf` / `TriVerts` (HBM3) | **4.0M verts / 1.8M tris** | Headroom for uber-batched mega-meshes + 900K-vert / 390K-tri 7-plane clip tail |
| Indirect dispatch buffer (`ind_disp_buf`) | **32 KB / slot (`16,384` slots)** | Holds up to 128 indirect sub-draw headers + GPU atomic slice allocators |
| Clear / Blit slots | **4,096 slots** | High-pass-count deferred/bloom/shadow cascades |
| `tile_hiz` (per Depth Image) | **64 KB @ 1080p** | `120 × 68` tiles × 8 bytes (`(zmin, zmax)`), L2-resident |
| Descriptor-set cache pools | **16K sets / ring + `ind_prep_ds_cache`** | Zero-update `STORAGE_BUFFER_DYNAMIC` caching across all 18-binding internal DS layouts |

---

## 3. Why DX11 (`DXVK`) and DX12 (`vkd3d-proton`) Behave Differently on a Unified Vulkan ICD

SmRast-VK implements **one single Vulkan ICD** — there are no separate DX11 or DX12 rasterizers. The stark difference in bring-up difficulty comes from the **Vulkan API surface and SPIR-V complexity** emitted by the two Windows translation layers:

| Dimension | DX11 (`DXVK`) | DX12 (`vkd3d-proton`) | Engineering Impact on SmRast-VK |
|---|---|---|---|
| **Resource Binding** | Classic fixed-slot Descriptor Sets (bounded CBV/SRV/UAV tables) | Massive **Bindless arrays** (`nonuniformEXT`), `VK_EXT_mutable_descriptor_type`, `VK_KHR_push_descriptor`, and `VK_EXT_descriptor_buffer` (GPU pointers) | Required full emulation of mutable descriptors, descriptor indexing, push descriptors (incl. `WithTemplate`/`2KHR`), and dynamic SSBOs |
| **SPIR-V IR** | Translated from DXBC (SM 5.0); clean, regular GLSL-like IR | Translated from DXIL (SM 6.0–6.6); heavy `GL_EXT_buffer_reference` (raw 64-bit pointers), `scalar_block_layout`, 16-bit types, Wave/Subgroup ops, Root Constants | Required `#extension` hoisting above injected preambles in `spirv_lower.cpp` and careful SSBO lowering |
| **Queues & Sync** | Single immediate context; simple binary semaphores & fences | Multi-queue (Direct / Async Compute / Copy) + **out-of-order cross-thread Timeline Semaphore Wait/Signal** (`vkQueueSubmit2`) | Virtualizing multiple queues onto one physical `qf[2]` requires deadlock-free Timeline/Present scheduling |
| **Pipeline State** | Predominantly traditional `VkRenderPass` + `VkFramebuffer` | Heavy `VK_KHR_dynamic_rendering`, `VK_EXT_extended_dynamic_state1/2/3`, `VK_KHR_maintenance5/6`, `VK_EXT_graphics_pipeline_library`, `DrawIndirectCount` | Every missing dynamic state or maintenance feature blocks commercial DX12 startup |

---

## 4. `v0.7.0 → v0.8.0` GPU Profiling & Visual Bug Case Study: `3.7 FPS → 59 FPS → 81.3 FPS` (Zero Flicker)

Starting from `v0.6.0` (`3.71 FPS`), our systematic GPU timestamp profiling (`SmXrayGpuScope` inside `SMRAST_XRAY=1`) and frame-by-frame visual root-cause analysis drove both **22× real-game performance gains** and **zero-artifact rendering**:

| Stage / Defect Diagnosed on Live 1M-Tri 3D Scene (`1280×720`) | Before (`v0.6.0` / `v0.7.0`) | Fix Implemented (`v0.7.0` / `v0.8.0`) | After (`v0.8.0`) |
|---|---|---|---|
| **Giant White Blocks & Screen-Wide Flickering** (`v0.7.0` visual defects) | `diff_max = 4.9`, severe white trapezoids / needles | **7-Plane CUDARaster Guard-Band Clipping (`±4w`)** + **`VK_EXT_depth_clip_enable`** + **Painter's Order** (Fine + Fused Quad) + `flat`/`noperspective` varyings | **`diff_max = 1.00–2.02` (`0` spikes > 3.0), ZERO white blocks** |
| **1,000-Draw Per-Draw Push-Constant / Dyn-UBO Draws** | `42.53 ms` (`23.5 FPS`, unbatchable in `v0.7.0`) | **Per-Draw Push-Constant & Dynamic-UBO Metadata SSBO Uber-Batching** (`draw_id` in `TriSetup.idx.w[16..23]`, SPIRV-Cross SSBO unpacker) | **`2.54–2.92 ms` (`342–394 FPS`, `14.6×` speedup!)** |
| **52/83 Redundant `SyncA2I` G-Buffer Copies/Frame** | `WSI gpu = 14.0 ms`, `SyncA2I = 83/f` (`v0.7.0`) | **Zero-Copy Linear-Aliased Color G-Buffers** (`VK_IMAGE_TILING_LINEAR \| ALIAS_BIT`, 14 H20 formats, `rt_stride_px[8]`) + exact `ds_bound_imgs` lazy sync | **`WSI gpu = 9.1–10.0 ms`, `SyncA2I = 27–31/f` (`52` copies/f skipped!)** |
| **WSI X11 Readback & Xvfb Blit Stall** | `76–99 ms` (`v0.6`) → `2.4 ms` (`v0.7`) | **Double-Buffered MIT-SHM Ping-Pong Ring** (`shm_ptrs[2]`, `xlib_shm_imgs[2]`) + 2-slot Async DMA Ring + `HOST_CACHED` + 32-bit swizzle | **`0.3–0.8 ms`** (`WSI x11`) |
| **Single-Thread Giant-Triangle Binning** (`3,555` tiles) | `7.7 ms` (`v0.6`) | **64-Way Warp-Cooperative Big-Tri Binning** (`s_big_*[64]`) | **`< 0.3 ms`** |
| **48 Post-Process / UI Quads + 15 RP Clears** | `~30 ms` (`v0.6`) | **Fused Quad FS** (Painter's order) + **DMA `vkCmdFillBuffer`** | **`2.05 ms` combined** |
| **End-to-End Live Commercial 3D 1M-Tri Scene** (shared H20) | `269.5 ms` (`3.71 FPS`, `v0.6`) / `17.0 ms` (`58.8 FPS`, `v0.7`) | All `v0.8.0` visual + zero-copy + SHM optimizations combined | **`12.3–13.1 ms` (`76.0–81.3 FPS`, `677×` vs CPU Lavapipe)** |

---

## 5. Hard-Won Engineering Lessons & Pitfall Log (`v0.1 → v0.8.0`)

Record of every major bug, silicon/driver quirk, and architectural bottleneck encountered so future iterations never re-learn these the hard way:

### 5.1 Driver & Silicon Quirks (NVIDIA Hopper / Driver `535.x`)
1. **Compute UBO Constant-Bank Aliasing at `DescriptorSet >= 5`**:
   - *Fix*: Permanently use `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` (`std430 readonly buffer`) for all internal SmRast parameter blocks (`SmVsParams` / `SmFsParams`). Never use `UNIFORM_BUFFER` on high set indices.
2. **H20 Graphics Queue DMA Hang (`Xid 69 / 109`)**:
   - *Fix*: Strictly bind all internal and pass-through submissions to pure Compute `qf[2]` (`VK_QUEUE_COMPUTE_BIT`), while advertising virtual `qf[0]` (`GRAPHICS | COMPUTE | TRANSFER`).
3. **H20 `VK_IMAGE_TILING_LINEAR` + `VK_IMAGE_CREATE_ALIAS_BIT` Native Capabilities (`v0.8.0`)**:
   - *Discovery*: All 14 standard color formats (`RGBA8`/`BGRA8` UNORM & SRGB, `RGBA16F`, `B10G11R11F`, `R32F`, `R16F`, `R8`, `RG8`, `RG16F`, `RG32F`, `RGBA32F`, `A2B10G10R10`) natively support `LINEAR` + `SAMPLED_IMAGE` + `FILTER_LINEAR` + `MUTABLE_FORMAT` + `ALIAS_BIT` with `subresourceLayout.offset == 0`, 32B-aligned `rowPitch`, and `imr.memoryTypeBits & bmr.memoryTypeBits == 0x1f` on H20 — allowing `attach_buf` and `real_img` to alias the exact same `VkDeviceMemory` with zero copies!
   - *Caution*: Always compute `row_stride_px = rowPitch / bpp` (NOT `width`, which fails for odd/non-aligned render target widths like `1257×707`!) and propagate `rt_stride_px[8]` into FS, Clear, Blit, Copy, and MSAA resolve.
4. **Background Thread Teardown `SIGSEGV` in `libnvidia-eglcore.so.535`**:
   - *Fix*: Always perform explicit `vkDeviceWaitIdle` → `vkDestroyDevice` → `vkDestroyInstance` teardown in test binaries and WSI shutdown paths.
5. **Portable Backend ICD Probing**:
   - *Fix*: Probe `SMRAST_BACKEND_ICD` first, then standard `ld.so` names (`libGLX_nvidia.so.0`, system paths, `libvulkan_radeon.so`). Never hardcode container paths.

### 5.2 Correctness & Spec Traps
6. **Giant White-Block Polygons & Grazing Near-Plane Fixed-Point Overflow (`v0.8.0`)**:
   - *Trap*: 2-plane clipping (`z >= 0` & `w - z >= 0`) alone allowed triangles with tiny `w` (`1e-7`) or off-screen vertices (`|x/w| > 10^5`) to overflow 4-bit subpixel `int32` fixed-point coordinates in `smrast_fine.comp`, inverting edge function signs and painting massive white/grey screen-spanning trapezoids! Additionally, failing to parse `VkPipelineRasterizationDepthClipStateCreateInfoEXT` (`VK_EXT_depth_clip_enable`) falsely clipped shadow/skybox passes that disable depth clip independently of `depthClampEnable`.
   - *Fix*: Implement full **7-Plane CUDARaster Homogeneous Guard-Band Clipping** (`w >= 1e-4`, `z >= 0`, `w - z >= 0`, `±4w` XY guard-band) and independently parse `pipe->depth_clip` (`rflags |= 2048u` from `!pipe->depth_clip`).
7. **Non-Deterministic Tie-Breaking / Painter's Order Flickering (`v0.8.0`)**:
   - *Trap*: For depth-test-disabled or equal-depth overlapping UI/decal triangles, `atomicMin` on `(0 << 32) | tid` picked the *smallest* `tid` (first-drawn wins instead of last-drawn wins), and `fused_quad` broke on the first matching triangle (`k=0`), causing flickering and inverted UI layering.
   - *Fix*: Pack `(uint64_t(~tid) << 32) | uint64_t(tid)` so `atomicMin` deterministically selects the **largest `tid` (last-drawn)** in `smrast_bin.comp`/`smrast_fine.comp`, and pack `(tid & 0x7Fu) << 24u` into `TriSetup.idx.w` for max-order tie-breaking in `fused_quad`.
8. **`flat` / `noperspective` Varyings & Unaligned `R16_*` Vertex Loads (`v0.8.0`)**:
   - *Trap*: Perspective-correct interpolation corrupted `flat` float IDs/attributes and `noperspective` screen-space gradients; `R16_*` vertex formats with 2-byte offset alignment dropped the lower half-word when using word-aligned loads.
   - *Fix*: Reflect `spvc_compiler_has_decoration(..., SpvDecorationFlat / NoPerspective)` in `spirv_lower.cpp` and emit exact unaligned `load_u(0)` shifts for `R16_*`.
9. **Iterator Invalidation in `sm_CmdPipelineBarrier2` (`v0.8.0`)**:
   - *Trap*: Iterating `scb->pending_sync_imgs` while calling `sm_flush_pending_syncs` erased elements from the same vector during iteration.
   - *Fix*: Copy `pending_sync_imgs` into a local vector before selective flushing.
10. **`gl_FragCoord.w` Spec Inversion (`v0.7.0`)**:
    - *Fix*: Set `sm_gl_FragCoord = vec4(fx, fy, pz, wsum)` matching Vulkan spec §15.4 (`1 / w_c`).
11. **Uber-Batching Correctness Hazards (`v0.7.0 / v0.8.0`)**:
    - *Fix*: Reflect `fs_uses_push_constants`, `fs_batched_push_ok`, `fs_batched_dyn_ubo_ok`, and `fs_has_discard_or_depth` in `spirv_lower.cpp`; flush open draw batches in `sm_CmdClearAttachments`; guard `is_single_small = (!B.is_indirect && K == 1u && B.draws[0].total_tris <= 8u)`.
12. **Discard/Demote Immunity, sRGB Blend Decode, Read-Only Stencil Sync & Ring Sizing (`v0.5–v0.7`)**:
    - Self-clear `sm_vis_px[pidx]` before user FS code; decode `sm_srgb_to_lin()` before blending; clamp half-floats via `sm_safe_f16()`; maintain companion `R32_UINT` stencil SRV image + `stn_can_write` tracking; preserve `49,152` draw slots and invalidate DS caches on `vkDestroyBuffer`/`vkDestroyImage`.

### 5.3 Performance Optimization Lessons (`v0.5 → v0.8`)
13. **Always Profile GPU Timestamps Before Optimizing**:
    - `SmXrayGpuScope` (`vkCmdWriteTimestamp`) pinpointed giant-triangle serialization (`6.33 ms`) in `v0.7.0`, and `[SmRast-PROF]` pinpointed `83` `SyncA2I` copies/frame (`~4.5 ms`) and `2.4 ms` X11 single-SHM wait in `v0.8.0`.
14. **Eliminate Data Movement Before Micro-Tuning Shaders**:
    - Zero-copy linear-aliased color G-Buffers removed 52 GPU-to-GPU image copies/frame (`WSI gpu`: `14.0 ms → 9.1 ms`, `+37%` live FPS), and Per-Draw Metadata SSBOs collapsed 1,000 cross-material barriers into 4 batches (`42.53 ms → 2.92 ms`, `14.6×` speedup).
15. **Double-Buffer Host Presentation to Eliminate X11 Bubbles**:
    - Combining a 2-slot async GPU→Host DMA ring (`present_cmds[2]`/`present_fences[2]`) with a 2-slot MIT-SHM ping-pong ring (`shm_ptrs[2]`) shrank `WSI x11` from `76–99 ms` (`v0.6`) → `2.4 ms` (`v0.7`) → **`0.3–0.8 ms`** (`v0.8`).

---

## 6. Version History & Verified H20 Baseline

| Version | Major Milestone | Verified H20 Metrics (`qf[2]`, 78 SMs) |
|---|---|---|
| `v0.1 – v0.2` | Initial Vulkan Compute Rasterizer ICD + SPIR-V lowerer + CuRast micro-tri + CUDARaster tile binning | 1M-tri 1080p PBR **470–680 FPS** vs CPU Lavapipe **10 FPS** (`PSNR = 46.54 dB`); `vkcube` 60/60 via MIT-SHM WSI |
| `v0.3.0` | Commercial DX11 3A bring-up (DXVK 3.1.1), `smrast_depth_only.comp`, `smrast_blit.comp`, Vulkan 1.4 emulation | 60+ commercial DX11 pipelines 0 failures; 1,260+ continuous frames |
| `v0.4.0` | DX12 (`vkd3d-proton 3.1.0`, SM 6.6) end-to-end, 8-MRT × 21 formats, 8-bit Stencil, `VK_KHR_push_descriptor`, `DrawIndirectCount`, 49K draw slots | `DX12_VKD3D_TEST PASS`, `DEFERRED_MRT_PIXEL PASS`; 230+ commercial DX11 & 20+ commercial DX12 pipelines 0 failures |
| `v0.5.0` | Sampled Stencil Aspect SRVs, watertight Top-Left + 4-bit snapping, Depth Bias / Depth Clamp / Rasterizer Discard, 2-Level Tile Hi-Z, `vertex_attribute_divisor` | `V05_FEATURE_TEST PASS` (`2304/2304` watertight pixels); 1M PBR `477–681 FPS` |
| `v0.6.0` | Hybrid Active-Tile Indirect Dispatch, Self-Clearing `VisBuf`/`tile_counts`, Zero-Update Dynamic SSBO DS Cache, Persistent Pre-Binning Coarse Hi-Z, Primitive Restart, Lines/Points | `V06_FEATURE_TEST PASS`; 1,000-Draw 1080p `56.72–61.07 ms` (`16–18 FPS`); AAA 3D scene `3.71 FPS` |
| `v0.7.0` | Direct (≤256) & Indirect (≤128) Multi-Draw Uber-Batching, 64-Way Warp-Coop Big-Tri Binning, Fused Quad FS, DMA Fast-Clear, Lazy Attachment Sync + `stn_can_write`, 1-Frame Pipelined MIT-SHM Ring, `gl_FragCoord.w` & 2-Plane Clipping, `vkCmdWriteTimestamp` X-Ray | `V07_FEATURE_TEST PASS`; 1,000-Draw Uber-Batch `2.84 ms` Direct (`352.7 FPS`) / `5.03 ms` Indirect (`198.7 FPS`); 1M-Tri PBR `934.4 FPS`; AAA 3D Scene `58.8–70.1 FPS` |
| **`v0.8.0`** | **7-Plane CUDARaster Guard-Band Clipping (`±4w`) + `VK_EXT_depth_clip_enable`, Painter's Order (Fine + Fused Quad), `flat`/`noperspective` Varyings, Per-Draw Push-Constant & Dynamic-UBO SSBO Cross-Material Uber-Batching, Zero-Copy Linear-Aliased Color G-Buffers (`14` H20 formats, `rt_stride_px[8]`) + Exact Bound-Set Lazy Sync, Double-Buffered MIT-SHM Ping-Pong Ring** | **`V08_FEATURE_TEST PASS` (10/10 suites PASS, 0 warnings)**; **1,000-Draw Per-Draw Push-Constant: `2.57–2.92 ms` (`342–389 FPS`, `14.6×` vs v0.7)**; **1,000-Draw Per-Draw Dynamic-UBO: `2.54 ms` (`393.6 FPS`)**; **1,000-Draw Direct/Indirect: `2.50 ms` (`399.9 FPS`) / `4.20 ms` (`238.2 FPS`)**; **Commercial AAA 3D 1M-Tri Scene: `76.0–81.3 FPS` (`+37%` vs v0.7, `633×–677×` vs 48-Core CPU Lavapipe, `0` flicker spikes, `0` white blocks)**; **3D Menu: `67.6–71.1 FPS`** (`WSI x11=0.3–0.8 ms`) |

---

## 7. Technical Roadmap (`v0.9.0+`)

### Phase 1 — `v0.9.0`: Mipmapped Linear Aliasing & Depth-SRV Zero-Copy (Target: `100+ FPS` Heavy Open-World)
1. **Mipmapped Linear-Aliased Pyramids & Zero-Copy Depth SRV**: Extend zero-copy linear aliasing to multi-mip bloom/Hi-Z chains and native linear/storage depth buffers to eliminate the remaining `~27–31` `SyncA2I` and `~11` `SyncI2A` copies per frame.
2. **Multi-Pipeline Indirect Draw Fusion**: Fuse adjacent indirect draws sharing identical FS binary across varying vertex strides/bindings via GPU descriptor indexing.

### Phase 2 — `v0.9.x`: Full Commercial DX12 3A Presentation & Modern Vulkan Coverage
1. **Out-of-Order Timeline Semaphore Scheduler**: Handle `vkd3d-proton`'s cross-thread wait-before-signal `vkQueueSubmit2` patterns cleanly on multiplexed `qf[2]` without blocking the Present thread.
2. **`VK_EXT_descriptor_buffer` & BDA Support**: Lower buffer-device-address descriptor reads or safely advertise optimal fallback profile for `vkd3d-proton`.
3. **Compute-Emulated Geometry & Tessellation Stages + 4× MSAA Resolve**: Unblock engines relying on GS particle expansion / hull-domain terrain tessellation and native MSAA render targets (`AlphaToCoverage`, `SampleMask`).

### Phase 3 — `v1.0.0`: Zero-Copy Streaming & Cross-Vendor Compute GPUs
1. **Zero-Copy HBM3 → NVENC / DMA-BUF Output**: Bypass X11 MIT-SHM CPU readback completely for low-latency cloud game streaming and high-throughput RGB capture.
2. **AMD MI300X (`RADV` / `AMDVLK`, Wave64) Validation**: Audit subgroup quad shuffle masks, 64-way warp-cooperative bin assumptions, and LDS tile layouts for Wave64 parity on CDNA3 silicon.
