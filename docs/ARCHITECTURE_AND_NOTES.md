# SmRast-VK: Architecture, Engineering Lessons & Technical Roadmap

> **Living technical memory** for **SmRast-VK** — a pure-Compute **Vulkan 1.3 (+ emulated Vulkan 1.4)** Software Rasterizer Wrapper ICD (`libvulkan_smrast.so`) designed for compute-only datacenter GPUs (**NVIDIA H20 / H100**, **AMD MI300X**).
>
> *Last updated: `v0.7.0`.*

---

## 1. Core Thesis & Hardware Motivation

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (graphics queue submissions hang DMA on H20), forcing legacy graphics stacks onto CPU software rasterization (Mesa Lavapipe / `llvmpipe`, typically 0.1–10 FPS on complex 3D scenes while saturating dozens of CPU cores).

However, a compute-only H20 retains massive graphics-capable silicon on its pure Compute Queue (`VK_QUEUE_COMPUTE_BIT`, `qf[2]`):
- **78 Hopper SMs** (`FP32` / `INT32` / `FP16`)
- **96 GB HBM3** (~4 TB/s bandwidth)
- **Hardware 64-bit buffer atomics** (`VK_KHR_shader_atomic_int64`)
- **1-cycle Subgroup Quad Shuffles** (`VK_KHR_shader_subgroup_quad`)
- **Functional Hardware Texture Mapping Units (TMUs)** supporting `textureGrad()` anisotropic / mipmapped sampling inside Compute Shaders

**SmRast-VK's Hybrid Principle**:
1. **Pass through everything silicon already does natively**: Compute Shaders (`vkCmdDispatch`), HBM3 allocations, DMA transfers (`vkCmdFillBuffer` / `vkCmdCopyBuffer`), hardware TMU sampling, Timeline Semaphores, and Sync2 events run natively on `qf[2]`.
2. **Replace ONLY the missing graphics rasterizer & ROP fixed-function pipeline**: Intercept `VkPipeline` (graphics) creation and `vkCmdDraw*`, lower arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders in-process, and execute a multi-draw uber-batched Compute Rasterizer directly on the GPU SMs.

---

## 2. Architecture & Reference Lineage (`v0.7.0`)

SmRast-VK fuses four classic GPU software rasterization architectures with a production Vulkan 1.3/1.4 ICD, live SPIR-V lowering compiler, and GPU-driven multi-draw uber-batching:

| Classic Reference | Technique Adapted | SmRast-VK v0.7 Implementation |
|---|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Coarse `16×16` screen-space tile binning, Sutherland-Hodgman clipping, watertight Top-Left fill rule | `shaders/smrast_bin.comp` + `v0.7` **64-Way Warp-Cooperative Big-Tri Binning**, **2-Plane Exact Clipping** (`z>=0` & `w-z>=0`), **Zero-Atomic Coop Small-Tri Binner**, **Pre-Binning Coarse Hi-Z** (`shaders/smrast_hiz_update.comp`), & **GPU Active-Tile Compaction** |
| **CuRast / compute_rasterizer** (Schütz / UE5 Nanite) | Lock-free 64-bit `atomicMin(depth << 32 \| tri_id)` direct visibility-buffer fast-path for micro-triangles (`bbox ≤ 64 px`) | `shaders/smrast_bin.comp` micro-triangle fast-path driving **`934 FPS` / `930 Mtris/s`** on 1M-tri 1080p PBR, with `discard;`-safe self-clearing `VisBuf` |
| **cuRE & RetroWarp** (Kenzel / Themaister) | 100% pure Vulkan Compute graphics pipeline (zero ROPs), `16×16` shared-memory Tile Fine rasterizer, subgroup quad derivatives | `shaders/smrast_fine.comp` (2-level Tile Hi-Z + self-clearing `tile_counts`) + `src/spirv_lower.cpp` (**Fused Quad FS** skipping Fine/`VisBuf`, branchless `subgroupQuadSwap` `dFdx`/`dFdy`, native TMU `textureGrad`) |
| **Original ICD, Uber-Batcher & Compiler Layer** | Drop-in Khronos ICD + in-process SPIR-V → Compute compiler + Direct/Indirect Multi-Draw Uber-Batching + Pipelined MIT-SHM WSI | `src/smrast_icd.cpp`, `src/spirv_lower.cpp`, `src/smrast_raster_engine.cpp`, `src/smrast_wsi.cpp` |

### 2.1 End-to-End Multi-Draw Uber-Batched Compute Pipeline (`v0.7.0`)

```
RenderPass Begin (LOAD_OP_CLEAR / LOAD_OP_LOAD)
  └─ [0] Hardware DMA Fast-Clear (vkCmdFillBuffer)  [fallback: smrast_clear.comp]
         • Uniform 4B/8B/16B color, D32 depth + Hi-Z, and packed stencil cleared via DMA
           (0 descriptor updates, 0 compute shader launch; 15 clears = 0.13 ms total)
         • VisBuf & tile_counts remain invariant-clean via self-clearing (skips ~75% of RP clears)

Multi-Draw Uber-Batcher (src/smrast_icd.cpp):
  • Direct Batch:   coalesces up to 256 consecutive opaque draws (validated via SPIR-V
                    fs_uses_push_constants & fs_has_discard_or_depth reflection)
  • Indirect Batch: coalesces up to 128 consecutive vkCmdDraw[Indexed]Indirect[Count] draws
                    inside 32 KB ind_disp_buf using GPU atomic vert/tri slice allocators
  • Fused Quad FS:  for <=2-tri draws without depth test/write, skips Fine & VisBuf completely!

Per Flushed Batch (1..256 Direct OR 1..128 Indirect Draws):
  ├─ [0b] smrast_indirect_prep.comp  (indirect batch only, cached via ind_prep_ds_cache)
  │       Allocates contiguous ClipVertBuf / TriVerts slices on-GPU via atomicAdd(disp_cmds[12..13])
  │       and populates per-draw VS & Bin VkDispatchIndirectCommand structs.
  ├─ [1] K Barrier-Free Parallel Lowered Compute Vertex Shaders  (src/spirv_lower.cpp)
  │       • Up to 8 Vertex Buffers + VK_EXT_vertex_attribute_divisor (0 / >1)
  │       • Full vertex format unpackers (incl. USCALED/SSCALED/A2R10G10B10, R8/16 SNORM/SINT)
  │       • UINT8 / UINT16 / UINT32 indices + exact 0x7FC0DEAD NaN Primitive Restart sentinel
  │       • Writes Clip-Space position + interpolated varyings into 4.0M-vert HBM3 ClipVertBuf
  ├─ [2] K Parallel smrast_bin.comp Dispatches  (OR smrast_prim_bin.comp for Lines/Points)
  │       • Discards 0x7FC0DEAD restart sentinels & NaN/Inf vertices before clipping
  │       • Exact 2-Plane Sutherland-Hodgman Clipping (z >= 0 & w - z >= 0, Reverse-Z & w < 0 safe)
  │       • Hardware-accurate Depth Bias & depthClampEnable
  │       • Pre-Binning Persistent Coarse Hi-Z early-reject per 16×16 tile (Standard-Z & Reverse-Z)
  │       • 3-Tier Work Distribution:
  │           1) Micro-tris (bbox <= 64 px): direct 64-bit atomicMin into VisBuf (CuRast)
  │           2) Normal tris (<= 32 tiles): per-thread 16×16 tile_bins[] append
  │           3) Giant tris (> 32 tiles): 64-Way Warp-Cooperative shared-memory binning
  │              (s_big_tid/bb/z[64] -> all 64 threads cooperate row-major, 64× speedup!)
  │       • Small-Tri Coop Mode (flag bit 16, <=8 tris): zero-atomic shared-memory active_tiles[]
  ├─ [3] 1× smrast_fine.comp  (SKIPPED completely in Fused Quad FS mode!)
  │       • 1 workgroup = 1 active 16×16 tile (256 threads), Indirect OR Coherent-2D grid
  │       • 1-read Persistent Tile Hi-Z + Watertight Top-Left Fill-Rule & 4-bit snapping
  │       • Writes winning (depth << 32 | tri_id) into VisBuf; self-clears tile_counts/active
  ├─ [4] 1× Lowered Quad-Swizzled Compute FS  OR  smrast_depth_only.comp
  │       • Fused Quad FS (direct_2d == 2u, flag bit 17): uniform tile AABB early-out + inline
  │         edge testing directly from ClipVertBuf — zero Fine pass, zero VisBuf R/W!
  │       • Normal/Uber-Batch FS: reads VisBuf[pidx] & immediately self-clears (discard-safe!)
  │       • Spec-exact gl_FragCoord.w = 1 / w_clip, exact gl_FrontFacing, subgroupQuadSwap dFdx/dFdy
  │       • Native hardware TMU textureGrad(), 8-bit Stencil Test/Op (stn_can_write tracked)
  │       • 8 MRTs × 21 RT formats (incl. sRGB blend read decode, sm_safe_f16, 1B/2B atomics)
  └─ [5] 1× smrast_hiz_update.comp  (depth-writing batches only)
          Incrementally refreshes persistent 16×16 Coarse Hi-Z (tile_zmin, tile_zmax).

Lazy Sync & Pipelined Present:
  ├─ attach_pending_sync defers attach_buf -> real_img sync until barrier/read/copy/dispatch/CB-end
  ├─ [6] smrast_blit.comp — pure-compute scaled & bilinear vkCmdBlitImage
  └─ src/smrast_wsi.cpp — 2-slot Async Pipelined Present Ring (present_cmds[2]/present_fences[2])
                          + HOST_CACHED staging + 32-bit R<->B swizzle + XCB/Xlib MIT-SHM
```

### 2.2 Key Internal Memory & Slot Capacities (`include/smrast_internal.h`)

| Resource | Capacity (`v0.7.0`) | Purpose |
|---|---:|---|
| Draw parameter UBO/SSBO ring | **49,152 slots** | Supports full 3A open-world frames (`~14.6K` draws / `~215` render passes without wrap) |
| Direct / Indirect Uber-Batch width | **256 direct / 128 indirect** | Collapses consecutive opaque/indirect draws into single Fine+FS+Hi-Z chains |
| `ClipVertBuf` / `TriVerts` (HBM3) | **4.0M verts / 1.8M tris** | Headroom for uber-batched mega-meshes + 900K-vert / 390K-tri 2-plane clip tail |
| Indirect dispatch buffer (`ind_disp_buf`) | **32 KB / slot (`16,384` slots)** | Holds up to 128 indirect sub-draw headers + GPU atomic slice allocators |
| Clear / Blit slots | **4,096 slots** | High-pass-count deferred/bloom/shadow cascades |
| `tile_hiz` (per Depth Image) | **64 KB @ 1080p** | `120 × 68` tiles × 8 bytes (`(zmin, zmax)`), L2-resident |
| Descriptor-set cache pools | **16K sets / ring + `ind_prep_ds_cache`** | Zero-update `STORAGE_BUFFER_DYNAMIC` caching across all internal DS layouts |

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

## 4. `v0.7.0` GPU Profiling Case Study: `3.7 FPS → 70 FPS` on a Commercial 1M-Tri 3D Scene

The `v0.7.0` speedup (`16×–19×` on real 3D AAA scenes, `31.8×` on DoF intro, `21.5×` on 1,000-draw synthetic) was driven systematically by adding an on-demand **`vkCmdWriteTimestamp` GPU scope profiler** (`SmXrayGpuScope`) inside `SMRAST_XRAY=1` and attacking each measured millisecond in priority order:

| Stage / Bottleneck Diagnosed on Live 1M-Tri 3D Scene (`1280×720`) | Before (`v0.6.0` / Early `v0.7`) | Fix Implemented in `v0.7.0` | After (`v0.7.0`) |
|---|---:|---|---:|
| **WSI X11 Readback & Present Stall** | `76–99 ms` | `HOST_CACHED` staging + vectorized 32-bit R↔B swizzle + Xlib `XShmPutImage` + 2-slot Async Pipelined Present Ring | **`0.5–1.3 ms`** |
| **Single-Thread Giant-Triangle Binning** (`2` indirect draws spanning 3,555 tiles serialized on 1 thread) | `7.7 ms` (`4.30 + 2.03 + 1.4 ms`) | **64-Way Warp-Cooperative Big-Tri Binning** (`>32` tiles enqueue to shared memory, 64 threads cooperate row-major) | **`< 0.3 ms`** |
| **48 Post-Process / UI Quads** (`Fine` + `VisBuf` R/W + `CmdDispatchIndirect`) | `~18 ms` (~1 GB HBM3 traffic) | **Fused Quad FS** (`direct_2d == 2u`, bit 17) — inline edge test in lowered FS with tile-AABB early-out | **`1.92 ms`** (all 48 quads) |
| **15 RenderPass Clears + `VisBuf`/`tile_counts` Re-Clears** | `~12 ms` | **Hardware DMA `vkCmdFillBuffer`** + invariant self-clearing (47/62 RP clears skipped) | **`0.13 ms`** (all 15 clears) |
| **Redundant `Attach → Image` Syncs (incl. Read-Only Stencil)** | `~15 ms` (85+ syncs) | **`attach_pending_sync` lazy deferral** + **`stn_can_write` tracking** (eliminates 30+ read-only stencil syncs) | **`~1.5 ms`** |
| **1,000-Draw Inter-Draw Compute Grid Barriers** | `61.07 ms` (`16.4 FPS`) | **Direct (≤256) & Indirect (≤128) Multi-Draw Uber-Batching** | **`2.84 ms` Direct / `5.03 ms` Indirect** |
| **End-to-End 3D 1M-Tri Commercial Scene** (`shared H20`) | **`269.5 ms` (`3.71 FPS`)** | All `v0.7.0` optimizations combined | **`14.3–17.0 ms` (`58.8–70.1 FPS`)** |

---

## 5. Hard-Won Engineering Lessons & Pitfall Log (`v0.1 → v0.7.0`)

Record of every major bug, silicon/driver quirk, and architectural bottleneck encountered so future iterations never re-learn these the hard way:

### 5.1 Driver & Silicon Quirks (NVIDIA Hopper / Driver `535.x`)
1. **Compute UBO Constant-Bank Aliasing at `DescriptorSet >= 5`**:
   - *Fix*: Permanently use `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` (`std430 readonly buffer`) for all internal SmRast parameter blocks (`SmVsParams` / `SmFsParams`). Never use `UNIFORM_BUFFER` on high set indices.
2. **H20 Graphics Queue DMA Hang (`Xid 69 / 109`)**:
   - *Fix*: Strictly bind all internal and pass-through submissions to pure Compute `qf[2]` (`VK_QUEUE_COMPUTE_BIT`), while advertising virtual `qf[0]` (`GRAPHICS | COMPUTE | TRANSFER`).
3. **Background Thread Teardown `SIGSEGV` in `libnvidia-eglcore.so.535`**:
   - *Fix*: Always perform explicit `vkDeviceWaitIdle` → `vkDestroyDevice` → `vkDestroyInstance` teardown in test binaries and WSI shutdown paths.
4. **Portable Backend ICD Probing**:
   - *Fix*: Probe `SMRAST_BACKEND_ICD` first, then standard `ld.so` names (`libGLX_nvidia.so.0`, system paths, `libvulkan_radeon.so`). Never hardcode container paths.

### 5.2 Correctness & Spec Traps
5. **`gl_FragCoord.w` Spec Inversion (`v0.7.0`)**:
   - *Trap*: Lowered FS previously set `gl_FragCoord.w = inv_wsum` (`w_clip`) instead of `wsum` (`1.0 / w_clip`). Shaders doing deferred world-position reconstruction or soft-particle depth fades produced floating rectangular noise artifacts!
   - *Fix*: Set `sm_gl_FragCoord = vec4(fx, fy, pz, wsum)` matching Vulkan spec §15.4 (`1 / w_c`).
6. **Primitive Restart Sentinel Collision & Reverse-Z Far-Plane / Negative-`w` Clipping (`v0.7.0`)**:
   - *Trap*: Using `w = -2.0` as Primitive Restart sentinel silently discarded valid geometry passing behind the camera with `w ≈ -2.0`; single-plane `z >= 0` clipping missed Reverse-Z far plane (`w - z >= 0`); `NaN`/`Inf` vertices overflowed `int(floor(...))` tile bounds.
   - *Fix*: Use exact quiet-NaN bit pattern `0x7FC0DEADu` for Primitive Restart, implement full **2-Plane Sutherland-Hodgman Clipping (`z >= 0` & `w - z >= 0`)**, guard `isnan`/`isinf` + `clamp` before `int(floor(...))`, and reserve `+900K verts / +390K tris` clip tail in `sm_ensure_geom_capacity`.
7. **Uber-Batching Correctness Hazards (`v0.7.0`)**:
   - *Traps*: (a) Batching draws whose FS uses `discard;` or writes `gl_FragDepth` breaks single-visibility-entry resolution; (b) `vkCmdClearAttachments` mid-RenderPass must flush any open draw batch first; (c) `K=1` Indirect draws must NEVER trigger the `small_tri_coop` (`<=8` tris) direct fast-path because GPU indirect `tri_count` is unknown on CPU!
   - *Fix*: Reflect `fs_uses_push_constants` and `fs_has_discard_or_depth` in `spirv_lower.cpp`; call `sm_flush_draw_batch(scb)` in `sm_CmdClearAttachments`; guard `is_single_small = (!B.is_indirect && K == 1u && B.draws[0].total_tris <= 8u)`.
8. **Discard/Demote Immunity for Self-Clearing `VisBuf`**:
   - *Fix*: Clear `sm_vis_px[pidx] = 0xFFFFFFFFFFFFFFFFul` **immediately after reading `vis_val`** at the top of lowered FS and `smrast_depth_only.comp`, *before* any user fragment code runs.
9. **sRGB Blend Readback & Half-Float Dynamic NaN Safety (`v0.7.0`)**:
   - *Fix*: Decode `sm_srgb_to_lin()` before blending onto sRGB RTs and clamp dynamic `fc == 7u/19u` (`R16G16B16A16_SFLOAT` / `B10G11R11_UFLOAT`) via `sm_safe_f16()`.
10. **Sampled Stencil Aspect SRVs & Read-Only Stencil Sync (`v0.5 / v0.7`)**:
    - *Fix*: Maintain a companion `R32_UINT` shadow `VkImage` preserving `VkComponentMapping` swizzle, and track `stn_can_write(front) || stn_can_write(back)` so read-only stencil tests never mark `pass_wrote_stn = true` (eliminating 30+ redundant image syncs/frame).
11. **Ring Buffer Sizing & Dangling Buffer Handles**:
    - *Fix*: Keep `49,152` draw slots / `16,384` indirect slots / `4,096` clear-blit slots; zero `vtx_bufs`/`idx_buf` in `sm_BeginCommandBuffer`; invalidate DS caches on `vkDestroyBuffer`/`vkDestroyImage`.
12. **SPIR-V Lowering Stability (`spirv_lower.cpp`)**:
    - Thread-local `libshaderc` + atomic write-and-rename disk cache, `#extension` hoisting for `vkd3d-proton`, and `-O0` fallback on pathological shaders.

### 5.3 Performance Optimization Lessons (`v0.5 → v0.7`)
13. **Always Profile GPU Timestamps Before Optimizing**:
    - `SmXrayGpuScope` (`vkCmdWriteTimestamp`) immediately pinpointed that 2 single-thread giant-triangle indirect draws were costing `6.33 ms` (almost half of total GPU frame time!) while 1M-tri micro-triangle drawing cost only `2.46 ms`.
14. **Eliminate Synchronous Host-GPU Bubbles in WSI**:
    - Moving readback to `HOST_CACHED` memory + 32-bit R↔B swizzle + Xlib `XShmPutImage` + a 2-slot async present ring (`present_cmds[2]`, `present_fences[2]` for `>=3` swapchain images) shrank present overhead by `~80×` (`76–99 ms → 0.5–1.3 ms`) and unlocked full CPU/GPU frame overlap.
15. **Hybrid Dispatch Heuristics**:
    - `<=2` tris without depth → **Fused Quad FS** (no Fine, no VisBuf).
    - `<=8` direct tris with depth → **Coop Small-Tri Binner** (0 global atomics).
    - `<64K` tris & indirect → **Active-Tile Indirect Dispatch**.
    - `>=64K` tris → **Coherent-2D Direct Grid** (preserves L2/TMU 2D spatial locality).
    - `>32` tiles/tri → **64-Way Warp-Cooperative Binning** inside `smrast_bin.comp`.

---

## 6. Version History & Verified H20 Baseline

| Version | Major Milestone | Verified H20 Metrics (`qf[2]`, 78 SMs) |
|---|---|---|
| `v0.1 – v0.2` | Initial Vulkan Compute Rasterizer ICD + SPIR-V lowerer + CuRast micro-tri + CUDARaster tile binning | 1M-tri 1080p PBR **470–680 FPS** vs CPU Lavapipe **10 FPS** (`PSNR = 46.54 dB`); `vkcube` 60/60 via MIT-SHM WSI |
| `v0.3.0` | Commercial DX11 3A bring-up (DXVK 3.1.1), `smrast_depth_only.comp`, `smrast_blit.comp`, Vulkan 1.4 emulation | 60+ commercial DX11 pipelines 0 failures; 1,260+ continuous frames |
| `v0.4.0` | DX12 (`vkd3d-proton 3.1.0`, SM 6.6) end-to-end, 8-MRT × 21 formats, 8-bit Stencil, `VK_KHR_push_descriptor`, `DrawIndirectCount`, 49K draw slots | `DX12_VKD3D_TEST PASS`, `DEFERRED_MRT_PIXEL PASS`; 230+ commercial DX11 & 20+ commercial DX12 pipelines 0 failures |
| `v0.5.0` | Sampled Stencil Aspect SRVs, watertight Top-Left + 4-bit snapping, Depth Bias / Depth Clamp / Rasterizer Discard, 2-Level Tile Hi-Z, `vertex_attribute_divisor` | `V05_FEATURE_TEST PASS` (`2304/2304` watertight pixels); 1M PBR `477–681 FPS` |
| `v0.6.0` | Hybrid Active-Tile Indirect Dispatch, Self-Clearing `VisBuf`/`tile_counts`, Zero-Update Dynamic SSBO DS Cache, Persistent Pre-Binning Coarse Hi-Z, Primitive Restart, Lines/Points | `V06_FEATURE_TEST PASS`; 1,000-Draw 1080p `56.72–61.07 ms` (`16–18 FPS`); AAA 3D scene `3.71 FPS` |
| **`v0.7.0`** | **Direct (≤256) & Indirect (≤128) Multi-Draw Uber-Batching, 64-Way Warp-Coop Big-Tri Binning, Fused Quad FS, DMA Fast-Clear, Lazy Attachment Sync + `stn_can_write`, 1-Frame Pipelined MIT-SHM Present Ring, `gl_FragCoord.w` & 2-Plane Clipping Spec Fixes, `vkCmdWriteTimestamp` X-Ray Profiler** | **`V07_FEATURE_TEST PASS`**; **1,000-Draw 1080p Uber-Batch: `2.84 ms` Direct (`352.7 FPS`, `21.5×` vs v0.6) / `5.03 ms` Indirect (`198.7 FPS`)**; **1M-Tri 1080p PBR: `1.07 ms` (`934.4 FPS`, `930.1 Mtris/s`)**; **Commercial AAA 3D 1M-Tri Scene: `58.8–70.1 FPS` (`490×–584×` vs 48-Core CPU Lavapipe `0.12 FPS`)**; **DoF Intro: `283.2 FPS` (`31.8×` vs v0.6)**; **3D Menu: `57.2 FPS`**; all Vulkan/DX11/DX12 tests PASS |

---

## 7. Technical Roadmap (`v0.8.0+`)

### Phase 1 — `v0.8.0`: Per-Draw Push-Constant / Dynamic-UBO Uber-Batching (Target: `100+ FPS` Heavy Open-World)
1. **Per-Draw Metadata SSBO (`draw_id` in `VisBuf` / `TriVerts`)**: Extend uber-batching to coalesce draws that differ in Push Constants or Dynamic UBO offsets (e.g., per-object transforms/materials in unbatched 1,000-draw stress `test_v06_features`: target `39.8 ms → <4 ms`).
2. **Eliminate Remaining `SyncA2I` Overhead on Transient Render Targets**: Keep internal transient G-Buffer / bloom textures in native SmRast linear/tiled storage buffers when only read by lowered shaders, skipping `vkCmdCopyBufferToImage` entirely where possible.

### Phase 2 — `v0.8.x`: Full Commercial DX12 3A Presentation & Modern Vulkan Coverage
1. **Out-of-Order Timeline Semaphore Scheduler**: Handle `vkd3d-proton`'s cross-thread wait-before-signal `vkQueueSubmit2` patterns cleanly on multiplexed `qf[2]` without blocking the Present thread.
2. **`VK_EXT_descriptor_buffer` & BDA Support**: Lower buffer-device-address descriptor reads or safely advertise optimal fallback profile for `vkd3d-proton`.
3. **Compute-Emulated Geometry & Tessellation Stages + 4× MSAA Resolve**: Unblock engines relying on GS particle expansion / hull-domain terrain tessellation and native MSAA render targets (`AlphaToCoverage`, `SampleMask`).

### Phase 3 — `v0.9.0`: Zero-Copy Streaming & Cross-Vendor Compute GPUs
1. **Zero-Copy HBM3 → NVENC / DMA-BUF Output**: Bypass X11 MIT-SHM CPU readback completely for low-latency cloud game streaming and high-throughput RGB capture.
2. **AMD MI300X (`RADV` / `AMDVLK`, Wave64) Validation**: Audit subgroup quad shuffle masks, 64-way warp-cooperative bin assumptions, and LDS tile layouts for Wave64 parity on CDNA3 silicon.
