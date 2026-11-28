# SmRast-VK: Architecture, Engineering Lessons & Technical Roadmap

> **Living technical memory** for **SmRast-VK** — a pure-Compute **Vulkan 1.3 (+ emulated Vulkan 1.4)** Software Rasterizer Wrapper ICD (`libvulkan_smrast.so`) designed for compute-only datacenter GPUs (**NVIDIA H20 / H100**, **AMD MI300X**).
>
> *Last updated: `v1.1.0`.*

---

## 1. Core Thesis & Hardware Motivation

Datacenter Hopper GPUs (**NVIDIA H20 / H100**) and **AMD MI300X** disable or lack functioning hardware graphics/ROP pipelines (graphics queue submissions hang DMA on H20), forcing legacy graphics stacks onto CPU software rasterization (Mesa Lavapipe / `llvmpipe`, typically 0.1–10 FPS on complex 3D scenes while saturating dozens of CPU cores).

However, a compute-only H20 retains massive graphics-capable silicon on its pure Compute Queue (`VK_QUEUE_COMPUTE_BIT`, `qf[2]`):
- **78 Hopper SMs** (`FP32` / `INT32` / `FP16`)
- **96 GB HBM3** (~4 TB/s bandwidth)
- **Hardware 64-bit buffer atomics** (`VK_KHR_shader_atomic_int64`)
- **1-cycle Subgroup Quad Shuffles** (`VK_KHR_shader_subgroup_quad`)
- **Functional Hardware Texture Mapping Units (TMUs)** supporting `textureGrad()` anisotropic / mipmapped sampling inside Compute Shaders — including native `VK_IMAGE_TILING_LINEAR | VK_IMAGE_CREATE_ALIAS_BIT` sampling across all 14 standard single-mip color formats (`offset=0`, 32B-aligned `rowPitch`). *(Hardware probe confirms NVIDIA H20 returns `linMipMax=1` for color and `linMipMax=0` for `D32F`/`D32S8`.)*

**SmRast-VK's Hybrid Principle**:
1. **Pass through everything silicon already does natively**: Compute Shaders (`vkCmdDispatch`), HBM3 allocations, DMA transfers (`vkCmdFillBuffer` / `vkCmdCopyBuffer`), hardware TMU sampling (zero-copy over linear-aliased G-Buffers), Timeline Semaphores, and Sync2 events run natively on `qf[2]`.
2. **Replace ONLY the missing graphics rasterizer & ROP fixed-function pipeline**: Intercept `VkPipeline` (graphics) creation and `vkCmdDraw*`, lower arbitrary DX11/DX12/Vulkan VS & FS SPIR-V into Hopper Compute Shaders in-process, and execute a cross-material multi-draw uber-batched Compute Rasterizer directly on the GPU SMs.

---

## 2. Architecture & Reference Lineage (`v0.9.0`)

SmRast-VK fuses four classic GPU software rasterization architectures with a production Vulkan 1.3/1.4 ICD, live SPIR-V lowering compiler, zero-copy linear-aliased G-Buffers, and cross-material GPU multi-draw uber-batching:

| Classic Reference | Technique Adapted | SmRast-VK v0.9 Implementation |
|---|---|---|
| **CUDARaster** (Laine & Karras, NVIDIA Research) | Homogeneous guard-band clipping, exact triangle-vs-tile edge trivial-reject, coarse `16×16` tile binning, watertight Top-Left fill rule | `shaders/smrast_bin.comp` + `v0.9` **7-Plane Homogeneous Guard-Band Clipping** (`w>=1e-4`, `z>=0`, `w-z>=0`, `±4w` XY) + **`VK_EXT_depth_clip_enable`**, **Exact 3-Edge Tile Trivial-Reject**, **64-Way Warp-Cooperative Big-Tri Binning**, **Pre-Binning Coarse Hi-Z**, & **GPU Active-Tile Compaction** |
| **CuRast / compute_rasterizer** (Schütz / UE5 Nanite) | Lock-free 64-bit `atomicMin(depth << 32 \| tri_id)` direct visibility-buffer fast-path for micro-triangles | `shaders/smrast_bin.comp` **`≤128 px` Branchless Precomputed-Edge Micro-Tri Fast-Path** with deterministic **Vulkan Painter's Order** `(uint64_t(~tid) << 32) \| tid`, driving **`1,299.0 FPS` / `1.293 Gtris/s`** on 1M-tri 1080p PBR, with `discard;`-safe self-clearing `VisBuf` |
| **cuRE & RetroWarp** (Kenzel / Themaister) | 100% pure Vulkan Compute graphics pipeline (zero ROPs), `16×16` shared-memory Tile Fine rasterizer, subgroup quad derivatives | `shaders/smrast_fine.comp` (**Loader-Time Hi-Z Compaction + Precomputed Branchless Canonical Edges + Warp-Uniform Y-Band Skip + Intra-Tile Early-Z + Inter-Batch Live Hi-Z Tightening**) + `src/spirv_lower.cpp` (**Zero-Bin Inline-Setup Fused Quad FS** with inline 7-plane clip + barycentric remap, **Dynamic Compact `vert_stride_vec4`**, Per-Draw Push-Constant / Dynamic-UBO SSBO unpacking, `flat`/`noperspective`, branchless `subgroupQuadSwap` `dFdx`/`dFdy`) |
| **Original ICD, Uber-Batcher & Compiler Layer** | Drop-in Khronos ICD + in-process SPIR-V → Compute compiler + Cross-Material Uber-Batching + Zero-Copy Linear-Aliased G-Buffers + AVX2 Overlapped MIT-SHM WSI | `src/smrast_icd.cpp`, `src/spirv_lower.cpp`, `src/smrast_raster_engine.cpp`, `src/smrast_wsi.cpp` |

### 2.1 End-to-End Cross-Material Multi-Draw Compute Pipeline (`v0.9.0`)

```
RenderPass Begin (LOAD_OP_CLEAR / LOAD_OP_LOAD / LOAD_OP_DONT_CARE)
  ├─ LOAD_OP_CLEAR / DONT_CARE immediately cancels stale attach_pending_sync
  └─ [0] Hardware DMA Fast-Clear (vkCmdFillBuffer)  [fallback: smrast_clear.comp w/ rt_stride_px]
         • Empty-clear tracking (real_uniform_cleared) skips repeated 0-draw shadow/RT SyncA2I!
         • VisBuf & tile_counts remain invariant-clean via self-clearing (skips ~75% of RP clears)

Cross-Material Multi-Draw Uber-Batcher (src/smrast_icd.cpp):
  • Direct Batch:   coalesces up to 256 consecutive draws — INCLUDING draws with differing
                    FS Push Constants, Dynamic UBO offsets, or no-op identity blends (ONE/ZERO/ADD)!
  • Indirect Batch: coalesces up to 128 consecutive vkCmdDraw[Indexed]Indirect[Count] draws.
  • Zero-Bin Inline Fused Quad FS (direct_2d == 3u): for <=2-tri draws without depth test/write,
                    COMPLETELY SKIPS smrast_bin.comp, smrast_fine.comp, VisBuf, and 2 GPU barriers!

Per Flushed Batch / Single Draw:
  ├─ [0b] Zero-DMA smrast_indirect_prep.comp  (indirect draws only)
  │       Single indirect draws zero clip-tail & active-tile headers directly on-GPU (0 DMA updates!).
  ├─ [1] K Barrier-Free Parallel Lowered Compute Vertex Shaders  (src/spirv_lower.cpp)
  │       • Dynamic compact vert_stride_vec4 (2..16 vec4 reflected per pipeline -> 3x-5x less HBM3!)
  │       • Up to 8 VBs + VK_EXT_vertex_attribute_divisor + unaligned R16_* + 0x7FC0DEAD restart
  ├─ [2] K Parallel smrast_bin.comp Dispatches  (SKIPPED on Zero-Bin Inline Fused Quad FS!)
  │       • 7-Plane CUDARaster Homogeneous Guard-Band Clipping (±4w XY + w>=1e-4 + z∈[0,w])
  │       • VK_EXT_depth_clip_enable decoupled from depthClampEnable + Hardware Depth Bias
  │       • 3-Tier Work Distribution:
  │           1) Micro-tris (bbox <= 128 px): branchless precomputed-edge CuRast 64-bit atomicMin
  │           2) Normal tris (<= 32 tiles): exact 3-edge tile trivial-reject + Hi-Z -> tile_bins[]
  │           3) Giant tris (> 32 tiles): 64-Way Warp-Cooperative binning w/ exact tile trivial-reject
  ├─ [3] 1× smrast_fine.comp  (SKIPPED on Fused Quad FS!)
  │       • Loader threads compact Hi-Z-passing tris & precompute branchless canonical edges ONCE
  │       • Warp-Uniform Y-Band Early-Out (subgroupMin/Max(fy), portable Warp32/Wave64)
  │       • Intra-Tile Early-Z (best_d) + Inter-Batch Live Hi-Z Tightening across 64-tri batches!
  │       • Deterministic Painter's Order tie-breaking; self-clears tile_counts/active
  ├─ [4] 1× Lowered Quad-Swizzled Compute FS  OR  smrast_depth_only.comp
  │       • Zero-Bin Inline Fused Quad FS (direct_2d == 3u): 1-2 threads run 7-plane clip + setup
  │         into shared sm_fq_tris[16] + barycentric remap u0/u1/u2 (0 Bin dispatch, 0 TriSetupBuf!)
  │       • Cross-Material SSBO Unpacking (bindings 15/16/17 indexed by batch_push_base + sm_draw_id)
  │       • flat & noperspective varyings, spec-exact gl_FragCoord.w = 1/w_clip, hardware TMUs
  │       • 8 MRTs × 21 RT formats w/ per-RT rt_stride_px[8] + 8-bit Stencil + unified end barrier
  └─ [5] 1× smrast_hiz_update.comp  (depth-writing draws only)

Zero-Copy Linear-Aliased Sync & AVX2 Overlapped MIT-SHM Present:
  ├─ 14 single-mip color formats alias real_img & attach_buf memory -> ZERO SyncA2I / SyncI2A!
  ├─ Empty-clear SyncA2I skip eliminates repeated 0-draw 2048x2048 shadow map transfers
  └─ src/smrast_wsi.cpp — Fused wait-semaphore DMA Submit + HOST_CACHED + runtime AVX2
                          _mm256_shuffle_epi8 R<->B swizzle overlapped with Xvfb XShmPutImage!
```

### 2.2 Key Internal Memory & Slot Capacities (`include/smrast_internal.h`)

| Resource | Capacity (`v1.1.0`) | Purpose |
|---|---:|---|
| Draw parameter UBO/SSBO ring | **131,072 slots** | Supports ultra-dense 3A open-world frames (`~14.6K` draws / `~215` render passes with `>8×` headroom, zero wrap) |
| Direct / Indirect Uber-Batch width | **256 direct / 128 indirect** (`254` stencil-write) | Collapses consecutive draws across differing FS Push Constants, Dynamic UBOs, no-op blends, & `KEEP`/`REPLACE` stencil writes |
| Per-Draw Metadata SSBO Rings | **65,536 × 128B Push (`8 MB`) + 65,536 × 4B DynOff (`256 KB`)** | Host-coherent per-draw FS Push-Constant & Dynamic-UBO offset rings indexed by `batch_push_base + sm_draw_id` |
| `ClipVertBuf` / `TriVerts` (HBM3) | **4.0M verts / 1.8M tris** (dynamic `2..16` vec4 stride!) | Up to `8×` higher L2 efficiency via reflected per-pipeline `vert_stride_vec4` |
| Indirect dispatch buffer (`ind_disp_buf`) | **32 KB / slot (`16,384` slots)** | Holds up to 128 indirect sub-draw headers + GPU atomic slice allocators |
| Clear / Blit slots | **4,096 slots** | High-pass-count deferred/bloom/shadow cascades |
| `tile_hiz` (per Depth Image) | **64 KB @ 1080p** | `120 × 68` tiles × 8 bytes (`(zmin, zmax)`), L2-resident |
| Descriptor-set cache pools | **16K sets / ring + `ind_prep_ds_cache`** | Zero-update `STORAGE_BUFFER_DYNAMIC` caching across all 18-binding internal DS layouts |

---

## 3. Why DX11 (`DXVK`) and DX12 (`vkd3d-proton`) Behave Differently on a Unified Vulkan ICD

SmRast-VK implements **one single Vulkan ICD** — there are no separate DX11 or DX12 rasterizers. The difference in bring-up difficulty comes from the **Vulkan API surface and SPIR-V complexity** emitted by the two Windows translation layers:

| Dimension | DX11 (`DXVK`) | DX12 (`vkd3d-proton`) | Engineering Impact on SmRast-VK |
|---|---|---|---|
| **Resource Binding** | Classic fixed-slot Descriptor Sets (bounded CBV/SRV/UAV tables) | Massive **Bindless arrays** (`nonuniformEXT`), `VK_EXT_mutable_descriptor_type`, `VK_KHR_push_descriptor`, and `VK_EXT_descriptor_buffer` (GPU pointers) | Required full emulation of mutable descriptors, descriptor indexing, push descriptors (incl. `WithTemplate`/`2KHR`), and dynamic SSBOs |
| **SPIR-V IR** | Translated from DXBC (SM 5.0); clean, regular GLSL-like IR | Translated from DXIL (SM 6.0–6.6); heavy `GL_EXT_buffer_reference` (raw 64-bit pointers), `scalar_block_layout`, 16-bit types, Wave/Subgroup ops, Root Constants | Required `#extension` hoisting above injected preambles in `spirv_lower.cpp` and careful SSBO lowering |
| **Queues & Sync** | Single immediate context; simple binary semaphores & fences | Multi-queue (Direct / Async Compute / Copy) + **out-of-order cross-thread Timeline Semaphore Wait/Signal** (`vkQueueSubmit2`) | Virtualizing multiple queues onto one physical `qf[2]` requires deadlock-free Timeline/Present scheduling |
| **Pipeline State** | Predominantly traditional `VkRenderPass` + `VkFramebuffer` | Heavy `VK_KHR_dynamic_rendering`, `VK_EXT_extended_dynamic_state1/2/3`, `VK_KHR_maintenance5/6`, `VK_EXT_graphics_pipeline_library`, `DrawIndirectCount` | Every missing dynamic state or maintenance feature blocks commercial DX12 startup |

---

## 4. `v0.6.0 → v0.9.0` GPU Profiling Case Study: `3.7 FPS → 108.6 FPS` & `1.293 Gtris/s`

Every single performance leap from `3.71 FPS` (`v0.6.0`) → `58.8–70.1 FPS` (`v0.7.0`) → `76.0–81.3 FPS` (`v0.8.0`) → **`85.5–108.6 FPS` (`v0.9.0`)** was guided by live `vkCmdWriteTimestamp` (`SmXrayGpuScope`) profiling:

| Stage / Bottleneck Diagnosed on Live 1M-Tri 3D Scene | Before | Fix Implemented | After (`v0.9.0`) |
|---|---|---|---|
| **Hot Horizon Tile Serialization in `smrast_fine.comp`** (1M-tri instanced buildings) | `1.31–3.61 ms` (`v0.8.0`, #1 GPU bottleneck!) | **Loader-Time Hi-Z Compaction + Precomputed Branchless Canonical Edges + Warp-Uniform Y-Skip + Intra-Tile Early-Z + Inter-Batch Live Hi-Z + `128 px` CuRast + Exact Tile-Edge Trivial-Reject** | **`512`-layer overdraw = `1.014 ms`; 1M PBR = `0.77 ms` (`1,299 FPS`)** |
| **`ClipVertBuf` Fixed 256B/Vert Stride (`770 MB`/draw!)** | `VS = 0.77 ms` (`v0.8.0`) | **Dynamic Per-Pipeline `vert_stride_vec4` (`2..16` vec4)** | **`3×–5×` less HBM3 traffic** across VS, Bin & FS |
| **48 Post-Process / Mipmap Quads (`3` dispatches + `3` barriers each)** | `~18 ms` (`v0.6`) → `2.02 ms` (`v0.8`) | **Zero-Bin Inline-Setup `Fused Quad FS` (`direct_2d == 3u`)** + unified end barrier | **48 Bin dispatches + 48+ barriers eliminated!** |
| **36 Menu Indirect Draws (`2` DMA updates + `7` barriers each)** | `4.36 ms` (`v0.8.0`) | **Zero-DMA `smrast_indirect_prep.comp` + unified end barrier** | **Menu jumped `67–71 FPS → 76–80 FPS` (+16%)** |
| **52/83 Redundant `SyncA2I` Copies + Empty-Clear Shadow Maps** | `SyncA2I = 83/f` (`v0.7`) → `31/f` (`v0.8`) | **Linear-Aliased Color G-Buffers (`14` formats) + `real_uniform_cleared` Empty-Clear Skip** | **`SyncA2I = 23–28/f` (`54–55` skipped/frame!)** |
| **1,000-Draw Per-Draw Push-Constant / Dynamic-UBO Draws** | `42.53 ms` (`v0.7.0`) | **Per-Draw Push-Constant & Dynamic-UBO Metadata SSBO Uber-Batching** | **`2.35–2.75 ms` (`364–426 FPS`, `15.5×` speedup)** |
| **WSI Readback, Swizzle & X11 Present** | `76–99 ms` (`v0.6`) → `2.4 ms` (`v0.7`) | **Fused Wait-Sem DMA Ring + Runtime AVX2 `_mm256_shuffle_epi8` + Overlapped Ping-Pong MIT-SHM** | **`0.5–1.3 ms`**, Intro **`324.7 FPS`** |
| **Giant White Blocks & Screen-Wide Flickering** | `diff_max = 4.9` (`v0.7.0`) | **7-Plane CUDARaster Guard-Band Clipping (`±4w`) + `VK_EXT_depth_clip_enable` + Painter's Order** | **`diff_max = 0.40–1.45` (`0` spikes > 3.0), ZERO white blocks** |
| **End-to-End Live Commercial 3D 1M-Tri Scene** (shared H20) | **`3.71 FPS`** (`v0.6.0`) | All `v0.7–v0.9` optimizations combined | **`85.5–108.6 FPS` (`9.2–11.7 ms`, up to `905×` vs CPU Lavapipe)** |

---

## 5. Hard-Won Engineering Lessons & Pitfall Log (`v0.1 → v0.9.0`)

1. **Compute UBO Constant-Bank Aliasing at `DescriptorSet >= 5`**: Permanently use `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` for internal parameter blocks.
2. **H20 Graphics Queue DMA Hang (`Xid 69 / 109`)**: Strictly route all work to pure Compute `qf[2]` while exposing virtual `qf[0]`.
3. **H20 `VK_IMAGE_TILING_LINEAR` Capabilities (`v0.8 / v0.9`)**: Single-mip 2D color attachments (`14` formats) natively support `LINEAR | ALIAS_BIT | MUTABLE_FORMAT` (`offset=0`, 32B-aligned `rowPitch`), whereas mipmapped images (`linMipMax=1`) and depth formats (`linMipMax=0`) require `OPTIMAL` tiling. Always propagate per-RT `rt_stride_px[8] = rowPitch / bpp`.
4. **Always Probe Silicon & Profile Timestamps Before Optimizing (`v0.9.0`)**: A 5-second H20 `vkGetPhysicalDeviceImageFormatProperties` probe proved mipmapped/depth linear aliasing is unsupported in hardware, while `SMRAST_XRAY=1` immediately revealed `smrast_fine.comp` (`1.31–3.61 ms`) and `ClipVertBuf` `16`-vec4 stride (`770 MB`/draw) as the true bottlenecks — unlocking `1,299 FPS` PBR and `108.6 FPS` live 3D!
5. **Inline `Fused Quad FS` Must Still Support 7-Plane Clipping (`v0.9.0`)**: Depth-less 2-triangle draws (`!dtest && !dwrite`) can still contain 3D near-plane/negative-`w` vertices (caught immediately by `test_v07_features [1]`). Clipping `poly_pos[10]` + original-vertex barycentric weights `poly_u[10]` in shared memory gives 100% spec-exact 7-plane clipping with zero `ClipVertBuf` writes and zero `smrast_bin.comp` dispatches!
6. **IEEE-754 Bit-Exact Branchless Edge Equations (`v0.9.0`)**: Precomputing canonical `vec4(ax, ay, sgn*dx, sgn*dy)` in loader threads yields `e.z*(fy - e.y) - e.w*(fx - e.x)` which is IEEE-754 bit-identical to branching `edge_eval()` while removing 3 branches + 3 vector subtractions from every inner pixel iteration (preserving `6400/6400` and `2304/2304` watertightness).
7. **Overlapped Double-Buffered MIT-SHM Ping-Pong (`v0.9.0`)**: Because `shm_ptrs[sidx]` belonged to Frame `N-2` (already synchronized during Frame `N-1`), writing the AVX2 R↔B swizzle into `shm_ptrs[sidx]` *before* `XSync` overlaps CPU swizzle 100% with Xvfb's `XShmPutImage` compositing of Frame `N-1`.
8. **Giant White Blocks, Painter's Order, `gl_FragCoord.w`, Self-Clearing `VisBuf` & Lifecycle Safety (`v0.5–v0.8`)**: Always preserve 7-plane `±4w` guard-band clipping + `VK_EXT_depth_clip_enable`, `(uint64_t(~tid) << 32) | tid` Painter's order, `gl_FragCoord.w = 1/w_clip`, `0x7FC0DEAD` NaN restart, immediate `VisBuf` self-clear before user FS code, local-copy iteration before `sm_flush_pending_syncs`, and explicit `vkDeviceWaitIdle` teardown.
9. **Cross-Block `OpSampledImage` & Un-Inlined `-O0` Bindless SPIR-V Crashes NVIDIA `libnvidia-glvkspirv.so` (`v1.0.0`)**: When `SPIRV-Cross` emits conditional/ternary expressions inside `texture(sampler2D(tex, smp), cond ? a : b)` arguments, `glslang` emits `OpSampledImage` in the pre-branch block and `OpImageSample*` after the merge block, failing `shaderc -O` validation (`VUID-StandaloneSpirv-OpSampledImage-04651`). Falling back to raw `-O0` passes un-inlined `OpFunctionCall`s over `RuntimeDescriptorArray` straight into NVIDIA 535's `_nv002nvvm`, triggering a deterministic `SIGSEGV` (`si_addr=0x4`) across 5 distinct DX12 engines. Sinking every cross-block `OpSampledImage` right before its consumer(s) (`sm_sink_op_sampled_image()`) and running `spirv-opt -O` post-sink achieves **0/861 `spirv-val` errors** and **0 crashes**.
10. **Sparse-Binding Feature/Queue Parity & 8-Queue Hardware Multiplexing (`v1.0.0`)**: Forwarding `features.sparseBinding = VK_TRUE` without exposing `VK_QUEUE_SPARSE_BINDING_BIT` causes `vkd3d-proton` to leave its sparse queue pointer null and crash on `UpdateTileMappings`, while collapsing Direct/Compute/Copy/Upload/Sparse queues onto 1 physical queue causes cross-queue Timeline Semaphore deadlocks. Exposing all 8 independent hardware queues on H20 `qf[2]` (`flags = COMPUTE | TRANSFER | SPARSE_BINDING`) eliminates both classes of failure.
11. **`VK_KHR_dedicated_allocation` Unwrapping & 64-Bit `VkAccessFlags2` Preservation (`v1.0.0`)**: Advertising `VK_KHR_dedicated_allocation` requires unwrapping `VkMemoryDedicatedAllocateInfo` handles (`SmImage*`/`SmBuffer*` → real backend handles) and explicitly omitting image-dedicated allocation on `linear_aliased` RTs (whose `VkDeviceMemory` is bound to both `real_img` and `attach_buf`). Similarly, `sm_CmdPipelineBarrier2` must preserve upper-32-bit `VkAccessFlags2` (`SHADER_SAMPLED_READ_BIT`, `SHADER_STORAGE_READ/WRITE_BIT`) rather than truncating to 32-bit `VkAccessFlags`.
12. **Thread Cancellation Safety Around C++ Compiler Libraries (`v1.0.0`)**: Wine thread teardown (`pthread_cancel` → `__cxxabiv1::__forced_unwind`) firing inside `libspirv-cross-c-shared` or `libshaderc` C++ `catch (...)` blocks causes glibc to abort with `FATAL: exception not rethrown`. Guarding shader lowering and pipeline compilation with `pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, ...)` RAII guards defers cancellation cleanly until exit.
13. **Deferred Stencil `depthFailOp` & Null-FS Stencil Passes (`v1.1.0`)**: Modern deferred renderers use depth-failing far-plane sky quads or bounding light volumes (`depthFailOp = ZERO / INCR / DECR`) to stamp/clear stencil masks for subsequent albedo/lighting passes. Early-rejecting fragments in `smrast_bin.comp` / `smrast_fine.comp` when `depthFailOp != KEEP` skips stencil updates entirely, causing deferred albedo/lighting dropout. Tracking `needs_stencil_dfail` and executing the full Vulkan spec order (stencil test → depth test `dpass` → `failOp`/`depthFailOp`/`passOp`) in both lowered FS and `sm_build_null_fs_stencil_spv` fixes this cleanly.
14. **Lossless 31-Bit Reversed-Z Packing & 16-ULP Cross-Pass Depth Invariance (`v1.1.0`)**: Quantizing `depth` to `uint(d * 4294967295.0)` destroys sub-`1e-6` exponent precision required by Reversed-Z far geometry (`z ≈ 0.0005`), while independent VS shaders between a Depth-PrePass and a G-Buffer pass can drift by 1–8 ULPs due to cross-shader FMA contraction, producing 1-pixel holes on `GREATER_OR_EQUAL` tests. Switching `VisBuf` packing to lossless 31-bit IEEE-754 order-preserving bits `0x3F800000u - (floatBitsToUint(d) & 0x7FFFFFFFu)` (with NaN/range guards) plus a **16-ULP (`~1.9e-6` relative)** cross-pass invariance band on `EQUAL`/`LEQUAL`/`GEQUAL` eliminates 100% of static-geometry speckles without harming depth ordering.
15. **Small-RT (`<64 px`) Linear-Tiling Bloom Corruption on H20 (`v1.1.0`)**: On NVIDIA H20, aliasing `VK_IMAGE_TILING_LINEAR` buffers onto tiny (`40×22` `RGBA16F`) bloom pyramid render targets caused TMU linear sampling in the subsequent horizontal blur pass to read `-65504` (`-HALF_MAX`), exploding into saturated color blocks across the full bloom pyramid. Requiring `width >= 64 && height >= 64` (plus verifying `sl.rowPitch * H <= imr.size`) in `sm_can_linear_alias()` keeps zero-copy linear aliasing active on all full-size G-Buffers while keeping small mip/bloom chains 100% clean.
16. **Dual-Source Blending, `VK_FORMAT_R8_SNORM` & `131K` UBO Slot Headroom (`v1.1.0`)**: Volumetric fog/transmittance passes rely on dual-source blending (`location=0, index=1` + `ONE_MINUS_SRC1_ALPHA`), while normal/material caches use `VK_FORMAT_R8_SNORM` and dense open-world frames exceed `14.5K` draws/frame across 215+ render passes — requiring `SpvDecorationIndex` reflection, format code `21u`, `131,072` UBO ring slots, and `KEEP`/`REPLACE` stencil-write uber-batching (`max_k=254`).

---

## 6. Version History & Verified H20 Baseline

| Version | Major Milestone | Verified H20 Metrics (`qf[2]`, 78 SMs) |
|---|---|---|
| `v0.1 – v0.2` | Initial Vulkan Compute Rasterizer ICD + SPIR-V lowerer + CuRast micro-tri + CUDARaster tile binning | 1M-tri 1080p PBR **470–680 FPS** vs CPU Lavapipe **10 FPS** (`PSNR = 46.54 dB`); `vkcube` 60/60 |
| `v0.3.0` | Commercial DX11 3A bring-up (DXVK 3.1.1), `smrast_depth_only.comp`, `smrast_blit.comp`, Vulkan 1.4 emulation | 60+ commercial DX11 pipelines 0 failures; 1,260+ continuous frames |
| `v0.4.0` | DX12 (`vkd3d-proton 3.1.0`, SM 6.6) end-to-end, 8-MRT × 21 formats, 8-bit Stencil, `VK_KHR_push_descriptor`, `DrawIndirectCount`, 49K draw slots | `DX12_VKD3D_TEST PASS`, `DEFERRED_MRT_PIXEL PASS`; 230+ DX11 & 20+ DX12 pipelines 0 failures |
| `v0.5.0` | Sampled Stencil Aspect SRVs, watertight Top-Left + 4-bit snapping, Depth Bias / Depth Clamp / Rasterizer Discard, 2-Level Tile Hi-Z, `vertex_attribute_divisor` | `V05_FEATURE_TEST PASS` (`2304/2304` watertight); 1M PBR `477–681 FPS` |
| `v0.6.0` | Hybrid Active-Tile Indirect Dispatch, Self-Clearing `VisBuf`/`tile_counts`, Zero-Update Dynamic SSBO DS Cache, Persistent Pre-Binning Coarse Hi-Z, Primitive Restart, Lines/Points | `V06_FEATURE_TEST PASS`; 1,000-Draw 1080p `56.72–61.07 ms` (`16–18 FPS`); AAA 3D scene `3.71 FPS` |
| `v0.7.0` | Direct (≤256) & Indirect (≤128) Multi-Draw Uber-Batching, 64-Way Warp-Coop Big-Tri Binning, Fused Quad FS, DMA Fast-Clear, Lazy Attachment Sync + `stn_can_write`, Pipelined MIT-SHM, `vkCmdWriteTimestamp` X-Ray | `V07_FEATURE_TEST PASS`; 1,000-Draw Uber-Batch `2.84 ms` Direct / `5.03 ms` Indirect; 1M PBR `934.4 FPS`; AAA 3D Scene `58.8–70.1 FPS` |
| `v0.8.0` | 7-Plane CUDARaster Guard-Band Clipping (`±4w`) + `VK_EXT_depth_clip_enable`, Painter's Order, `flat`/`noperspective`, Per-Draw Push-Constant & Dynamic-UBO SSBO Uber-Batching, Zero-Copy Linear-Aliased Color G-Buffers (`14` formats), Double-Buffered MIT-SHM | `V08_FEATURE_TEST PASS`; 1,000-Draw Push-Constant/Dyn-UBO `2.54–2.92 ms` (`342–394 FPS`); AAA 3D Scene `76.0–81.3 FPS` (`0` flicker spikes, `0` white blocks); 3D Menu `67.6–71.1 FPS` |
| `v0.9.0` | Precomputed Branchless `smrast_fine.comp` (Loader-Time Hi-Z Compaction + Warp-Uniform Y-Skip + Intra-Tile Early-Z + Live Hi-Z), Exact 3-Edge Tile Trivial-Reject + `128 px` CuRast, Dynamic `vert_stride_vec4` (`2..16`), Zero-Bin Inline Fused Quad FS, Zero-DMA `smrast_indirect_prep.comp`, Empty-Clear `SyncA2I` Skip, Runtime AVX2 Overlapped MIT-SHM | `V09_FEATURE_TEST PASS`; 1M-Tri 1080p PBR `0.77 ms` (`1,299.0 FPS`, `1.293 Gtris/s`); 1,000-Draw Dyn-UBO `2.35 ms` (`426.1 FPS`); 512-Layer Overdraw `0.536–1.014 ms`; AAA 3D Scene `85.5–108.6 FPS` (`905×` vs CPU) |
| `v1.0.0` | Multi-Engine DX11/DX12 Production Hardening: Binary SPIR-V `OpSampledImage` Basic-Block Sinker + Post-Sink `spirv-opt -O`, Bare `gl_ClipDistance[8]`/`gl_CullDistance[8]`/`sampler*ArrayShadow` Lowering, 8-Queue Hardware Pool + Native `vkQueueBindSparse` + Cross-Queue Timeline Semaphores, `VK_KHR_dedicated_allocation` Unwrapping + `linear_aliased` Req Parity, 64-Bit `VkAccessFlags2` & VK1.3 Layouts, Hardware `SmQueryPool` GPU Timestamps, Multi-Thread `res_mtx` + `pthread_setcancelstate` Guards | `V10_FEATURE_TEST PASS` (12/12 suites PASS, 0 warnings); `861/861` lowered SPVs `spirv-val` clean; 8,000+ commercial DX12/DX11 pipelines compiled with 0 failures & 0 crashes across 10+ engines; live DX11 (`85–108.6 FPS` & `258–297 FPS`) & live DX12 (`30–60 FPS` VSync-capped & `287 FPS` uncapped) |
| **`v1.1.0`** | **Deferred Rendering & Precision Hardening: Dual-Source Blending (`location=0, index=1` + `SRC1_*`), Full Vulkan-Spec Stencil `depthFailOp` (`needs_stencil_dfail` + null-FS stencil), Lossless 31-Bit Reversed-Z + 16-ULP Cross-Pass Depth Invariance + NaN Guard, Small-RT (`<64 px`) Linear-Aliasing Guard, `VK_FORMAT_R8_SNORM`, Stencil-Write Uber-Batching (`max_k=254`), `131,072` UBO Ring Slots, Push-Descriptor `ever_bound_in_ds`, & Discard Hi-Z Skip** | **`V11_FEATURE_TEST PASS` (13/13 suites PASS, `0` warnings)**; **1M-tri 1080p PBR `1,276.8 FPS` (`0.78 ms`)**; **1,000-Draw Push-Constant `2.44 ms` (`409.4 FPS`)**; **live 14,500-draw / 215-pass commercial deferred 3D open-world DX11 gameplay (`4.6–12.9 FPS` @ 720p pure-compute H20, zero Depth-PrePass speckles, zero bloom corruption, zero flicker)** |

---

## 7. Technical Roadmap (`v1.2.0+`)

1. **Advanced DX12 Stages & Descriptor Buffer**: `VK_EXT_descriptor_buffer` / BDA lowering, compute-emulated Geometry, Hull/Domain Tessellation & Mesh Shader stages, and 4× MSAA resolve.
2. **Vertex-Reuse Indexed Shading Cache (`VS` Deduplication)**: Shade unique index-buffer vertices once per meshlet/draw instead of per-index expansion (`3×–5×` fewer VS invocations on high-poly instanced meshes).
3. **Zero-Copy Streaming & Cross-Vendor Compute GPUs**: Direct HBM3 → NVENC / DMA-BUF hardware video encoding bridge and Wave64 verification on AMD MI300X (`RADV` / `AMDVLK`).
