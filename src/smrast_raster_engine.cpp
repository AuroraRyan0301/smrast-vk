// smrast_raster_engine.cpp — SmRast-VK v0.3 HBM3 Rasterizer Buffers & Compute Pipelines
// Supports: MRT (4x Color + HDR bpp), GPU-Driven DrawIndirect Prep, On-Demand Geometry Buffer Growth (<15ms init!)
#include "smrast_internal.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>

static bool load_spv_file(const char* path, std::vector<uint32_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        fprintf(stderr, "[SmRast] ERROR: cannot open shader %s\n", path);
        return false;
    }
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    out.resize(sz / 4);
    f.read((char*)out.data(), sz);
    return true;
}

uint32_t sm_find_mem_type(SmPhysicalDevice* pd, uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < pd->mem_props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && ((pd->mem_props.memoryTypes[i].propertyFlags & req) == req))
            return i;
    }
    for (uint32_t i = 0; i < pd->mem_props.memoryTypeCount; ++i) {
        if (bits & (1u << i)) return i;
    }
    return 0;
}

static bool create_buf(SmDevice* dev, VkDeviceSize sz, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags mem_props, VkBuffer& buf, VkDeviceMemory& mem, void** mapped = nullptr) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, VK_SHARING_MODE_EXCLUSIVE};
    if (dev->fn.CreateBuffer(dev->real_dev, &bci, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{};
    dev->fn.GetBufferMemoryRequirements(dev->real_dev, buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                             sm_find_mem_type(dev->pd, mr.memoryTypeBits, mem_props)};
    if (dev->fn.AllocateMemory(dev->real_dev, &mai, nullptr, &mem) != VK_SUCCESS) return false;
    if (dev->fn.BindBufferMemory(dev->real_dev, buf, mem, 0) != VK_SUCCESS) return false;
    if (mapped) {
        if (dev->fn.MapMemory(dev->real_dev, mem, 0, sz, 0, mapped) != VK_SUCCESS) return false;
    }
    return true;
}

bool sm_ensure_image_attach_buf(SmDevice* dev, SmImage* img) {
    if (!img) return false;
    std::lock_guard<std::recursive_mutex> rlk(dev->res_mtx);
    if (img->attach_buf == VK_NULL_HANDLE) {
        uint32_t bpp = img->is_depth ? 4u : sm_format_bpp(img->format);
        VkDeviceSize sz = (VkDeviceSize)img->extent.width * img->extent.height * bpp;
        sz = (sz + 15ull) & ~15ull; // 16-byte aligned for 1B/2B u32 atomic words
        if (sz < 256) sz = 256;
        if (!create_buf(dev, sz,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        img->attach_buf, img->attach_mem)) return false;
    }
    if (img->is_depth && img->hiz_buf == VK_NULL_HANDLE && img->extent.width > 0 && img->extent.height > 0) {
        uint32_t tx = (img->extent.width + 15u) / 16u;
        uint32_t ty = (img->extent.height + 15u) / 16u;
        VkDeviceSize hsz = std::max<VkDeviceSize>(256, (VkDeviceSize)tx * ty * 8u);
        create_buf(dev, hsz,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   img->hiz_buf, img->hiz_mem);
    }
    return true;
}

bool sm_ensure_image_stencil_buf(SmDevice* dev, SmImage* img) {
    if (!img) return false;
    std::lock_guard<std::recursive_mutex> rlk(dev->res_mtx);
    if (img->stencil_buf == VK_NULL_HANDLE) {
        VkDeviceSize sz = (VkDeviceSize)img->extent.width * img->extent.height * 4u;
        if (sz < 256) sz = 256;
        if (!create_buf(dev, sz,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        img->stencil_buf, img->stencil_mem)) return false;
    }
    // Allocate companion R32_UINT shadow VkImage for hardware-sampled VK_IMAGE_ASPECT_STENCIL_BIT SRVs!
    if (img->real_stencil_img == VK_NULL_HANDLE && img->extent.width > 0 && img->extent.height > 0) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT,
                              {img->extent.width, img->extent.height, 1},
                              std::max(1u, img->mip_levels), std::max(1u, img->array_layers),
                              VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
        if (dev->fn.CreateImage(dev->real_dev, &ici, nullptr, &img->real_stencil_img) == VK_SUCCESS) {
            VkMemoryRequirements mr{};
            dev->fn.GetImageMemoryRequirements(dev->real_dev, img->real_stencil_img, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                                     sm_find_mem_type(dev->pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
            if (dev->fn.AllocateMemory(dev->real_dev, &mai, nullptr, &img->real_stencil_mem) == VK_SUCCESS) {
                dev->fn.BindImageMemory(dev->real_dev, img->real_stencil_img, img->real_stencil_mem, 0);
            } else {
                dev->fn.DestroyImage(dev->real_dev, img->real_stencil_img, nullptr);
                img->real_stencil_img = VK_NULL_HANDLE;
            }
        }
    }
    return true;
}

bool sm_ensure_geom_capacity(SmDevice* dev, uint32_t need_verts, uint32_t need_tris) {
    auto& R = dev->rast;
    if (need_verts <= R.cap_verts && need_tris <= R.cap_tris && R.clip_vert_buf != VK_NULL_HANDLE)
        return true;

    uint32_t nv = std::min(SM_MAX_VERTS_PER_DRAW, std::max(need_verts, std::max(R.cap_verts * 2u, 524288u)));
    uint32_t nt = std::min(SM_MAX_TRIS_PER_DRAW,  std::max(need_tris,  std::max(R.cap_tris  * 2u, 262144u)));
    // Always reserve extra tail space for near-plane clipped vertices/triangles!
    if (nv > SM_MAX_DRAW_VERTS) nv = SM_MAX_VERTS_PER_DRAW;
    if (nt > SM_MAX_DRAW_TRIS)  nt = SM_MAX_TRIS_PER_DRAW;

    if (R.clip_vert_buf) {
        // Retire old buffers safely without destroying them while recorded command buffers may still reference them!
        R.retired_bufs.push_back({R.clip_vert_buf, R.clip_vert_mem});
        R.retired_bufs.push_back({R.tri_setup_buf, R.tri_setup_mem});
        R.clip_vert_buf = VK_NULL_HANDLE; R.clip_vert_mem = VK_NULL_HANDLE;
        R.tri_setup_buf = VK_NULL_HANDLE; R.tri_setup_mem = VK_NULL_HANDLE;
    }
    VkBufferUsageFlags ssbo_u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!create_buf(dev, (VkDeviceSize)nv * SM_VERT_STRIDE_VEC4 * 16, ssbo_u,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.clip_vert_buf, R.clip_vert_mem)) return false;
    if (!create_buf(dev, (VkDeviceSize)nt * 80, ssbo_u,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.tri_setup_buf, R.tri_setup_mem)) return false;
    R.cap_verts = nv;
    R.cap_tris  = nt;
    return true;
}

static VkPipeline make_internal_pipe(SmDevice* dev, const char* spv_path, VkPipelineLayout pl) {
    std::vector<uint32_t> spv;
    if (!load_spv_file(spv_path, spv)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size() * 4, spv.data()};
    VkShaderModule sm = VK_NULL_HANDLE;
    if (dev->fn.CreateShaderModule(dev->real_dev, &smci, nullptr, &sm) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"},
        pl, VK_NULL_HANDLE, -1};
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = dev->fn.CreateComputePipelines(dev->real_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    dev->fn.DestroyShaderModule(dev->real_dev, sm, nullptr);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[SmRast] ERROR: CreateComputePipelines(%s) failed: %d\n", spv_path, r);
        return VK_NULL_HANDLE;
    }
    return pipe;
}

bool sm_init_raster_engine(SmDevice* dev) {
    auto& R = dev->rast;
    const char* shader_dir = getenv("SMRAST_SHADER_DIR");
    if (!shader_dir || !shader_dir[0]) shader_dir = "./build";

    // 1. DescriptorSetLayouts for Clear (8-MRT+Depth+Vis+Stencil+UBO+HiZ+TileCount), Bin, Fine, IndirectPrep, DepthOnly, Blit, HiZUpdate, VS-int, FS-int
    {
        VkDescriptorSetLayoutBinding b[14];
        for (uint32_t i = 0; i < 11; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[11] = {11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[12] = {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[13] = {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 14, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.clear_dsl);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.clear_dsl, 0, nullptr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.clear_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[9];
        for (uint32_t i = 0; i < 9; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 9, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.bin_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.bin_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.bin_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[7];
        for (uint32_t i = 0; i < 7; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fine_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.fine_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.fine_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[5];
        for (uint32_t i = 0; i < 5; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.ind_prep_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.ind_prep_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.ind_prep_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[4];
        for (uint32_t i = 0; i < 4; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.depth_only_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.depth_only_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.depth_only_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[2];
        for (uint32_t i = 0; i < 2; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.blit_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.blit_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.blit_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.hiz_update_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.hiz_update_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.hiz_update_pl);
    }
    {
        // VS internal set: 0=ClipVert, 1=IdxBuf, 2..9=VB0..VB7, 10=VsParams (DYNAMIC SSBO for zero per-draw vkUpdateDescriptorSets!)
        VkDescriptorSetLayoutBinding b[11];
        for (uint32_t i = 0; i < 10; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 11, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.vs_int_dsl);
    }
    {
        // FS internal set: 0=ClipVert, 1=TriSetup, 2=VisBuf, 3..10=Color0..7, 11=DepthBuf, 12=StencilBuf,
        //                  13=FsParams (DYNAMIC!), 14=ActiveTiles, 15=BatchPushBuf, 16=BatchDynUboRaw, 17=BatchDynOffBuf
        VkDescriptorSetLayoutBinding b[18];
        for (uint32_t i = 0; i < 18; ++i) b[i] = {i, (i == 13u) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 18, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fs_int_dsl);
    }

    // 2. Load compiled internal Compute Rasterizer SPIR-V shaders onto H20 qf[2]
    char p_clear[512], p_bin[512], p_pbin[512], p_fine[512], p_ind[512], p_dep[512], p_blit[512], p_hiz[512];
    snprintf(p_clear, sizeof(p_clear), "%s/smrast_clear.comp.spv", shader_dir);
    snprintf(p_bin,   sizeof(p_bin),   "%s/smrast_bin.comp.spv",   shader_dir);
    snprintf(p_pbin,  sizeof(p_pbin),  "%s/smrast_prim_bin.comp.spv", shader_dir);
    snprintf(p_fine,  sizeof(p_fine),  "%s/smrast_fine.comp.spv",  shader_dir);
    snprintf(p_ind,   sizeof(p_ind),   "%s/smrast_indirect_prep.comp.spv", shader_dir);
    snprintf(p_dep,   sizeof(p_dep),   "%s/smrast_depth_only.comp.spv", shader_dir);
    snprintf(p_blit,  sizeof(p_blit),  "%s/smrast_blit.comp.spv",  shader_dir);
    snprintf(p_hiz,   sizeof(p_hiz),   "%s/smrast_hiz_update.comp.spv", shader_dir);
    R.clear_pipe      = make_internal_pipe(dev, p_clear, R.clear_pl);
    R.bin_pipe        = make_internal_pipe(dev, p_bin,   R.bin_pl);
    R.prim_bin_pipe   = make_internal_pipe(dev, p_pbin,  R.bin_pl);
    R.fine_pipe       = make_internal_pipe(dev, p_fine,  R.fine_pl);
    R.ind_prep_pipe   = make_internal_pipe(dev, p_ind,   R.ind_prep_pl);
    R.depth_only_pipe = make_internal_pipe(dev, p_dep,   R.depth_only_pl);
    R.blit_pipe       = make_internal_pipe(dev, p_blit,  R.blit_pl);
    R.hiz_update_pipe = make_internal_pipe(dev, p_hiz,   R.hiz_update_pl);
    if (!R.clear_pipe || !R.bin_pipe || !R.prim_bin_pipe || !R.fine_pipe || !R.ind_prep_pipe || !R.depth_only_pipe || !R.blit_pipe || !R.hiz_update_pipe) return false;

    // 3. Pre-allocate 4.0M verts + 1.8M tris on H20 HBM3 (~1.16 GB < 1.2% of 96 GB) so 128-draw Indirect Uber-Batches never realloc or overflow!
    if (!sm_ensure_geom_capacity(dev, 4000000, 1800000)) return false;

    VkBufferUsageFlags ssbo_u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!create_buf(dev, 4096, ssbo_u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.dummy_buf, R.dummy_mem)) return false;
    // 32 KB indirect dispatch + batched indirect metadata table (64B header + 256 x 64B per-draw slots!)
    if (!create_buf(dev, 32768, ssbo_u | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.ind_disp_buf, R.ind_disp_mem)) return false;

    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // VS & FS param buffers use STORAGE_BUFFER_BIT (zero UBO bank contention at set>=5 for vkd3d-proton!)
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmVsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, R.vs_ubo_buf, R.vs_ubo_mem, &R.vs_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmFsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, R.fs_ubo_buf, R.fs_ubo_mem, &R.fs_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_AUX_SLOTS * sizeof(SmClearUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, R.clr_ubo_buf, R.clr_ubo_mem, &R.clr_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * 128u,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, R.batch_push_buf, R.batch_push_mem, &R.batch_push_ptr)) return false;
    void* dyn_off_p = nullptr;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, R.batch_dyn_off_buf, R.batch_dyn_off_mem, &dyn_off_p)) return false;
    R.batch_dyn_off_ptr = reinterpret_cast<uint32_t*>(dyn_off_p);

    // 4. Batch-allocate cached DS ring pools (16,384 cached sets/stage w/ dynamic SSBO offsets -> 0 vkUpdateDescriptorSets/draw!)
    uint32_t ND = SmRasterEngine::MAX_DS_RING;
    uint32_t NI = SmRasterEngine::MAX_IND_SLOTS;
    uint32_t NA = SmRasterEngine::MAX_AUX_SLOTS;
    uint32_t total_sets = ND * 6 + NI + NA * 2;
    uint32_t total_ssbo = ND * (10 + 17 + 9 + 7 + 4 + 3) + NI * 5 + NA * (13 + 2);
    uint32_t total_dyn_ssbo = ND * 2;
    VkDescriptorPoolSize ps[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, total_ssbo},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, total_dyn_ssbo},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NA},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, total_sets, 3, ps};
    if (dev->fn.CreateDescriptorPool(dev->real_dev, &dpci, nullptr, &R.desc_pool) != VK_SUCCESS) return false;

    auto alloc_batch = [&](VkDescriptorSetLayout dsl, uint32_t count, std::vector<VkDescriptorSet>& out) -> bool {
        out.resize(count, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> layouts(count, dsl);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, R.desc_pool, count, layouts.data()};
        return dev->fn.AllocateDescriptorSets(dev->real_dev, &ai, out.data()) == VK_SUCCESS;
    };
    if (!alloc_batch(R.clear_dsl,      NA, R.clear_ds)      ||
        !alloc_batch(R.blit_dsl,       NA, R.blit_ds)       ||
        !alloc_batch(R.ind_prep_dsl,   NI, R.ind_prep_ds)   ||
        !alloc_batch(R.bin_dsl,        ND, R.bin_ds)        ||
        !alloc_batch(R.fine_dsl,       ND, R.fine_ds)       ||
        !alloc_batch(R.depth_only_dsl, ND, R.depth_only_ds) ||
        !alloc_batch(R.hiz_update_dsl, ND, R.hiz_update_ds) ||
        !alloc_batch(R.vs_int_dsl,     ND, R.vs_int_ds)     ||
        !alloc_batch(R.fs_int_dsl,     ND, R.fs_int_ds)) return false;

    // Pre-size vis_buf to 2048x2048 so 1080p/2K shadow maps never reallocate mid-frame across DX12 worker threads!
    return sm_ensure_vis_size(dev, 2048, 2048);
}

bool sm_ensure_vis_size(SmDevice* dev, uint32_t w, uint32_t h) {
    std::lock_guard<std::recursive_mutex> rlk(dev->res_mtx);
    auto& R = dev->rast;
    if (w <= R.cur_w && h <= R.cur_h && R.vis_buf != VK_NULL_HANDLE) return true;
    uint32_t nw = std::max(w, R.cur_w);
    uint32_t nh = std::max(h, R.cur_h);
    uint32_t tx = (nw + 15) / 16;
    uint32_t ty = (nh + 15) / 16;
    uint32_t num_tiles = tx * ty;

    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer new_vis = VK_NULL_HANDLE, new_tc = VK_NULL_HANDLE, new_tb = VK_NULL_HANDLE, new_at = VK_NULL_HANDLE;
    VkDeviceMemory new_vis_m = VK_NULL_HANDLE, new_tc_m = VK_NULL_HANDLE, new_tb_m = VK_NULL_HANDLE, new_at_m = VK_NULL_HANDLE;
    if (!create_buf(dev, (VkDeviceSize)nw * nh * 8, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_vis, new_vis_m)) return false;
    if (!create_buf(dev, (VkDeviceSize)(num_tiles * 2 + 16) * 4, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_tc, new_tc_m)) return false;
    if (!create_buf(dev, (VkDeviceSize)num_tiles * SM_MAX_TRIS_PER_TILE * 4, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_tb, new_tb_m)) return false;
    if (!create_buf(dev, std::max<VkDeviceSize>(256, (VkDeviceSize)num_tiles * 4), u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, new_at, new_at_m)) return false;

    {
        std::lock_guard<std::mutex> dlk(dev->mtx);
        if (R.vis_buf) {
            R.retired_bufs.push_back({R.vis_buf, R.vis_mem});
            R.retired_bufs.push_back({R.tile_count_buf, R.tile_count_mem});
            R.retired_bufs.push_back({R.tile_bins_buf, R.tile_bins_mem});
            R.retired_bufs.push_back({R.active_tiles_buf, R.active_tiles_mem});
        }
        R.vis_buf = new_vis; R.vis_mem = new_vis_m;
        R.tile_count_buf = new_tc; R.tile_count_mem = new_tc_m;
        R.tile_bins_buf = new_tb; R.tile_bins_mem = new_tb_m;
        R.active_tiles_buf = new_at; R.active_tiles_mem = new_at_m;
        R.cur_w = nw; R.cur_h = nh;
        R.tiles_x = tx; R.tiles_y = ty;
        R.vis_inited = false;
        R.bin_ds_cache.clear();
        R.fine_ds_cache.clear();
        R.fs_ds_cache.clear();
        R.dep_ds_cache.clear();
        R.hiz_ds_cache.clear();
        R.ind_prep_ds_cache.clear();
    }
    return true;
}

void sm_destroy_raster_engine(SmDevice* dev) {
    auto& R = dev->rast;
    if (!dev->real_dev) return;
    dev->fn.DeviceWaitIdle(dev->real_dev);
    auto dpipe = [&](VkPipeline& p) { if (p) { dev->fn.DestroyPipeline(dev->real_dev, p, nullptr); p = VK_NULL_HANDLE; } };
    dpipe(R.clear_pipe); dpipe(R.bin_pipe); dpipe(R.prim_bin_pipe); dpipe(R.fine_pipe); dpipe(R.ind_prep_pipe);
    dpipe(R.depth_only_pipe); dpipe(R.blit_pipe); dpipe(R.hiz_update_pipe);
    for (auto& rb : R.retired_bufs) {
        if (rb.first)  dev->fn.DestroyBuffer(dev->real_dev, rb.first, nullptr);
        if (rb.second) dev->fn.FreeMemory(dev->real_dev, rb.second, nullptr);
    }
    R.retired_bufs.clear();
}
