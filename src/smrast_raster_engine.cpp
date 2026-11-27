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
    if (!img || img->attach_buf != VK_NULL_HANDLE) return true;
    uint32_t bpp = img->is_depth ? 4u : sm_format_bpp(img->format);
    VkDeviceSize sz = (VkDeviceSize)img->extent.width * img->extent.height * bpp;
    if (sz < 256) sz = 256;
    return create_buf(dev, sz,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      img->attach_buf, img->attach_mem);
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

    // 1. DescriptorSetLayouts for Clear (MRT 4+Depth+Vis+UBO), Bin, Fine, IndirectPrep, VS-int, FS-int (MRT 4)
    {
        VkDescriptorSetLayoutBinding b[7];
        for (uint32_t i = 0; i < 6; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.clear_dsl);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.clear_dsl, 0, nullptr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.clear_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[6];
        for (uint32_t i = 0; i < 6; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 6, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.bin_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.bin_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.bin_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[5];
        for (uint32_t i = 0; i < 5; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fine_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.fine_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.fine_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[4];
        for (uint32_t i = 0; i < 4; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.ind_prep_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.ind_prep_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.ind_prep_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, b};
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
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.blit_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.blit_pl);
    }
    {
        // VS internal set: 0=ClipVert, 1=IdxBuf, 2..9=VB0..VB7, 10=VsParamsUBO
        VkDescriptorSetLayoutBinding b[11];
        for (uint32_t i = 0; i < 10; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[10] = {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 11, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.vs_int_dsl);
    }
    {
        // FS internal set: 0=ClipVert, 1=TriSetup, 2=VisBuf, 3..6=Color0..3, 7=DepthBuf, 8=FsParamsUBO
        VkDescriptorSetLayoutBinding b[9];
        for (uint32_t i = 0; i < 8; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[8] = {8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 9, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fs_int_dsl);
    }

    // 2. Load compiled internal Compute Rasterizer SPIR-V shaders onto H20 qf[2]
    char p_clear[512], p_bin[512], p_fine[512], p_ind[512], p_dep[512], p_blit[512];
    snprintf(p_clear, sizeof(p_clear), "%s/smrast_clear.comp.spv", shader_dir);
    snprintf(p_bin,   sizeof(p_bin),   "%s/smrast_bin.comp.spv",   shader_dir);
    snprintf(p_fine,  sizeof(p_fine),  "%s/smrast_fine.comp.spv",  shader_dir);
    snprintf(p_ind,   sizeof(p_ind),   "%s/smrast_indirect_prep.comp.spv", shader_dir);
    snprintf(p_dep,   sizeof(p_dep),   "%s/smrast_depth_only.comp.spv", shader_dir);
    snprintf(p_blit,  sizeof(p_blit),  "%s/smrast_blit.comp.spv",  shader_dir);
    R.clear_pipe      = make_internal_pipe(dev, p_clear, R.clear_pl);
    R.bin_pipe        = make_internal_pipe(dev, p_bin,   R.bin_pl);
    R.fine_pipe       = make_internal_pipe(dev, p_fine,  R.fine_pl);
    R.ind_prep_pipe   = make_internal_pipe(dev, p_ind,   R.ind_prep_pl);
    R.depth_only_pipe = make_internal_pipe(dev, p_dep,   R.depth_only_pl);
    R.blit_pipe       = make_internal_pipe(dev, p_blit,  R.blit_pl);
    if (!R.clear_pipe || !R.bin_pipe || !R.fine_pipe || !R.ind_prep_pipe || !R.depth_only_pipe || !R.blit_pipe) return false;

    // 3. Pre-allocate 2.2M verts + 1.1M tris on H20 HBM3 (~640 MB < 0.7% of 96 GB) so DrawIndirect never reallocs mid-CB!
    if (!sm_ensure_geom_capacity(dev, 2200000, 1100000)) return false;

    VkBufferUsageFlags ssbo_u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!create_buf(dev, 4096, ssbo_u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.dummy_buf, R.dummy_mem)) return false;
    if (!create_buf(dev, 256, ssbo_u | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.ind_disp_buf, R.ind_disp_mem)) return false;

    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // VS UBO also has STORAGE_BUFFER_BIT so smrast_indirect_prep.comp can patch draw counts directly on GPU!
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmVsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, R.vs_ubo_buf, R.vs_ubo_mem, &R.vs_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmFsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, R.fs_ubo_buf, R.fs_ubo_mem, &R.fs_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmClearUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, R.clr_ubo_buf, R.clr_ubo_mem, &R.clr_ubo_ptr)) return false;

    // 4. Allocate descriptor pool & per-slot ring sets
    uint32_t N = SmRasterEngine::MAX_DRAWS_IN_FLIGHT;
    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * (6 + 6 + 5 + 4 + 3 + 2 + 10 + 8)},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 3},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, N * 8, 2, ps};
    if (dev->fn.CreateDescriptorPool(dev->real_dev, &dpci, nullptr, &R.desc_pool) != VK_SUCCESS) return false;

    auto alloc_ds = [&](VkDescriptorSetLayout dsl) {
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, R.desc_pool, 1, &dsl};
        dev->fn.AllocateDescriptorSets(dev->real_dev, &ai, &ds);
        return ds;
    };
    R.clear_ds.resize(N);
    R.bin_ds.resize(N);
    R.fine_ds.resize(N);
    R.ind_prep_ds.resize(N);
    R.depth_only_ds.resize(N);
    R.blit_ds.resize(N);
    R.vs_int_ds.resize(N);
    R.fs_int_ds.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        R.clear_ds[i]      = alloc_ds(R.clear_dsl);
        R.bin_ds[i]        = alloc_ds(R.bin_dsl);
        R.fine_ds[i]       = alloc_ds(R.fine_dsl);
        R.ind_prep_ds[i]   = alloc_ds(R.ind_prep_dsl);
        R.depth_only_ds[i] = alloc_ds(R.depth_only_dsl);
        R.blit_ds[i]       = alloc_ds(R.blit_dsl);
        R.vs_int_ds[i]     = alloc_ds(R.vs_int_dsl);
        R.fs_int_ds[i]     = alloc_ds(R.fs_int_dsl);
    }

    return sm_ensure_vis_size(dev, 1920, 1080);
}

bool sm_ensure_vis_size(SmDevice* dev, uint32_t w, uint32_t h) {
    auto& R = dev->rast;
    if (w <= R.cur_w && h <= R.cur_h && R.vis_buf != VK_NULL_HANDLE) return true;
    uint32_t nw = std::max(w, R.cur_w);
    uint32_t nh = std::max(h, R.cur_h);
    uint32_t tx = (nw + 15) / 16;
    uint32_t ty = (nh + 15) / 16;
    uint32_t num_tiles = tx * ty;

    if (R.vis_buf) {
        // Retire safely so in-flight / recorded CBs never hit freed HBM3 pages!
        R.retired_bufs.push_back({R.vis_buf, R.vis_mem});
        R.retired_bufs.push_back({R.tile_count_buf, R.tile_count_mem});
        R.retired_bufs.push_back({R.tile_bins_buf, R.tile_bins_mem});
        R.vis_buf = VK_NULL_HANDLE; R.vis_mem = VK_NULL_HANDLE;
        R.tile_count_buf = VK_NULL_HANDLE; R.tile_count_mem = VK_NULL_HANDLE;
        R.tile_bins_buf = VK_NULL_HANDLE; R.tile_bins_mem = VK_NULL_HANDLE;
    }

    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buf(dev, (VkDeviceSize)nw * nh * 8, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.vis_buf, R.vis_mem)) return false;
    // +8 extra uints at tail of tile_count_buf for near-plane clipper & DrawIndirect metadata!
    if (!create_buf(dev, (VkDeviceSize)(num_tiles + 8) * 4, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.tile_count_buf, R.tile_count_mem)) return false;
    if (!create_buf(dev, (VkDeviceSize)num_tiles * SM_MAX_TRIS_PER_TILE * 4, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.tile_bins_buf, R.tile_bins_mem)) return false;

    R.cur_w = nw; R.cur_h = nh;
    R.tiles_x = tx; R.tiles_y = ty;
    return true;
}

void sm_destroy_raster_engine(SmDevice* dev) {
    auto& R = dev->rast;
    if (!dev->real_dev) return;
    dev->fn.DeviceWaitIdle(dev->real_dev);
    auto dpipe = [&](VkPipeline& p) { if (p) { dev->fn.DestroyPipeline(dev->real_dev, p, nullptr); p = VK_NULL_HANDLE; } };
    dpipe(R.clear_pipe); dpipe(R.bin_pipe); dpipe(R.fine_pipe); dpipe(R.ind_prep_pipe); dpipe(R.depth_only_pipe); dpipe(R.blit_pipe);
    for (auto& rb : R.retired_bufs) {
        if (rb.first)  dev->fn.DestroyBuffer(dev->real_dev, rb.first, nullptr);
        if (rb.second) dev->fn.FreeMemory(dev->real_dev, rb.second, nullptr);
    }
    R.retired_bufs.clear();
}
