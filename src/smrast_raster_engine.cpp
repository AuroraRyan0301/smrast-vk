// smrast_raster_engine.cpp — H20/H100/MI300X HBM3 Rasterizer Buffer & Compute Pipeline Initialization
#include "smrast_internal.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

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
    VkDeviceSize sz = (VkDeviceSize)img->extent.width * img->extent.height * 4;
    if (sz < 256) sz = 256;
    return create_buf(dev, sz,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      img->attach_buf, img->attach_mem);
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

    // 1. Create DescriptorSetLayouts for Clear, Bin, Fine, VS-internal, FS-internal
    {
        VkDescriptorSetLayoutBinding b[3] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.clear_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.clear_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.clear_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[6];
        for (uint32_t i = 0; i < 6; ++i)
            b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 6, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.bin_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.bin_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.bin_pl);
    }
    {
        VkDescriptorSetLayoutBinding b[5];
        for (uint32_t i = 0; i < 5; ++i)
            b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fine_dsl);
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &R.fine_dsl, 1, &pcr};
        dev->fn.CreatePipelineLayout(dev->real_dev, &plci, nullptr, &R.fine_pl);
    }
    {
        // VS internal set: 0=ClipVert, 1=IdxBuf, 2..5=VB0..3, 6=VsParamsUBO
        VkDescriptorSetLayoutBinding b[7];
        for (uint32_t i = 0; i < 6; ++i)
            b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.vs_int_dsl);
    }
    {
        // FS internal set: 0=ClipVert, 1=TriSetup, 2=VisBuf, 3=ColorBuf, 4=DepthBuf, 5=FsParamsUBO
        VkDescriptorSetLayoutBinding b[6];
        for (uint32_t i = 0; i < 5; ++i)
            b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[5] = {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 6, b};
        dev->fn.CreateDescriptorSetLayout(dev->real_dev, &ci, nullptr, &R.fs_int_dsl);
    }

    // 2. Load compiled internal Compute Rasterizer SPIR-V shaders onto H20 qf[2]
    char p_clear[512], p_bin[512], p_fine[512];
    snprintf(p_clear, sizeof(p_clear), "%s/smrast_clear.comp.spv", shader_dir);
    snprintf(p_bin,   sizeof(p_bin),   "%s/smrast_bin.comp.spv",   shader_dir);
    snprintf(p_fine,  sizeof(p_fine),  "%s/smrast_fine.comp.spv",  shader_dir);
    R.clear_pipe = make_internal_pipe(dev, p_clear, R.clear_pl);
    R.bin_pipe   = make_internal_pipe(dev, p_bin,   R.bin_pl);
    R.fine_pipe  = make_internal_pipe(dev, p_fine,  R.fine_pl);
    if (!R.clear_pipe || !R.bin_pipe || !R.fine_pipe) return false;

    // 3. Allocate persistent H20 HBM3 geometry & UBO ring buffers
    //    (H20 has 96 GB HBM3! Allocating ~400 MB for 2M tris/draw + 4K visbuf is <0.4% of VRAM!)
    VkBufferUsageFlags ssbo_u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!create_buf(dev, (VkDeviceSize)SM_MAX_VERTS_PER_DRAW * SM_VERT_STRIDE_VEC4 * 16, ssbo_u,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.clip_vert_buf, R.clip_vert_mem)) return false;
    // TriSetup = 5 vec4 = 80 bytes/tri
    if (!create_buf(dev, (VkDeviceSize)SM_MAX_TRIS_PER_DRAW * 80, ssbo_u,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.tri_setup_buf, R.tri_setup_mem)) return false;
    if (!create_buf(dev, 4096, ssbo_u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.dummy_buf, R.dummy_mem)) return false;

    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmVsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, R.vs_ubo_buf, R.vs_ubo_mem, &R.vs_ubo_ptr)) return false;
    if (!create_buf(dev, (VkDeviceSize)SmRasterEngine::MAX_DRAWS_IN_FLIGHT * sizeof(SmFsParamsUBO),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, R.fs_ubo_buf, R.fs_ubo_mem, &R.fs_ubo_ptr)) return false;

    // 4. Allocate descriptor pool & per-slot ring sets (eliminates multi-pass descriptor clobbering!)
    uint32_t N = SmRasterEngine::MAX_DRAWS_IN_FLIGHT;
    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * (3 + 6 + 5 + 6 + 5)},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, N * 5, 2, ps};
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
    R.vs_int_ds.resize(N);
    R.fs_int_ds.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        R.clear_ds[i]  = alloc_ds(R.clear_dsl);
        R.bin_ds[i]    = alloc_ds(R.bin_dsl);
        R.fine_ds[i]   = alloc_ds(R.fine_dsl);
        R.vs_int_ds[i] = alloc_ds(R.vs_int_dsl);
        R.fs_int_ds[i] = alloc_ds(R.fs_int_dsl);
    }

    // Initialize visibility & tile bins for up to 1920x1080 upfront
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
        dev->fn.DeviceWaitIdle(dev->real_dev);
        dev->fn.DestroyBuffer(dev->real_dev, R.vis_buf, nullptr);
        dev->fn.FreeMemory(dev->real_dev, R.vis_mem, nullptr);
        dev->fn.DestroyBuffer(dev->real_dev, R.tile_count_buf, nullptr);
        dev->fn.FreeMemory(dev->real_dev, R.tile_count_mem, nullptr);
        dev->fn.DestroyBuffer(dev->real_dev, R.tile_bins_buf, nullptr);
        dev->fn.FreeMemory(dev->real_dev, R.tile_bins_mem, nullptr);
    }

    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buf(dev, (VkDeviceSize)nw * nh * 8, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.vis_buf, R.vis_mem)) return false;
    // +4 extra uints at tail of tile_count_buf for near-plane clipper atomic counters!
    if (!create_buf(dev, (VkDeviceSize)(num_tiles + 4) * 4, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, R.tile_count_buf, R.tile_count_mem)) return false;
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
    dpipe(R.clear_pipe); dpipe(R.bin_pipe); dpipe(R.fine_pipe);
}
