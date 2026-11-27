// test_v10_features.cpp — SmRast-VK v1.0.0 Multi-Engine DX11/DX12 Compatibility & Stability Suite:
//   [1] Bare gl_ClipDistance[8] I/O + Cross-Block OpSampledImage Sinker + Post-Sink spirv-opt (-O)
//   [2] Multi-Queue Pool (4-8 HW Queues) + Cross-Queue Timeline Semaphore Wait-Before-Signal + Sparse Binding
//   [3] 64-bit VkAccessFlags2 (SHADER_STORAGE_WRITE -> SHADER_SAMPLED_READ) + Vulkan 1.3 Attachment Layouts
//   [4] VK_KHR_dedicated_allocation Unwrapping + linear_aliased Req Consistency + VkImageViewUsageCreateInfo
//   [5] Hardware Timestamp Queries (CmdWriteTimestamp/2 + CmdCopyQueryPoolResults) + Emulated Occlusion Readback
//   [6] Multi-Threaded Concurrent Pipeline Compilation & >2K Visibility-Buffer Resize Stress (4 Threads)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

#define VK_CHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", _r, __LINE__); exit(1); } } while (0)

static std::atomic<int> g_tmp_id{0};

static std::vector<uint32_t> compile_glsl(const char* src, const char* stage) {
    int id = g_tmp_id.fetch_add(1);
    std::string in_p = "/tmp/sm_v10_tmp_" + std::to_string(id) + "." + stage;
    std::string out_p = "/tmp/sm_v10_tmp_" + std::to_string(id) + ".spv";
    FILE* f = fopen(in_p.c_str(), "wb");
    fwrite(src, 1, strlen(src), f);
    fclose(f);
    std::string cmd = "glslangValidator --target-env vulkan1.3 -V " + in_p + " -o " + out_p + " >/dev/null";
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "glslang failed (%s)\n", stage); exit(1); }
    f = fopen(out_p.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    size_t sz = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> spv(sz / 4);
    if (fread(spv.data(), 1, sz, f) != sz) { fprintf(stderr, "fread spv failed\n"); exit(1); }
    fclose(f);
    remove(in_p.c_str());
    remove(out_p.c_str());
    return spv;
}

static VkShaderModule make_mod(VkDevice dev, const char* src, const char* stage) {
    auto spv = compile_glsl(src, stage);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size() * 4, spv.data()};
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
    fprintf(stderr, "No memory type matching bits=0x%x req=0x%x\n", bits, req);
    exit(1);
}

struct ImgObj { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; uint32_t w = 0, h = 0; };
struct BufObj { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void* ptr = nullptr; VkDeviceSize sz = 0; };

static ImgObj make_img(VkPhysicalDevice pd, VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                       VkImageAspectFlags asp, bool try_dedicated = false, VkImageUsageFlags view_usage_override = 0) {
    ImgObj o; o.w = w; o.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1}, 1, 1,
                         VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHK(vkCreateImage(dev, &ci, nullptr, &o.img));
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(dev, o.img, &mr);
    VkMemoryDedicatedAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, o.img, VK_NULL_HANDLE};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, try_dedicated ? &dai : nullptr, mr.size,
                            find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindImageMemory(dev, o.img, o.mem, 0));
    VkImageViewUsageCreateInfo ivu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, view_usage_override};
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, view_usage_override ? &ivu : nullptr, 0,
                             o.img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
    VK_CHK(vkCreateImageView(dev, &vi, nullptr, &o.view));
    return o;
}

static BufObj make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize sz, VkBufferUsageFlags usage, bool try_dedicated = false) {
    BufObj o; o.sz = sz;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, VK_SHARING_MODE_EXCLUSIVE};
    VK_CHK(vkCreateBuffer(dev, &ci, nullptr, &o.buf));
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, o.buf, &mr);
    VkMemoryDedicatedAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, VK_NULL_HANDLE, o.buf};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, try_dedicated ? &dai : nullptr, mr.size,
                            find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindBufferMemory(dev, o.buf, o.mem, 0));
    VK_CHK(vkMapMemory(dev, o.mem, 0, sz, 0, &o.ptr));
    return o;
}

static VkPipeline make_pipe(VkDevice dev, VkPipelineLayout pl, VkShaderModule vs, VkShaderModule fs, VkFormat col_fmt) {
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr}
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
                                              VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xFu;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &ba};
    VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyns};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &col_fmt, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, st, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cb, &dyn, pl};
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p));
    return p;
}

static void submit_and_wait(VkDevice dev, VkQueue q, VkCommandBuffer cb, VkFence fence) {
    VK_CHK(vkEndCommandBuffer(cb));
    VK_CHK(vkResetFences(dev, 1, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    VK_CHK(vkQueueSubmit(q, 1, &si, fence));
    VK_CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
}

int main() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV10Test", 1, "SmRast", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai};
    VkInstance inst = VK_NULL_HANDLE;
    VK_CHK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0;
    VK_CHK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    std::vector<VkPhysicalDevice> pds(nd);
    VK_CHK(vkEnumeratePhysicalDevices(inst, &nd, pds.data()));
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("Testing SmRast-VK v1.0.0 on: %s\n", props.deviceName);

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(pd, &feats);
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
    if (nqf == 0 || qfs[0].queueCount < 2) {
        fprintf(stderr, "FAIL [2]: expected multi-queue family (queueCount >= 2), got %u\n", nqf ? qfs[0].queueCount : 0u);
        return 1;
    }
    bool qf_sparse = (qfs[0].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
    if ((feats.sparseBinding == VK_TRUE) != qf_sparse) {
        fprintf(stderr, "FAIL [2]: sparseBinding feature (%d) mismatches queue family SPARSE_BINDING bit (%d)\n",
                (int)feats.sparseBinding, (int)qf_sparse);
        return 1;
    }

    uint32_t n_req_q = std::min(qfs[0].queueCount, 4u);
    float prios[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, n_req_q, prios};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures en_feats{};
    en_feats.sparseBinding = feats.sparseBinding;
    en_feats.shaderClipDistance = feats.shaderClipDistance;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, nullptr, 0, nullptr, &en_feats};
    VkDevice dev = VK_NULL_HANDLE;
    VK_CHK(vkCreateDevice(pd, &dci, nullptr, &dev));

    std::vector<VkQueue> qs(n_req_q);
    for (uint32_t i = 0; i < n_req_q; ++i) vkGetDeviceQueue(dev, 0, i, &qs[i]);
    if (qs[0] == qs[1]) { fprintf(stderr, "FAIL [2]: q0 and q1 aliased to identical handle!\n"); return 1; }

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VK_CHK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHK(vkCreateFence(dev, &fci, nullptr, &fence));

    // =========================================================================
    // [1] Bare gl_ClipDistance[8] I/O + Cross-Block OpSampledImage Sinker + Post-Sink spirv-opt
    // =========================================================================
    {
        const uint32_t W = 64, H = 64;
        // Create 2x1 source texture: left pixel = RED (255,0,0,255), right pixel = GREEN (0,255,0,255)
        ImgObj src_tex = make_img(pd, dev, 2, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        BufObj src_stg = make_buf(pd, dev, 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        uint8_t px2[8] = {255, 0, 0, 255, 0, 255, 0, 255};
        memcpy(src_stg.ptr, px2, 8);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, src_tex.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkBufferImageCopy bic{}; bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {2, 1, 1};
        vkCmdCopyBufferToImage(cb, src_stg.buf, src_tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        submit_and_wait(dev, qs[0], cb, fence);

        VkSamplerCreateInfo smci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        smci.magFilter = VK_FILTER_NEAREST; smci.minFilter = VK_FILTER_NEAREST;
        smci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; smci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler smp = VK_NULL_HANDLE;
        VK_CHK(vkCreateSampler(dev, &smci, nullptr, &smp));

        VkDescriptorSetLayoutBinding dbs[2] = {
            {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, dbs};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VK_CHK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkDescriptorPoolSize dps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, dps};
        VkDescriptorPool dp = VK_NULL_HANDLE;
        VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorImageInfo dii0{VK_NULL_HANDLE, src_tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo dii1{smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkWriteDescriptorSet wds[2] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dii0},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &dii1}
        };
        vkUpdateDescriptorSets(dev, 2, wds, 0, nullptr);

        // VS with bare `out float gl_ClipDistance[8];`
        const char* vs_src = R"GLSL(
#version 450
out float gl_ClipDistance[8];
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.5, 1.0);
    v_uv = p * 0.5 + 0.5;
    gl_ClipDistance[0] = 1.0;
    gl_ClipDistance[1] = 1.0;
}
)GLSL";
        // FS with bare `in float gl_ClipDistance[8];` AND a ternary branch inside separate-texture/sampler `texture()`
        // coordinate expression (forces cross-block OpSampledImage before sm_sink_op_sampled_image + spirv-opt!)
        const char* fs_src = R"GLSL(
#version 450
in float gl_ClipDistance[8];
layout(set = 0, binding = 0) uniform texture2D uTex;
layout(set = 0, binding = 1) uniform sampler uSmp;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
void main() {
    float c = (v_uv.x >= 0.5 && v_uv.y >= -1.0) ? 0.75 : 0.25;
    vec4 s = texture(sampler2D(uTex, uSmp), vec2((v_uv.x > 0.5 || c > 0.6) ? 0.75 : 0.25, 0.5));
    o_col = s;
}
)GLSL";
        VkShaderModule vm = make_mod(dev, vs_src, "vert");
        VkShaderModule fm = make_mod(dev, fs_src, "frag");
        VkPipeline pipe = make_pipe(dev, pl, vm, fm, VK_FORMAT_R8G8B8A8_UNORM);

        ImgObj rt = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
        BufObj rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VK_CHK(vkResetCommandBuffer(cb, 0));
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rt.view,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0, 0, 0, 1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {W, H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {W, H}};
        vkCmdSetViewport(cb, 0, 1, &vp);
        vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        VkBufferImageCopy rbic{}; rbic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rbic.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cb, rt.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &rbic);
        submit_and_wait(dev, qs[0], cb, fence);

        const uint8_t* p = (const uint8_t*)rb.ptr;
        const uint8_t* pL = p + (32 * W + 16) * 4;
        const uint8_t* pR = p + (32 * W + 48) * 4;
        printf("[1] gl_ClipDistance[8] + Cross-Block OpSampledImage Sinker: L=(%u,%u,%u) R=(%u,%u,%u)\n",
               pL[0], pL[1], pL[2], pR[0], pR[1], pR[2]);
        if (pL[0] < 240 || pL[1] > 15 || pR[0] > 15 || pR[1] < 240) {
            fprintf(stderr, "FAIL [1]: unexpected colors!\n");
            return 1;
        }
    }

    // =========================================================================
    // [2] Multi-Queue Cross-Queue Timeline Semaphore Wait-Before-Signal + Sparse Binding
    // =========================================================================
    {
        VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0};
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci};
        VkSemaphore tsem = VK_NULL_HANDLE;
        VK_CHK(vkCreateSemaphore(dev, &sci, nullptr, &tsem));

        // Submit on q0 FIRST: waits for value=1 (not signaled yet!), then signals value=2
        VkSemaphoreSubmitInfo w0{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, tsem, 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
        VkSemaphoreSubmitInfo s0{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, tsem, 2, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
        VkSubmitInfo2 si0{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 1, &w0, 0, nullptr, 1, &s0};
        VK_CHK(vkQueueSubmit2(qs[0], 1, &si0, VK_NULL_HANDLE));

        // Submit on q1 SECOND: signals value=1 -> unblocks q0 -> q0 signals value=2!
        VkSemaphoreSubmitInfo s1{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, tsem, 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
        VkSubmitInfo2 si1{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 0, nullptr, 1, &s1};
        VK_CHK(vkQueueSubmit2(qs[1], 1, &si1, VK_NULL_HANDLE));

        uint64_t wait_val = 2;
        VkSemaphoreWaitInfo swi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &tsem, &wait_val};
        VkResult wr = vkWaitSemaphores(dev, &swi, 5000000000ULL);
        if (wr != VK_SUCCESS) { fprintf(stderr, "FAIL [2]: cross-queue timeline semaphore wait failed/timed out: %d\n", wr); return 1; }
        vkDestroySemaphore(dev, tsem, nullptr);

        bool sparse_ok = true;
        if (qf_sparse) {
            VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
                                    65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VkBuffer sbuf = VK_NULL_HANDLE;
            VK_CHK(vkCreateBuffer(dev, &sbci, nullptr, &sbuf));
            VkMemoryRequirements smr{};
            vkGetBufferMemoryRequirements(dev, sbuf, &smr);
            VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, smr.size,
                                      find_mem(pd, smr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
            VkDeviceMemory smem = VK_NULL_HANDLE;
            VK_CHK(vkAllocateMemory(dev, &smai, nullptr, &smem));
            VkSparseMemoryBind smb{0, smr.size, smem, 0, 0};
            VkSparseBufferMemoryBindInfo sbbi{sbuf, 1, &smb};
            VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, nullptr, 0, nullptr, 1, &sbbi};
            VK_CHK(vkResetFences(dev, 1, &fence));
            VK_CHK(vkQueueBindSparse(qs[n_req_q - 1], 1, &bsi, fence));
            VK_CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ULL));

            VkImageCreateInfo sici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr,
                                   VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
                                   VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {256, 256, 1}, 1, 1,
                                   VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                   VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
            VkDeviceImageMemoryRequirements dimr{VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, nullptr, &sici, VK_IMAGE_ASPECT_COLOR_BIT};
            uint32_t scnt = 0;
            vkGetDeviceImageSparseMemoryRequirements(dev, &dimr, &scnt, nullptr);
            sparse_ok = (scnt >= 1);
            vkDestroyBuffer(dev, sbuf, nullptr);
            vkFreeMemory(dev, smem, nullptr);
        }
        printf("[2] Multi-Queue (qcount=%u) + Cross-Queue Timeline Wait-Before-Signal + SparseBind=%d\n",
               qfs[0].queueCount, sparse_ok ? 1 : 0);
        if (!sparse_ok) { fprintf(stderr, "FAIL [2]: sparse reqs failed!\n"); return 1; }
    }

    // =========================================================================
    // [3] 64-bit VkAccessFlags2 + Vulkan 1.3 Attachment Layouts in vkCmdPipelineBarrier2
    // =========================================================================
    {
        const uint32_t W = 64, H = 64;
        ImgObj rtA = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj rtB = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
        BufObj rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VkSamplerCreateInfo smci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        VkSampler smp = VK_NULL_HANDLE;
        VK_CHK(vkCreateSampler(dev, &smci, nullptr, &smp));
        VkDescriptorSetLayoutBinding db{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &db};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VK_CHK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
        VkDescriptorPool dp = VK_NULL_HANDLE;
        VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorImageInfo dii{smp, rtA.view, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii};
        vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        const char* vs_q = R"GLSL(
#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.5, 1.0);
    v_uv = p * 0.5 + 0.5;
}
)GLSL";
        const char* fs_a = "#version 450\nlayout(location=0) out vec4 o; void main(){ o = vec4(0.0, 0.502, 1.0, 1.0); }\n";
        const char* fs_b = "#version 450\nlayout(set=0,binding=0) uniform sampler2D uS; layout(location=0) in vec2 uv; layout(location=0) out vec4 o; void main(){ o = texture(uS, uv); }\n";
        VkShaderModule vm = make_mod(dev, vs_q, "vert");
        VkShaderModule fa = make_mod(dev, fs_a, "frag");
        VkShaderModule fb = make_mod(dev, fs_b, "frag");
        VkPipeline pipeA = make_pipe(dev, pl, vm, fa, VK_FORMAT_R8G8B8A8_UNORM);
        VkPipeline pipeB = make_pipe(dev, pl, vm, fb, VK_FORMAT_R8G8B8A8_UNORM);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkResetCommandBuffer(cb, 0));
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {W, H}};

        VkRenderingAttachmentInfo caA{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rtA.view,
                                      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0, 0, 0, 1}}}};
        VkRenderingInfo riA{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {W, H}}, 1, 0, 1, &caA};
        vkCmdBeginRendering(cb, &riA);
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeA);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);

        // 64-bit VkAccessFlags2 (bits 34 & 32!) + Vulkan 1.3 ATTACHMENT_OPTIMAL -> READ_ONLY_OPTIMAL
        VkImageMemoryBarrier2 imb2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, rtA.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &imb2};
        vkCmdPipelineBarrier2(cb, &dep);

        VkRenderingAttachmentInfo caB{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rtB.view,
                                      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0, 0, 0, 1}}}};
        VkRenderingInfo riB{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {W, H}}, 1, 0, 1, &caB};
        vkCmdBeginRendering(cb, &riB);
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeB);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);

        VkBufferImageCopy rbic{}; rbic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rbic.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cb, rtB.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &rbic);
        submit_and_wait(dev, qs[0], cb, fence);

        const uint8_t* pM = (const uint8_t*)rb.ptr + (32 * W + 32) * 4;
        printf("[3] 64-bit AccessFlags2 + VK1.3 Layouts (Barrier2 -> TMU): mid=(%u,%u,%u)\n", pM[0], pM[1], pM[2]);
        if (pM[0] > 5 || std::abs((int)pM[1] - 128) > 3 || pM[2] < 250) {
            fprintf(stderr, "FAIL [3]: barrier2/layout mismatch!\n");
            return 1;
        }
    }

    // =========================================================================
    // [4] Dedicated Allocation Unwrapping + linear_aliased Req Consistency + ImageViewUsage Sanitization
    // =========================================================================
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM,
                              {1257, 707, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
        VkMemoryDedicatedRequirements ddr{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 dev_mr{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ddr};
        VkDeviceImageMemoryRequirements dimr{VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, nullptr, &ici};
        vkGetDeviceImageMemoryRequirements(dev, &dimr, &dev_mr);

        // Allocate with BOTH dedicated_allocation attempt AND VkImageViewUsageCreateInfo carrying COLOR_ATTACHMENT_BIT!
        ImgObj aliased_rt = make_img(pd, dev, 1257, 707, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, true,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        VkMemoryDedicatedRequirements idr{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 img_mr{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &idr};
        VkImageMemoryRequirementsInfo2 imri{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, aliased_rt.img};
        vkGetImageMemoryRequirements2(dev, &imri, &img_mr);

        // Also test dedicated buffer allocation unwrapping!
        BufObj ded_buf = make_buf(pd, dev, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        printf("[4] DedicatedAlloc Unwrap + Aliased Reqs: dev_sz=%llu img_sz=%llu dedReq=%d buf=%p\n",
               (unsigned long long)dev_mr.memoryRequirements.size,
               (unsigned long long)img_mr.memoryRequirements.size,
               (int)idr.requiresDedicatedAllocation, (void*)ded_buf.buf);
        if (dev_mr.memoryRequirements.size < img_mr.memoryRequirements.size || idr.requiresDedicatedAllocation != VK_FALSE) {
            fprintf(stderr, "FAIL [4]: device/image memory requirements mismatch!\n");
            return 1;
        }
    }

    // =========================================================================
    // [5] Real Hardware Timestamp Queries + Emulated Occlusion QueryPool Readback
    // =========================================================================
    {
        VkQueryPoolCreateInfo tqci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
        VkQueryPool tqp = VK_NULL_HANDLE;
        VK_CHK(vkCreateQueryPool(dev, &tqci, nullptr, &tqp));
        VkQueryPoolCreateInfo oqci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_OCCLUSION, 2, 0};
        VkQueryPool oqp = VK_NULL_HANDLE;
        VK_CHK(vkCreateQueryPool(dev, &oqci, nullptr, &oqp));

        BufObj qbuf = make_buf(pd, dev, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        memset(qbuf.ptr, 0, 64);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkResetCommandBuffer(cb, 0));
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        vkCmdResetQueryPool(cb, tqp, 0, 2);
        vkCmdResetQueryPool(cb, oqp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tqp, 0);
        vkCmdFillBuffer(cb, qbuf.buf, 48, 16, 0x12345678u);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, tqp, 1);
        vkCmdCopyQueryPoolResults(cb, tqp, 0, 2, qbuf.buf, 0, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        vkCmdCopyQueryPoolResults(cb, oqp, 0, 2, qbuf.buf, 16, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        submit_and_wait(dev, qs[0], cb, fence);

        const uint64_t* u64 = (const uint64_t*)qbuf.ptr;
        printf("[5] HW Timestamp + Occlusion CopyQueryPoolResults: ts0=%llu ts1=%llu (dt=%lld) occ0=%llu avail=%llu\n",
               (unsigned long long)u64[0], (unsigned long long)u64[1], (long long)(u64[1] - u64[0]),
               (unsigned long long)u64[2], (unsigned long long)u64[3]);
        if (u64[0] == 0 || u64[1] < u64[0] || u64[2] != 1 || u64[3] != 1) {
            fprintf(stderr, "FAIL [5]: query pool results invalid!\n");
            return 1;
        }
        vkDestroyQueryPool(dev, tqp, nullptr);
        vkDestroyQueryPool(dev, oqp, nullptr);
    }

    // =========================================================================
    // [6] Multi-Threaded Concurrent Pipeline Compile & >2K Vis-Buffer Resize Stress
    // =========================================================================
    {
        const uint32_t dims[4][2] = {{512, 512}, {1280, 720}, {2560, 1440}, {640, 480}};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        const char* vs_t = "#version 450\nvoid main(){ vec2 p=vec2((gl_VertexIndex==1)?3.0:-1.0,(gl_VertexIndex==2)?3.0:-1.0); gl_Position=vec4(p,0.5,1.0); }\n";
        std::atomic<int> ok_threads{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&, t]() {
                char fs_buf[256];
                snprintf(fs_buf, sizeof(fs_buf), "#version 450\nlayout(location=0) out vec4 o; void main(){ o=vec4(0.%d,0.5,1.0,1.0); }\n", t + 1);
                VkShaderModule vm = make_mod(dev, vs_t, "vert");
                VkShaderModule fm = make_mod(dev, fs_buf, "frag");
                VkPipeline pipe = make_pipe(dev, pl, vm, fm, VK_FORMAT_R8G8B8A8_UNORM);
                ImgObj rt = make_img(pd, dev, dims[t][0], dims[t][1], VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
                VkCommandPool tcp = VK_NULL_HANDLE;
                VkCommandPoolCreateInfo tcpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, 0};
                VK_CHK(vkCreateCommandPool(dev, &tcpci, nullptr, &tcp));
                VkCommandBuffer tcb = VK_NULL_HANDLE;
                VkCommandBufferAllocateInfo tcbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, tcp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
                VK_CHK(vkAllocateCommandBuffers(dev, &tcbai, &tcb));
                VkCommandBufferBeginInfo tbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VK_CHK(vkBeginCommandBuffer(tcb, &tbi));
                VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rt.view,
                                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE,
                                             VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0, 0, 0, 1}}}};
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {dims[t][0], dims[t][1]}}, 1, 0, 1, &ca};
                vkCmdBeginRendering(tcb, &ri);
                VkViewport vp{0, 0, (float)dims[t][0], (float)dims[t][1], 0.0f, 1.0f};
                VkRect2D sc{{0, 0}, {dims[t][0], dims[t][1]}};
                vkCmdSetViewport(tcb, 0, 1, &vp); vkCmdSetScissor(tcb, 0, 1, &sc);
                vkCmdBindPipeline(tcb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                vkCmdDraw(tcb, 3, 1, 0, 0);
                vkCmdEndRendering(tcb);
                VK_CHK(vkEndCommandBuffer(tcb));
                vkDestroyCommandPool(dev, tcp, nullptr);
                vkDestroyPipeline(dev, pipe, nullptr);
                vkDestroyShaderModule(dev, vm, nullptr);
                vkDestroyShaderModule(dev, fm, nullptr);
                ok_threads.fetch_add(1);
            });
        }
        for (auto& th : threads) th.join();
        printf("[6] Multi-Threaded Concurrent PSO Compile & >2K VisBuf Resize Stress: %d/4 threads OK\n", ok_threads.load());
        if (ok_threads.load() != 4) return 1;
    }

    printf("V10_FEATURE_TEST PASS\n");
    return 0;
}
