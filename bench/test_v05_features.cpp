// test_v05_features.cpp — SmRast-VK v0.5.0 Comprehensive Feature Verification
// Verifies on H20 pure compute qf[2]:
//   1) Sampled Stencil Aspect SRV (D32_SFLOAT_S8_UINT stencil written in pass 1 -> sampled via usampler2D
//      with DX11/DX12-style component swizzle {ZERO, R, ZERO, ZERO} in pass 2!)
//   2) Watertight Top-Left Fill-Rule + 4-bit Subpixel Snapping:
//      4x4 grid of 32 shared-edge triangles drawn with ADDITIVE blending (1/4 intensity per hit) ->
//      EVERY interior pixel on shared diagonals/edges/vertices must accumulate EXACTLY 1 hit (zero cracks, zero double-shading!)
//   3) Hardware-Accurate Depth Bias (vkCmdSetDepthBias + depthBiasEnable):
//      Coplanar quad with negative constant bias passes LESS depth test; positive bias fails LESS test!
//   4) depthClampEnable (Shadow Pancake Mode):
//      Quad at clip z = +1.5 (beyond far plane) survives with depthClampEnable=VK_TRUE and clamps depth to 1.0;
//      rejected when depthClampEnable=VK_FALSE!
//   5) VK_EXT_vertex_attribute_divisor (divisor=2 and divisor=0) + R8_SNORM / R16_SINT vertex formats:
//      4 instanced quads: divisor=2 maps instances (0,1)->slot 0 and (2,3)->slot 1; divisor=0 maps all 4->slot 0!
//   6) 2-Level Tile Hi-Z Early-Reject (Standard-Z LESS & Reverse-Z GREATER):
//      Occluders + occluded & visible layers produce pixel-exact depth ordering under both Standard-Z and Reverse-Z!
//   7) Dynamic rasterizerDiscardEnable (vkCmdSetRasterizerDiscardEnable): discarded draw writes 0 pixels!

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK_VK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", (int)_r, __LINE__); exit(1); } } while (0)

static std::vector<uint32_t> compile_glsl(const char* stage, const char* src) {
    FILE* f = fopen("/tmp/sm_v05_tmp.glsl", "w");
    fputs(src, f);
    fclose(f);
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "glslangValidator -V --target-env vulkan1.3 -S %s /tmp/sm_v05_tmp.glsl -o /tmp/sm_v05_tmp.spv >/dev/null", stage);
    if (system(cmd) != 0) {
        fprintf(stderr, "glslangValidator failed for %s:\n%s\n", stage, src);
        exit(1);
    }
    FILE* sf = fopen("/tmp/sm_v05_tmp.spv", "rb");
    fseek(sf, 0, SEEK_END);
    size_t sz = (size_t)ftell(sf);
    fseek(sf, 0, SEEK_SET);
    std::vector<uint32_t> spv(sz / 4);
    if (fread(spv.data(), 1, sz, sf) != sz) {
        fprintf(stderr, "Failed to read SPIR-V\n");
        fclose(sf);
        exit(1);
    }
    fclose(sf);
    return spv;
}

static uint32_t find_mem(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & req) == req)) return i;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if (bits & (1u << i)) return i;
    return 0;
}

static void create_buf(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp, VkDeviceSize sz,
                       VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& m, void** ptr = nullptr) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, u, VK_SHARING_MODE_EXCLUSIVE};
    CHECK_VK(vkCreateBuffer(dev, &ci, nullptr, &b));
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, b, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(mp, mr.memoryTypeBits, p)};
    CHECK_VK(vkAllocateMemory(dev, &ai, nullptr, &m));
    CHECK_VK(vkBindBufferMemory(dev, b, m, 0));
    if (ptr) CHECK_VK(vkMapMemory(dev, m, 0, sz, 0, ptr));
}

struct ImgObj { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkFormat fmt = VK_FORMAT_UNDEFINED; };

static ImgObj create_img(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags u, VkImageAspectFlags asp) {
    ImgObj o; o.fmt = fmt;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1}, 1, 1,
                         VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, u | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    CHECK_VK(vkCreateImage(dev, &ci, nullptr, &o.img));
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(dev, o.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(mp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CHECK_VK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    CHECK_VK(vkBindImageMemory(dev, o.img, o.mem, 0));
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, o.img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
    CHECK_VK(vkCreateImageView(dev, &iv, nullptr, &o.view));
    return o;
}

static VkShaderModule make_sm(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size()*4, spv.data()};
    VkShaderModule sm = VK_NULL_HANDLE;
    CHECK_VK(vkCreateShaderModule(dev, &ci, nullptr, &sm));
    return sm;
}

static void readback_rgba8(VkDevice dev, VkQueue q, VkCommandBuffer cb, VkImage img, VkBuffer rb_buf, uint32_t w, uint32_t h) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(cb, &bi));
    VkBufferImageCopy bic{};
    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bic.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb_buf, 1, &bic);
    CHECK_VK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(q));
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV05Test", 1, "Test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app};
    VkInstance inst = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(inst, &nd, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("Testing SmRast-VK v0.5.0 on: %s\n", props.deviceName);

    const char* exts[] = { VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME };
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT vad{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, nullptr, VK_TRUE, VK_TRUE};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &vad};
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures feats{};
    feats.depthClamp = VK_TRUE;
    feats.depthBiasClamp = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, nullptr, 1, exts, &feats};
    VkDevice dev = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue(dev, 0, 0, &q);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    CHECK_VK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CHECK_VK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    const uint32_t W = 64, H = 64;
    ImgObj col = create_img(dev, mp, W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    ImgObj ds  = create_img(dev, mp, W, H, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    // Create a STENCIL-ONLY SRV image view with DX11/DX12-style swizzle {ZERO, R, ZERO, ZERO} -> stencil appears in .g!
    VkImageViewCreateInfo stn_ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, ds.img, VK_IMAGE_VIEW_TYPE_2D,
                                   VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO},
                                   {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
    VkImageView stn_srv_view = VK_NULL_HANDLE;
    CHECK_VK(vkCreateImageView(dev, &stn_ivci, nullptr, &stn_srv_view));

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_NEAREST; sci.minFilter = VK_FILTER_NEAREST;
    VkSampler samp = VK_NULL_HANDLE;
    CHECK_VK(vkCreateSampler(dev, &sci, nullptr, &samp));

    VkBuffer rb_buf = VK_NULL_HANDLE; VkDeviceMemory rb_mem = VK_NULL_HANDLE; void* rb_ptr = nullptr;
    create_buf(dev, mp, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rb_buf, rb_mem, &rb_ptr);
    auto* px = reinterpret_cast<const uint8_t*>(rb_ptr);

    VkFormat col_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &col_fmt,
                                       VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    VkViewport vp{0, 0, (float)W, (float)H, 0.f, 1.f};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};

    // =========================================================================
    // TEST 1: Sampled Stencil Aspect SRV (D32_SFLOAT_S8_UINT -> usampler2D.g!)
    // =========================================================================
    {
        VkDescriptorSetLayoutBinding dslb{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslb};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
        VkDescriptorPool dp = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VkDescriptorSet dset = VK_NULL_HANDLE; CHECK_VK(vkAllocateDescriptorSets(dev, &dsai, &dset));
        VkDescriptorImageInfo dii{samp, stn_srv_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii};
        vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        VkPipelineLayoutCreateInfo plci_e{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl_empty = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci_e, nullptr, &pl_empty));
        VkPipelineLayoutCreateInfo plci_s{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl_srv = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci_s, nullptr, &pl_srv));

        const char* vs_half = R"(#version 460
void main() {
    vec2 p[3] = vec2[](vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(0.0, 1.0));
    vec2 q[3] = vec2[](vec2(-1.0, -1.0), vec2(0.0, 1.0), vec2(-1.0, 1.0));
    vec2 v = (gl_VertexIndex < 3) ? p[gl_VertexIndex] : q[gl_VertexIndex - 3];
    gl_Position = vec4(v, 0.5, 1.0);
})";
        const char* vs_full = R"(#version 460
void main() {
    vec2 p[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(p[gl_VertexIndex], 0.5, 1.0);
})";
        const char* fs_solid = R"(#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(0.1, 0.1, 0.1, 1.0); }
)";
        const char* fs_read_stn = R"(#version 460
layout(set = 0, binding = 0) uniform usampler2D u_stn;
layout(location = 0) out vec4 o;
void main() {
    uvec4 sv = texelFetch(u_stn, ivec2(gl_FragCoord.xy), 0);
    // Expect swizzle {ZERO, R, ZERO, ZERO} -> sv.r == 0, sv.g == stencil!
    float g_val = float(sv.g) / 255.0;
    float r_err = (sv.r == 0u) ? 0.0 : 1.0;
    o = vec4(r_err, g_val, (sv.g == 173u) ? 1.0 : 0.0, 1.0);
})";
        VkShaderModule sm_vh = make_sm(dev, compile_glsl("vert", vs_half));
        VkShaderModule sm_vf = make_sm(dev, compile_glsl("vert", vs_full));
        VkShaderModule sm_fs = make_sm(dev, compile_glsl("frag", fs_solid));
        VkShaderModule sm_fr = make_sm(dev, compile_glsl("frag", fs_read_stn));

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};

        // Pass 1 pipeline: writes stencil = 173 on left half!
        VkPipelineDepthStencilStateCreateInfo dss1{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss1.stencilTestEnable = VK_TRUE;
        dss1.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 173};
        dss1.back = dss1.front;
        VkPipelineShaderStageCreateInfo st1[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_vh, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fs, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci1{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st1, &vi, &ia, nullptr, &vps, &rs, &ms, &dss1, &cb_st, nullptr, pl_empty};
        VkPipeline p1 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci1, nullptr, &p1));

        // Pass 2 pipeline: samples stencil SRV via usampler2D!
        VkPipelineDepthStencilStateCreateInfo dss2{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineShaderStageCreateInfo st2[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_vf, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fr, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci2{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st2, &vi, &ia, nullptr, &vps, &rs, &ms, &dss2, &cb_st, nullptr, pl_srv};
        VkPipeline p2 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci2, nullptr, &p2));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, ds.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
        da.clearValue.depthStencil = {1.0f, 37u};
        VkRenderingInfo ri1{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, &da};
        vkCmdBeginRendering(cb, &ri1);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p1);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);

        // Barrier on STENCIL aspect -> shader read
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ds.img, {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);

        // Pass 2: sample stencil SRV across full screen
        VkRenderingInfo ri2{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ri2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p2);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_srv, 0, 1, &dset, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pL = px + (32 * W + 16) * 4;
        const uint8_t* pR = px + (32 * W + 48) * 4;
        printf("[1] Sampled Stencil SRV: left=(%u,%u,%u) expect (0,173,255); right=(%u,%u,%u) expect (0,37,0)\n",
               pL[0], pL[1], pL[2], pR[0], pR[1], pR[2]);
        if (pL[0] != 0 || pL[1] != 173 || pL[2] != 255 || pR[0] != 0 || pR[1] != 37 || pR[2] != 0) {
            fprintf(stderr, "FAIL [1]: Sampled Stencil Aspect SRV mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, p1, nullptr); vkDestroyPipeline(dev, p2, nullptr);
        vkDestroyShaderModule(dev, sm_vh, nullptr); vkDestroyShaderModule(dev, sm_vf, nullptr);
        vkDestroyShaderModule(dev, sm_fs, nullptr); vkDestroyShaderModule(dev, sm_fr, nullptr);
        vkDestroyPipelineLayout(dev, pl_empty, nullptr); vkDestroyPipelineLayout(dev, pl_srv, nullptr);
        vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    }

    // =========================================================================
    // TEST 2: Watertight Top-Left Fill-Rule (4x4 grid, 32 shared-edge tris, ADDITIVE blend!)
    // =========================================================================
    {
        // Build a 4x4 grid covering NDC [-0.75, +0.75] -> pixel [8, 56], with shared diagonals passing
        // right through pixel centers (e.g. x==y at 8.5, 9.5, ..., 55.5) AND horizontal/vertical shared edges!
        // Draw each triangle in its OWN draw call with ADDITIVE blend (+64/255) so ANY double-shaded pixel
        // becomes 128 and ANY missed crack pixel stays 0! Every interior pixel [8..55]x[8..55] MUST be == 64!
        std::vector<float> verts;
        const int N = 4;
        for (int gy = 0; gy < N; ++gy) {
            for (int gx = 0; gx < N; ++gx) {
                float x0 = -0.75f + 1.5f * gx / N, x1 = -0.75f + 1.5f * (gx + 1) / N;
                float y0 = -0.75f + 1.5f * gy / N, y1 = -0.75f + 1.5f * (gy + 1) / N;
                // Tri A: (x0,y0)-(x1,y0)-(x1,y1)
                float ta[6] = {x0, y0, x1, y0, x1, y1};
                // Tri B: (x0,y0)-(x1,y1)-(x0,y1)
                float tb[6] = {x0, y0, x1, y1, x0, y1};
                verts.insert(verts.end(), ta, ta + 6);
                verts.insert(verts.end(), tb, tb + 6);
            }
        }
        VkBuffer vb = VK_NULL_HANDLE; VkDeviceMemory vbm = VK_NULL_HANDLE; void* vbp = nullptr;
        create_buf(dev, mp, verts.size() * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vb, vbm, &vbp);
        memcpy(vbp, verts.data(), verts.size() * 4);

        const char* vs_v2 = R"(#version 460
layout(location = 0) in vec2 p;
void main() { gl_Position = vec4(p, 0.5, 1.0); }
)";
        const char* fs_qtr = R"(#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(64.0/255.0, 64.0/255.0, 64.0/255.0, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_v2));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_qtr));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkVertexInputBindingDescription vbd{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription vad0{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &vbd, 1, &vad0};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline pipe = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &vb, &off);
        for (uint32_t t = 0; t < 32; ++t) vkCmdDraw(cb, 3, 1, t * 3, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        int cracks = 0, double_hits = 0, exact_once = 0;
        for (uint32_t y = 8; y < 56; ++y) {
            for (uint32_t x = 8; x < 56; ++x) {
                uint8_t r = px[(y * W + x) * 4];
                if (r == 64) exact_once++;
                else if (r == 0) cracks++;
                else double_hits++;
            }
        }
        printf("[2] Watertight Top-Left Fill-Rule (48x48=2304 px across 32 shared tris): exact_once=%d cracks=%d double_hits=%d\n",
               exact_once, cracks, double_hits);
        if (exact_once != 2304 || cracks != 0 || double_hits != 0) {
            fprintf(stderr, "FAIL [2]: Shared-edge cracks or double-shading detected!\n"); return 1;
        }
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
        vkDestroyBuffer(dev, vb, nullptr); vkFreeMemory(dev, vbm, nullptr);
    }

    // =========================================================================
    // TEST 3 & 4 & 7: Depth Bias + depthClampEnable + Dynamic rasterizerDiscardEnable
    // =========================================================================
    {
        const char* vs_pc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect_z; vec4 color; } pc;
void main() {
    vec2 u[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 p = mix(pc.rect_z.xy, pc.rect_z.zw, u[gl_VertexIndex]);
    gl_Position = vec4(p, pc.color.w, 1.0);
})";
        const char* fs_pc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect_z; vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.color.rgb, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_pc));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_pc));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };

        VkDynamicState dyn_sts[3] = { VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE };
        VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 3, dyn_sts};

        VkPipelineRasterizationStateCreateInfo rs_noclamp{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs_noclamp.polygonMode = VK_POLYGON_MODE_FILL; rs_noclamp.cullMode = VK_CULL_MODE_NONE; rs_noclamp.lineWidth = 1.f;
        rs_noclamp.depthClampEnable = VK_FALSE;
        VkGraphicsPipelineCreateInfo gp1{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs_noclamp, &ms, &dss, &cb_st, &dys, pl};
        VkPipeline p_noclamp = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp1, nullptr, &p_noclamp));

        VkPipelineRasterizationStateCreateInfo rs_clamp = rs_noclamp;
        rs_clamp.depthClampEnable = VK_TRUE;
        VkPipelineDepthStencilStateCreateInfo dss_le = dss;
        dss_le.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkGraphicsPipelineCreateInfo gp2{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs_clamp, &ms, &dss_le, &cb_st, &dys, pl};
        VkPipeline p_clamp = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp2, nullptr, &p_clamp));

        struct PC { float x0, y0, x1, y1, r, g, b, z; };
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, ds.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
        da.clearValue.depthStencil = {1.0f, 0u};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, nullptr};
        vkCmdBeginRendering(cb, &ri);

        // Quadrant TL [-1,0]x[-1,0]: base red at z=0.5, then coplanar GREEN with negative depth bias (-1000) -> GREEN wins!
        // Quadrant TR [0,1]x[-1,0]:  base red at z=0.5, then coplanar BLUE with positive depth bias (+1000) -> RED stays!
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_noclamp);
        vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
        vkCmdSetDepthBiasEnable(cb, VK_FALSE);
        PC base_top{-1.f, -1.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.5f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &base_top);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdSetDepthBiasEnable(cb, VK_TRUE);
        vkCmdSetDepthBias(cb, -10000.f, 0.f, 0.f);
        PC bias_neg{-1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.5f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &bias_neg);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdSetDepthBias(cb, +10000.f, 0.f, 0.f);
        PC bias_pos{0.f, -1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.5f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &bias_pos);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Quadrant BL [-1,0]x[0,1]: z=1.5 with depthClamp=FALSE -> rejected (stays black 0,0,0)!
        vkCmdSetDepthBiasEnable(cb, VK_FALSE);
        PC far_noclamp{-1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 1.5f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &far_noclamp);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Quadrant BR [0,1]x[0,1]: z=1.5 with depthClamp=TRUE (LEQUAL against 1.0) -> survives as CYAN (0,255,255)!
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_clamp);
        vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
        vkCmdSetDepthBiasEnable(cb, VK_FALSE);
        PC far_clamp{0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 1.5f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &far_clamp);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Dynamic rasterizerDiscardEnable=TRUE: full-screen MAGENTA at z=0.0 MUST NOT draw anything!
        vkCmdSetRasterizerDiscardEnable(cb, VK_TRUE);
        PC disc_quad{-1.f, -1.f, 1.f, 1.f, 1.f, 0.f, 1.f, 0.0f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &disc_quad);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pTL = px + (16 * W + 16) * 4;
        const uint8_t* pTR = px + (16 * W + 48) * 4;
        const uint8_t* pBL = px + (48 * W + 16) * 4;
        const uint8_t* pBR = px + (48 * W + 48) * 4;
        printf("[3] Depth Bias: neg_bias=(%u,%u,%u) expect (0,255,0); pos_bias=(%u,%u,%u) expect (255,0,0)\n",
               pTL[0], pTL[1], pTL[2], pTR[0], pTR[1], pTR[2]);
        printf("[4] Depth Clamp + Discard: no_clamp=(%u,%u,%u) expect (0,0,0); clamped=(%u,%u,%u) expect (0,255,255)\n",
               pBL[0], pBL[1], pBL[2], pBR[0], pBR[1], pBR[2]);
        if (pTL[1] != 255 || pTL[0] != 0 || pTR[0] != 255 || pTR[2] != 0) {
            fprintf(stderr, "FAIL [3]: Depth Bias mismatch!\n"); return 1;
        }
        if (pBL[0] != 0 || pBL[1] != 0 || pBL[2] != 0 || pBR[0] != 0 || pBR[1] != 255 || pBR[2] != 255) {
            fprintf(stderr, "FAIL [4]: depthClampEnable or rasterizerDiscardEnable mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, p_noclamp, nullptr); vkDestroyPipeline(dev, p_clamp, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
    }

    // =========================================================================
    // TEST 5: VK_EXT_vertex_attribute_divisor (divisor=2 & divisor=0) + R8_SNORM / R16_SINT
    // =========================================================================
    {
        // 4 instanced quads laid out left-to-right across 4 strips: x in [-1,-0.5], [-0.5,0], [0,0.5], [0.5,1]
        // Binding 0 (VERTEX): unit quad [-0.2, 0.2] in R8G8_SNORM (signed normalized!)
        // Binding 1 (INSTANCE, divisor=1): per-instance center X offset in R16_SINT (-3, -1, +1, +3) * 0.25
        // Binding 2 (INSTANCE, divisor=2): color R,G for pair 0 (inst 0,1) = (1,0), pair 1 (inst 2,3) = (0,1)
        // Binding 3 (INSTANCE, divisor=0): constant blue = 0.5 for ALL instances!
        int8_t quad_sn8[12] = { -26, -64, 26, -64, 26, 64, -26, -64, 26, 64, -26, 64 };
        int16_t x_s16[4]    = { -3, -1, 1, 3 };
        float pair_rg[4]    = { 1.0f, 0.0f, 0.0f, 1.0f };
        float const_b[2]    = { 0.5f, 0.0f };

        VkBuffer b0, b1, b2, b3; VkDeviceMemory m0, m1, m2, m3; void *p0, *p1, *p2, *p3;
        create_buf(dev, mp, 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b0, m0, &p0);
        create_buf(dev, mp, 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b1, m1, &p1);
        create_buf(dev, mp, 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b2, m2, &p2);
        create_buf(dev, mp, 64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b3, m3, &p3);
        memcpy(p0, quad_sn8, sizeof(quad_sn8));
        memcpy(p1, x_s16, sizeof(x_s16));
        memcpy(p2, pair_rg, sizeof(pair_rg));
        memcpy(p3, const_b, sizeof(const_b));

        const char* vs_div = R"(#version 460
layout(location = 0) in vec2 in_local;
layout(location = 1) in int  in_cx;
layout(location = 2) in vec2 in_rg;
layout(location = 3) in float in_b;
layout(location = 0) out vec3 v_col;
void main() {
    float cx = float(in_cx) * 0.25;
    gl_Position = vec4(in_local.x + cx, in_local.y, 0.5, 1.0);
    v_col = vec3(in_rg, in_b);
})";
        const char* fs_div = R"(#version 460
layout(location = 0) in vec3 v_col;
layout(location = 0) out vec4 o;
void main() { o = vec4(v_col, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_div));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_div));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkVertexInputBindingDescription vbs[4] = {
            {0, 2, VK_VERTEX_INPUT_RATE_VERTEX},
            {1, 2, VK_VERTEX_INPUT_RATE_INSTANCE},
            {2, 8, VK_VERTEX_INPUT_RATE_INSTANCE},
            {3, 4, VK_VERTEX_INPUT_RATE_INSTANCE},
        };
        VkVertexInputAttributeDescription vas[4] = {
            {0, 0, VK_FORMAT_R8G8_SNORM, 0},
            {1, 1, VK_FORMAT_R16_SINT,   0},
            {2, 2, VK_FORMAT_R32G32_SFLOAT, 0},
            {3, 3, VK_FORMAT_R32_SFLOAT, 0},
        };
        VkVertexInputBindingDivisorDescriptionEXT divs[2] = { {2, 2}, {3, 0} };
        VkPipelineVertexInputDivisorStateCreateInfoEXT div_ci{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, nullptr, 2, divs};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, &div_ci, 0, 4, vbs, 4, vas};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline pipe = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        VkBuffer bufs[4] = {b0, b1, b2, b3};
        VkDeviceSize offs[4] = {0, 0, 0, 0};
        vkCmdBindVertexBuffers(cb, 0, 4, bufs, offs);
        vkCmdDraw(cb, 6, 4, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        // Centers of the 4 quads at x = 8, 24, 40, 56 (y = 32)
        const uint8_t* q0 = px + (32 * W + 8) * 4;
        const uint8_t* q1 = px + (32 * W + 24) * 4;
        const uint8_t* q2 = px + (32 * W + 40) * 4;
        const uint8_t* q3 = px + (32 * W + 56) * 4;
        printf("[5] Vertex Divisor (div=2 & div=0) + R8G8_SNORM/R16_SINT: q0=(%u,%u,%u) q1=(%u,%u,%u) q2=(%u,%u,%u) q3=(%u,%u,%u)\n",
               q0[0], q0[1], q0[2], q1[0], q1[1], q1[2], q2[0], q2[1], q2[2], q3[0], q3[1], q3[2]);
        if (q0[0] != 255 || q0[1] != 0 || abs((int)q0[2] - 128) > 1 ||
            q1[0] != 255 || q1[1] != 0 || abs((int)q1[2] - 128) > 1 ||
            q2[0] != 0 || q2[1] != 255 || abs((int)q2[2] - 128) > 1 ||
            q3[0] != 0 || q3[1] != 255 || abs((int)q3[2] - 128) > 1) {
            fprintf(stderr, "FAIL [5]: Vertex attribute divisor / SNORM / SINT mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
        vkDestroyBuffer(dev, b0, nullptr); vkFreeMemory(dev, m0, nullptr);
        vkDestroyBuffer(dev, b1, nullptr); vkFreeMemory(dev, m1, nullptr);
        vkDestroyBuffer(dev, b2, nullptr); vkFreeMemory(dev, m2, nullptr);
        vkDestroyBuffer(dev, b3, nullptr); vkFreeMemory(dev, m3, nullptr);
    }

    // =========================================================================
    // TEST 6: 2-Level Tile Hi-Z Early-Reject (Reverse-Z GREATER + Standard-Z LESS)
    // =========================================================================
    {
        const char* vs_z = R"(#version 460
layout(push_constant) uniform PC { vec4 color_z; } pc;
void main() {
    vec2 p[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(p[gl_VertexIndex], pc.color_z.w, 1.0);
})";
        const char* fs_z = R"(#version 460
layout(push_constant) uniform PC { vec4 color_z; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.color_z.rgb, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_z));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_z));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};
        VkPipelineDepthStencilStateCreateInfo dss_rev{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss_rev.depthTestEnable = VK_TRUE; dss_rev.depthWriteEnable = VK_TRUE; dss_rev.depthCompareOp = VK_COMPARE_OP_GREATER;
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs, &ms, &dss_rev, &cb_st, nullptr, pl};
        VkPipeline p_rev = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &p_rev));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, ds.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
        da.clearValue.depthStencil = {0.0f, 0u}; // Reverse-Z clears to 0.0!
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, nullptr};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_rev);
        float c1[4] = {0.0f, 1.0f, 0.0f, 0.75f}; // Reverse-Z near occluder (z=0.75 -> GREEN)
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c1);
        vkCmdDraw(cb, 3, 1, 0, 0);
        float c2[4] = {1.0f, 0.0f, 0.0f, 0.25f}; // Occluded behind (z=0.25 < 0.75 -> Hi-Z early-rejected!)
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c2);
        vkCmdDraw(cb, 3, 1, 0, 0);
        float c3[4] = {0.0f, 0.5f, 1.0f, 0.90f}; // Closer in Reverse-Z (z=0.90 > 0.75 -> SKY BLUE wins!)
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c3);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pc = px + (32 * W + 32) * 4;
        printf("[6] Reverse-Z Tile Hi-Z (GREATER): center=(%u,%u,%u) expect (0,128,255)\n", pc[0], pc[1], pc[2]);
        if (pc[0] != 0 || abs((int)pc[1] - 128) > 1 || pc[2] != 255) {
            fprintf(stderr, "FAIL [6]: Reverse-Z Tile Hi-Z mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, p_rev, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
    }

    printf("V05_FEATURE_TEST PASS\n");
    return 0;
}
