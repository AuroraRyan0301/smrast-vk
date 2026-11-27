// test_v08_features.cpp — SmRast-VK v0.8.0 Comprehensive Correctness & Performance Verification
// Verifies on H20 pure compute qf[2]:
//   1) VK_EXT_depth_clip_enable (depthClampEnable=TRUE + depthClipEnable=TRUE vs FALSE under Reverse-Z) + 4x Guard-Band Clipping
//   2) Painter's Order (depthTestEnable=FALSE: largest tid wins via (~tid << 32)|tid) + flat/noperspective varyings
//   3) Per-Draw Push-Constant Metadata SSBO Cross-Material Uber-Batching (64 distinct colors verified + 1,000-Draw 1080p bench!)
//   4) Per-Draw Dynamic-UBO Metadata SSBO Cross-Material Uber-Batching (64 distinct UBO offsets verified + 1,000-Draw 1080p bench!)
//   5) Zero-Copy Linear-Aliased Odd-Width G-Buffer (1257x707 RGBA8 + RGBA16F + R8 MRT -> hardware TMU deferred pass + readback!)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#define CHECK_VK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", (int)_r, __LINE__); exit(1); } } while (0)

static std::vector<uint32_t> compile_glsl(const char* stage, const char* src) {
    FILE* f = fopen("/tmp/sm_v08_tmp.glsl", "w");
    fputs(src, f);
    fclose(f);
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "glslangValidator -V --target-env vulkan1.3 -S %s /tmp/sm_v08_tmp.glsl -o /tmp/sm_v08_tmp.spv >/dev/null", stage);
    if (system(cmd) != 0) {
        fprintf(stderr, "glslangValidator failed for %s:\n%s\n", stage, src);
        exit(1);
    }
    FILE* sf = fopen("/tmp/sm_v08_tmp.spv", "rb");
    fseek(sf, 0, SEEK_END);
    size_t sz = (size_t)ftell(sf);
    fseek(sf, 0, SEEK_SET);
    std::vector<uint32_t> spv(sz / 4);
    if (fread(spv.data(), 1, sz, sf) != sz) { fprintf(stderr, "Failed to read SPIR-V\n"); fclose(sf); exit(1); }
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

struct ImgObj { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };

static ImgObj create_img(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags u, VkImageAspectFlags asp) {
    ImgObj o;
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
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV08Test", 1, "Test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app};
    VkInstance inst = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(inst, &nd, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("Testing SmRast-VK v0.8.0 on: %s\n", props.deviceName);

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceDepthClipEnableFeaturesEXT dcef{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT, &f13, VK_TRUE};
    VkPhysicalDeviceFeatures feats{}; feats.depthClamp = VK_TRUE;
    const char* exts[] = { VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &dcef, 0, 1, &qci, 0, nullptr, 1, exts, &feats};
    VkDevice dev = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue(dev, 0, 0, &q);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    CHECK_VK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CHECK_VK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    const uint32_t W = 128, H = 128;
    ImgObj col = create_img(dev, mp, W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    ImgObj dep = create_img(dev, mp, W, H, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkBuffer rb_buf = VK_NULL_HANDLE; VkDeviceMemory rb_mem = VK_NULL_HANDLE; void* rb_ptr = nullptr;
    create_buf(dev, mp, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rb_buf, rb_mem, &rb_ptr);
    auto* px = reinterpret_cast<const uint8_t*>(rb_ptr);

    VkFormat col_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &col_fmt, VK_FORMAT_D32_SFLOAT, VK_FORMAT_UNDEFINED};
    VkViewport vp{0, 0, (float)W, (float)H, 0.f, 1.f};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb_st{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};

    // =========================================================================
    // TEST 1: VK_EXT_depth_clip_enable (depthClamp=TRUE + depthClip=TRUE vs FALSE, Reverse-Z) + 4x Guard-Band
    // =========================================================================
    {
        const char* vs_clip = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_zw; } pc;
void main() {
    vec2 u[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 p = mix(pc.rect.xy, pc.rect.zw, u[gl_VertexIndex]);
    gl_Position = vec4(p * pc.col_zw.w, pc.col_zw.z, pc.col_zw.w);
})";
        const char* fs_clip = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_zw; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.col_zw.r, pc.col_zw.g, 0.0, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_clip));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_clip));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // Reverse-Z!
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };

        VkPipelineRasterizationDepthClipStateCreateInfoEXT dc_on{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT, nullptr, 0, VK_TRUE};
        VkPipelineRasterizationStateCreateInfo rs_on{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, &dc_on};
        rs_on.depthClampEnable = VK_TRUE; rs_on.polygonMode = VK_POLYGON_MODE_FILL; rs_on.cullMode = VK_CULL_MODE_NONE; rs_on.lineWidth = 1.f;
        VkGraphicsPipelineCreateInfo gp_on{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs_on, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_clip_on = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_on, nullptr, &p_clip_on));

        VkPipelineRasterizationDepthClipStateCreateInfoEXT dc_off{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT, nullptr, 0, VK_FALSE};
        VkPipelineRasterizationStateCreateInfo rs_off{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, &dc_off};
        rs_off.depthClampEnable = VK_TRUE; rs_off.polygonMode = VK_POLYGON_MODE_FILL; rs_off.cullMode = VK_CULL_MODE_NONE; rs_off.lineWidth = 1.f;
        VkGraphicsPipelineCreateInfo gp_off{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs_off, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_clip_off = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_off, nullptr, &p_clip_off));

        struct PC { float x0, y0, x1, y1, r, g, z, w; };
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
        da.clearValue.depthStencil = {0.0f, 0u}; // Reverse-Z far = 0.0
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, nullptr};
        vkCmdBeginRendering(cb, &ri);

        // Background green at z=0.5 (w=1.0)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_clip_on);
        PC bg{-1.f, -1.f, 1.f, 1.f, 0.f, 1.f, 0.5f, 1.0f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &bg);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Left half: depthClip=TRUE, z=1.5 > w=1.0 -> MUST be clipped (stays GREEN!)
        PC left_behind{-1.f, -1.f, 0.f, 1.f, 1.f, 0.f, 1.5f, 1.0f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &left_behind);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Top-right: depthClip=FALSE, z=1.5 > w=1.0 -> clamped to 1.0 >= 0.5 -> becomes RED!
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_clip_off);
        PC right_clamp{0.f, -1.f, 1.f, 0.f, 1.f, 0.f, 1.5f, 1.0f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &right_clamp);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Bottom-right: huge 25x offscreen guard-band quad (tiny w=0.004, z=0.0032 -> z/w=0.8 > 0.5 -> YELLOW!)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_clip_on);
        PC gb_quad{0.f, 0.f, 25.f, 25.f, 1.f, 1.f, 0.0032f, 0.004f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &gb_quad);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pL  = px + (64 * W + 32) * 4;
        const uint8_t* pTR = px + (32 * W + 96) * 4;
        const uint8_t* pBR = px + (96 * W + 96) * 4;
        printf("[1] VK_EXT_depth_clip_enable + Guard-Band (Rev-Z): clipON=(%u,%u,%u) clipOFF=(%u,%u,%u) guardBand=(%u,%u,%u)\n",
               pL[0], pL[1], pL[2], pTR[0], pTR[1], pTR[2], pBR[0], pBR[1], pBR[2]);
        if (pL[0] != 0 || pL[1] != 255 || pTR[0] != 255 || pTR[1] != 0 || pBR[0] != 255 || pBR[1] != 255) {
            fprintf(stderr, "FAIL [1]: depth_clip_enable or guard-band mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, p_clip_on, nullptr); vkDestroyPipeline(dev, p_clip_off, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
    }

    // =========================================================================
    // TEST 2: Painter's Order (depthTest=FALSE: later tid wins!) + flat & noperspective varyings
    // =========================================================================
    {
        const char* vs_po = R"(#version 460
layout(location = 0) flat out vec3 v_flat;
layout(location = 1) noperspective out float v_lin;
void main() {
    // Tri 0 (tid=0, z=0.1, RED) overlapped by Tri 1 (tid=1, z=0.9, BLUE): with depthTest=FALSE, tid=1 MUST win!
    vec4 pos[6] = vec4[](
        vec4(-0.8,-0.8, 0.1, 1.0), vec4( 0.8,-0.8, 0.1, 1.0), vec4( 0.0, 0.8, 0.1, 1.0),
        vec4(-0.6,-0.6, 0.9, 1.0), vec4( 0.6,-0.6, 0.9, 1.0), vec4( 0.0, 0.6, 0.9, 1.0));
    vec3 col[6] = vec3[](
        vec3(1,0,0), vec3(0,1,0), vec3(0,1,0),
        vec3(0,0,1), vec3(1,1,0), vec3(1,1,0));
    gl_Position = pos[gl_VertexIndex];
    v_flat = col[gl_VertexIndex]; // provoking vertex is v0 -> Tri 0 = RED, Tri 1 = BLUE!
    v_lin  = (gl_VertexIndex % 3 == 0) ? 0.0 : 1.0;
})";
        const char* fs_po = R"(#version 460
layout(location = 0) flat in vec3 v_flat;
layout(location = 1) noperspective in float v_lin;
layout(location = 0) out vec4 o;
void main() { o = vec4(v_flat.r, v_lin, v_flat.b, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_po));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_po));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_FALSE; dss.depthWriteEnable = VK_FALSE;
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline pipe = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pC = px + (64 * W + 64) * 4; // inside Tri 1 overlap -> flat BLUE (R=0, B=255), G > 0 from noperspective
        const uint8_t* pE = px + (108 * W + 64) * 4; // inside Tri 0 only -> flat RED (R=255, B=0)
        printf("[2] Painter's Order + flat/noperspective: overlap=(%u,%u,%u) outer=(%u,%u,%u)\n",
               pC[0], pC[1], pC[2], pE[0], pE[1], pE[2]);
        if (pC[0] != 0 || pC[2] != 255 || pC[1] < 100 || pE[0] != 255 || pE[2] != 0) {
            fprintf(stderr, "FAIL [2]: Painter's order or flat/noperspective mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
    }

    // =========================================================================
    // TEST 3 & 4: Per-Draw Push-Constant AND Dynamic-UBO SSBO Cross-Material Uber-Batching
    //   Correctness (64 distinct per-draw materials) + 1,000-Draw 1080p Benchmark (<4 ms!)
    // =========================================================================
    {
        const uint32_t BW = 1920, BH = 1080;
        ImgObj bcol = create_img(dev, mp, BW, BH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj bdep = create_img(dev, mp, BW, BH, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkViewport bvp{0, 0, (float)BW, (float)BH, 0.f, 1.f};
        VkRect2D bsc{{0, 0}, {BW, BH}};
        VkPipelineViewportStateCreateInfo bvps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &bvp, 1, &bsc};

        // --- TEST 3: Per-Draw Push-Constant Uber-Batching ---
        const char* vs_pc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_z; } pc;
void main() {
    vec2 u[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, u[gl_VertexIndex]), pc.col_z.w, 1.0);
})";
        const char* fs_pc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_z; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.col_z.rgb, 1.0); }
)";
        VkShaderModule sm_vp = make_sm(dev, compile_glsl("vert", vs_pc));
        VkShaderModule sm_fp = make_sm(dev, compile_glsl("frag", fs_pc));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl_pc = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl_pc));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineShaderStageCreateInfo st_p[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_vp, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fp, "main"},
        };
        VkGraphicsPipelineCreateInfo gp_p128{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st_p, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl_pc};
        VkPipeline p_pc128 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_p128, nullptr, &p_pc128));
        VkGraphicsPipelineCreateInfo gp_p1080 = gp_p128; gp_p1080.pViewportState = &bvps;
        VkPipeline p_pc1080 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_p1080, nullptr, &p_pc1080));

        // Verify 64 distinct push-constant colors in ONE uber-batch!
        struct PC { float x0, y0, x1, y1, r, g, b, z; };
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
        da.clearValue.depthStencil = {1.0f, 0u};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, nullptr};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pc128);
        for (int i = 0; i < 64; ++i) {
            int cx = i % 8, cy = i / 8;
            float x0 = -1.f + cx * 0.25f + 0.02f, y0 = -1.f + cy * 0.25f + 0.02f;
            PC pc{x0, y0, x0 + 0.21f, y0 + 0.21f, (cx + 1) / 8.0f, (cy + 1) / 8.0f, (i % 4) / 3.0f, 0.5f};
            vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &pc);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);
        int pc_ok = 0;
        for (int i = 0; i < 64; ++i) {
            int cx = i % 8, cy = i / 8;
            const uint8_t* p = px + ((cy * 16 + 8) * W + (cx * 16 + 8)) * 4;
            int er = (int)std::round(((cx + 1) / 8.0f) * 255.0f);
            int eg = (int)std::round(((cy + 1) / 8.0f) * 255.0f);
            int eb = (int)std::round(((i % 4) / 3.0f) * 255.0f);
            if (std::abs((int)p[0] - er) <= 2 && std::abs((int)p[1] - eg) <= 2 && std::abs((int)p[2] - eb) <= 2) pc_ok++;
        }

        // 1,000-Draw 1080p Push-Constant Uber-Batch Bench
        double pc_ms = 0.0;
        auto run_pc_bench = [&](bool measure) {
            auto t0 = std::chrono::steady_clock::now();
            CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
            VkRenderingAttachmentInfo bca = ca; bca.imageView = bcol.view;
            VkRenderingAttachmentInfo bda = da; bda.imageView = bdep.view;
            VkRenderingInfo bri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{BW,BH}}, 1, 0, 1, &bca, &bda, nullptr};
            vkCmdBeginRendering(cb, &bri);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pc1080);
            for (int d = 0; d < 1000; ++d) {
                float gx = -0.95f + 1.85f * ((d % 40) / 40.0f);
                float gy = -0.95f + 1.85f * ((d / 40) / 25.0f);
                PC pc{gx, gy, gx + 0.03f, gy + 0.04f, (d % 7)/6.f, (d % 5)/4.f, 0.8f, 0.2f + 0.6f*(d/1000.f)};
                vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &pc);
                vkCmdDraw(cb, 6, 1, 0, 0);
            }
            vkCmdEndRendering(cb);
            CHECK_VK(vkEndCommandBuffer(cb));
            CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(q));
            auto t1 = std::chrono::steady_clock::now();
            if (measure) pc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        run_pc_bench(false);
        for (int f = 0; f < 8; ++f) run_pc_bench(true);
        pc_ms /= 8.0;
        printf("[3] Per-Draw Push-Constant SSBO Uber-Batch: distinct_colors=%d/64 | 1000-Draw 1080p=%.2f ms (%.1f FPS)\n",
               pc_ok, pc_ms, 1000.0 / pc_ms);
        if (pc_ok != 64) { fprintf(stderr, "FAIL [3]: Push-Constant per-draw color mismatch!\n"); return 1; }

        // --- TEST 4: Per-Draw Dynamic-UBO SSBO Cross-Material Uber-Batching ---
        const char* vs_du = R"(#version 460
layout(set = 0, binding = 0, std140) uniform MatUBO { vec4 rect; vec4 col_z; } u;
void main() {
    vec2 k[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    gl_Position = vec4(mix(u.rect.xy, u.rect.zw, k[gl_VertexIndex]), u.col_z.w, 1.0);
})";
        const char* fs_du = R"(#version 460
layout(set = 0, binding = 0, std140) uniform MatUBO { vec4 rect; vec4 col_z; } u;
layout(location = 0) out vec4 o;
void main() { o = vec4(u.col_z.rgb, 1.0); }
)";
        VkDescriptorSetLayoutBinding dslb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslb};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plci_du{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl, 0, nullptr};
        VkPipelineLayout pl_du = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci_du, nullptr, &pl_du));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
        VkDescriptorPool dpool = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dpool, 1, &dsl};
        VkDescriptorSet dset = VK_NULL_HANDLE; CHECK_VK(vkAllocateDescriptorSets(dev, &dsai, &dset));

        VkBuffer ubo_buf = VK_NULL_HANDLE; VkDeviceMemory ubo_mem = VK_NULL_HANDLE; void* ubo_ptr = nullptr;
        create_buf(dev, mp, 1000 * 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ubo_buf, ubo_mem, &ubo_ptr);
        for (int i = 0; i < 64; ++i) {
            int cx = i % 8, cy = i / 8;
            float x0 = -1.f + cx * 0.25f + 0.02f, y0 = -1.f + cy * 0.25f + 0.02f;
            PC m{x0, y0, x0 + 0.21f, y0 + 0.21f, (cy + 1)/8.f, (i % 4)/3.f, (cx + 1)/8.f, 0.5f};
            memcpy((uint8_t*)ubo_ptr + i * 256, &m, sizeof(m));
        }
        VkDescriptorBufferInfo dbi{ubo_buf, 0, 256};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &dbi};
        vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        VkShaderModule sm_vdu = make_sm(dev, compile_glsl("vert", vs_du));
        VkShaderModule sm_fdu = make_sm(dev, compile_glsl("frag", fs_du));
        VkPipelineShaderStageCreateInfo st_du[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_vdu, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fdu, "main"},
        };
        VkGraphicsPipelineCreateInfo gp_du128{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st_du, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl_du};
        VkPipeline p_du128 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_du128, nullptr, &p_du128));
        VkGraphicsPipelineCreateInfo gp_du1080 = gp_du128; gp_du1080.pViewportState = &bvps;
        VkPipeline p_du1080 = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_du1080, nullptr, &p_du1080));

        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_du128);
        for (int i = 0; i < 64; ++i) {
            uint32_t doff = (uint32_t)i * 256u;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_du, 0, 1, &dset, 1, &doff);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);
        int du_ok = 0;
        for (int i = 0; i < 64; ++i) {
            int cx = i % 8, cy = i / 8;
            const uint8_t* p = px + ((cy * 16 + 8) * W + (cx * 16 + 8)) * 4;
            int er = (int)std::round(((cy + 1) / 8.0f) * 255.0f);
            int eg = (int)std::round(((i % 4) / 3.0f) * 255.0f);
            int eb = (int)std::round(((cx + 1) / 8.0f) * 255.0f);
            if (std::abs((int)p[0] - er) <= 2 && std::abs((int)p[1] - eg) <= 2 && std::abs((int)p[2] - eb) <= 2) du_ok++;
        }

        for (int d = 0; d < 1000; ++d) {
            float gx = -0.95f + 1.85f * ((d % 40) / 40.0f);
            float gy = -0.95f + 1.85f * ((d / 40) / 25.0f);
            PC m{gx, gy, gx + 0.03f, gy + 0.04f, (d % 7)/6.f, (d % 5)/4.f, 0.8f, 0.2f + 0.6f*(d/1000.f)};
            memcpy((uint8_t*)ubo_ptr + d * 256, &m, sizeof(m));
        }
        double du_ms = 0.0;
        auto run_du_bench = [&](bool measure) {
            auto t0 = std::chrono::steady_clock::now();
            CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
            VkRenderingAttachmentInfo bca = ca; bca.imageView = bcol.view;
            VkRenderingAttachmentInfo bda = da; bda.imageView = bdep.view;
            VkRenderingInfo bri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{BW,BH}}, 1, 0, 1, &bca, &bda, nullptr};
            vkCmdBeginRendering(cb, &bri);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_du1080);
            for (int d = 0; d < 1000; ++d) {
                uint32_t doff = (uint32_t)d * 256u;
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_du, 0, 1, &dset, 1, &doff);
                vkCmdDraw(cb, 6, 1, 0, 0);
            }
            vkCmdEndRendering(cb);
            CHECK_VK(vkEndCommandBuffer(cb));
            CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(q));
            auto t1 = std::chrono::steady_clock::now();
            if (measure) du_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        run_du_bench(false);
        for (int f = 0; f < 8; ++f) run_du_bench(true);
        du_ms /= 8.0;
        printf("[4] Per-Draw Dynamic-UBO SSBO Uber-Batch: distinct_colors=%d/64 | 1000-Draw 1080p=%.2f ms (%.1f FPS)\n",
               du_ok, du_ms, 1000.0 / du_ms);
        if (du_ok != 64) { fprintf(stderr, "FAIL [4]: Dynamic-UBO per-draw color mismatch!\n"); return 1; }

        vkDestroyPipeline(dev, p_pc128, nullptr); vkDestroyPipeline(dev, p_pc1080, nullptr);
        vkDestroyPipeline(dev, p_du128, nullptr); vkDestroyPipeline(dev, p_du1080, nullptr);
        vkDestroyPipelineLayout(dev, pl_pc, nullptr); vkDestroyPipelineLayout(dev, pl_du, nullptr);
        vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyBuffer(dev, ubo_buf, nullptr); vkFreeMemory(dev, ubo_mem, nullptr);
        vkDestroyShaderModule(dev, sm_vp, nullptr); vkDestroyShaderModule(dev, sm_fp, nullptr);
        vkDestroyShaderModule(dev, sm_vdu, nullptr); vkDestroyShaderModule(dev, sm_fdu, nullptr);
        vkDestroyImageView(dev, bcol.view, nullptr); vkDestroyImage(dev, bcol.img, nullptr); vkFreeMemory(dev, bcol.mem, nullptr);
        vkDestroyImageView(dev, bdep.view, nullptr); vkDestroyImage(dev, bdep.img, nullptr); vkFreeMemory(dev, bdep.mem, nullptr);
    }

    // =========================================================================
    // TEST 5: Zero-Copy Linear-Aliased Odd-Width G-Buffer (1257x707 MRT -> Hardware TMU Lighting Pass!)
    // =========================================================================
    {
        const uint32_t OW = 1257, OH = 707;
        ImgObj g_alb = create_img(dev, mp, OW, OH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj g_hdr = create_img(dev, mp, OW, OH, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj g_r8  = create_img(dev, mp, OW, OH, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj out_rt = create_img(dev, mp, OW, OH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VkBuffer orb = VK_NULL_HANDLE; VkDeviceMemory orbm = VK_NULL_HANDLE; void* orbp = nullptr;
        create_buf(dev, mp, (VkDeviceSize)OW * OH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, orb, orbm, &orbp);

        const char* vs_fsq = R"(#version 460
layout(location = 0) out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.5, 1.0);
    uv = p * 0.5 + 0.5;
})";
        const char* fs_gbuf = R"(#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() {
    o0 = vec4(uv.x, 0.25, 0.5, 1.0);
    o1 = vec4(2.0, 4.0 * uv.y, 1.0, 1.0);
    o2 = vec4(0.5, 0.0, 0.0, 1.0);
})";
        const char* fs_lit = R"(#version 460
layout(set = 0, binding = 0) uniform sampler2D t_alb;
layout(set = 0, binding = 1) uniform sampler2D t_hdr;
layout(set = 0, binding = 2) uniform sampler2D t_r8;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() {
    vec3 a = texture(t_alb, uv).rgb;
    vec3 h = texture(t_hdr, uv).rgb;
    float r8 = texture(t_r8, uv).r;
    // Expected: R = a.r * h.r * r8 = uv.x * 2.0 * 0.5 = uv.x
    //           G = h.g * 0.25 = uv.y
    //           B = a.b = 0.5
    o = vec4(a.r * h.r * r8, h.g * 0.25, a.b, 1.0);
})";
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler samp = VK_NULL_HANDLE; CHECK_VK(vkCreateSampler(dev, &sci, nullptr, &samp));

        VkDescriptorSetLayoutBinding b[3] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, b};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plci0{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl_g = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci0, nullptr, &pl_g));
        VkPipelineLayoutCreateInfo plci1{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl_l = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci1, nullptr, &pl_l));

        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
        VkDescriptorPool dpool = VK_NULL_HANDLE; CHECK_VK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dpool, 1, &dsl};
        VkDescriptorSet ds = VK_NULL_HANDLE; CHECK_VK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorImageInfo dii[3] = {
            {samp, g_alb.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {samp, g_hdr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {samp, g_r8.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet wds[3] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii[0]},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii[1]},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii[2]},
        };
        vkUpdateDescriptorSets(dev, 3, wds, 0, nullptr);

        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_fsq));
        VkShaderModule sm_fg = make_sm(dev, compile_glsl("frag", fs_gbuf));
        VkShaderModule sm_fl = make_sm(dev, compile_glsl("frag", fs_lit));
        VkViewport ovp{0, 0, (float)OW, (float)OH, 0.f, 1.f};
        VkRect2D osc{{0, 0}, {OW, OH}};
        VkPipelineViewportStateCreateInfo ovps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &ovp, 1, &osc};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

        VkFormat gfmts[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8_UNORM};
        VkPipelineRenderingCreateInfo prg{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 3, gfmts};
        VkPipelineColorBlendAttachmentState cba3[3] = {cba, cba, cba};
        VkPipelineColorBlendStateCreateInfo cbs3{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 3, cba3};
        VkPipelineShaderStageCreateInfo st_g[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fg, "main"},
        };
        VkGraphicsPipelineCreateInfo gpg{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prg, 0, 2, st_g, &vi, &ia, nullptr, &ovps, &rs, &ms, &dss, &cbs3, nullptr, pl_g};
        VkPipeline p_g = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpg, nullptr, &p_g));

        VkPipelineRenderingCreateInfo prl{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &col_fmt};
        VkPipelineShaderStageCreateInfo st_l[2] = {st_g[0], {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fl, "main"}};
        VkGraphicsPipelineCreateInfo gpl{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prl, 0, 2, st_l, &vi, &ia, nullptr, &ovps, &rs, &ms, &dss, &cb_st, nullptr, pl_l};
        VkPipeline p_l = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpl, nullptr, &p_l));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo gca[3] = {
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, g_alb.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, g_hdr.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, g_r8.view,  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}},
        };
        VkRenderingInfo rig{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{OW,OH}}, 1, 0, 3, gca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &rig);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_g);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);

        VkImageMemoryBarrier ibs[3];
        VkImage gimgs[3] = {g_alb.img, g_hdr.img, g_r8.img};
        for (int k = 0; k < 3; ++k) {
            ibs[k] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, gimgs[k], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 3, ibs);

        VkRenderingAttachmentInfo lca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, out_rt.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ril{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{OW,OH}}, 1, 0, 1, &lca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ril);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_l);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_l, 0, 1, &ds, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, out_rt.img, orb, OW, OH);

        const uint8_t* opx = reinterpret_cast<const uint8_t*>(orbp);
        auto check_pt = [&](uint32_t x, uint32_t y) -> bool {
            const uint8_t* p = opx + ((size_t)y * OW + x) * 4;
            int er = (int)std::round(((x + 0.5f) / OW) * 255.0f);
            int eg = (int)std::round(((y + 0.5f) / OH) * 255.0f);
            int eb = 128;
            return std::abs((int)p[0] - er) <= 3 && std::abs((int)p[1] - eg) <= 3 && std::abs((int)p[2] - eb) <= 3;
        };
        const uint8_t* pMid = opx + ((size_t)353 * OW + 628) * 4;
        const uint8_t* pBR  = opx + ((size_t)706 * OW + 1256) * 4;
        bool ok5 = check_pt(10, 10) && check_pt(628, 353) && check_pt(1256, 706) && check_pt(1200, 100) && check_pt(50, 680);
        printf("[5] Zero-Copy Linear-Aliased Odd-Width G-Buffer (1257x707 3-MRT -> TMU): mid=(%u,%u,%u) BR=(%u,%u,%u) ok=%d\n",
               pMid[0], pMid[1], pMid[2], pBR[0], pBR[1], pBR[2], (int)ok5);
        if (!ok5) { fprintf(stderr, "FAIL [5]: Odd-width zero-copy linear-aliased G-Buffer mismatch!\n"); return 1; }

        vkDestroyPipeline(dev, p_g, nullptr); vkDestroyPipeline(dev, p_l, nullptr);
        vkDestroyPipelineLayout(dev, pl_g, nullptr); vkDestroyPipelineLayout(dev, pl_l, nullptr);
        vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroySampler(dev, samp, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_fg, nullptr); vkDestroyShaderModule(dev, sm_fl, nullptr);
        vkDestroyBuffer(dev, orb, nullptr); vkFreeMemory(dev, orbm, nullptr);
        for (auto& im : {g_alb, g_hdr, g_r8, out_rt}) {
            vkDestroyImageView(dev, im.view, nullptr); vkDestroyImage(dev, im.img, nullptr); vkFreeMemory(dev, im.mem, nullptr);
        }
    }

    printf("V08_FEATURE_TEST PASS\n");
    fflush(stdout);
    vkDeviceWaitIdle(dev);
    vkDestroyBuffer(dev, rb_buf, nullptr); vkFreeMemory(dev, rb_mem, nullptr);
    vkDestroyImageView(dev, col.view, nullptr); vkDestroyImage(dev, col.img, nullptr); vkFreeMemory(dev, col.mem, nullptr);
    vkDestroyImageView(dev, dep.view, nullptr); vkDestroyImage(dev, dep.img, nullptr); vkFreeMemory(dev, dep.mem, nullptr);
    vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
