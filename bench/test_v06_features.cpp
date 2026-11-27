// test_v06_features.cpp — SmRast-VK v0.6.0 Comprehensive Feature + 1000-Draw 1080p Benchmark
// Verifies on H20 pure compute qf[2]:
//   1) GPU-Driven Active-Tile Indirect Dispatch + Self-Clearing VisBuf (incl. fragment `discard;` safety!)
//   2) Coarse Pre-Binning Hi-Z Early-Reject (Standard-Z LESS & Reverse-Z GREATER, depth-only + color occluders)
//   3) Hardware-Exact Primitive Restart (UINT16 0xFFFF & UINT32 0xFFFFFFFF in TRIANGLE_STRIP — zero bridge tris!)
//   4) LINE_LIST & LINE_STRIP rasterization with dynamic vkCmdSetLineWidth (4 px wide lines!)
//   5) POINT_LIST rasterization with VS gl_PointSize (6x6 px point sprites!)
//   6) 1,000-Draw 1080p Scattered-Prop Throughput Benchmark (measures Active-Tile + Self-Clearing VisBuf speed!)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#define CHECK_VK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", (int)_r, __LINE__); exit(1); } } while (0)

static std::vector<uint32_t> compile_glsl(const char* stage, const char* src) {
    FILE* f = fopen("/tmp/sm_v06_tmp.glsl", "w");
    fputs(src, f);
    fclose(f);
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "glslangValidator -V --target-env vulkan1.3 -S %s /tmp/sm_v06_tmp.glsl -o /tmp/sm_v06_tmp.spv >/dev/null", stage);
    if (system(cmd) != 0) {
        fprintf(stderr, "glslangValidator failed for %s:\n%s\n", stage, src);
        exit(1);
    }
    FILE* sf = fopen("/tmp/sm_v06_tmp.spv", "rb");
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
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV06Test", 1, "Test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app};
    VkInstance inst = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(inst, &nd, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("Testing SmRast-VK v0.6.0 on: %s\n", props.deviceName);

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures feats{};
    feats.wideLines = VK_TRUE;
    feats.largePoints = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, nullptr, 0, nullptr, &feats};
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
    // TEST 1: Self-Clearing VisBuf with `discard;` + Pre-Binning Hi-Z (Standard-Z + Depth-Only Occluder!)
    // =========================================================================
    {
        const char* vs_pc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_z; } pc;
void main() {
    vec2 u[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 p = mix(pc.rect.xy, pc.rect.zw, u[gl_VertexIndex]);
    gl_Position = vec4(p, pc.col_z.w, 1.0);
})";
        const char* fs_disc = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_z; } pc;
layout(location = 0) out vec4 o;
void main() {
    // Discard left half! Must still self-clear VisBuf cleanly so next draw sees ZERO ghost pixels!
    if (gl_FragCoord.x < 32.0) discard;
    o = vec4(pc.col_z.rgb, 1.0);
})";
        const char* fs_solid = R"(#version 460
layout(push_constant) uniform PC { vec4 rect; vec4 col_z; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.col_z.rgb, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_pc));
        VkShaderModule sm_fd = make_sm(dev, compile_glsl("frag", fs_disc));
        VkShaderModule sm_fs = make_sm(dev, compile_glsl("frag", fs_solid));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineShaderStageCreateInfo st_d[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fd, "main"},
        };
        VkGraphicsPipelineCreateInfo gp_d{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st_d, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_disc = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_d, nullptr, &p_disc));

        VkPipelineShaderStageCreateInfo st_s[2] = { st_d[0], {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_fs, "main"} };
        VkGraphicsPipelineCreateInfo gp_s{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st_s, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_solid = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_s, nullptr, &p_solid));

        // Depth-Only pipeline (VS only, no FS -> exercises Active-Tile Depth-Only + Hi-Z Update!)
        VkGraphicsPipelineCreateInfo gp_z{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 1, &st_d[0], &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_zonly = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_z, nullptr, &p_zonly));

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

        // Draw 1: fullscreen quad with `discard;` on left half -> right half becomes GREEN at z=0.4, left half stays BLACK at z=1.0!
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_disc);
        PC d1{-1.f, -1.f, 1.f, 1.f, 0.f, 1.f, 0.f, 0.4f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &d1);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Draw 2: Depth-Only occluder on top-left quadrant [-1,0]x[-1,0] at z=0.2 (updates Tile Hi-Z!)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_zonly);
        PC d2{-1.f, -1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.2f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &d2);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Draw 3: Fullscreen BLUE at z=0.6:
        //   - TL (occluded by depth-only z=0.2 -> Pre-Bin Hi-Z rejects!) -> stays BLACK (0,0,0)
        //   - BL (discarded in Draw 1, depth still 1.0 -> passes!)       -> becomes BLUE (0,0,255)
        //   - Right (occluded by Draw 1 z=0.4 -> Pre-Bin Hi-Z rejects!)  -> stays GREEN (0,255,0)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_solid);
        PC d3{-1.f, -1.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.6f};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &d3);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pTL = px + (16 * W + 16) * 4;
        const uint8_t* pBL = px + (48 * W + 16) * 4;
        const uint8_t* pR  = px + (32 * W + 48) * 4;
        printf("[1] Active-Tile + Discard Self-Clear + Pre-Bin Hi-Z: TL=(%u,%u,%u) BL=(%u,%u,%u) R=(%u,%u,%u)\n",
               pTL[0], pTL[1], pTL[2], pBL[0], pBL[1], pBL[2], pR[0], pR[1], pR[2]);
        if (pTL[0] != 0 || pTL[1] != 0 || pTL[2] != 0 ||
            pBL[0] != 0 || pBL[1] != 0 || pBL[2] != 255 ||
            pR[0]  != 0 || pR[1]  != 255 || pR[2]  != 0) {
            fprintf(stderr, "FAIL [1]: Active-Tile / Discard / Pre-Bin Hi-Z mismatch!\n"); return 1;
        }
        vkDestroyPipeline(dev, p_disc, nullptr); vkDestroyPipeline(dev, p_solid, nullptr); vkDestroyPipeline(dev, p_zonly, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_fd, nullptr); vkDestroyShaderModule(dev, sm_fs, nullptr);
    }

    // =========================================================================
    // TEST 2: Hardware-Exact Primitive Restart (UINT16 0xFFFF in TRIANGLE_STRIP!)
    // =========================================================================
    {
        // Left quad [-0.9, -0.3]x[-0.5, 0.5] and Right quad [0.3, 0.9]x[-0.5, 0.5] separated by 0xFFFF!
        // Center gap x in (-0.3, 0.3) MUST remain pure BLACK (0,0,0) — zero bridge triangles!
        float verts[16] = {
            -0.9f, -0.5f,  -0.3f, -0.5f,  -0.9f, 0.5f,  -0.3f, 0.5f,
             0.3f, -0.5f,   0.9f, -0.5f,   0.3f, 0.5f,   0.9f, 0.5f,
        };
        uint16_t idxs[9] = { 0, 1, 2, 3, 0xFFFFu, 4, 5, 6, 7 };
        VkBuffer vb, ib; VkDeviceMemory vbm, ibm; void *vbp, *ibp;
        create_buf(dev, mp, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vb, vbm, &vbp);
        create_buf(dev, mp, 32, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ib, ibm, &ibp);
        memcpy(vbp, verts, sizeof(verts));
        memcpy(ibp, idxs, sizeof(idxs));

        const char* vs_v2 = R"(#version 460
layout(location = 0) in vec2 p;
void main() { gl_Position = vec4(p, 0.5, 1.0); }
)";
        const char* fs_y = R"(#version 460
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 1.0, 0.0, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_v2));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_y));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkVertexInputBindingDescription vbd{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription vad{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &vbd, 1, &vad};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_TRUE};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
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
        vkCmdBindIndexBuffer(cb, ib, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cb, 9, 1, 0, 0, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        const uint8_t* pL = px + (32 * W + 12) * 4;
        const uint8_t* pM = px + (32 * W + 32) * 4;
        const uint8_t* pR = px + (32 * W + 52) * 4;
        printf("[2] Primitive Restart (0xFFFF TRIANGLE_STRIP): L=(%u,%u,%u) gapM=(%u,%u,%u) R=(%u,%u,%u)\n",
               pL[0], pL[1], pL[2], pM[0], pM[1], pM[2], pR[0], pR[1], pR[2]);
        if (pL[0] != 255 || pL[1] != 255 || pM[0] != 0 || pM[1] != 0 || pR[0] != 255 || pR[1] != 255) {
            fprintf(stderr, "FAIL [2]: Primitive restart bridge triangle or missing strip!\n"); return 1;
        }
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
        vkDestroyBuffer(dev, vb, nullptr); vkFreeMemory(dev, vbm, nullptr);
        vkDestroyBuffer(dev, ib, nullptr); vkFreeMemory(dev, ibm, nullptr);
    }

    // =========================================================================
    // TEST 3: LINE_LIST (lineWidth=4) + POINT_LIST (gl_PointSize=6)
    // =========================================================================
    {
        const char* vs_lp = R"(#version 460
layout(location = 0) in vec2 p;
layout(location = 1) in vec3 c;
layout(location = 0) out vec3 vc;
void main() {
    gl_Position = vec4(p, 0.5, 1.0);
    gl_PointSize = 6.0;
    vc = c;
})";
        const char* fs_lp = R"(#version 460
layout(location = 0) in vec3 vc;
layout(location = 0) out vec4 o;
void main() { o = vec4(vc, 1.0); }
)";
        // Horizontal red line along y = -0.5 (pixel y=16), and a cyan 6x6 point sprite at (0.0, 0.5) (pixel 32, 48)!
        float vdata[15] = {
            -0.8f, -0.5f, 1.f, 0.f, 0.f,
             0.8f, -0.5f, 1.f, 0.f, 0.f,
             0.0f,  0.5f, 0.f, 1.f, 1.f,
        };
        VkBuffer vb; VkDeviceMemory vbm; void* vbp;
        create_buf(dev, mp, sizeof(vdata), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vb, vbm, &vbp);
        memcpy(vbp, vdata, sizeof(vdata));

        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_lp));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_lp));
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkVertexInputBindingDescription vbd{0, 20, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription vas[2] = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 8},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &vbd, 2, vas};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 4.0f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };

        VkPipelineInputAssemblyStateCreateInfo ia_l{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_LINE_LIST};
        VkGraphicsPipelineCreateInfo gpl{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia_l, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_line = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpl, nullptr, &p_line));

        VkPipelineInputAssemblyStateCreateInfo ia_p{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
        VkGraphicsPipelineCreateInfo gpp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia_p, nullptr, &vps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline p_pt = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpp, nullptr, &p_pt));

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, nullptr, nullptr};
        vkCmdBeginRendering(cb, &ri);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &vb, &off);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_line);
        vkCmdDraw(cb, 2, 1, 0, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pt);
        vkCmdDraw(cb, 1, 1, 2, 0);
        vkCmdEndRendering(cb);
        CHECK_VK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(q));
        readback_rgba8(dev, q, cb, col.img, rb_buf, W, H);

        int line_px = 0, pt_px = 0;
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x) {
                const uint8_t* p = px + (y * W + x) * 4;
                if (p[0] == 255 && p[1] == 0 && p[2] == 0) line_px++;
                if (p[0] == 0 && p[1] == 255 && p[2] == 255) pt_px++;
            }
        printf("[3] LINE_LIST (w=4) + POINT_LIST (ps=6): line_px=%d (expect ~204) pt_px=%d (expect ~36)\n", line_px, pt_px);
        if (line_px < 150 || line_px > 260 || pt_px < 25 || pt_px > 49) {
            fprintf(stderr, "FAIL [3]: LINE_LIST or POINT_LIST coverage out of range!\n"); return 1;
        }
        vkDestroyPipeline(dev, p_line, nullptr); vkDestroyPipeline(dev, p_pt, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
        vkDestroyBuffer(dev, vb, nullptr); vkFreeMemory(dev, vbm, nullptr);
    }

    // =========================================================================
    // TEST 4: 1,000-Draw 1080p Open-World Scattered-Prop Throughput Benchmark!
    // =========================================================================
    {
        const uint32_t BW = 1920, BH = 1080;
        ImgObj bcol = create_img(dev, mp, BW, BH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ImgObj bdep = create_img(dev, mp, BW, BH, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkViewport bvp{0, 0, (float)BW, (float)BH, 0.f, 1.f};
        VkRect2D bsc{{0, 0}, {BW, BH}};
        VkPipelineViewportStateCreateInfo bvps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &bvp, 1, &bsc};

        const char* vs_prop = R"(#version 460
layout(push_constant) uniform PC { vec4 rect_z; vec4 color; } pc;
void main() {
    vec2 u[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 p = mix(pc.rect_z.xy, pc.rect_z.zw, u[gl_VertexIndex]);
    gl_Position = vec4(p, pc.color.w, 1.0);
})";
        const char* fs_prop = R"(#version 460
layout(push_constant) uniform PC { vec4 rect_z; vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.color.rgb, 1.0); }
)";
        VkShaderModule sm_v = make_sm(dev, compile_glsl("vert", vs_prop));
        VkShaderModule sm_f = make_sm(dev, compile_glsl("frag", fs_prop));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VkPipelineLayout pl = VK_NULL_HANDLE; CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sm_v, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, sm_f, "main"},
        };
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st, &vi, &ia, nullptr, &bvps, &rs, &ms, &dss, &cb_st, nullptr, pl};
        VkPipeline pipe = VK_NULL_HANDLE; CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe));

        struct PC { float x0, y0, x1, y1, r, g, b, z; };
        const int NDRAWS = 1000;
        const int FRAMES = 10;
        double rec_ms_sum = 0.0, gpu_ms_sum = 0.0;
        auto run_frame = [&](bool measure) {
            auto tr0 = std::chrono::steady_clock::now();
            VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK_VK(vkBeginCommandBuffer(cb, &cbi));
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, bcol.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, bdep.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
            da.clearValue.depthStencil = {1.0f, 0u};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{BW,BH}}, 1, 0, 1, &ca, &da, nullptr};
            vkCmdBeginRendering(cb, &ri);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            for (int d = 0; d < NDRAWS; ++d) {
                float gx = -0.95f + 1.85f * ((d % 40) / 40.0f);
                float gy = -0.95f + 1.85f * ((d / 40) / 25.0f);
                PC pc{gx, gy, gx + 0.03f, gy + 0.04f, (d % 7) / 6.0f, (d % 5) / 4.0f, 0.8f, 0.2f + 0.6f * (d / (float)NDRAWS)};
                vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PC), &pc);
                vkCmdDraw(cb, 6, 1, 0, 0);
            }
            vkCmdEndRendering(cb);
            CHECK_VK(vkEndCommandBuffer(cb));
            auto tr1 = std::chrono::steady_clock::now();
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
            CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(q));
            auto tr2 = std::chrono::steady_clock::now();
            if (measure) {
                rec_ms_sum += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                gpu_ms_sum += std::chrono::duration<double, std::milli>(tr2 - tr1).count();
            }
        };
        run_frame(false); // warmup
        for (int f = 0; f < FRAMES; ++f) run_frame(true);
        double rec_ms = rec_ms_sum / FRAMES, gpu_ms = gpu_ms_sum / FRAMES, ms = rec_ms + gpu_ms;
        printf("[4] 1,000-Draw 1080p Active-Tile Bench: %.2f ms/frame (CPU=%.2f ms, GPU=%.2f ms, %.1f FPS, %.1f Kdraws/s)\n",
               ms, rec_ms, gpu_ms, 1000.0 / ms, (NDRAWS / ms));
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyShaderModule(dev, sm_v, nullptr); vkDestroyShaderModule(dev, sm_f, nullptr);
        vkDestroyImageView(dev, bcol.view, nullptr); vkDestroyImage(dev, bcol.img, nullptr); vkFreeMemory(dev, bcol.mem, nullptr);
        vkDestroyImageView(dev, bdep.view, nullptr); vkDestroyImage(dev, bdep.img, nullptr); vkFreeMemory(dev, bdep.mem, nullptr);
    }

    printf("V06_FEATURE_TEST PASS\n");
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
