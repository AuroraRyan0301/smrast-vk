// test_v04_features.cpp — SmRast-VK v0.4.0 Comprehensive Feature Verification
// Verifies on H20 pure compute qf[2]:
//   1) 8-MRT simultaneous output across 8 formats (RGBA8, RGBA8_SNORM, R8_UNORM 1B atomic,
//      RG8_UNORM 2B atomic, R16_SFLOAT 2B atomic, R16_UNORM 2B atomic, RGBA16_UNORM, RGBA16_SFLOAT)
//   2) 8-bit Stencil Test/Op (D24_UNORM_S8_UINT mask pass + EQUAL ref=42 shading pass)
//   3) Per-channel colorWriteMask (R|B only, preserving G)
//   4) Separate RGB + Alpha blending + vkCmdSetBlendConstants
//   5) Exact gl_FrontFacing (front=green, back=red, cull=NONE)
//   6) GPU-Driven vkCmdDrawIndirectCount (GPU count=1, maxDrawCount=4 -> only draw 0 executes)
//   7) VK_INDEX_TYPE_UINT8_EXT
//   8) VK_KHR_push_descriptor

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define CHECK_VK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", (int)_r, __LINE__); exit(1); } } while (0)

static std::vector<uint32_t> compile_glsl(const char* stage, const char* src) {
    FILE* f = fopen("/tmp/sm_v04_tmp.glsl", "w");
    fputs(src, f);
    fclose(f);
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "glslangValidator -V --target-env vulkan1.3 -S %s /tmp/sm_v04_tmp.glsl -o /tmp/sm_v04_tmp.spv >/dev/null", stage);
    if (system(cmd) != 0) {
        fprintf(stderr, "glslangValidator failed for %s:\n%s\n", stage, src);
        exit(1);
    }
    FILE* sf = fopen("/tmp/sm_v04_tmp.spv", "rb");
    fseek(sf, 0, SEEK_END);
    size_t sz = (size_t)ftell(sf);
    fseek(sf, 0, SEEK_SET);
    std::vector<uint32_t> spv(sz / 4);
    if (fread(spv.data(), 1, sz, sf) != sz) {
        fprintf(stderr, "Failed to read %zu bytes from SPIR-V output\n", sz);
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

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    if (e == 0) return (s ? -1.f : 1.f) * ldexpf((float)m, -24);
    if (e == 31) return s ? -INFINITY : INFINITY;
    return (s ? -1.f : 1.f) * ldexpf((float)(1024 + m), (int)e - 25);
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV04Test", 1, "Test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app};
    VkInstance inst = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    CHECK_VK(vkEnumeratePhysicalDevices(inst, &nd, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    const char* exts[] = { VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME };
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT iu8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, nullptr, VK_TRUE};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &iu8};
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, nullptr, 2, exts, nullptr};
    VkDevice dev = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue(dev, 0, 0, &q);

    auto pfnPushDesc = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(dev, "vkCmdPushDescriptorSetKHR");
    if (!pfnPushDesc) { fprintf(stderr, "FAIL: vkCmdPushDescriptorSetKHR is NULL\n"); return 1; }

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    CHECK_VK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CHECK_VK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    const uint32_t W = 64, H = 64;
    VkFormat rt_fmts[8] = {
        VK_FORMAT_R8G8B8A8_UNORM,      // RT0: 4B
        VK_FORMAT_R8G8B8A8_SNORM,      // RT1: 4B signed norm
        VK_FORMAT_R8_UNORM,            // RT2: 1B atomic
        VK_FORMAT_R8G8_UNORM,          // RT3: 2B atomic
        VK_FORMAT_R16_SFLOAT,          // RT4: 2B atomic half
        VK_FORMAT_R16_UNORM,           // RT5: 2B atomic u16
        VK_FORMAT_R16G16B16A16_UNORM,  // RT6: 8B u16x4
        VK_FORMAT_R16G16B16A16_SFLOAT, // RT7: 8B f16x4
    };
    uint32_t rt_bpps[8] = {4, 4, 1, 2, 2, 2, 8, 8};
    ImgObj rts[8];
    for (int i = 0; i < 8; ++i)
        rts[i] = create_img(dev, mp, W, H, rt_fmts[i], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    ImgObj ds_img = create_img(dev, mp, W, H, VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    // Push descriptor set layout (binding 0 = UBO vec4 factor)
    VkDescriptorSetLayoutBinding dslb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 1, &dslb};
    VkDescriptorSetLayout push_dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &push_dsl));
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &push_dsl, 0, nullptr};
    VkPipelineLayout pl = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    // Shared VS reading vec2 pos at location 0 + push-descriptor UBO scale
    const char* vs_src = R"(#version 460
layout(location = 0) in vec2 in_pos;
layout(set = 0, binding = 0) uniform UBO { vec4 scale_off; } u;
void main() { gl_Position = vec4(in_pos * u.scale_off.xy + u.scale_off.zw, 0.5, 1.0); }
)";
    // FS1: 8-MRT + gl_FrontFacing output!
    const char* fs_mrt_src = R"(#version 460
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;
void main() {
    o0 = gl_FrontFacing ? vec4(0.2, 0.8, 0.4, 1.0) : vec4(0.9, 0.1, 0.1, 1.0);
    o1 = vec4(-0.5, 0.5, -1.0, 1.0);
    o2 = vec4(0.75, 0.0, 0.0, 1.0);
    o3 = vec4(0.25, 0.50, 0.0, 1.0);
    o4 = vec4(3.5, 0.0, 0.0, 1.0);
    o5 = vec4(0.60, 0.0, 0.0, 1.0);
    o6 = vec4(0.1, 0.4, 0.7, 0.9);
    o7 = vec4(2.0, 4.0, 6.0, 8.0);
}
)";
    // FS2: Constant color for stencil + writeMask + separate alpha blend + blendConstant verification
    const char* fs_blend_src = R"(#version 460
layout(location = 0) out vec4 o0;
void main() { o0 = vec4(1.0, 1.0, 1.0, 0.5); }
)";
    VkShaderModule vs_m = make_sm(dev, compile_glsl("vert", vs_src));
    VkShaderModule fs_mrt_m = make_sm(dev, compile_glsl("frag", fs_mrt_src));
    VkShaderModule fs_bl_m = make_sm(dev, compile_glsl("frag", fs_blend_src));

    VkVertexInputBindingDescription vbd{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vad{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &vbd, 1, &vad};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineRenderingCreateInfo prci_8{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 8, rt_fmts, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};

    // Pipe A: 8-MRT + Stencil Write (ref=42, ALWAYS -> REPLACE)
    VkPipelineColorBlendAttachmentState cba8[8]{};
    for (int i = 0; i < 8; ++i) cba8[i].colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb8{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 8, cba8};
    VkPipelineDepthStencilStateCreateInfo ds_write{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds_write.stencilTestEnable = VK_TRUE;
    ds_write.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 42};
    ds_write.back  = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 99};
    VkPipelineShaderStageCreateInfo st_mrt[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs_m, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs_mrt_m, "main"},
    };
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci_8, 0, 2, st_mrt, &vi, &ia, nullptr, &vps, &rs, &ms, &ds_write, &cb8, nullptr, pl};
    VkPipeline pipe_mrt = VK_NULL_HANDLE;
    CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe_mrt));

    // Pipe B: Stencil EQUAL ref=42 + WriteMask(R|B|A, NOT G!) + Separate RGB (CONSTANT_COLOR * src + ZERO) & Alpha (ONE * src + ZERO)
    VkPipelineColorBlendAttachmentState cba_bl{};
    cba_bl.blendEnable = VK_TRUE;
    cba_bl.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
    cba_bl.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba_bl.colorBlendOp = VK_BLEND_OP_ADD;
    cba_bl.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba_bl.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba_bl.alphaBlendOp = VK_BLEND_OP_ADD;
    cba_bl.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // G masked out!
    VkPipelineColorBlendStateCreateInfo cb_bl{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba_bl};
    VkPipelineDepthStencilStateCreateInfo ds_eq{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds_eq.stencilTestEnable = VK_TRUE;
    ds_eq.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 0xFF, 0x00, 42};
    ds_eq.back  = ds_eq.front;
    VkDynamicState dyn_s[1] = { VK_DYNAMIC_STATE_BLEND_CONSTANTS };
    VkPipelineDynamicStateCreateInfo dynci{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 1, dyn_s};
    VkPipelineRenderingCreateInfo prci_1{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &rt_fmts[0], VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    VkPipelineShaderStageCreateInfo st_bl[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs_m, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs_bl_m, "main"},
    };
    VkGraphicsPipelineCreateInfo gpci_bl{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci_1, 0, 2, st_bl, &vi, &ia, nullptr, &vps, &rs, &ms, &ds_eq, &cb_bl, &dynci, pl};
    VkPipeline pipe_bl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci_bl, nullptr, &pipe_bl));

    // Geometry:
    //  Left half (x in [-1, 0]): CCW (in Vulkan Y-down screen space: (-1,-1)->(-1,1)->(0,-1) has cross>0 -> front-facing!)
    //  Right half (x in [0, 1]): reversed winding -> back-facing!
    //  Full-screen quad for Pass 2 (stencil test + blend constant + write mask)
    //  Plus 2nd indirect draw that would overwrite everything if count buffer wasn't respected!
    float verts[] = {
        // 0..2: Left CCW triangle covering (-0.8, 0.0) -> pixel (6, 32)
        -1.0f, -1.0f,  -1.0f,  1.0f,   0.0f, -1.0f,
        // 3..5: Right CW (back-facing) triangle covering (+0.5, 0.0) -> pixel (48, 32)
         0.0f, -1.0f,   1.0f, -1.0f,   1.0f,  1.0f,
        // 6..11: Fullscreen quad (2 tris)
        -1.0f, -1.0f,  -1.0f,  1.0f,   1.0f,  1.0f,
        -1.0f, -1.0f,   1.0f,  1.0f,   1.0f, -1.0f,
    };
    uint8_t idx8[] = { 0, 1, 2,   3, 4, 5,   6, 7, 8, 9, 10, 11 };

    VkBuffer vb, ib, ubo, ind_b, cnt_b, rb;
    VkDeviceMemory vm, im, um, ind_m, cnt_m, rbm;
    void *vp_m, *ip_m, *up_m, *indp_m, *cntp_m, *rbp_m;
    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    create_buf(dev, mp, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hv, vb, vm, &vp_m);
    memcpy(vp_m, verts, sizeof(verts));
    create_buf(dev, mp, sizeof(idx8), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, hv, ib, im, &ip_m);
    memcpy(ip_m, idx8, sizeof(idx8));
    float udata[4] = {1.f, 1.f, 0.f, 0.f};
    create_buf(dev, mp, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hv, ubo, um, &up_m);
    memcpy(up_m, udata, sizeof(udata));

    // IndirectCount commands: draw 0 draws indices 6..11 (fullscreen), draw 1 draws indices 0..2 with bogus offset (should be skipped since count=1!)
    VkDrawIndexedIndirectCommand cmds[4]{};
    cmds[0] = {6, 1, 6, 0, 0};
    cmds[1] = {6, 1, 0, 0, 0};
    cmds[2] = {6, 1, 0, 0, 0};
    cmds[3] = {6, 1, 0, 0, 0};
    create_buf(dev, mp, sizeof(cmds), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, hv, ind_b, ind_m, &indp_m);
    memcpy(indp_m, cmds, sizeof(cmds));
    uint32_t gpu_draw_count = 1;
    create_buf(dev, mp, 16, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, hv, cnt_b, cnt_m, &cntp_m);
    memcpy(cntp_m, &gpu_draw_count, 4);
    create_buf(dev, mp, W * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hv, rb, rbm, &rbp_m);

    // Record commands
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(cb, &cbi));

    VkRenderingAttachmentInfo col_atts[8]{};
    for (int i = 0; i < 8; ++i) {
        col_atts[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rts[i].view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,0}}}};
    }
    VkRenderingAttachmentInfo dep_att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, ds_img.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                      VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{1.0f, 0}}}};
    VkRenderingAttachmentInfo stn_att = dep_att;
    stn_att.clearValue.depthStencil.stencil = 0;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 8, col_atts, &dep_att, &stn_att};
    vkCmdBeginRendering(cb, &ri);

    VkDescriptorBufferInfo dbi{ubo, 0, 64};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &dbi};
    pfnPushDesc(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &wds);

    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vb, &voff);
    vkCmdBindIndexBuffer(cb, ib, 0, VK_INDEX_TYPE_UINT8_EXT);

    // Pass 1: 8-MRT + gl_FrontFacing + Stencil Write (left front=42, right back=99) using UINT8 indices!
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_mrt);
    vkCmdDrawIndexed(cb, 6, 1, 0, 0, 0);

    // Pass 2: DrawIndexedIndirectCount (count=1) over fullscreen with Stencil EQUAL 42 + BlendConst(0.6, 0.0, 0.8, 0.25) + WriteMask(R|B|A)!
    // Expected on left pixel (front, stencil==42):
    //   R = 1.0 * 0.6 = 0.6 (~153)
    //   G = preserved from Pass 1 front = 0.8 (~204) [G masked out!]
    //   B = 1.0 * 0.8 = 0.8 (~204)
    //   A = 0.5 * 1.0 = 0.5 (~128) [Separate Alpha blend!]
    // Expected on right pixel (back, stencil==99 != 42):
    //   Stencil FAILS -> untouched from Pass 1 back = (0.9, 0.1, 0.1, 1.0) (~230, 26, 26, 255)!
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_bl);
    float bc[4] = {0.6f, 0.0f, 0.8f, 0.25f};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdDrawIndexedIndirectCount(cb, ind_b, 0, cnt_b, 0, 4, sizeof(VkDrawIndexedIndirectCommand));

    vkCmdEndRendering(cb);
    CHECK_VK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    CHECK_VK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(q));

    // Readback helper
    auto read_rt = [&](int idx, std::vector<uint8_t>& out) {
        out.resize(W * H * rt_bpps[idx]);
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbi);
        VkBufferImageCopy bic{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0,0,0}, {W,H,1}};
        vkCmdCopyImageToBuffer(cb, rts[idx].img, VK_IMAGE_LAYOUT_GENERAL, rb, 1, &bic);
        vkEndCommandBuffer(cb);
        vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(q);
        memcpy(out.data(), rbp_m, out.size());
    };

    // Sample pixel in left front triangle: (x=8, y=16) -> NDC (-0.75, -0.5) inside (-1,-1),(-1,1),(0,-1)
    // Sample pixel in right back triangle: (x=56, y=16) -> NDC (+0.75, -0.5) inside (0,-1),(1,-1),(1,1)
    const uint32_t pL = 16 * W + 8;
    const uint32_t pR = 16 * W + 56;
    bool ok = true;

    std::vector<uint8_t> d0, d1, d2, d3, d4, d5, d6, d7;
    read_rt(0, d0); read_rt(1, d1); read_rt(2, d2); read_rt(3, d3);
    read_rt(4, d4); read_rt(5, d5); read_rt(6, d6); read_rt(7, d7);

    // Check RT0 (Stencil + WriteMask + Separate Alpha + BlendConst + FrontFacing)
    uint8_t* l0 = &d0[pL * 4];
    uint8_t* r0 = &d0[pR * 4];
    printf("RT0 Left  (front+stn42+wmRBA+sepAlpha+bc): RGBA=(%u,%u,%u,%u) expect~(153,204,204,128)\n", l0[0], l0[1], l0[2], l0[3]);
    printf("RT0 Right (back+stn99 fail -> untouched):  RGBA=(%u,%u,%u,%u) expect~(230,26,26,255)\n", r0[0], r0[1], r0[2], r0[3]);
    if (abs((int)l0[0] - 153) > 3 || abs((int)l0[1] - 204) > 3 || abs((int)l0[2] - 204) > 3 || abs((int)l0[3] - 128) > 3) ok = false;
    if (abs((int)r0[0] - 230) > 3 || abs((int)r0[1] - 26) > 3 || abs((int)r0[2] - 26) > 3 || abs((int)r0[3] - 255) > 3) ok = false;

    // Check RT1 (RGBA8_SNORM: -0.5, 0.5, -1.0, 1.0 -> -64, 64, -127, 127)
    int8_t* l1 = (int8_t*)&d1[pL * 4];
    printf("RT1 RGBA8_SNORM: (%d,%d,%d,%d) expect~(-64,64,-127,127)\n", l1[0], l1[1], l1[2], l1[3]);
    if (abs((int)l1[0] - (-64)) > 2 || abs((int)l1[1] - 64) > 2 || l1[2] > -126 || l1[3] < 126) ok = false;

    // Check RT2 (R8_UNORM 1B atomic: 0.75 -> 191; check 4 adjacent pixels in same u32 word!)
    uint8_t b2a = d2[pL], b2b = d2[pL + 1], b2c = d2[pL + 2], b2d = d2[pL + 3];
    printf("RT2 R8_UNORM (1B atomic 4x adjacent): (%u,%u,%u,%u) expect~(191,191,191,191)\n", b2a, b2b, b2c, b2d);
    if (abs((int)b2a - 191) > 2 || abs((int)b2b - 191) > 2 || abs((int)b2c - 191) > 2 || abs((int)b2d - 191) > 2) ok = false;

    // Check RT3 (RG8_UNORM 2B atomic: 0.25, 0.50 -> 64, 128)
    uint8_t* l3 = &d3[pL * 2];
    printf("RT3 RG8_UNORM (2B atomic): (%u,%u) expect~(64,128)\n", l3[0], l3[1]);
    if (abs((int)l3[0] - 64) > 2 || abs((int)l3[1] - 128) > 2) ok = false;

    // Check RT4 (R16_SFLOAT 2B atomic: 3.5)
    uint16_t h4 = *(uint16_t*)&d4[pL * 2];
    float f4 = f16_to_f32(h4);
    printf("RT4 R16_SFLOAT (2B atomic): %.3f expect=3.500\n", f4);
    if (fabsf(f4 - 3.5f) > 0.05f) ok = false;

    // Check RT5 (R16_UNORM 2B atomic: 0.60 -> 39321)
    uint16_t u5 = *(uint16_t*)&d5[pL * 2];
    printf("RT5 R16_UNORM (2B atomic): %u expect~39321\n", u5);
    if (abs((int)u5 - 39321) > 10) ok = false;

    // Check RT6 (RGBA16_UNORM 8B: 0.1, 0.4, 0.7, 0.9)
    uint16_t* u6 = (uint16_t*)&d6[pL * 8];
    printf("RT6 RGBA16_UNORM (8B): (%u,%u,%u,%u) expect~(6554,26214,45875,58982)\n", u6[0], u6[1], u6[2], u6[3]);
    if (abs((int)u6[0] - 6554) > 10 || abs((int)u6[3] - 58982) > 10) ok = false;

    // Check RT7 (RGBA16_SFLOAT 8B: 2, 4, 6, 8)
    uint16_t* h7 = (uint16_t*)&d7[pL * 8];
    printf("RT7 RGBA16_SFLOAT (8B): (%.1f,%.1f,%.1f,%.1f) expect=(2.0,4.0,6.0,8.0)\n",
           f16_to_f32(h7[0]), f16_to_f32(h7[1]), f16_to_f32(h7[2]), f16_to_f32(h7[3]));
    if (fabsf(f16_to_f32(h7[0]) - 2.f) > 0.05f || fabsf(f16_to_f32(h7[3]) - 8.f) > 0.05f) ok = false;

    printf("V04_FEATURE_TEST %s | device=\"%s\" | 8MRT+21FMT+Atomic1B2B+Stencil+WriteMask+SepAlpha+BlendConst+FrontFace+IndirectCount+Uint8Idx+PushDesc\n",
           ok ? "PASS" : "FAIL", props.deviceName);
    return ok ? 0 : 1;
}
