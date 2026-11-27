// test_v09_features.cpp — SmRast-VK v0.9.0 Comprehensive Correctness & Throughput Verification Suite:
//   [1] Dynamic Compact vert_stride_vec4 (2-vec4 minimal vs 5-vec4 multi-varying vs 16-vec4 gl_PointSize)
//   [2] Zero-Bin Inline-Setup Fused Quad FS (direct_2d==3u): 2D fullscreen quad + 7-plane near/neg-W clip + Painter's Order
//   [3] Precomputed Branchless Fine Rasterizer + Intra-Tile Early-Z + Inter-Batch Live Hi-Z (512-Layer Overdraw Stress!)
//   [4] CUDARaster Exact Tile-Edge Trivial-Reject + 128px CuRast Micro-Tri Watertightness (0 cracks, 0 double-hits)
//   [5] Empty-Clear SyncA2I Skip Correctness (repeated empty depth/color clears -> draw invalidation -> TMU readback)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#define VK_CHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", _r, __LINE__); exit(1); } } while (0)

static std::vector<uint32_t> compile_glsl(const char* src, const char* stage) {
    std::string in_p = std::string("/tmp/sm_v09_tmp.") + stage;
    std::string out_p = "/tmp/sm_v09_tmp.spv";
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
    return spv;
}

static VkShaderModule make_mod(VkDevice dev, const char* src, const char* stage) {
    auto spv = compile_glsl(src, stage);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size()*4, spv.data()};
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) return i;
    fprintf(stderr, "No memory type\n"); exit(1);
}

struct ImgObj { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; uint32_t w = 0, h = 0; };
struct BufObj { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void* ptr = nullptr; VkDeviceSize sz = 0; };

static ImgObj make_img(VkPhysicalDevice pd, VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags asp) {
    ImgObj o; o.w = w; o.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1}, 1, 1,
                         VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHK(vkCreateImage(dev, &ci, nullptr, &o.img));
    VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(dev, o.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindImageMemory(dev, o.img, o.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, o.img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
    VK_CHK(vkCreateImageView(dev, &vi, nullptr, &o.view));
    return o;
}

static BufObj make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize sz, VkBufferUsageFlags usage) {
    BufObj o; o.sz = sz;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, VK_SHARING_MODE_EXCLUSIVE};
    VK_CHK(vkCreateBuffer(dev, &ci, nullptr, &o.buf));
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, o.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                            find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindBufferMemory(dev, o.buf, o.mem, 0));
    VK_CHK(vkMapMemory(dev, o.mem, 0, sz, 0, &o.ptr));
    return o;
}

static VkPipeline make_pipe(VkDevice dev, VkPipelineLayout pl, VkShaderModule vs, VkShaderModule fs,
                            VkFormat col_fmt, VkFormat dep_fmt, bool dtest, bool dwrite, VkCompareOp dop,
                            bool blend_add = false, VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                            uint32_t vstride = 0, const VkVertexInputAttributeDescription* attrs = nullptr, uint32_t nattrs = 0) {
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr}
    };
    VkVertexInputBindingDescription vb{0, vstride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0,
                                            vstride ? 1u : 0u, vstride ? &vb : nullptr, nattrs, attrs};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, topo, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
                                              VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0,
                                             dtest ? VK_TRUE : VK_FALSE, dwrite ? VK_TRUE : VK_FALSE, dop, VK_FALSE, VK_FALSE};
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xFu;
    if (blend_add) {
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; ba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &ba};
    VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyns};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &col_fmt, dep_fmt, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, st, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cb, &dyn, pl};
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p));
    return p;
}

static void run_cb(VkDevice dev, VkQueue q, VkCommandBuffer cb, VkFence fence) {
    VK_CHK(vkEndCommandBuffer(cb));
    VK_CHK(vkResetFences(dev, 1, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    VK_CHK(vkQueueSubmit(q, 1, &si, fence));
    VK_CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
}

int main() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "test_v09", 1, "test_v09", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai};
    VkInstance inst = VK_NULL_HANDLE;
    VK_CHK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    VK_CHK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    printf("Testing SmRast-VK v0.9.0 on: %s\n", props.deviceName);

    float qp = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &qp};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci};
    VkDevice dev = VK_NULL_HANDLE; VK_CHK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue(dev, 0, 0, &q);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VK_CHK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    VK_CHK(vkCreateFence(dev, &fci, nullptr, &fence));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl_empty = VK_NULL_HANDLE;
    VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl_empty));

    const uint32_t W = 256, H = 256;
    ImgObj col = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    ImgObj dep = make_img(pd, dev, W, H, VK_FORMAT_D32_SFLOAT,
                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    BufObj rb = make_buf(pd, dev, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    auto readback = [&](ImgObj& src_img) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkBufferImageCopy bic{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {src_img.w, src_img.h, 1}};
        vkCmdCopyImageToBuffer(cb, src_img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &bic);
        run_cb(dev, q, cb, fence);
        return (const uint8_t*)rb.ptr;
    };

    // =========================================================================
    // [1] Dynamic Compact vert_stride_vec4 (2-vec4 minimal, 5-vec4 varyings, 16-vec4 gl_PointSize)
    // =========================================================================
    {
        auto vs_min = make_mod(dev, R"(#version 460
            void main() {
                vec2 p[3] = vec2[](vec2(-0.9,-0.9), vec2(0.0,-0.9), vec2(-0.9,0.0));
                gl_Position = vec4(p[gl_VertexIndex], 0.5, 1.0);
            })", "vert");
        auto fs_min = make_mod(dev, R"(#version 460
            layout(location=0) out vec4 o;
            void main() { o = vec4(1.0, 0.0, 0.0, 1.0); })", "frag");
        auto vs_mul = make_mod(dev, R"(#version 460
            layout(location=0) out vec4 v0;
            layout(location=3) out vec4 v3;
            void main() {
                vec2 p[3] = vec2[](vec2(0.1,-0.9), vec2(0.9,-0.9), vec2(0.9,0.0));
                gl_Position = vec4(p[gl_VertexIndex], 0.5, 1.0);
                v0 = vec4(0.0, 1.0, 0.0, 1.0);
                v3 = vec4(0.0, 0.0, 1.0, 1.0);
            })", "vert");
        auto fs_mul = make_mod(dev, R"(#version 460
            layout(location=0) in vec4 v0;
            layout(location=3) in vec4 v3;
            layout(location=0) out vec4 o;
            void main() { o = vec4(v0.g * 0.0, v0.g, v3.b, 1.0); })", "frag");
        auto vs_pt = make_mod(dev, R"(#version 460
            void main() { gl_Position = vec4(0.0, 0.5, 0.5, 1.0); gl_PointSize = 8.0; })", "vert");
        auto fs_pt = make_mod(dev, R"(#version 460
            layout(location=0) out vec4 o;
            void main() { o = vec4(1.0, 1.0, 0.0, 1.0); })", "frag");

        VkPipeline p_min = make_pipe(dev, pl_empty, vs_min, fs_min, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false, false, VK_COMPARE_OP_ALWAYS);
        VkPipeline p_mul = make_pipe(dev, pl_empty, vs_mul, fs_mul, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false, false, VK_COMPARE_OP_ALWAYS);
        VkPipeline p_pt  = make_pipe(dev, pl_empty, vs_pt,  fs_pt,  VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false, false, VK_COMPARE_OP_ALWAYS, false, VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0,0,(float)W,(float)H,0,1}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_min); vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_mul); vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pt);  vkCmdDraw(cb, 1, 1, 0, 0);
        vkCmdEndRendering(cb);
        run_cb(dev, q, cb, fence);

        const uint8_t* px = readback(col);
        const uint8_t* p1 = px + (32 * W + 32) * 4;     // red (stride=2)
        const uint8_t* p2 = px + (32 * W + 220) * 4;    // cyan (stride=5)
        uint32_t pt_cnt = 0;
        for (uint32_t y = 180; y < 204; ++y)
            for (uint32_t x = 116; x < 140; ++x) {
                const uint8_t* qpx = px + (y * W + x) * 4;
                if (qpx[0] > 200 && qpx[1] > 200 && qpx[2] < 20) pt_cnt++;
            }
        printf("[1] Dynamic vert_stride_vec4 (2/5/16): min=(%u,%u,%u) mul=(%u,%u,%u) pt_px=%u\n",
               p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], pt_cnt);
        if (p1[0] < 240 || p2[1] < 240 || p2[2] < 240 || pt_cnt < 56) { fprintf(stderr, "FAIL [1]\n"); return 1; }
        vkDestroyPipeline(dev, p_min, nullptr); vkDestroyPipeline(dev, p_mul, nullptr); vkDestroyPipeline(dev, p_pt, nullptr);
        vkDestroyShaderModule(dev, vs_min, nullptr); vkDestroyShaderModule(dev, fs_min, nullptr);
        vkDestroyShaderModule(dev, vs_mul, nullptr); vkDestroyShaderModule(dev, fs_mul, nullptr);
        vkDestroyShaderModule(dev, vs_pt, nullptr); vkDestroyShaderModule(dev, fs_pt, nullptr);
    }

    // =========================================================================
    // [2] Zero-Bin Inline Fused Quad FS (direct_2d==3u): 7-plane clip + Painter's Order
    // =========================================================================
    {
        auto vs = make_mod(dev, R"(#version 460
            layout(location=0) out vec3 vcol;
            void main() {
                // Tri 0: negative-w near-clipped quad (green); Tri 1: overlapping smaller front tri (magenta)
                vec4 p[6] = vec4[](
                    vec4(-0.8, -0.8, 0.5, 1.0), vec4(0.8, -0.8, 0.5, 1.0), vec4(0.0, 2.0, -0.5, -1.0),
                    vec4(-0.3, -0.3, 0.5, 1.0), vec4(0.3, -0.3, 0.5, 1.0), vec4(0.0, 0.3, 0.5, 1.0)
                );
                vec3 c[6] = vec3[](
                    vec3(0,1,0), vec3(0,1,0), vec3(0,1,0),
                    vec3(1,0,1), vec3(1,0,1), vec3(1,0,1)
                );
                gl_Position = p[gl_VertexIndex];
                vcol = c[gl_VertexIndex];
            })", "vert");
        auto fs = make_mod(dev, R"(#version 460
            layout(location=0) in vec3 vcol;
            layout(location=0) out vec4 o;
            void main() { o = vec4(vcol, 1.0); })", "frag");
        VkPipeline p = make_pipe(dev, pl_empty, vs, fs, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false, false, VK_COMPARE_OP_ALWAYS);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0,0,(float)W,(float)H,0,1}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        vkCmdDraw(cb, 6, 1, 0, 0); // 2 tris, no depth -> direct_2d==3u Zero-Bin Inline Fused Quad!
        vkCmdEndRendering(cb);
        run_cb(dev, q, cb, fence);
        const uint8_t* px = readback(col);
        const uint8_t* mid = px + (128 * W + 128) * 4;   // magenta (Painter's Order!)
        const uint8_t* top = px + (40 * W + 128) * 4;    // green (clipped negative-w tri!)
        printf("[2] Zero-Bin Inline Fused Quad (7-plane clip + Painter's Order): mid=(%u,%u,%u) clip=(%u,%u,%u)\n",
               mid[0],mid[1],mid[2], top[0],top[1],top[2]);
        if (mid[0] < 240 || mid[2] < 240 || top[1] < 240) { fprintf(stderr, "FAIL [2]\n"); return 1; }
        vkDestroyPipeline(dev, p, nullptr); vkDestroyShaderModule(dev, vs, nullptr); vkDestroyShaderModule(dev, fs, nullptr);
    }

    // =========================================================================
    // [3] Precomputed Branchless Fine + Intra-Tile Early-Z + Inter-Batch Live Hi-Z (512-Layer Reverse-Z Overdraw!)
    // =========================================================================
    {
        auto vs = make_mod(dev, R"(#version 460
            layout(location=0) out vec3 vcol;
            void main() {
                uint inst = gl_InstanceIndex; // 0..511, front-to-back in Reverse-Z (0.999 -> 0.1)
                float z = 0.999 - float(inst) * (0.8 / 512.0);
                vec2 q[6] = vec2[](vec2(-0.9,-0.9), vec2(0.9,-0.9), vec2(0.9,0.9), vec2(-0.9,-0.9), vec2(0.9,0.9), vec2(-0.9,0.9));
                gl_Position = vec4(q[gl_VertexIndex], z, 1.0);
                vcol = (inst == 0u) ? vec3(0.0, 0.8, 1.0) : vec3(1.0, 0.0, 0.0);
            })", "vert");
        auto fs = make_mod(dev, R"(#version 460
            layout(location=0) in vec3 vcol;
            layout(location=0) out vec4 o;
            void main() { o = vec4(vcol, 1.0); })", "frag");
        VkPipeline p = make_pipe(dev, pl_empty, vs, fs, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, true, true, VK_COMPARE_OP_GREATER);
        auto run_overdraw = [&]() {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHK(vkBeginCommandBuffer(cb, &bi));
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0.0f, 0}}}};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da};
            vkCmdBeginRendering(cb, &ri);
            VkViewport vp{0,0,(float)W,(float)H,0,1}; VkRect2D sc{{0,0},{W,H}};
            vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
            vkCmdDraw(cb, 6, 512, 0, 0); // 1,024 full-tile overlapping tris -> 16 Fine batches with Live Hi-Z Tightening!
            vkCmdEndRendering(cb);
            run_cb(dev, q, cb, fence);
        };
        run_overdraw(); // warmup
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < 20; ++it) run_overdraw();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
        const uint8_t* px = readback(col);
        const uint8_t* mid = px + (128 * W + 128) * 4;
        printf("[3] 512-Layer Overdraw Fine+LiveHiZ (Rev-Z): mid=(%u,%u,%u) avg=%.3f ms\n", mid[0], mid[1], mid[2], ms);
        if (mid[0] > 10 || mid[1] < 190 || mid[2] < 240) { fprintf(stderr, "FAIL [3]\n"); return 1; }
        vkDestroyPipeline(dev, p, nullptr); vkDestroyShaderModule(dev, vs, nullptr); vkDestroyShaderModule(dev, fs, nullptr);
    }

    // =========================================================================
    // [4] 128px CuRast Micro-Tri + Tile-Edge Trivial-Reject Watertight Grid (10x10 cells = 100px each!)
    // =========================================================================
    {
        auto vs = make_mod(dev, R"(#version 460
            void main() {
                // 8x8 grid of 10x10 pixel quads (100px bbox -> hits v0.9 <=128px CuRast fast-path!)
                uint cell = gl_VertexIndex / 6u;
                uint v    = gl_VertexIndex % 6u;
                uint gx = cell % 8u, gy = cell / 8u;
                float x0 = 64.0 + float(gx) * 10.0, y0 = 64.0 + float(gy) * 10.0;
                vec2 o[6] = vec2[](vec2(0,0), vec2(10,0), vec2(10,10), vec2(0,0), vec2(10,10), vec2(0,10));
                vec2 sp = vec2(x0, y0) + o[v];
                gl_Position = vec4((sp / 128.0) - 1.0, 0.5, 1.0);
            })", "vert");
        auto fs = make_mod(dev, R"(#version 460
            layout(location=0) out vec4 o;
            void main() { o = vec4(100.0/255.0, 0.0, 0.0, 1.0); })", "frag");
        VkPipeline p = make_pipe(dev, pl_empty, vs, fs, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false, false, VK_COMPARE_OP_ALWAYS, true);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0,0,(float)W,(float)H,0,1}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        vkCmdDraw(cb, 8 * 8 * 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        run_cb(dev, q, cb, fence);
        const uint8_t* px = readback(col);
        uint32_t exact = 0, cracks = 0, dbl = 0;
        for (uint32_t y = 64; y < 144; ++y) {
            for (uint32_t x = 64; x < 144; ++x) {
                uint8_t r = px[(y * W + x) * 4];
                if (r >= 98 && r <= 102) exact++;
                else if (r == 0) cracks++;
                else dbl++;
            }
        }
        printf("[4] 128px CuRast Watertight 80x80 Grid (10x10 quads): exact=%u/6400 cracks=%u dbl=%u\n", exact, cracks, dbl);
        if (exact != 6400 || cracks != 0 || dbl != 0) { fprintf(stderr, "FAIL [4]\n"); return 1; }
        vkDestroyPipeline(dev, p, nullptr); vkDestroyShaderModule(dev, vs, nullptr); vkDestroyShaderModule(dev, fs, nullptr);
    }

    // =========================================================================
    // [5] Empty-Clear SyncA2I Skip + Draw Invalidation -> Hardware TMU Sampled Depth Readback
    // =========================================================================
    {
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo dci_l{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VK_CHK(vkCreateDescriptorSetLayout(dev, &dci_l, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plc2{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
        VkPipelineLayout pl_tex = VK_NULL_HANDLE; VK_CHK(vkCreatePipelineLayout(dev, &plc2, nullptr, &pl_tex));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &ps};
        VkDescriptorPool dp = VK_NULL_HANDLE; VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VkDescriptorSet ds = VK_NULL_HANDLE; VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0, VK_FILTER_NEAREST, VK_FILTER_NEAREST};
        VkSampler samp = VK_NULL_HANDLE; VK_CHK(vkCreateSampler(dev, &sci, nullptr, &samp));
        VkDescriptorImageInfo dii{samp, dep.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii};
        vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        auto vs_d = make_mod(dev, R"(#version 460
            void main() {
                vec2 p[3] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));
                gl_Position = vec4(p[gl_VertexIndex], 0.25, 1.0);
            })", "vert");
        auto fs_d = make_mod(dev, R"(#version 460
            layout(location=0) out vec4 o;
            void main() { o = vec4(1.0); })", "frag");
        auto vs_q = make_mod(dev, R"(#version 460
            layout(location=0) out vec2 uv;
            void main() {
                vec2 p[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
                gl_Position = vec4(p[gl_VertexIndex], 0.5, 1.0);
                uv = p[gl_VertexIndex] * 0.5 + 0.5;
            })", "vert");
        auto fs_q = make_mod(dev, R"(#version 460
            layout(set=0, binding=0) uniform sampler2D t_dep;
            layout(location=0) in vec2 uv;
            layout(location=0) out vec4 o;
            void main() { float d = texture(t_dep, uv).r; o = vec4(d, d, d, 1.0); })", "frag");

        VkPipeline p_dep = make_pipe(dev, pl_empty, vs_d, fs_d, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, true, true, VK_COMPARE_OP_LESS);
        VkPipeline p_smp = make_pipe(dev, pl_tex,   vs_q, fs_q, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED,  false, false, VK_COMPARE_OP_ALWAYS);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHK(vkBeginCommandBuffer(cb, &bi));
        VkViewport vp{0,0,(float)W,(float)H,0,1}; VkRect2D sc{{0,0},{W,H}};
        // Pass A1 & A2: two consecutive empty clears to 0.75 (A2 hits empty-clear SyncA2I skip!)
        for (int k = 0; k < 2; ++k) {
            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0.75f, 0}}}};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 0, nullptr, &da};
            vkCmdBeginRendering(cb, &ri);
            vkCmdEndRendering(cb);
        }
        // Pass B: draw depth=0.25 triangle in center (must invalidate real_uniform_cleared!)
        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0.75f, 0}}}};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da};
            vkCmdBeginRendering(cb, &ri);
            vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_dep);
            vkCmdDraw(cb, 3, 1, 0, 0);
            vkCmdEndRendering(cb);
        }
        // Barrier + Sample depth via Hardware TMU!
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                 dep.img, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        {
            VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
            vkCmdBeginRendering(cb, &ri);
            vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_smp);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_tex, 0, 1, &ds, 0, nullptr);
            vkCmdDraw(cb, 3, 1, 0, 0);
            vkCmdEndRendering(cb);
        }
        run_cb(dev, q, cb, fence);
        const uint8_t* px = readback(col);
        uint8_t bg  = px[(10 * W + 10) * 4];    // expect ~191 (0.75 * 255)
        uint8_t mid = px[(128 * W + 128) * 4];  // expect ~64  (0.25 * 255)
        printf("[5] Empty-Clear SyncA2I Skip + Invalidation -> TMU: bg=%u(~191) mid=%u(~64)\n", bg, mid);
        if (abs((int)bg - 191) > 4 || abs((int)mid - 64) > 4) { fprintf(stderr, "FAIL [5]\n"); return 1; }
        vkDestroyPipeline(dev, p_dep, nullptr); vkDestroyPipeline(dev, p_smp, nullptr);
        vkDestroyShaderModule(dev, vs_d, nullptr); vkDestroyShaderModule(dev, fs_d, nullptr);
        vkDestroyShaderModule(dev, vs_q, nullptr); vkDestroyShaderModule(dev, fs_q, nullptr);
        vkDestroySampler(dev, samp, nullptr); vkDestroyDescriptorPool(dev, dp, nullptr);
        vkDestroyPipelineLayout(dev, pl_tex, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    }

    printf("V09_FEATURE_TEST PASS\n");
    vkDeviceWaitIdle(dev);
    vkDestroyBuffer(dev, rb.buf, nullptr); vkFreeMemory(dev, rb.mem, nullptr);
    vkDestroyImageView(dev, col.view, nullptr); vkDestroyImage(dev, col.img, nullptr); vkFreeMemory(dev, col.mem, nullptr);
    vkDestroyImageView(dev, dep.view, nullptr); vkDestroyImage(dev, dep.img, nullptr); vkFreeMemory(dev, dep.mem, nullptr);
    vkDestroyPipelineLayout(dev, pl_empty, nullptr);
    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDevice(dev, dev ? nullptr : nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
