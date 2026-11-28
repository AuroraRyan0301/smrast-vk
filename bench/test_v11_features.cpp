// test_v11_features.cpp — SmRast-VK v1.1.0 Deferred Rendering & Precision Regression Suite:
//   [1] Dual-Source Blending (location=0, index=1 + ONE / ONE_MINUS_SRC1_ALPHA transmittance)
//   [2] Stencil depthFailOp (VK_STENCIL_OP_ZERO on depth-fail vs REPLACE on depth-pass)
//   [3] Lossless 31-bit Reversed-Z + Cross-Pass Depth Invariance (<=16 ULP) + NaN Safety
//   [4] VK_FORMAT_R8_SNORM Clear, Render & Signed Readback
//   [5] Stencil-Write Uber-Batching (32 draws) + Small-RT (<64px) 40x22 RGBA16F Blur Chain

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <atomic>

#define VK_CHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", _r, __LINE__); exit(1); } } while (0)

static std::atomic<int> g_tmp_id{0};

static std::vector<uint32_t> compile_glsl(const char* src, const char* stage) {
    int id = g_tmp_id.fetch_add(1);
    std::string in_p = "/tmp/sm_v11_tmp_" + std::to_string(id) + "." + stage;
    std::string out_p = "/tmp/sm_v11_tmp_" + std::to_string(id) + ".spv";
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

static ImgObj make_img(VkPhysicalDevice pd, VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags asp) {
    ImgObj o; o.w = w; o.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1}, 1, 1,
                         VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHK(vkCreateImage(dev, &ci, nullptr, &o.img));
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(dev, o.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                            find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindImageMemory(dev, o.img, o.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0,
                             o.img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
    VK_CHK(vkCreateImageView(dev, &vi, nullptr, &o.view));
    return o;
}

static BufObj make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize sz, VkBufferUsageFlags usage) {
    BufObj o; o.sz = sz;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, VK_SHARING_MODE_EXCLUSIVE};
    VK_CHK(vkCreateBuffer(dev, &ci, nullptr, &o.buf));
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, o.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                            find_mem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHK(vkAllocateMemory(dev, &ai, nullptr, &o.mem));
    VK_CHK(vkBindBufferMemory(dev, o.buf, o.mem, 0));
    VK_CHK(vkMapMemory(dev, o.mem, 0, sz, 0, &o.ptr));
    return o;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u, f;
    if (e == 0) { if (m == 0) f = s << 31; else { while (!(m & 1024u)) { m <<= 1; e--; } e++; m &= ~1024u; f = (s << 31) | ((e + 112u) << 23) | (m << 13); } }
    else if (e == 31) f = (s << 31) | (255u << 23) | (m << 13);
    else f = (s << 31) | ((e + 112u) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static void submit_and_wait(VkDevice dev, VkQueue q, VkCommandBuffer cb) {
    VK_CHK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    VK_CHK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    VK_CHK(vkQueueWaitIdle(q));
}

static void copy_img_to_buf(VkCommandBuffer cb, const ImgObj& img, VkBuffer dst) {
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
    VkBufferImageCopy bic{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {img.w, img.h, 1}};
    vkCmdCopyImageToBuffer(cb, img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &bic);
}

int main() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastV11Test", 1, "SmRast", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai};
    VkInstance inst = VK_NULL_HANDLE;
    VK_CHK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t pc = 1; VkPhysicalDevice pd = VK_NULL_HANDLE;
    VK_CHK(vkEnumeratePhysicalDevices(inst, &pc, &pd));
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    printf("Testing SmRast-VK v1.1.0 on: %s\n", props.deviceName);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures feats{};
    feats.dualSrcBlend = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, nullptr, 0, nullptr, &feats};
    VkDevice dev = VK_NULL_HANDLE;
    VK_CHK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue(dev, 0, 0, &q);

    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VK_CHK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    // Common full-screen quad VS with push-constant rect (x0,y0,x1,y1) and z
    const char* vs_rect_src = R"(
        #version 460
        layout(push_constant) uniform PC { vec4 rect; float zval; float pad0; float pad1; float pad2; } pc;
        void main() {
            vec2 uvs[6] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
            vec2 uv = uvs[gl_VertexIndex % 6];
            vec2 pos = mix(pc.rect.xy, pc.rect.zw, uv);
            gl_Position = vec4(pos, pc.zval, 1.0);
        }
    )";
    VkShaderModule vs_rect = make_mod(dev, vs_rect_src, "vert");
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
    VkPipelineLayout pl_pc = VK_NULL_HANDLE;
    VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl_pc));

    auto build_pipe = [&](VkShaderModule vs, VkShaderModule fs, VkPipelineLayout pl,
                          VkFormat col_fmt, VkFormat dep_fmt,
                          const VkPipelineDepthStencilStateCreateInfo& ds,
                          const VkPipelineColorBlendAttachmentState& ba) -> VkPipeline {
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
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &ba};
        VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyns};
        VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0,
            col_fmt != VK_FORMAT_UNDEFINED ? 1u : 0u, col_fmt != VK_FORMAT_UNDEFINED ? &col_fmt : nullptr,
            dep_fmt, (dep_fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) ? dep_fmt : VK_FORMAT_UNDEFINED};
        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, st, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cbs, &dyn, pl};
        VkPipeline p = VK_NULL_HANDLE;
        VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p));
        return p;
    };

    // =========================================================================
    // [1] Dual-Source Blending (location=0, index=1 + ONE + ONE_MINUS_SRC1_ALPHA)
    // =========================================================================
    {
        const uint32_t W = 64, H = 64;
        auto col = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const char* fs_dual = R"(
            #version 460
            layout(location = 0, index = 0) out vec4 o0;
            layout(location = 0, index = 1) out vec4 o1;
            void main() {
                o0 = vec4(0.5, 0.25, 0.0, 1.0);
                o1 = vec4(0.0, 0.0, 0.0, 0.75); // transmittance = 1 - 0.75 = 0.25
            }
        )";
        VkShaderModule fs = make_mod(dev, fs_dual, "frag");
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp = VK_BLEND_OP_ADD;
        ba.colorWriteMask = 0xFu;
        VkPipeline p = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, ds, ba);

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0.2f, 0.4f, 0.8f, 1.0f}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        float pc_data[8] = {-1.f, -1.f, 1.f, 1.f, 0.5f, 0, 0, 0};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_data);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        copy_img_to_buf(cb, col, rb.buf);
        submit_and_wait(dev, q, cb);

        const uint8_t* px = (const uint8_t*)rb.ptr + ((H/2) * W + (W/2)) * 4;
        // Expected: (0.5 + 0.2*0.25, 0.25 + 0.4*0.25, 0.0 + 0.8*0.25) = (0.55, 0.35, 0.20) = (140, 89, 51)
        printf("[1] Dual-Source Blending (ONE + ONE_MINUS_SRC1_ALPHA): px=(%u,%u,%u) expect~(140,89,51)\n", px[0], px[1], px[2]);
        if (abs((int)px[0] - 140) > 2 || abs((int)px[1] - 89) > 2 || abs((int)px[2] - 51) > 2) {
            fprintf(stderr, "FAIL [1]: dual-source blending mismatch!\n"); exit(1);
        }
    }

    // =========================================================================
    // [2] Stencil depthFailOp (Deferred Light-Volume / Sky-Quad Masking)
    // =========================================================================
    {
        const uint32_t W = 64, H = 64;
        auto col = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto dep = make_img(pd, dev, W, H, VK_FORMAT_D32_SFLOAT_S8_UINT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
        auto rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const char* fs_col = R"(
            #version 460
            layout(push_constant) uniform PC { vec4 rect; float zval; float r; float g; float b; } pc;
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(pc.r, pc.g, pc.b, 1.0); }
        )";
        VkShaderModule fs = make_mod(dev, fs_col, "frag");
        VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xFu;

        // Pass A: Depth occluder (left half, z=0.5, GREATER_OR_EQUAL, depthWrite=1)
        VkPipelineDepthStencilStateCreateInfo ds_occ{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_occ.depthTestEnable = VK_TRUE; ds_occ.depthWriteEnable = VK_TRUE; ds_occ.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        VkPipeline p_occ = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, ds_occ, ba);

        // Pass B: Far-plane sky quad (z=0.0, GREATER_OR_EQUAL, depthWrite=0, passOp=REPLACE(1), depthFailOp=ZERO!)
        VkPipelineDepthStencilStateCreateInfo ds_sky{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_sky.depthTestEnable = VK_TRUE; ds_sky.depthWriteEnable = VK_FALSE; ds_sky.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        ds_sky.stencilTestEnable = VK_TRUE;
        ds_sky.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_ALWAYS, 0x01u, 0x01u, 0x01u};
        ds_sky.back = ds_sky.front;
        VkPipelineColorBlendAttachmentState ba_none{}; ba_none.colorWriteMask = 0u;
        VkPipeline p_sky = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, ds_sky, ba_none);

        // Pass C: Stencil-masked composite (renders GREEN only where (stencil & 1) == 0!)
        VkPipelineDepthStencilStateCreateInfo ds_cmp{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_cmp.stencilTestEnable = VK_TRUE;
        ds_cmp.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 0x01u, 0x00u, 0x00u};
        ds_cmp.back = ds_cmp.front;
        VkPipeline p_cmp = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, ds_cmp, ba);

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0.f, 0.f, 0.f, 1.f}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        da.clearValue.depthStencil = {0.0f, 0x01u};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, &da};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);

        // Draw left-half occluder at z=0.5 (color red, overwritten to green in Pass C if depthFailOp worked!)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_occ);
        float pc_occ[8] = {-1.f, -1.f, 0.f, 1.f, 0.5f, 1.f, 0.f, 0.f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_occ);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Draw far-plane quad at z=0.0 -> clears stencil bit 0 on left half (depth fail!)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_sky);
        float pc_sky[8] = {-1.f, -1.f, 1.f, 1.f, 0.0f, 0.f, 0.f, 1.f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_sky);
        vkCmdDraw(cb, 6, 1, 0, 0);

        // Draw fullscreen green where stencil == 0
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_cmp);
        float pc_cmp[8] = {-1.f, -1.f, 1.f, 1.f, 0.0f, 0.f, 1.f, 0.f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_cmp);
        vkCmdDraw(cb, 6, 1, 0, 0);

        vkCmdEndRendering(cb);
        copy_img_to_buf(cb, col, rb.buf);
        submit_and_wait(dev, q, cb);

        const uint8_t* pxL = (const uint8_t*)rb.ptr + ((H/2) * W + (W/4)) * 4;
        const uint8_t* pxR = (const uint8_t*)rb.ptr + ((H/2) * W + (3*W/4)) * 4;
        printf("[2] Stencil depthFailOp=ZERO: L=(%u,%u,%u) [expect 0,255,0] R=(%u,%u,%u) [expect 0,0,0]\n",
               pxL[0], pxL[1], pxL[2], pxR[0], pxR[1], pxR[2]);
        if (pxL[1] < 250 || pxL[0] > 5 || pxR[0] > 5 || pxR[1] > 5 || pxR[2] > 5) {
            fprintf(stderr, "FAIL [2]: stencil depthFailOp did not execute properly!\n"); exit(1);
        }
    }

    // =========================================================================
    // [3] Lossless 31-bit Reversed-Z + Cross-Pass Invariance (4 ULP drift) + NaN Guard
    // =========================================================================
    {
        const uint32_t W = 160, H = 160;
        auto col = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto dep = make_img(pd, dev, W, H, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        auto rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const char* fs_col = R"(
            #version 460
            layout(push_constant) uniform PC { vec4 rect; float zval; float r; float g; float b; } pc;
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(pc.r, pc.g, pc.b, 1.0); }
        )";
        VkShaderModule fs = make_mod(dev, fs_col, "frag");
        VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xFu;
        VkPipelineColorBlendAttachmentState ba_none{}; ba_none.colorWriteMask = 0u;
        VkPipelineDepthStencilStateCreateInfo ds_pre{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_pre.depthTestEnable = VK_TRUE; ds_pre.depthWriteEnable = VK_TRUE; ds_pre.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        VkPipeline p_pre = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, ds_pre, ba_none);

        VkPipelineDepthStencilStateCreateInfo ds_gb{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_gb.depthTestEnable = VK_TRUE; ds_gb.depthWriteEnable = VK_FALSE; ds_gb.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        VkPipeline p_gb = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, ds_gb, ba);

        VkPipeline p_nan = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, ds_pre, ba);

        float z_pre = 0.0005f;
        uint32_t zu = 0; memcpy(&zu, &z_pre, 4);
        zu -= 4u; // simulate 4-ULP cross-pass VS contraction drift
        float z_gb = 0.f; memcpy(&z_gb, &zu, 4);
        float z_nan = NAN;

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0},{W,H}};

        // Pass 1: Depth Pre-Pass writes z=0.0005 (full-screen tile path)
        VkRenderingAttachmentInfo ca1{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{1.f, 1.f, 1.f, 1.f}}}};
        VkRenderingAttachmentInfo da1{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        da1.clearValue.depthStencil = {0.0f, 0u};
        VkRenderingInfo ri1{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca1, &da1};
        vkCmdBeginRendering(cb, &ri1);
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pre);
        float pc1[8] = {-1.f, -1.f, 1.f, 1.f, z_pre, 1.f, 0.f, 0.f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc1);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);

        // Pass 2: G-Buffer Pass writes z=z_pre - 4 ULP (blue) -> must pass cross-pass GREATER_OR_EQUAL invariance!
        // Plus Pass 3 (pipeline switch): NaN quad must be safely rejected and never overwrite valid pixels!
        VkRenderingAttachmentInfo ca2 = ca1; ca2.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkRenderingAttachmentInfo da2 = da1; da2.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkRenderingInfo ri2{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca2, &da2};
        vkCmdBeginRendering(cb, &ri2);
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_gb);
        float pc2[8] = {-1.f, -1.f, 1.f, 1.f, z_gb, 0.2f, 0.6f, 0.9f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc2);
        vkCmdDraw(cb, 6, 1, 0, 0);
        // NaN quad on distinct pipeline (flushes separately) must not clobber blue!
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_nan);
        float pc3[8] = {-1.f, -1.f, 1.f, 1.f, z_nan, 1.f, 1.f, 0.f};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc3);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);

        copy_img_to_buf(cb, col, rb.buf);
        submit_and_wait(dev, q, cb);

        uint32_t ok_px = 0;
        for (uint32_t i = 0; i < W * H; ++i) {
            const uint8_t* p = (const uint8_t*)rb.ptr + i * 4;
            if (abs((int)p[0] - 51) <= 2 && abs((int)p[1] - 153) <= 2 && abs((int)p[2] - 230) <= 2) ok_px++;
        }
        printf("[3] Lossless 31-bit Reversed-Z + 4-ULP Cross-Pass Invariance + NaN Guard: %u/%u pixels OK\n", ok_px, W * H);
        if (ok_px != W * H) { fprintf(stderr, "FAIL [3]: Reversed-Z invariance or NaN guard failed!\n"); exit(1); }
    }

    // =========================================================================
    // [4] VK_FORMAT_R8_SNORM Clear & Render
    // =========================================================================
    {
        const uint32_t W = 64, H = 64;
        auto col = make_img(pd, dev, W, H, VK_FORMAT_R8_SNORM,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto rb = make_buf(pd, dev, W * H, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const char* fs_sn = R"(
            #version 460
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(0.5, 0.0, 0.0, 1.0); }
        )";
        VkShaderModule fs = make_mod(dev, fs_sn, "frag");
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xFu;
        VkPipeline p_sn = build_pipe(vs_rect, fs, pl_pc, VK_FORMAT_R8_SNORM, VK_FORMAT_UNDEFINED, ds, ba);

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{-0.5f, 0.f, 0.f, 0.f}}}};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_sn);
        float pc_r[8] = {0.f, -1.f, 1.f, 1.f, 0.5f, 0, 0, 0};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_r);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        copy_img_to_buf(cb, col, rb.buf);
        submit_and_wait(dev, q, cb);

        int8_t vL = ((const int8_t*)rb.ptr)[(H/2) * W + (W/4)];
        int8_t vR = ((const int8_t*)rb.ptr)[(H/2) * W + (3*W/4)];
        printf("[4] VK_FORMAT_R8_SNORM Clear & Render: L=%d [expect ~-64] R=%d [expect ~+64]\n", (int)vL, (int)vR);
        if (abs((int)vL + 64) > 2 || abs((int)vR - 64) > 2) { fprintf(stderr, "FAIL [4]: R8_SNORM mismatch!\n"); exit(1); }
    }

    // =========================================================================
    // [5] Stencil-Write Uber-Batching (32 draws) + Small-RT (40x22) RGBA16F Blur Chain
    // =========================================================================
    {
        // Part A: 32 batched stencil-write draws
        const uint32_t W = 64, H = 64;
        auto col = make_img(pd, dev, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto dep = make_img(pd, dev, W, H, VK_FORMAT_D32_SFLOAT_S8_UINT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
        auto rb = make_buf(pd, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const char* fs_tag = R"(
            #version 460
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(0.1, 0.2, 0.8, 1.0); }
        )";
        const char* fs_chk = R"(
            #version 460
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(0.0, 1.0, 0.5, 1.0); }
        )";
        VkShaderModule f1 = make_mod(dev, fs_tag, "frag");
        VkShaderModule f2 = make_mod(dev, fs_chk, "frag");
        VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xFu;

        VkPipelineDepthStencilStateCreateInfo ds_bat{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_bat.depthTestEnable = VK_TRUE; ds_bat.depthWriteEnable = VK_TRUE; ds_bat.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        ds_bat.stencilTestEnable = VK_TRUE;
        ds_bat.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFFu, 0x03u, 0x03u};
        ds_bat.back = ds_bat.front;
        VkPipeline p_bat = build_pipe(vs_rect, f1, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, ds_bat, ba);

        VkPipelineDepthStencilStateCreateInfo ds_eq{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds_eq.stencilTestEnable = VK_TRUE;
        ds_eq.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 0x03u, 0x00u, 0x03u};
        ds_eq.back = ds_eq.front;
        VkPipeline p_eq = build_pipe(vs_rect, f2, pl_pc, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, ds_eq, ba);

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, col.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,1}}}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep.view,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        da.clearValue.depthStencil = {0.0f, 0u};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{W,H}}, 1, 0, 1, &ca, &da, &da};
        vkCmdBeginRendering(cb, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0},{W,H}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_bat);
        for (int k = 0; k < 32; ++k) {
            float x0 = -1.0f + 2.0f * (float)k / 32.0f;
            float x1 = -1.0f + 2.0f * (float)(k + 1) / 32.0f;
            float pc_k[8] = {x0, -1.f, x1, 1.f, 0.5f, 0, 0, 0};
            vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_k);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_eq);
        float pc_all[8] = {-1.f, -1.f, 1.f, 1.f, 0.5f, 0, 0, 0};
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_all);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        copy_img_to_buf(cb, col, rb.buf);
        submit_and_wait(dev, q, cb);

        uint32_t tagged = 0;
        for (uint32_t i = 0; i < W * H; ++i) {
            const uint8_t* p = (const uint8_t*)rb.ptr + i * 4;
            if (p[0] < 5 && p[1] > 250 && abs((int)p[2] - 128) <= 2) tagged++;
        }

        // Part B: 40x22 RGBA16F bloom RT -> TMU blur into second 40x22 RGBA16F RT (small-RT linear-alias guard!)
        const uint32_t BW = 40, BH = 22;
        auto b0 = make_img(pd, dev, BW, BH, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto b1 = make_img(pd, dev, BW, BH, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        auto brb = make_buf(pd, dev, BW * BH * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        VkSampler samp = VK_NULL_HANDLE;
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHK(vkCreateSampler(dev, &sci, nullptr, &samp));

        VkDescriptorSetLayoutBinding dslb{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslb};
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VK_CHK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
        VkPipelineLayoutCreateInfo plci2{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl, 1, &pcr};
        VkPipelineLayout pl_samp = VK_NULL_HANDLE; VK_CHK(vkCreatePipelineLayout(dev, &plci2, nullptr, &pl_samp));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
        VkDescriptorPool dp = VK_NULL_HANDLE; VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        VkDescriptorSet dset = VK_NULL_HANDLE; VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &dset));
        VkDescriptorImageInfo dii{samp, b0.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii};
        vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        const char* fs_fill16 = R"(
            #version 460
            layout(location = 0) out vec4 o0;
            void main() { o0 = vec4(0.25, 0.5, 0.75, 1.0); }
        )";
        const char* fs_blur16 = R"(
            #version 460
            layout(set = 0, binding = 0) uniform sampler2D uSrc;
            layout(location = 0) out vec4 o0;
            void main() {
                vec2 uv = gl_FragCoord.xy / vec2(40.0, 22.0);
                vec4 s = 0.25 * texture(uSrc, uv - vec2(1.0/40.0, 0.0))
                       + 0.50 * texture(uSrc, uv)
                       + 0.25 * texture(uSrc, uv + vec2(1.0/40.0, 0.0));
                o0 = s;
            }
        )";
        VkPipelineDepthStencilStateCreateInfo ds_off{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipeline p_f16 = build_pipe(vs_rect, make_mod(dev, fs_fill16, "frag"), pl_pc, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, ds_off, ba);
        VkPipeline p_b16 = build_pipe(vs_rect, make_mod(dev, fs_blur16, "frag"), pl_samp, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, ds_off, ba);

        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkViewport bvp{0, 0, (float)BW, (float)BH, 0.0f, 1.0f}; VkRect2D bsc{{0,0},{BW,BH}};
        VkRenderingAttachmentInfo ca0{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, b0.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,0}}}};
        VkRenderingInfo ri0{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{BW,BH}}, 1, 0, 1, &ca0};
        vkCmdBeginRendering(cb, &ri0);
        vkCmdSetViewport(cb, 0, 1, &bvp); vkCmdSetScissor(cb, 0, 1, &bsc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_f16);
        vkCmdPushConstants(cb, pl_pc, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_all);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);

        VkImageMemoryBarrier bbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, b0.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bbar);

        VkRenderingAttachmentInfo ca1{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, b1.view,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {{{0,0,0,0}}}};
        VkRenderingInfo ri1{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0},{BW,BH}}, 1, 0, 1, &ca1};
        vkCmdBeginRendering(cb, &ri1);
        vkCmdSetViewport(cb, 0, 1, &bvp); vkCmdSetScissor(cb, 0, 1, &bsc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_b16);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl_samp, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cb, pl_samp, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc_all);
        vkCmdDraw(cb, 6, 1, 0, 0);
        vkCmdEndRendering(cb);
        copy_img_to_buf(cb, b1, brb.buf);
        submit_and_wait(dev, q, cb);

        uint32_t bad_hdr = 0;
        const uint16_t* hp = (const uint16_t*)brb.ptr;
        for (uint32_t i = 0; i < BW * BH; ++i) {
            float r = f16_to_f32(hp[i*4+0]), g = f16_to_f32(hp[i*4+1]), b = f16_to_f32(hp[i*4+2]);
            if (std::isnan(r) || std::isnan(g) || std::isnan(b) ||
                fabsf(r - 0.25f) > 0.02f || fabsf(g - 0.50f) > 0.02f || fabsf(b - 0.75f) > 0.02f) bad_hdr++;
        }
        printf("[5] Stencil-Write Uber-Batch (32 draws): %u/%u tagged | 40x22 RGBA16F Blur: bad=%u/%u\n",
               tagged, W * H, bad_hdr, BW * BH);
        if (tagged != W * H || bad_hdr != 0) {
            fprintf(stderr, "FAIL [5]: stencil batch or 40x22 RGBA16F blur failed!\n"); exit(1);
        }
    }

    printf("V11_FEATURE_TEST PASS\n");
    return 0;
}
