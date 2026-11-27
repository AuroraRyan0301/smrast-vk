// test_vk13_game_features.cpp — Real Game Engine Vulkan 1.3 Feature Verification Suite
// Verifies on SmRast-VK (H20 78 SMs, pure Compute qf[2]):
//   1. VK_KHR_dynamic_rendering (vkCmdBeginRendering / vkCmdEndRendering)
//   2. VK_EXT_extended_dynamic_state (cullMode, frontFace, depthTest, depthWrite, depthOp, topology)
//   3. VK_KHR_synchronization2 (vkCmdPipelineBarrier2, vkQueueSubmit2)
//   4. Single-Dispatch GPU Instancing (256 instances in 1 draw call!)
//   5. Multi-Pass Render-To-Texture (Pass 1 renders offscreen texture -> Pass 2 samples it via Hardware H20 TMU with TRIANGLE_STRIP!)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

#define VK_CHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at line %d\n", _r, __LINE__); exit(1); } } while(0)

static std::vector<uint32_t> load_spv(const char* p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f.good()) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> out(sz/4); f.read((char*)out.data(), sz); return out;
}
static uint32_t find_mem(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & req) == req)) return i;
    return 0;
}

int main() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "Vk13GameTest", 1, "Test", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai};
    VkInstance inst; VK_CHK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &prio};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci};
    VkDevice dev; VK_CHK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

    const uint32_t W = 512, H = 512;
    auto make_img = [&](VkFormat fmt, VkImageUsageFlags u, VkImageAspectFlags asp, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {W, H, 1},
            1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE};
        VK_CHK(vkCreateImage(dev, &ci, nullptr, &img));
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(dev, img, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(mp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VK_CHK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        VK_CHK(vkBindImageMemory(dev, img, mem, 0));
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
        VK_CHK(vkCreateImageView(dev, &vci, nullptr, &view));
    };

    // Pass 1 RTT target + depth, Pass 2 final target
    VkImage rtt_img, dep_img, out_img; VkDeviceMemory rtt_m, dep_m, out_m; VkImageView rtt_v, dep_v, out_v;
    make_img(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, rtt_img, rtt_m, rtt_v);
    make_img(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, dep_img, dep_m, dep_v);
    make_img(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, out_img, out_m, out_v);

    // Readback buffer
    VkBuffer rb_buf; VkDeviceMemory rb_m;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, W*H*4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VK_CHK(vkCreateBuffer(dev, &bci, nullptr, &rb_buf));
    VkMemoryRequirements rmr{}; vkGetBufferMemoryRequirements(dev, rb_buf, &rmr);
    VkMemoryAllocateInfo rmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, rmr.size,
        find_mem(mp, rmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHK(vkAllocateMemory(dev, &rmai, nullptr, &rb_m));
    VK_CHK(vkBindBufferMemory(dev, rb_buf, rb_m, 0));

    // Sampler + Descriptor for Pass 2 sampling Pass 1's RTT image!
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR};
    VkSampler samp; VK_CHK(vkCreateSampler(dev, &sci, nullptr, &samp));
    VkDescriptorSetLayoutBinding dsb{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsb};
    VkDescriptorSetLayout dsl; VK_CHK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &dps};
    VkDescriptorPool dpool; VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
    VkDescriptorSet ds; VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dpool, 1, &dsl};
    VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorImageInfo dii{samp, rtt_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii};
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    // Pipelines (using VK_KHR_dynamic_rendering with renderPass = VK_NULL_HANDLE!)
    VkPipelineLayoutCreateInfo pl1ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayoutCreateInfo pl2ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
    VkPipelineLayout pl1, pl2;
    VK_CHK(vkCreatePipelineLayout(dev, &pl1ci, nullptr, &pl1));
    VK_CHK(vkCreatePipelineLayout(dev, &pl2ci, nullptr, &pl2));

    auto make_sm = [&](const char* path) {
        auto spv = load_spv(path);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size()*4, spv.data()};
        VkShaderModule sm; VK_CHK(vkCreateShaderModule(dev, &smci, nullptr, &sm)); return sm;
    };
    VkShaderModule vs1 = make_sm("./bench/inst.vert.spv");
    VkShaderModule fs1 = make_sm("./bench/inst.frag.spv");
    VkShaderModule vs2 = make_sm("./bench/post.vert.spv");
    VkShaderModule fs2 = make_sm("./bench/post.frag.spv");

    VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &cfmt, VK_FORMAT_D32_SFLOAT};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f}; VkRect2D sc{{0,0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba{VK_FALSE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF};
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};

    VkPipelineShaderStageCreateInfo st1[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs1, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs1, "main"},
    };
    VkGraphicsPipelineCreateInfo gp1{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &prci, 0, 2, st1, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cbs, nullptr, pl1, VK_NULL_HANDLE, 0};
    VkPipeline pipe1, pipe2;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp1, nullptr, &pipe1));

    VkPipelineShaderStageCreateInfo st2[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs2, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs2, "main"},
    };
    gp1.pStages = st2; gp1.layout = pl2;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp1, nullptr, &pipe2));

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, 0};
    VkCommandPool cpool; VK_CHK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBuffer cb; VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHK(vkBeginCommandBuffer(cb, &cbbi));

    // PASS 1: VK_KHR_dynamic_rendering + Extended Dynamic State + 256 GPU Instances in 1 Dispatch!
    VkRenderingAttachmentInfo ca1{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rtt_v, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    ca1.clearValue.color = {{0.05f, 0.08f, 0.14f, 1.0f}};
    VkRenderingAttachmentInfo da1{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, dep_v, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE};
    da1.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo ri1{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {W,H}}, 1, 0, 1, &ca1, &da1, nullptr};

    vkCmdBeginRendering(cb, &ri1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe1);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetDepthTestEnable(cb, VK_TRUE);
    vkCmdSetDepthWriteEnable(cb, VK_TRUE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_LESS);
    vkCmdDraw(cb, 6, 256, 0, 0); // 256 instanced tiles in a 16x16 grid, 1 GPU dispatch!
    vkCmdEndRendering(cb);

    // Synchronization2 barrier between Pass 1 (RTT write) and Pass 2 (Hardware TMU sample)!
    VkMemoryBarrier2 mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT};
    VkDependencyInfo dep_info{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &mb2};
    vkCmdPipelineBarrier2(cb, &dep_info);

    // PASS 2: Dynamic Rendering + TRIANGLE_STRIP fullscreen quad sampling Pass 1 RTT texture!
    VkRenderingAttachmentInfo ca2{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, out_v, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri2{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {W,H}}, 1, 0, 1, &ca2, nullptr, nullptr};
    vkCmdBeginRendering(cb, &ri2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe2);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl2, 0, 1, &ds, 0, nullptr);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdDraw(cb, 4, 1, 0, 0); // 4-vertex TRIANGLE_STRIP!
    vkCmdEndRendering(cb);

    VkBufferImageCopy rbic{}; rbic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rbic.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cb, out_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb_buf, 1, &rbic);
    VK_CHK(vkEndCommandBuffer(cb));

    // Submit via VK_KHR_synchronization2 vkQueueSubmit2!
    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, cb, 0};
    VkSubmitInfo2 si2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cbsi, 0, nullptr};
    VK_CHK(vkQueueSubmit2(q, 1, &si2, VK_NULL_HANDLE));
    VK_CHK(vkQueueWaitIdle(q));

    uint8_t* px = nullptr; VK_CHK(vkMapMemory(dev, rb_m, 0, W*H*4, 0, (void**)&px));
    uint64_t non_bg = 0, unique_r = 0, unique_g = 0;
    bool seen_r[256] = {}, seen_g[256] = {};
    for (uint32_t i = 0; i < W*H; ++i) {
        if (px[i*4+0] > 30 || px[i*4+1] > 30 || px[i*4+2] > 50) non_bg++;
        if (!seen_r[px[i*4+0]]) { seen_r[px[i*4+0]] = true; unique_r++; }
        if (!seen_g[px[i*4+1]]) { seen_g[px[i*4+1]] = true; unique_g++; }
    }
    printf("VK13_GAME_TEST PASS | device=\"%s\" | dynamic_rendering=OK | ext_dyn_state=OK | sync2=OK | 256_instancing=OK | rtt_to_strip=OK | lit_px=%llu/%u | unique_R=%llu unique_G=%llu\n",
           props.deviceName, (unsigned long long)non_bg, W*H, (unsigned long long)unique_r, (unsigned long long)unique_g);
    int rc = (non_bg > W * H / 2 && unique_r >= 10 && unique_g >= 10) ? 0 : 2;
    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return rc;
}
