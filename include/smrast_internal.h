#pragma once
// smrast_internal.h — SmRast-VK v0.3: Vulkan 1.3 Graphics-on-Compute Wrapper ICD for H20 / H100 / MI300X
// Runs 100% on backend VK_QUEUE_COMPUTE_BIT (zero hardware rasterizer, ROP, or RT Core touched!)
// Features: CUDARaster 16x16 binning + CuRast 64-bit atomic VisBuf + cuRE/RetroWarp shared-mem fine rasterizer,
//           MRT (up to 4 Color Attachments + HDR RGBA16F/R11G11B10F/R32F/RGBA32F), GPU-Driven DrawIndirect,
//           Sutherland-Hodgman Near-Plane Clipper, In-Process libshaderc Lowering, Zero-Copy MIT-SHM WSI!

#define VK_USE_PLATFORM_XCB_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>
#include <xcb/shm.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef ICD_LOADER_MAGIC
#define ICD_LOADER_MAGIC 0x01CDC0DE
#endif

static constexpr uint32_t SM_MAX_MRT            = 8;
static constexpr uint32_t SM_MAX_VARYINGS       = 15;
static constexpr uint32_t SM_VERT_STRIDE_VEC4   = 1 + SM_MAX_VARYINGS; // 16 vec4 = 256 bytes
static constexpr uint32_t SM_MAX_DRAW_VERTS     = 6 * 1024 * 1024;     // 6.29M regular verts/draw (>2M tris!)
static constexpr uint32_t SM_MAX_VERTS_PER_DRAW = 7 * 1024 * 1024;     // +1M near-clipped verts
static constexpr uint32_t SM_MAX_DRAW_TRIS      = 2100000;             // 2.1M regular tris/draw
static constexpr uint32_t SM_MAX_TRIS_PER_DRAW  = 2500000;             // +400K near-clipped tris
static constexpr uint32_t SM_MAX_TRIS_PER_TILE  = 2048;                // 2048 macro tris/tile
static constexpr uint32_t SM_MAX_VTX_BINDINGS   = 8;

// Render-Target Format Encoding shared between Host & Compute Shaders:
// 0=RGBA8_UNORM, 1=BGRA8_UNORM, 2=RGBA8_SRGB, 3=BGRA8_SRGB,
// 4=RGBA16_SFLOAT (8B), 5=B10G11R11_UFLOAT (4B), 6=R32_SFLOAT (4B),
// 7=RG16_SFLOAT (4B), 8=RGBA32_SFLOAT (16B), 9=A2B10G10R10_UNORM (4B),
// 10=RG32_SFLOAT (8B), 11=R32_UINT/SINT (4B), 12=RGBA8_SNORM (4B),
// 13=RG16_UNORM (4B), 14=RG16_SNORM (4B), 15=RGBA16_UNORM (8B), 16=RGBA16_SNORM (8B),
// 17=R8_UNORM/UINT (1B), 18=RG8_UNORM (2B), 19=R16_SFLOAT (2B), 20=R16_UNORM/UINT (2B)
static inline uint32_t sm_format_code(VkFormat f) {
    switch (f) {
        case VK_FORMAT_B8G8R8A8_UNORM:          return 1u;
        case VK_FORMAT_R8G8B8A8_SRGB:           return 2u;
        case VK_FORMAT_B8G8R8A8_SRGB:           return 3u;
        case VK_FORMAT_R16G16B16A16_SFLOAT:     return 4u;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return 5u;
        case VK_FORMAT_R32_SFLOAT:              return 6u;
        case VK_FORMAT_R16G16_SFLOAT:           return 7u;
        case VK_FORMAT_R32G32B32A32_SFLOAT:     return 8u;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:return 9u;
        case VK_FORMAT_R32G32_SFLOAT:           return 10u;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:                return 11u;
        case VK_FORMAT_R8G8B8A8_SNORM:          return 12u;
        case VK_FORMAT_R16G16_UNORM:            return 13u;
        case VK_FORMAT_R16G16_SNORM:            return 14u;
        case VK_FORMAT_R16G16B16A16_UNORM:      return 15u;
        case VK_FORMAT_R16G16B16A16_SNORM:      return 16u;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:                 return 17u;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:              return 18u;
        case VK_FORMAT_R16_SFLOAT:              return 19u;
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:                return 20u;
        default:                                return 0u;
    }
}
static inline uint32_t sm_format_bpp_code(VkFormat f) {
    uint32_t c = sm_format_code(f);
    if (c == 4u || c == 10u || c == 15u || c == 16u) return 1u; // 8 bytes/px (2 u32)
    if (c == 8u)                                     return 2u; // 16 bytes/px (4 u32)
    if (c == 17u)                                    return 3u; // 1 byte/px
    if (c == 18u || c == 19u || c == 20u)            return 4u; // 2 bytes/px
    return 0u;                                                  // 4 bytes/px (1 u32)
}
static inline uint32_t sm_format_bpp(VkFormat f) {
    uint32_t bc = sm_format_bpp_code(f);
    if (bc == 1u) return 8u;
    if (bc == 2u) return 16u;
    if (bc == 3u) return 1u;
    if (bc == 4u) return 2u;
    return 4u;
}
static inline uint32_t sm_format_exact_bpp(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_SRGB:
            return 1u;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_D16_UNORM:
            return 2u;
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT:
            return 8u;
        case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16u;
        default:
            return 4u;
    }
}

struct SmInstance;
struct SmPhysicalDevice;
struct SmDevice;
struct SmQueue;
struct SmCommandBuffer;

struct BackendInstFns {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkCreateInstance CreateInstance = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures GetPhysicalDeviceFeatures = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2 GetPhysicalDeviceQueueFamilyProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties2 GetPhysicalDeviceMemoryProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties GetPhysicalDeviceImageFormatProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkCreateDevice CreateDevice = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
};

struct BackendDevFns {
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkQueueSubmit2 QueueSubmit2 = nullptr;
    PFN_vkQueueWaitIdle QueueWaitIdle = nullptr;

    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges = nullptr;

    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;

    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout = nullptr;

    PFN_vkCreateImageView CreateImageView = nullptr;
    PFN_vkDestroyImageView DestroyImageView = nullptr;
    PFN_vkCreateSampler CreateSampler = nullptr;
    PFN_vkDestroySampler DestroySampler = nullptr;

    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines CreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
    PFN_vkCreatePipelineCache CreatePipelineCache = nullptr;
    PFN_vkDestroyPipelineCache DestroyPipelineCache = nullptr;
    PFN_vkGetPipelineCacheData GetPipelineCacheData = nullptr;

    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkResetDescriptorPool ResetDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets FreeDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;

    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkResetCommandPool ResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers FreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkResetCommandBuffer ResetCommandBuffer = nullptr;

    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkGetFenceStatus GetFenceStatus = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkCreateSemaphore CreateSemaphore = nullptr;
    PFN_vkDestroySemaphore DestroySemaphore = nullptr;

    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants CmdPushConstants = nullptr;
    PFN_vkCmdPushDescriptorSetKHR CmdPushDescriptorSetKHR = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCmdDispatchIndirect CmdDispatchIndirect = nullptr;
    PFN_vkCmdCopyBuffer CmdCopyBuffer = nullptr;
    PFN_vkCmdCopyImage CmdCopyImage = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkCmdFillBuffer CmdFillBuffer = nullptr;
    PFN_vkCmdUpdateBuffer CmdUpdateBuffer = nullptr;
    PFN_vkCmdClearColorImage CmdClearColorImage = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
};

struct SmInstance {
    VK_LOADER_DATA loader_data;
    void* backend_dl = nullptr;
    BackendInstFns fn;
    VkInstance real_inst = VK_NULL_HANDLE;
    std::vector<SmPhysicalDevice*> phys_devs;
};

struct SmPhysicalDevice {
    VK_LOADER_DATA loader_data;
    SmInstance* inst = nullptr;
    VkPhysicalDevice real_pd = VK_NULL_HANDLE;
    uint32_t backend_compute_qf = 2;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
};

struct SmShaderModule {
    VkShaderModule real_mod = VK_NULL_HANDLE;
    std::vector<uint32_t> spirv;
};

struct SmDescriptorUpdateTemplate {
    VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    std::vector<VkDescriptorUpdateTemplateEntry> entries;
};

struct SmBuffer {
    VkBuffer real_buf = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
};

struct SmImage {
    VkImage real_img = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{1, 1, 1};
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
    VkImageUsageFlags usage = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    bool is_swapchain = false;
    bool is_depth = false;
    bool has_stencil = false;
    bool attach_dirty = false;
    int32_t attach_mip = -1;
    int32_t attach_layer = -1;
    bool stencil_dirty = false;
    int32_t stencil_mip = -1;
    int32_t stencil_layer = -1;
    VkBuffer attach_buf = VK_NULL_HANDLE;
    VkDeviceMemory attach_mem = VK_NULL_HANDLE;
    VkBuffer hiz_buf = VK_NULL_HANDLE;
    VkDeviceMemory hiz_mem = VK_NULL_HANDLE;
    bool hiz_valid = false;
    bool hiz_has_occluders = false;
    uint32_t hiz_tx = 0, hiz_ty = 0;
    VkBuffer stencil_buf = VK_NULL_HANDLE;
    VkDeviceMemory stencil_mem = VK_NULL_HANDLE;
    VkImage real_stencil_img = VK_NULL_HANDLE;
    VkDeviceMemory real_stencil_mem = VK_NULL_HANDLE;
    VkBuffer host_buf = VK_NULL_HANDLE;
    VkDeviceMemory host_mem = VK_NULL_HANDLE;
    void* host_ptr = nullptr;
};

struct SmImageView {
    VkImageView real_view = VK_NULL_HANDLE;
    SmImage* image = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t base_mip = 0;
    uint32_t level_count = 1;
    uint32_t base_layer = 0;
    uint32_t layer_count = 1;
    bool is_stencil_view = false;
};

struct SmRenderPass {
    std::vector<VkAttachmentDescription> attachments;
    int color_attach_idxs[SM_MAX_MRT] = {-1,-1,-1,-1,-1,-1,-1,-1};
    uint32_t color_attach_count = 0;
    int depth_attach_idx = -1;
};

struct SmFramebuffer {
    SmRenderPass* rp = nullptr;
    std::vector<SmImageView*> attachments;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct SmPipelineLayout {
    VkPipelineLayout real_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> app_dsls;
    std::vector<VkPushConstantRange> push_ranges;
    VkPipelineLayout sm_vs_layout = VK_NULL_HANDLE;
    VkPipelineLayout sm_fs_layout = VK_NULL_HANDLE;
    uint32_t sm_set_index = 0;
};

struct SmVertexInputState {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attrs;
    uint32_t divisors[SM_MAX_VTX_BINDINGS] = {1, 1, 1, 1, 1, 1, 1, 1};
};

struct SmPipeline {
    bool is_compute = false;
    VkPipeline real_compute_pipe = VK_NULL_HANDLE;

    SmPipelineLayout* layout = nullptr;
    VkPipeline vs_comp_pipe = VK_NULL_HANDLE;
    VkPipeline fs_comp_pipe = VK_NULL_HANDLE;
    SmVertexInputState vtx_input;

    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool prim_restart = false;
    bool vs_writes_point_size = false;
    float line_width = 1.0f;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_clamp = false;
    bool rasterizer_discard = false;
    bool depth_bias_enable = false;
    float depth_bias_const = 0.0f;
    float depth_bias_clamp = 0.0f;
    float depth_bias_slope = 0.0f;
    bool depth_test = false;
    bool depth_write = false;
    VkCompareOp depth_op = VK_COMPARE_OP_LESS;
    bool stencil_test = false;
    VkStencilOpState stencil_front{};
    VkStencilOpState stencil_back{};

    // Per-MRT blending & write-mask state (up to 8 independent RTs!)
    bool blend_enable[SM_MAX_MRT] = {false,false,false,false,false,false,false,false};
    VkBlendFactor src_color_blend[SM_MAX_MRT] = {
        VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA};
    VkBlendFactor dst_color_blend[SM_MAX_MRT] = {
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA};
    VkBlendOp color_blend_op[SM_MAX_MRT] = {
        VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD,
        VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD};
    VkBlendFactor src_alpha_blend[SM_MAX_MRT] = {
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE};
    VkBlendFactor dst_alpha_blend[SM_MAX_MRT] = {
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO,
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO};
    VkBlendOp alpha_blend_op[SM_MAX_MRT] = {
        VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD,
        VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD};
    uint32_t color_write_mask[SM_MAX_MRT] = {0xFu,0xFu,0xFu,0xFu,0xFu,0xFu,0xFu,0xFu};
    float blend_constants[4] = {0.f, 0.f, 0.f, 0.f};
    bool has_static_vp = false;
    VkViewport static_vp{};
    bool has_static_scissor = false;
    VkRect2D static_scissor{};
};

struct SmSurface {
    bool is_xcb = false;
    xcb_connection_t* xcb_conn = nullptr;
    xcb_window_t xcb_win = 0;
    Display* xlib_dpy = nullptr;
    Window xlib_win = 0;
};

struct SmSwapchain {
    SmDevice* dev = nullptr;
    SmSurface* surface = nullptr;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{800, 600};
    std::vector<SmImage*> images;
    uint32_t next_image = 0;
    VkCommandPool present_pool = VK_NULL_HANDLE;
    VkCommandBuffer present_cmd = VK_NULL_HANDLE;
    VkFence present_fence = VK_NULL_HANDLE;
    uint64_t present_frames = 0;
    // Zero-Copy MIT-SHM XCB presentation state
    bool shm_checked = false;
    bool use_shm = false;
    int shm_id = -1;
    xcb_shm_seg_t shm_seg = 0;
    uint8_t* shm_ptr = nullptr;
    size_t shm_bytes = 0;
    xcb_gcontext_t xcb_gc = 0;
};

struct alignas(256) SmVsParamsUBO {
    uint32_t vertex_count;
    uint32_t first_vertex;
    uint32_t instance_idx;
    uint32_t indexed;            // 0=non-indexed, 1=uint16, 2=uint32
    uint32_t first_index;
    int32_t  vertex_offset;
    uint32_t vert_stride_vec4;
    uint32_t idx_buf_offset;
    uint32_t vtx_offsets[4];     // VB 0..3
    uint32_t vtx_strides[4];     // VB 0..3
    uint32_t verts_per_instance; // word 16 (patched by smrast_indirect_prep.comp!)
    uint32_t vtx_inst_rate_mask; // bits 0..7 = instanced VB 0..7
    uint32_t prim_restart_en;    // 1 = emit restart sentinel (w = -2.0) for 0xFF/0xFFFF/0xFFFFFFFF
    uint32_t pad1;
    uint32_t vtx_offsets_hi[4];  // VB 4..7
    uint32_t vtx_strides_hi[4];  // VB 4..7
};

struct alignas(512) SmFsParamsUBO {
    uint32_t width;
    uint32_t height;
    uint32_t vert_stride_vec4;
    uint32_t depth_write;
    uint32_t rt_active_mask;     // bits 0..7 = RT0..7 attached
    uint32_t stencil_enable;     // bit0=stencil_test, bit1=has_stencil_buf
    uint32_t stencil_front_cfg;  // [0..2]=cmpOp, [3..5]=failOp, [6..8]=passOp, [9..11]=depthFailOp, [16..23]=cmpMask, [24..31]=writeMask
    uint32_t stencil_back_cfg;   // same layout for back faces
    uint32_t stencil_refs;       // [0..7]=frontRef, [8..15]=backRef
    uint32_t tiles_x;            // screen tile columns for Active-Tile Indirect Dispatch!
    uint32_t direct_2d;          // 1 = coherent 2D grid (gl_WorkGroupID.xy), 0 = active_tiles[gl_WorkGroupID.x]
    uint32_t pad2;
    float    blend_const[4];     // vec4 blend_const
    uint32_t rt_fmt[8];          // uvec4 rt_fmt[2]
    uint32_t rt_blend_en[8];     // uvec4 rt_blend_en[2]
    uint32_t rt_src_blend[8];    // uvec4 rt_src_blend[2]
    uint32_t rt_dst_blend[8];    // uvec4 rt_dst_blend[2]
    uint32_t rt_blend_op[8];     // uvec4 rt_blend_op[2]
    uint32_t rt_src_alpha[8];    // uvec4 rt_src_alpha[2]
    uint32_t rt_dst_alpha[8];    // uvec4 rt_dst_alpha[2]
    uint32_t rt_alpha_op[8];     // uvec4 rt_alpha_op[2]
    uint32_t rt_write_mask[8];   // uvec4 rt_write_mask[2]
};

struct alignas(256) SmClearUBO {
    uint32_t width;
    uint32_t height;
    uint32_t clear_mask;         // bits 0..7=RT0..7, bit8=depth, bit9=vis, bit10=stencil
    float    clear_depth;
    uint32_t clear_stencil;
    uint32_t tiles_x;
    uint32_t tiles_y;
    uint32_t hiz_en;
    uint32_t rt_bpp_code[8];     // uvec4[2]: 0=4B, 1=8B, 2=16B, 3=1B, 4=2B
    uint32_t rt_clear[8][4];     // uvec4[8]
};

struct SmRasterEngine {
    VkDescriptorSetLayout clear_dsl = VK_NULL_HANDLE;
    VkPipelineLayout clear_pl = VK_NULL_HANDLE;
    VkPipeline clear_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout bin_dsl = VK_NULL_HANDLE;
    VkPipelineLayout bin_pl = VK_NULL_HANDLE;
    VkPipeline bin_pipe = VK_NULL_HANDLE;
    VkPipeline prim_bin_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout fine_dsl = VK_NULL_HANDLE;
    VkPipelineLayout fine_pl = VK_NULL_HANDLE;
    VkPipeline fine_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout ind_prep_dsl = VK_NULL_HANDLE;
    VkPipelineLayout ind_prep_pl = VK_NULL_HANDLE;
    VkPipeline ind_prep_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout depth_only_dsl = VK_NULL_HANDLE;
    VkPipelineLayout depth_only_pl = VK_NULL_HANDLE;
    VkPipeline depth_only_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout blit_dsl = VK_NULL_HANDLE;
    VkPipelineLayout blit_pl = VK_NULL_HANDLE;
    VkPipeline blit_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout hiz_update_dsl = VK_NULL_HANDLE;
    VkPipelineLayout hiz_update_pl = VK_NULL_HANDLE;
    VkPipeline hiz_update_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout vs_int_dsl = VK_NULL_HANDLE;
    VkDescriptorSetLayout fs_int_dsl = VK_NULL_HANDLE;

    VkBuffer clip_vert_buf = VK_NULL_HANDLE;   VkDeviceMemory clip_vert_mem = VK_NULL_HANDLE;
    VkBuffer tri_setup_buf = VK_NULL_HANDLE;   VkDeviceMemory tri_setup_mem = VK_NULL_HANDLE;
    uint32_t cap_verts = 0;
    uint32_t cap_tris  = 0;

    VkBuffer tile_count_buf = VK_NULL_HANDLE;  VkDeviceMemory tile_count_mem = VK_NULL_HANDLE;
    VkBuffer tile_bins_buf = VK_NULL_HANDLE;   VkDeviceMemory tile_bins_mem = VK_NULL_HANDLE;
    VkBuffer active_tiles_buf = VK_NULL_HANDLE; VkDeviceMemory active_tiles_mem = VK_NULL_HANDLE;
    VkBuffer vis_buf = VK_NULL_HANDLE;         VkDeviceMemory vis_mem = VK_NULL_HANDLE;
    VkBuffer ind_disp_buf = VK_NULL_HANDLE;    VkDeviceMemory ind_disp_mem = VK_NULL_HANDLE;
    VkBuffer dummy_buf = VK_NULL_HANDLE;       VkDeviceMemory dummy_mem = VK_NULL_HANDLE;

    static constexpr uint32_t MAX_DRAWS_IN_FLIGHT = 49152;
    static constexpr uint32_t MAX_DS_RING         = 16384;
    static constexpr uint32_t MAX_IND_SLOTS       = 16384;
    static constexpr uint32_t MAX_AUX_SLOTS       = 4096;
    VkBuffer vs_ubo_buf = VK_NULL_HANDLE;      VkDeviceMemory vs_ubo_mem = VK_NULL_HANDLE; void* vs_ubo_ptr = nullptr;
    VkBuffer fs_ubo_buf = VK_NULL_HANDLE;      VkDeviceMemory fs_ubo_mem = VK_NULL_HANDLE; void* fs_ubo_ptr = nullptr;
    VkBuffer clr_ubo_buf = VK_NULL_HANDLE;     VkDeviceMemory clr_ubo_mem = VK_NULL_HANDLE; void* clr_ubo_ptr = nullptr;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> clear_ds;
    std::vector<VkDescriptorSet> bin_ds;
    std::vector<VkDescriptorSet> fine_ds;
    std::vector<VkDescriptorSet> ind_prep_ds;
    std::vector<VkDescriptorSet> depth_only_ds;
    std::vector<VkDescriptorSet> blit_ds;
    std::vector<VkDescriptorSet> hiz_update_ds;
    std::vector<VkDescriptorSet> vs_int_ds;
    std::vector<VkDescriptorSet> fs_int_ds;
    uint32_t draw_cursor = 0;
    uint32_t ind_cursor = 0;
    uint32_t clear_cursor = 0;
    uint32_t blit_cursor = 0;
    uint32_t vs_ds_cursor = 0, fs_ds_cursor = 0, bin_ds_cursor = 0, fine_ds_cursor = 0, dep_ds_cursor = 0, hiz_ds_cursor = 0;
    std::unordered_map<uint64_t, VkDescriptorSet> vs_ds_cache;
    std::unordered_map<uint64_t, VkDescriptorSet> fs_ds_cache;
    std::unordered_map<uint64_t, VkDescriptorSet> bin_ds_cache;
    std::unordered_map<uint64_t, VkDescriptorSet> fine_ds_cache;
    std::unordered_map<uint64_t, VkDescriptorSet> dep_ds_cache;
    std::unordered_map<uint64_t, VkDescriptorSet> hiz_ds_cache;

    uint32_t cur_w = 0, cur_h = 0;
    uint32_t tiles_x = 0, tiles_y = 0;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> retired_bufs;
};

struct SmQueue {
    VK_LOADER_DATA loader_data;
    SmDevice* dev = nullptr;
    VkQueue real_q = VK_NULL_HANDLE;
    std::mutex mtx;
};

struct SmDevice {
    VK_LOADER_DATA loader_data;
    SmPhysicalDevice* pd = nullptr;
    BackendDevFns fn;
    VkDevice real_dev = VK_NULL_HANDLE;
    SmQueue* main_queue = nullptr;
    SmRasterEngine rast;
    std::unordered_map<VkDescriptorSetLayout, uint32_t> dsl_dyn_counts;
    std::vector<SmImage*> storage_images;
    uint32_t compiled_gfx_pipes = 0;
    uint32_t failed_gfx_pipes = 0;
    std::mutex mtx;
};

struct SmPushDescSet {
    bool active = false;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs;
    std::vector<std::vector<VkDescriptorImageInfo>>  imgs;
    std::vector<std::vector<VkBufferView>>           bvs;
};

struct SmCommandBuffer {
    VK_LOADER_DATA loader_data;
    SmDevice* dev = nullptr;
    VkCommandBuffer real_cb = VK_NULL_HANDLE;

    bool in_render_pass = false;
    bool in_dyn_rendering = false;
    SmRenderPass* cur_rp = nullptr;
    SmFramebuffer* cur_fb = nullptr;
    SmImage* dyn_col_imgs[SM_MAX_MRT] = {nullptr};
    SmImageView* dyn_col_views[SM_MAX_MRT] = {nullptr};
    SmImage* dyn_res_imgs[SM_MAX_MRT] = {nullptr};
    uint32_t dyn_col_count = 0;
    SmImage* dyn_dep_img = nullptr;
    SmImageView* dyn_dep_view = nullptr;
    uint32_t dyn_w = 0, dyn_h = 0;
    SmPipeline* cur_gfx_pipe = nullptr;
    SmPipeline* cur_comp_pipe = nullptr;

    VkViewport cur_vp{0, 0, 1920, 1080, 0.0f, 1.0f};
    VkRect2D cur_scissor{{0, 0}, {1920, 1080}};
    bool has_vp = false;
    bool has_scissor = false;

    bool has_dyn_cull = false;        VkCullModeFlags dyn_cull = VK_CULL_MODE_NONE;
    bool has_dyn_front_face = false;  VkFrontFace dyn_front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool has_dyn_topo = false;        VkPrimitiveTopology dyn_topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool has_dyn_depth_test = false;  bool dyn_depth_test = false;
    bool has_dyn_depth_write = false; bool dyn_depth_write = false;
    bool has_dyn_depth_op = false;    VkCompareOp dyn_depth_op = VK_COMPARE_OP_LESS;
    bool has_dyn_depth_bias = false;  float dyn_depth_bias_const = 0.f, dyn_depth_bias_clamp = 0.f, dyn_depth_bias_slope = 0.f;
    bool has_dyn_depth_bias_en = false; bool dyn_depth_bias_en = false;
    bool has_dyn_rast_discard = false;  bool dyn_rast_discard = false;
    bool has_dyn_prim_restart = false;  bool dyn_prim_restart = false;
    bool has_dyn_line_width = false;    float dyn_line_width = 1.0f;
    bool has_dyn_blend_const = false; float dyn_blend_const[4] = {0.f, 0.f, 0.f, 0.f};

    bool has_dyn_stencil_test = false;       bool dyn_stencil_test = false;
    bool has_dyn_stencil_op[2] = {false, false};
    VkStencilOp dyn_stencil_fail[2] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    VkStencilOp dyn_stencil_pass[2] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    VkStencilOp dyn_stencil_dfail[2] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    VkCompareOp dyn_stencil_cmp[2] = {VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS};
    bool has_dyn_stencil_ref[2] = {false, false};  uint32_t dyn_stencil_ref[2] = {0, 0};
    bool has_dyn_stencil_cmask[2] = {false, false}; uint32_t dyn_stencil_cmask[2] = {0xFF, 0xFF};
    bool has_dyn_stencil_wmask[2] = {false, false}; uint32_t dyn_stencil_wmask[2] = {0xFF, 0xFF};

    static constexpr uint32_t MAX_DYN_OFFS_PER_SET = 32;
    VkDescriptorSet gfx_sets[16] = {VK_NULL_HANDLE};
    uint32_t gfx_set_mask = 0;
    uint32_t gfx_dyn_counts[16] = {0};
    uint32_t gfx_dyn_offs[16][MAX_DYN_OFFS_PER_SET] = {{0}};
    SmPushDescSet gfx_push_descs[16];
    uint8_t push_data[256] = {0};
    uint32_t push_size = 0;

    // App Compute state mirror (restored before vkCmdDispatch if internal rasterizer passes clobbered COMPUTE bind point!)
    VkDescriptorSet comp_sets[16] = {VK_NULL_HANDLE};
    uint32_t comp_set_mask = 0;
    uint32_t comp_dyn_counts[16] = {0};
    uint32_t comp_dyn_offs[16][MAX_DYN_OFFS_PER_SET] = {{0}};
    SmPushDescSet comp_push_descs[16];
    uint8_t comp_push_data[256] = {0};
    uint32_t comp_push_size = 0;
    bool comp_state_dirty = false;

    SmBuffer* vtx_bufs[SM_MAX_VTX_BINDINGS] = {nullptr};
    VkDeviceSize vtx_offsets[SM_MAX_VTX_BINDINGS] = {0};
    uint32_t dyn_vtx_strides[SM_MAX_VTX_BINDINGS] = {0};
    bool has_dyn_vtx_stride[SM_MAX_VTX_BINDINGS] = {false};
    SmBuffer* idx_buf = nullptr;
    VkDeviceSize idx_offset = 0;
    VkIndexType idx_type = VK_INDEX_TYPE_UINT32;
};

uint32_t sm_find_mem_type(SmPhysicalDevice* pd, uint32_t bits, VkMemoryPropertyFlags req);
bool sm_ensure_image_attach_buf(SmDevice* dev, SmImage* img);
bool sm_ensure_image_stencil_buf(SmDevice* dev, SmImage* img);
bool sm_ensure_geom_capacity(SmDevice* dev, uint32_t need_verts, uint32_t need_tris);
bool sm_init_raster_engine(SmDevice* dev);
void sm_destroy_raster_engine(SmDevice* dev);
bool sm_ensure_vis_size(SmDevice* dev, uint32_t w, uint32_t h);

bool sm_lower_vertex_spirv(const std::vector<uint32_t>& in_spv,
                           const SmVertexInputState& vtx_in,
                           const VkSpecializationInfo* spec,
                           uint32_t sm_set_idx,
                           std::vector<uint32_t>& out_comp_spv,
                           std::string& err,
                           bool* out_writes_point_size = nullptr);

struct SmFsCompileHints {
    int      rt_fmt[SM_MAX_MRT]     = {-1,-1,-1,-1,-1,-1,-1,-1};
    bool     blend_en[SM_MAX_MRT]   = {true,true,true,true,true,true,true,true};
    uint32_t write_mask[SM_MAX_MRT] = {0xFu,0xFu,0xFu,0xFu,0xFu,0xFu,0xFu,0xFu};
    bool     stencil_en             = false;
};

bool sm_lower_fragment_spirv(const std::vector<uint32_t>& in_spv,
                             const VkSpecializationInfo* spec,
                             uint32_t sm_set_idx,
                             const SmFsCompileHints& hints,
                             std::vector<uint32_t>& out_comp_spv,
                             std::string& err);

bool sm_build_null_fs_stencil_spv(uint32_t sm_set_idx,
                                  std::vector<uint32_t>& out_comp_spv,
                                  std::string& err);

void sm_wsi_blit_to_surface(SmSwapchain* sc, SmImage* img);
void sm_wsi_cleanup_swapchain(SmSwapchain* sc);
