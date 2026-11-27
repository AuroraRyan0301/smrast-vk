#pragma once
// smrast_internal.h — SmRast-VK: Vulkan 1.3 Graphics-on-Compute Wrapper ICD for H20 / H100 / MI300X
// Runs 100% on backend VK_QUEUE_COMPUTE_BIT (zero hardware rasterizer, ROP, or RT Core touched!)

#define VK_USE_PLATFORM_XCB_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>

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

// Max varyings per vertex (slot 0 = gl_Position, slots 1..15 = Location 0..14)
static constexpr uint32_t SM_MAX_VARYINGS       = 15;
static constexpr uint32_t SM_VERT_STRIDE_VEC4   = 1 + SM_MAX_VARYINGS; // 16 vec4 = 256 bytes
static constexpr uint32_t SM_MAX_DRAW_VERTS     = 6 * 1024 * 1024;     // 6.29M regular verts/draw (>2M tris!)
static constexpr uint32_t SM_MAX_VERTS_PER_DRAW = 7 * 1024 * 1024;     // +1M near-clipped verts (~1.75 GB HBM3)
static constexpr uint32_t SM_MAX_DRAW_TRIS      = 2100000;             // 2.1M regular tris/draw
static constexpr uint32_t SM_MAX_TRIS_PER_DRAW  = 2500000;             // +400K near-clipped tris
static constexpr uint32_t SM_MAX_TRIS_PER_TILE  = 2048;                // 2048 macro tris/tile
static constexpr uint32_t SM_MAX_VTX_BINDINGS   = 8;

struct SmInstance;
struct SmPhysicalDevice;
struct SmDevice;
struct SmQueue;
struct SmCommandBuffer;

// -----------------------------------------------------------------------------
// Backend ICD function pointer tables (loaded directly from libGLX_nvidia.so.0 / backend ICD)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Wrapper structs (ALL dispatchable handles begin with VK_LOADER_DATA!)
// -----------------------------------------------------------------------------
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
    uint32_t backend_compute_qf = 2; // Pure VK_QUEUE_COMPUTE_BIT queue family on H20!
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
};

// Tracked non-dispatchable objects (cast directly to/from Vulkan handle uint64_t)
struct SmShaderModule {
    VkShaderModule real_mod = VK_NULL_HANDLE;
    std::vector<uint32_t> spirv;
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
    VkImageUsageFlags usage = 0;
    bool is_swapchain = false;
    bool is_depth = false;
    // Linear HBM3 attachment mirror buffer (used when rendered into by Compute Fragment Shader)
    VkBuffer attach_buf = VK_NULL_HANDLE;
    VkDeviceMemory attach_mem = VK_NULL_HANDLE;
    // Host-coherent readback staging buffer (only allocated for swapchain images!)
    VkBuffer host_buf = VK_NULL_HANDLE;
    VkDeviceMemory host_mem = VK_NULL_HANDLE;
    void* host_ptr = nullptr;
};

struct SmImageView {
    VkImageView real_view = VK_NULL_HANDLE;
    SmImage* image = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct SmRenderPass {
    std::vector<VkAttachmentDescription> attachments;
    int color_attach_idx = -1;
    int depth_attach_idx = -1;
};

struct SmFramebuffer {
    SmRenderPass* rp = nullptr;
    std::vector<SmImageView*> attachments;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct SmPipelineLayout {
    VkPipelineLayout real_layout = VK_NULL_HANDLE; // App's original layout (for native vkCmdDispatch)
    std::vector<VkDescriptorSetLayout> app_dsls;
    std::vector<VkPushConstantRange> push_ranges;
    // Extended layout with SmRast internal descriptor set appended at set = app_dsls.size()
    VkPipelineLayout sm_vs_layout = VK_NULL_HANDLE;
    VkPipelineLayout sm_fs_layout = VK_NULL_HANDLE;
    uint32_t sm_set_index = 0;
};

struct SmVertexInputState {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attrs;
};

struct SmPipeline {
    bool is_compute = false;
    VkPipeline real_compute_pipe = VK_NULL_HANDLE;

    // Lowered graphics -> compute pipelines
    SmPipelineLayout* layout = nullptr;
    VkPipeline vs_comp_pipe = VK_NULL_HANDLE;
    VkPipeline fs_comp_pipe = VK_NULL_HANDLE;
    SmVertexInputState vtx_input;

    // Fixed-function rasterization & depth/blend state
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_test = false;
    bool depth_write = false;
    VkCompareOp depth_op = VK_COMPARE_OP_LESS;
    bool blend_enable = false;
    VkBlendFactor src_color_blend = VK_BLEND_FACTOR_SRC_ALPHA;
    VkBlendFactor dst_color_blend = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkBlendOp color_blend_op = VK_BLEND_OP_ADD;
    bool has_static_vp = false;
    VkViewport static_vp{};
    bool has_static_scissor = false;
    VkRect2D static_scissor{};
};

// WSI Surface & Swapchain
struct SmSurface {
    bool is_xcb = false;
    // XCB
    xcb_connection_t* xcb_conn = nullptr;
    xcb_window_t xcb_win = 0;
    // Xlib
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
    // Dedicated present command pool & buffer for fast GPU->Host readback
    VkCommandPool present_pool = VK_NULL_HANDLE;
    VkCommandBuffer present_cmd = VK_NULL_HANDLE;
    uint64_t present_frames = 0;
};

// Internal GPU Compute Rasterizer Resources per Device
struct SmRasterEngine {
    VkDescriptorSetLayout clear_dsl = VK_NULL_HANDLE;
    VkPipelineLayout clear_pl = VK_NULL_HANDLE;
    VkPipeline clear_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout bin_dsl = VK_NULL_HANDLE;
    VkPipelineLayout bin_pl = VK_NULL_HANDLE;
    VkPipeline bin_pipe = VK_NULL_HANDLE;

    VkDescriptorSetLayout fine_dsl = VK_NULL_HANDLE;
    VkPipelineLayout fine_pl = VK_NULL_HANDLE;
    VkPipeline fine_pipe = VK_NULL_HANDLE;

    // App-appended internal descriptor set layouts (bound at set = app_set_count)
    // VS internal set: binding 0=ClipVertBuf, 1=IndexBuf, 2=VtxBuf0, 3=VtxBuf1, 4=VtxBuf2, 5=VtxBuf3, 6=VsParamsUBO
    VkDescriptorSetLayout vs_int_dsl = VK_NULL_HANDLE;
    // FS internal set: binding 0=ClipVertBuf, 1=TriSetupBuf, 2=VisBuf, 3=ColorBuf, 4=DepthBuf, 5=FsParamsUBO
    VkDescriptorSetLayout fs_int_dsl = VK_NULL_HANDLE;

    // Persistent H20 HBM3 rasterization buffers
    VkBuffer clip_vert_buf = VK_NULL_HANDLE;   VkDeviceMemory clip_vert_mem = VK_NULL_HANDLE;
    VkBuffer tri_setup_buf = VK_NULL_HANDLE;   VkDeviceMemory tri_setup_mem = VK_NULL_HANDLE;
    VkBuffer tile_count_buf = VK_NULL_HANDLE;  VkDeviceMemory tile_count_mem = VK_NULL_HANDLE;
    VkBuffer tile_bins_buf = VK_NULL_HANDLE;   VkDeviceMemory tile_bins_mem = VK_NULL_HANDLE;
    VkBuffer vis_buf = VK_NULL_HANDLE;         VkDeviceMemory vis_mem = VK_NULL_HANDLE;
    VkBuffer dummy_buf = VK_NULL_HANDLE;       VkDeviceMemory dummy_mem = VK_NULL_HANDLE;

    // Ring buffer of host-visible Draw UBOs + per-slot Descriptor Sets (zero cross-pass hazards!)
    static constexpr uint32_t MAX_DRAWS_IN_FLIGHT = 1024;
    VkBuffer vs_ubo_buf = VK_NULL_HANDLE;      VkDeviceMemory vs_ubo_mem = VK_NULL_HANDLE; void* vs_ubo_ptr = nullptr;
    VkBuffer fs_ubo_buf = VK_NULL_HANDLE;      VkDeviceMemory fs_ubo_mem = VK_NULL_HANDLE; void* fs_ubo_ptr = nullptr;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> clear_ds;  // ring [MAX_DRAWS_IN_FLIGHT]
    std::vector<VkDescriptorSet> bin_ds;    // ring [MAX_DRAWS_IN_FLIGHT]
    std::vector<VkDescriptorSet> fine_ds;   // ring [MAX_DRAWS_IN_FLIGHT]
    std::vector<VkDescriptorSet> vs_int_ds; // ring [MAX_DRAWS_IN_FLIGHT]
    std::vector<VkDescriptorSet> fs_int_ds; // ring [MAX_DRAWS_IN_FLIGHT]
    uint32_t draw_cursor = 0;

    uint32_t cur_w = 0, cur_h = 0;
    uint32_t tiles_x = 0, tiles_y = 0;
};

struct alignas(256) SmVsParamsUBO {
    uint32_t vertex_count;     // total = verts_per_instance * instance_count
    uint32_t first_vertex;
    uint32_t instance_idx;     // first_instance
    uint32_t indexed;          // 0=non-indexed, 1=uint16, 2=uint32
    uint32_t first_index;
    int32_t  vertex_offset;
    uint32_t vert_stride_vec4;
    uint32_t idx_buf_offset;
    uint32_t vtx_offsets[4];
    uint32_t vtx_strides[4];
    uint32_t verts_per_instance;
    uint32_t vtx_inst_rate_mask; // bit b = 1 if binding b is VK_VERTEX_INPUT_RATE_INSTANCE
    uint32_t pad0;
    uint32_t pad1;
};

struct alignas(256) SmFsParamsUBO {
    uint32_t width;
    uint32_t height;
    uint32_t vert_stride_vec4;
    uint32_t format_flags;   // bit0=bgra, bit1=srgb
    uint32_t depth_write;
    uint32_t blend_enable;
    uint32_t src_blend;
    uint32_t dst_blend;
    uint32_t blend_op;
    uint32_t pad[3];
};

struct SmQueue {
    VK_LOADER_DATA loader_data;
    SmDevice* dev = nullptr;
    VkQueue real_q = VK_NULL_HANDLE;
};

struct SmDevice {
    VK_LOADER_DATA loader_data;
    SmPhysicalDevice* pd = nullptr;
    BackendDevFns fn;
    VkDevice real_dev = VK_NULL_HANDLE;
    SmQueue* main_queue = nullptr;
    SmRasterEngine rast;
    std::mutex mtx;
};

struct SmCommandBuffer {
    VK_LOADER_DATA loader_data;
    SmDevice* dev = nullptr;
    VkCommandBuffer real_cb = VK_NULL_HANDLE;

    // Active recorded graphics state
    bool in_render_pass = false;
    bool in_dyn_rendering = false;
    SmRenderPass* cur_rp = nullptr;
    SmFramebuffer* cur_fb = nullptr;
    SmImage* dyn_col_img = nullptr;
    SmImage* dyn_dep_img = nullptr;
    uint32_t dyn_w = 0, dyn_h = 0;
    SmPipeline* cur_gfx_pipe = nullptr;
    SmPipeline* cur_comp_pipe = nullptr;

    VkViewport cur_vp{0, 0, 1920, 1080, 0.0f, 1.0f};
    VkRect2D cur_scissor{{0, 0}, {1920, 1080}};
    bool has_vp = false;
    bool has_scissor = false;

    // Extended Dynamic State overrides (VK_EXT_extended_dynamic_state / Vulkan 1.3)
    bool has_dyn_cull = false;        VkCullModeFlags dyn_cull = VK_CULL_MODE_NONE;
    bool has_dyn_front_face = false;  VkFrontFace dyn_front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool has_dyn_topo = false;        VkPrimitiveTopology dyn_topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool has_dyn_depth_test = false;  bool dyn_depth_test = false;
    bool has_dyn_depth_write = false; bool dyn_depth_write = false;
    bool has_dyn_depth_op = false;    VkCompareOp dyn_depth_op = VK_COMPARE_OP_LESS;

    // Bound app descriptor sets for GRAPHICS point (+ dynamic buffer offsets!)
    VkDescriptorSet gfx_sets[16] = {VK_NULL_HANDLE};
    uint32_t gfx_set_mask = 0;
    uint32_t gfx_dyn_counts[16] = {0};
    uint32_t gfx_dyn_offs[16][8] = {{0}};
    // Bound push constants
    uint8_t push_data[256] = {0};
    uint32_t push_size = 0;

    // Bound vertex & index buffers
    SmBuffer* vtx_bufs[SM_MAX_VTX_BINDINGS] = {nullptr};
    VkDeviceSize vtx_offsets[SM_MAX_VTX_BINDINGS] = {0};
    uint32_t dyn_vtx_strides[SM_MAX_VTX_BINDINGS] = {0};
    bool has_dyn_vtx_stride[SM_MAX_VTX_BINDINGS] = {false};
    SmBuffer* idx_buf = nullptr;
    VkDeviceSize idx_offset = 0;
    VkIndexType idx_type = VK_INDEX_TYPE_UINT32;
};

// -----------------------------------------------------------------------------
// Helper functions implemented across translation units
// -----------------------------------------------------------------------------
uint32_t sm_find_mem_type(SmPhysicalDevice* pd, uint32_t bits, VkMemoryPropertyFlags req);
bool sm_ensure_image_attach_buf(SmDevice* dev, SmImage* img);
bool sm_init_raster_engine(SmDevice* dev);
void sm_destroy_raster_engine(SmDevice* dev);
bool sm_ensure_vis_size(SmDevice* dev, uint32_t w, uint32_t h);

// spirv_lower.cpp
bool sm_lower_vertex_spirv(const std::vector<uint32_t>& in_spv,
                           const SmVertexInputState& vtx_in,
                           uint32_t sm_set_idx,
                           std::vector<uint32_t>& out_comp_spv,
                           std::string& err);

bool sm_lower_fragment_spirv(const std::vector<uint32_t>& in_spv,
                             uint32_t sm_set_idx,
                             std::vector<uint32_t>& out_comp_spv,
                             std::string& err);

// smrast_wsi.cpp
void sm_wsi_blit_to_surface(SmSwapchain* sc, SmImage* img);
