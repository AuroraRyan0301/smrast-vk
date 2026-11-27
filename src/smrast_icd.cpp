// smrast_icd.cpp — SmRast-VK: Complete Vulkan 1.3 Graphics-on-Compute Wrapper ICD for H20 / H100 / MI300X
// Drop-in ICD (VK_ICD_FILENAMES=smrast_icd.json) that executes 100% of graphics rendering via
// GPU Compute Shaders on the backend's pure VK_QUEUE_COMPUTE_BIT queue (qf[2] on NVIDIA H20),
// while passing Compute Shaders, HBM3 Memory, Buffers, Hardware Textures/Samplers, and DMA straight through!

#include "smrast_internal.h"
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>

static std::atomic<uint32_t> g_sm_trace_cnt{0};
static std::atomic<int> g_sm_xray_frames{0};
static bool g_sm_xray_env = false;
static bool sm_trace_on() {
    static int en = -1;
    if (en < 0) en = getenv("SMRAST_TRACE") ? 1 : 0;
    return en != 0 && g_sm_trace_cnt.load(std::memory_order_relaxed) < 600;
}
#define SM_TRACE(fmt, ...) do { if (sm_trace_on()) { uint32_t _i = g_sm_trace_cnt.fetch_add(1); fprintf(stderr, "[SmTrace#%03u] " fmt "\n", _i, ##__VA_ARGS__); fflush(stderr); } } while (0)
#define SM_XRAY(fmt, ...) do { if (g_sm_xray_frames.load(std::memory_order_relaxed) > 0) { fprintf(stderr, "[SmXRay] " fmt "\n", ##__VA_ARGS__); } } while (0)

static void sm_crash_handler(int sig, siginfo_t* si, void* uctx) {
    ucontext_t* uc = reinterpret_cast<ucontext_t*>(uctx);
    uintptr_t rip = uc ? (uintptr_t)uc->uc_mcontext.gregs[REG_RIP] : 0;
    uintptr_t rsp = uc ? (uintptr_t)uc->uc_mcontext.gregs[REG_RSP] : 0;
    uintptr_t rbp = uc ? (uintptr_t)uc->uc_mcontext.gregs[REG_RBP] : 0;
    uintptr_t addr = si ? (uintptr_t)si->si_addr : 0;
    fprintf(stderr, "\n[SmRast] FATAL SIGNAL %d si_addr=%p RIP=%p RSP=%p RBP=%p\n",
            sig, (void*)addr, (void*)rip, (void*)rsp, (void*)rbp);
    Dl_info dli{};
    if (rip && dladdr((void*)rip, &dli)) {
        fprintf(stderr, "  RIP -> %s (%s + 0x%lx) [file_off=0x%lx]\n",
                dli.dli_fname ? dli.dli_fname : "?",
                dli.dli_sname ? dli.dli_sname : "?",
                (unsigned long)(rip - (uintptr_t)dli.dli_saddr),
                (unsigned long)(rip - (uintptr_t)dli.dli_fbase));
    }
    if (rsp) {
        const uintptr_t* sp = reinterpret_cast<const uintptr_t*>(rsp);
        for (int i = 0; i < 24; ++i) {
            Dl_info di{};
            if (sp[i] > 0x10000 && dladdr((void*)sp[i], &di) && di.dli_fname) {
                fprintf(stderr, "  RSP[%02d] = %p -> %s (%s + 0x%lx) [file_off=0x%lx]\n",
                        i, (void*)sp[i], di.dli_fname,
                        di.dli_sname ? di.dli_sname : "?",
                        di.dli_saddr ? (unsigned long)(sp[i] - (uintptr_t)di.dli_saddr) : 0ul,
                        (unsigned long)(sp[i] - (uintptr_t)di.dli_fbase));
            }
        }
    }
    fflush(stderr);
    signal(sig, SIG_DFL);
    raise(sig);
}

#define SM_EXPORT extern "C" __attribute__((visibility("default")))

// Helper to pack/unpack non-dispatchable pointers safely across 64-bit Vulkan handles
template<typename T, typename H> static inline T* from_h(H h) { return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(h)); }
template<typename H, typename T> static inline H to_h(T* p)   { return reinterpret_cast<H>(p); }

static uint32_t sanitize_stage_mask(VkPipelineStageFlags m) {
    const uint32_t gfx = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                         VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    uint32_t out = m & ~gfx;
    if (m & gfx) out |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (out == 0) out = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    return out;
}

static VkPipelineStageFlags2 sanitize_stage_mask2(VkPipelineStageFlags2 m) {
    const VkPipelineStageFlags2 gfx =
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
        VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    VkPipelineStageFlags2 out = m & ~gfx;
    if (m & gfx) out |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (out == 0) out = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return out;
}

static VkAccessFlags sanitize_access_mask(VkAccessFlags m) {
    const VkAccessFlags gfx = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                              VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
                              VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkAccessFlags out = m & ~gfx;
    if (m & gfx) out |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    return out;
}

static VkImageLayout sanitize_layout(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return VK_IMAGE_LAYOUT_GENERAL;
        default:
            return l;
    }
}

static uint32_t pack_clear_color(const VkClearColorValue& c, VkFormat fmt) {
    float r = std::clamp(c.float32[0], 0.0f, 1.0f);
    float g = std::clamp(c.float32[1], 0.0f, 1.0f);
    float b = std::clamp(c.float32[2], 0.0f, 1.0f);
    float a = std::clamp(c.float32[3], 0.0f, 1.0f);
    bool bgra = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
    uint32_t ur = (uint32_t)(r * 255.0f + 0.5f);
    uint32_t ug = (uint32_t)(g * 255.0f + 0.5f);
    uint32_t ub = (uint32_t)(b * 255.0f + 0.5f);
    uint32_t ua = (uint32_t)(a * 255.0f + 0.5f);
    return bgra ? (ub | (ug << 8) | (ur << 16) | (ua << 24))
                : (ur | (ug << 8) | (ub << 16) | (ua << 24));
}

// =============================================================================
// Instance & PhysicalDevice API
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumerateInstanceExtensionProperties(
        const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps) {
    if (pLayerName) return VK_ERROR_LAYER_NOT_PRESENT;
    static const VkExtensionProperties exts[] = {
        { VK_KHR_SURFACE_EXTENSION_NAME, 25 },
        { VK_KHR_XCB_SURFACE_EXTENSION_NAME, 6 },
        { VK_KHR_XLIB_SURFACE_EXTENSION_NAME, 6 },
        { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 2 },
        { VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, 1 },
        { VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, 1 },
        { VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME, 1 },
        { VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME, 1 },
        { VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME, 1 },
        { VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, 4 },
        { VK_EXT_DEBUG_UTILS_EXTENSION_NAME, 2 },
    };
    const uint32_t n = sizeof(exts) / sizeof(exts[0]);
    if (!pProps) { *pCount = n; return VK_SUCCESS; }
    uint32_t copy = std::min(*pCount, n);
    memcpy(pProps, exts, copy * sizeof(VkExtensionProperties));
    *pCount = copy;
    return (copy < n) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumerateInstanceVersion(uint32_t* pApiVersion) {
    *pApiVersion = VK_API_VERSION_1_3;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateInstance(
        const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks*, VkInstance* pInstance) {
    if (getenv("SMRAST_XRAY")) {
        g_sm_xray_env = true;
    }
    if (getenv("SMRAST_CRASH_BT")) {
        struct sigaction sa{};
        sa.sa_sigaction = sm_crash_handler;
        sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGABRT, &sa, nullptr);
        sigaction(SIGFPE,  &sa, nullptr);
    }
    const char* backend_path = getenv("SMRAST_BACKEND_ICD");
    void* dl = nullptr;
    if (backend_path && backend_path[0]) {
        dl = dlopen(backend_path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    } else {
        static const char* kCandidates[] = {
            "libGLX_nvidia.so.0",
            "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
            "/usr/lib64/libGLX_nvidia.so.0",
            "libvulkan_radeon.so",
            "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so",
        };
        for (const char* cand : kCandidates) {
            dl = dlopen(cand, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
            if (dl) { backend_path = cand; break; }
        }
    }
    if (!dl) {
        fprintf(stderr, "[SmRast] ERROR: failed to load backend Vulkan ICD (set SMRAST_BACKEND_ICD=/path/to/backend_icd.so): %s\n", dlerror());
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto gipa = (PFN_vkGetInstanceProcAddr)dlsym(dl, "vk_icdGetInstanceProcAddr");
    if (!gipa) gipa = (PFN_vkGetInstanceProcAddr)dlsym(dl, "vkGetInstanceProcAddr");
    if (!gipa) return VK_ERROR_INITIALIZATION_FAILED;

    auto create_inst = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_inst) return VK_ERROR_INITIALIZATION_FAILED;

    // Create headless compute-only backend instance (strip WSI surface extensions so backend never touches X11!)
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastBackend", 1, "SmRast", 1, VK_API_VERSION_1_3};
    if (pCreateInfo && pCreateInfo->pApplicationInfo) {
        ai = *pCreateInfo->pApplicationInfo;
        if (ai.apiVersion < VK_API_VERSION_1_3) ai.apiVersion = VK_API_VERSION_1_3;
    }
    VkInstanceCreateInfo bci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai, 0, nullptr, 0, nullptr};
    VkInstance real_inst = VK_NULL_HANDLE;
    VkResult r = create_inst(&bci, nullptr, &real_inst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[SmRast] ERROR: backend vkCreateInstance failed: %d\n", r);
        return r;
    }

    auto* inst = new SmInstance();
    set_loader_magic_value(inst);
    inst->backend_dl = dl;
    inst->real_inst = real_inst;
    inst->fn.gipa = gipa;
#define LOAD_I(name) inst->fn.name = (PFN_vk##name)gipa(real_inst, "vk" #name)
    LOAD_I(DestroyInstance);
    LOAD_I(EnumeratePhysicalDevices);
    LOAD_I(GetPhysicalDeviceProperties);
    LOAD_I(GetPhysicalDeviceProperties2);
    LOAD_I(GetPhysicalDeviceFeatures);
    LOAD_I(GetPhysicalDeviceFeatures2);
    LOAD_I(GetPhysicalDeviceQueueFamilyProperties);
    LOAD_I(GetPhysicalDeviceQueueFamilyProperties2);
    LOAD_I(GetPhysicalDeviceMemoryProperties);
    LOAD_I(GetPhysicalDeviceMemoryProperties2);
    LOAD_I(GetPhysicalDeviceFormatProperties);
    LOAD_I(GetPhysicalDeviceFormatProperties2);
    LOAD_I(GetPhysicalDeviceImageFormatProperties);
    LOAD_I(EnumerateDeviceExtensionProperties);
    LOAD_I(CreateDevice);
    LOAD_I(GetDeviceProcAddr);
#undef LOAD_I

    uint32_t nd = 0;
    inst->fn.EnumeratePhysicalDevices(real_inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> rpds(nd);
    inst->fn.EnumeratePhysicalDevices(real_inst, &nd, rpds.data());
    for (VkPhysicalDevice rpd : rpds) {
        auto* spd = new SmPhysicalDevice();
        set_loader_magic_value(spd);
        spd->inst = inst;
        spd->real_pd = rpd;
        inst->fn.GetPhysicalDeviceProperties(rpd, &spd->props);
        inst->fn.GetPhysicalDeviceMemoryProperties(rpd, &spd->mem_props);
        // Disable DEVICE_LOCAL|HOST_VISIBLE (BAR1/HVV) CPU mapping so Wine/DXVK never page-faults on H20 BAR1!
        for (uint32_t mt = 0; mt < spd->mem_props.memoryTypeCount; ++mt) {
            VkMemoryPropertyFlags& pf = spd->mem_props.memoryTypes[mt].propertyFlags;
            if ((pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (pf & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                pf = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            }
        }

        // Find pure VK_QUEUE_COMPUTE_BIT queue family (avoids broken GH100 hardware graphics queue!)
        uint32_t nqf = 0;
        inst->fn.GetPhysicalDeviceQueueFamilyProperties(rpd, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf);
        inst->fn.GetPhysicalDeviceQueueFamilyProperties(rpd, &nqf, qfs.data());
        spd->backend_compute_qf = 0;
        for (uint32_t i = 0; i < nqf; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                spd->backend_compute_qf = i;
                break;
            }
        }
        char new_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
        snprintf(new_name, sizeof(new_name), "%.200s (SmRast-VK Compute Rasterizer)", spd->props.deviceName);
        strncpy(spd->props.deviceName, new_name, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
        spd->props.deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] = 0;
        fprintf(stderr, "[SmRast] Initialized on %s (backend pure compute qf=%u)\n",
                spd->props.deviceName, spd->backend_compute_qf);
        inst->phys_devs.push_back(spd);
    }

    *pInstance = reinterpret_cast<VkInstance>(inst);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL sm_DestroyInstance(VkInstance instance, const VkAllocationCallbacks*) {
    auto* inst = reinterpret_cast<SmInstance*>(instance);
    if (!inst) return;
    for (auto* pd : inst->phys_devs) delete pd;
    if (inst->fn.DestroyInstance && inst->real_inst) inst->fn.DestroyInstance(inst->real_inst, nullptr);
    delete inst;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumeratePhysicalDevices(
        VkInstance instance, uint32_t* pCount, VkPhysicalDevice* pDevices) {
    auto* inst = reinterpret_cast<SmInstance*>(instance);
    uint32_t n = (uint32_t)inst->phys_devs.size();
    if (!pDevices) { *pCount = n; return VK_SUCCESS; }
    uint32_t copy = std::min(*pCount, n);
    for (uint32_t i = 0; i < copy; ++i) pDevices[i] = reinterpret_cast<VkPhysicalDevice>(inst->phys_devs[i]);
    *pCount = copy;
    return (copy < n) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    *p = spd->props;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    spd->inst->fn.GetPhysicalDeviceProperties2(spd->real_pd, p);
    p->properties = spd->props;
    for (auto* next = reinterpret_cast<VkBaseOutStructure*>(p->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR) {
            auto* m5 = reinterpret_cast<VkPhysicalDeviceMaintenance5PropertiesKHR*>(next);
            m5->earlyFragmentMultisampleCoverageAfterSampleCounting = VK_FALSE;
            m5->earlyFragmentSampleMaskTestBeforeSampleCounting = VK_FALSE;
            m5->depthStencilSwizzleOneSupport = VK_TRUE;
            m5->polygonModePointSize = VK_TRUE;
            m5->nonStrictSinglePixelWideLinesUseParallelogram = VK_FALSE;
            m5->nonStrictWideLinesUseParallelogram = VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES_KHR) {
            auto* m6 = reinterpret_cast<VkPhysicalDeviceMaintenance6PropertiesKHR*>(next);
            m6->blockTexelViewCompatibleMultipleLayers = VK_TRUE;
            m6->maxCombinedImageSamplerDescriptorCount = 1;
            m6->fragmentShadingRateClampCombinerInputs = VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR) {
            auto* pdp = reinterpret_cast<VkPhysicalDevicePushDescriptorPropertiesKHR*>(next);
            if (pdp->maxPushDescriptors == 0) pdp->maxPushDescriptors = 32;
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceFeatures(VkPhysicalDevice pd, VkPhysicalDeviceFeatures* f) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    spd->inst->fn.GetPhysicalDeviceFeatures(spd->real_pd, f);
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    spd->inst->fn.GetPhysicalDeviceFeatures2(spd->real_pd, f);
    for (auto* next = reinterpret_cast<VkBaseOutStructure*>(f->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR) {
            reinterpret_cast<VkPhysicalDeviceMaintenance5FeaturesKHR*>(next)->maintenance5 = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR) {
            reinterpret_cast<VkPhysicalDeviceMaintenance6FeaturesKHR*>(next)->maintenance6 = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT) {
            reinterpret_cast<VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(next)->extendedDynamicState = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT) {
            auto* eds2 = reinterpret_cast<VkPhysicalDeviceExtendedDynamicState2FeaturesEXT*>(next);
            eds2->extendedDynamicState2 = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT) {
            reinterpret_cast<VkPhysicalDeviceIndexTypeUint8FeaturesEXT*>(next)->indexTypeUint8 = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT) {
            reinterpret_cast<VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT*>(next)->dynamicRenderingUnusedAttachments = VK_TRUE;
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceQueueFamilyProperties(
        VkPhysicalDevice pd, uint32_t* pCount, VkQueueFamilyProperties* pProps) {
    (void)pd;
    // Expose 1 unified virtual queue family advertising GRAPHICS | COMPUTE | TRANSFER (backed by H20 qf[2]!)
    if (!pProps) { *pCount = 1; return; }
    if (*pCount >= 1) {
        pProps[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        pProps[0].queueCount = 1;
        pProps[0].timestampValidBits = 64;
        pProps[0].minImageTransferGranularity = {1, 1, 1};
        *pCount = 1;
    }
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceQueueFamilyProperties2(
        VkPhysicalDevice pd, uint32_t* pCount, VkQueueFamilyProperties2* pProps) {
    if (!pProps) { *pCount = 1; return; }
    if (*pCount >= 1) {
        uint32_t one = 1;
        sm_GetPhysicalDeviceQueueFamilyProperties(pd, &one, &pProps[0].queueFamilyProperties);
        *pCount = 1;
    }
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceMemoryProperties(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties* m) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    *m = spd->mem_props;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* m) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    spd->inst->fn.GetPhysicalDeviceMemoryProperties2(spd->real_pd, m);
    m->memoryProperties = spd->mem_props;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceFormatProperties(VkPhysicalDevice pd, VkFormat fmt, VkFormatProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    spd->inst->fn.GetPhysicalDeviceFormatProperties(spd->real_pd, fmt, p);
    // Ensure color/depth attachment & sampled/transfer bits are advertised cleanly
    p->optimalTilingFeatures |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D16_UNORM ||
        fmt == VK_FORMAT_D32_SFLOAT_S8_UINT || fmt == VK_FORMAT_D16_UNORM_S8_UINT ||
        fmt == VK_FORMAT_X8_D24_UNORM_PACK32 || fmt == VK_FORMAT_S8_UINT)
        p->optimalTilingFeatures |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd, VkFormat fmt, VkFormatProperties2* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    if (spd->inst->fn.GetPhysicalDeviceFormatProperties2) {
        spd->inst->fn.GetPhysicalDeviceFormatProperties2(spd->real_pd, fmt, p);
    } else {
        spd->inst->fn.GetPhysicalDeviceFormatProperties(spd->real_pd, fmt, &p->formatProperties);
    }
    VkFormatFeatureFlags extra = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                 VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                 VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D16_UNORM ||
        fmt == VK_FORMAT_D32_SFLOAT_S8_UINT || fmt == VK_FORMAT_D16_UNORM_S8_UINT ||
        fmt == VK_FORMAT_X8_D24_UNORM_PACK32 || fmt == VK_FORMAT_S8_UINT)
        extra |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    p->formatProperties.optimalTilingFeatures |= extra;
    for (auto* next = reinterpret_cast<VkBaseOutStructure*>(p->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3) {
            auto* fp3 = reinterpret_cast<VkFormatProperties3*>(next);
            fp3->optimalTilingFeatures |= (VkFormatFeatureFlags2)extra | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
        }
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceImageFormatProperties(
        VkPhysicalDevice pd, VkFormat fmt, VkImageType type, VkImageTiling tiling,
        VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    VkImageUsageFlags bu = (usage & ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return spd->inst->fn.GetPhysicalDeviceImageFormatProperties(spd->real_pd, fmt, type, tiling, bu, flags, p);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceImageFormatProperties2(
        VkPhysicalDevice pd, const VkPhysicalDeviceImageFormatInfo2* info, VkImageFormatProperties2* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    VkPhysicalDeviceImageFormatInfo2 bi = *info;
    bi.usage = (bi.usage & ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
             | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    auto fn = (PFN_vkGetPhysicalDeviceImageFormatProperties2)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceImageFormatProperties2");
    if (fn) return fn(spd->real_pd, &bi, p);
    return sm_GetPhysicalDeviceImageFormatProperties(pd, bi.format, bi.type, bi.tiling, bi.usage, bi.flags, &p->imageFormatProperties);
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceSparseImageFormatProperties(
        VkPhysicalDevice pd, VkFormat fmt, VkImageType type, VkSampleCountFlagBits samples,
        VkImageUsageFlags usage, VkImageTiling tiling, uint32_t* count, VkSparseImageFormatProperties* props) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    auto fn = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceSparseImageFormatProperties");
    if (fn) { fn(spd->real_pd, fmt, type, samples, usage, tiling, count, props); return; }
    *count = 0;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceSparseImageFormatProperties2(
        VkPhysicalDevice pd, const VkPhysicalDeviceSparseImageFormatInfo2* info, uint32_t* count, VkSparseImageFormatProperties2* props) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    auto fn = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceSparseImageFormatProperties2");
    if (fn) { fn(spd->real_pd, info, count, props); return; }
    *count = 0;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumeratePhysicalDeviceGroups(
        VkInstance instance, uint32_t* pCount, VkPhysicalDeviceGroupProperties* pProps) {
    auto* inst = reinterpret_cast<SmInstance*>(instance);
    uint32_t n = (uint32_t)inst->phys_devs.size();
    if (!pProps) { *pCount = n; return VK_SUCCESS; }
    uint32_t c = std::min(*pCount, n);
    for (uint32_t i = 0; i < c; ++i) {
        pProps[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
        pProps[i].physicalDeviceCount = 1;
        pProps[i].physicalDevices[0] = reinterpret_cast<VkPhysicalDevice>(inst->phys_devs[i]);
        pProps[i].subsetAllocation = VK_FALSE;
    }
    *pCount = c;
    return (c < n) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceExternalBufferProperties(
        VkPhysicalDevice pd, const VkPhysicalDeviceExternalBufferInfo* info, VkExternalBufferProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    auto fn = (PFN_vkGetPhysicalDeviceExternalBufferProperties)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceExternalBufferProperties");
    if (fn) fn(spd->real_pd, info, p);
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceExternalFenceProperties(
        VkPhysicalDevice pd, const VkPhysicalDeviceExternalFenceInfo* info, VkExternalFenceProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    auto fn = (PFN_vkGetPhysicalDeviceExternalFenceProperties)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceExternalFenceProperties");
    if (fn) fn(spd->real_pd, info, p);
}

static VKAPI_ATTR void VKAPI_CALL sm_GetPhysicalDeviceExternalSemaphoreProperties(
        VkPhysicalDevice pd, const VkPhysicalDeviceExternalSemaphoreInfo* info, VkExternalSemaphoreProperties* p) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    auto fn = (PFN_vkGetPhysicalDeviceExternalSemaphoreProperties)spd->inst->fn.gipa(spd->inst->real_inst, "vkGetPhysicalDeviceExternalSemaphoreProperties");
    if (fn) fn(spd->real_pd, info, p);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceToolProperties(
        VkPhysicalDevice, uint32_t* pCount, VkPhysicalDeviceToolProperties*) {
    *pCount = 0;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumerateInstanceLayerProperties(uint32_t* pCount, VkLayerProperties*) {
    *pCount = 0;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t* pCount, VkLayerProperties*) {
    *pCount = 0;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_EnumerateDeviceExtensionProperties(
        VkPhysicalDevice pd, const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps) {
    if (pLayerName) return VK_ERROR_LAYER_NOT_PRESENT;
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    uint32_t bn = 0;
    spd->inst->fn.EnumerateDeviceExtensionProperties(spd->real_pd, nullptr, &bn, nullptr);
    std::vector<VkExtensionProperties> raw_exts(bn);
    spd->inst->fn.EnumerateDeviceExtensionProperties(spd->real_pd, nullptr, &bn, raw_exts.data());
    // Strict allowlist of device extensions fully supported or emulated by SmRast-VK (prevents null fn ptr calls!)
    auto is_allowed = [](const char* n) {
        static const char* kAllow[] = {
            "VK_KHR_swapchain",
            "VK_KHR_swapchain_mutable_format",
            "VK_KHR_load_store_op_none",
            "VK_KHR_maintenance1", "VK_KHR_maintenance2", "VK_KHR_maintenance3", "VK_KHR_maintenance4",
            "VK_KHR_maintenance5", "VK_KHR_maintenance6",
            "VK_KHR_push_descriptor",
            "VK_KHR_shader_atomic_int64",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_16bit_storage",
            "VK_KHR_8bit_storage",
            "VK_KHR_shader_subgroup_extended_types",
            "VK_KHR_shader_subgroup_uniform_control_flow",
            "VK_KHR_shader_integer_dot_product",
            "VK_KHR_shader_non_semantic_info",
            "VK_KHR_shader_terminate_invocation",
            "VK_KHR_shader_clock",
            "VK_KHR_workgroup_memory_explicit_layout",
            "VK_KHR_format_feature_flags2",
            "VK_KHR_dynamic_rendering",
            "VK_KHR_synchronization2",
            "VK_KHR_create_renderpass2",
            "VK_KHR_timeline_semaphore",
            "VK_KHR_buffer_device_address",
            "VK_KHR_descriptor_update_template",
            "VK_KHR_copy_commands2",
            "VK_KHR_draw_indirect_count",
            "VK_KHR_get_memory_requirements2",
            "VK_KHR_bind_memory2",
            "VK_KHR_dedicated_allocation",
            "VK_KHR_sampler_mirror_clamp_to_edge",
            "VK_KHR_image_format_list",
            "VK_KHR_driver_properties",
            "VK_KHR_shader_float_controls",
            "VK_KHR_spirv_1_4",
            "VK_KHR_shader_draw_parameters",
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_uniform_buffer_standard_layout",
            "VK_KHR_vulkan_memory_model",
            "VK_KHR_separate_depth_stencil_layouts",
            "VK_KHR_imageless_framebuffer",
            "VK_KHR_depth_stencil_resolve",
            "VK_KHR_zero_initialize_workgroup_memory",
            "VK_EXT_descriptor_indexing",
            "VK_EXT_index_type_uint8",
            "VK_EXT_mutable_descriptor_type",
            "VK_EXT_shader_subgroup_ballot",
            "VK_EXT_shader_subgroup_vote",
            "VK_EXT_conditional_rendering",
            "VK_EXT_border_color_swizzle",
            "VK_EXT_image_2d_view_of_3d",
            "VK_EXT_image_view_min_lod",
            "VK_EXT_texel_buffer_alignment",
            "VK_EXT_robustness2",
            "VK_EXT_extended_dynamic_state",
            "VK_EXT_extended_dynamic_state2",
            "VK_EXT_vertex_attribute_divisor",
            "VK_EXT_custom_border_color",
            "VK_EXT_depth_clip_enable",
            "VK_EXT_transform_feedback",
            "VK_EXT_memory_budget",
            "VK_EXT_scalar_block_layout",
            "VK_EXT_sampler_filter_minmax",
            "VK_EXT_shader_viewport_index_layer",
            "VK_EXT_host_query_reset",
            "VK_EXT_subgroup_size_control",
            "VK_EXT_inline_uniform_block",
            "VK_EXT_pipeline_creation_cache_control",
            "VK_EXT_shader_demote_to_helper_invocation",
            "VK_EXT_dynamic_rendering_unused_attachments",
            "VK_EXT_4444_formats",
        };
        for (const char* a : kAllow) if (strcmp(n, a) == 0) return true;
        return false;
    };
    std::vector<VkExtensionProperties> exts;
    exts.reserve(64);
    bool has_sc = false, has_scm = false, has_lson = false, has_m5 = false, has_m6 = false, has_drua = false;
    for (const auto& e : raw_exts) {
        if (!is_allowed(e.extensionName)) continue;
        if (strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) has_sc = true;
        if (strcmp(e.extensionName, "VK_KHR_swapchain_mutable_format") == 0) has_scm = true;
        if (strcmp(e.extensionName, "VK_KHR_load_store_op_none") == 0) has_lson = true;
        if (strcmp(e.extensionName, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) has_m5 = true;
        if (strcmp(e.extensionName, VK_KHR_MAINTENANCE_6_EXTENSION_NAME) == 0) has_m6 = true;
        if (strcmp(e.extensionName, "VK_EXT_dynamic_rendering_unused_attachments") == 0) has_drua = true;
        exts.push_back(e);
    }
    auto add_ext = [&](const char* name, uint32_t ver) {
        VkExtensionProperties ep{};
        strncpy(ep.extensionName, name, VK_MAX_EXTENSION_NAME_SIZE - 1);
        ep.specVersion = ver;
        exts.push_back(ep);
    };
    if (!has_sc)   add_ext(VK_KHR_SWAPCHAIN_EXTENSION_NAME, 70);
    if (!has_scm)  add_ext("VK_KHR_swapchain_mutable_format", 1);
    if (!has_lson) add_ext("VK_KHR_load_store_op_none", 1);
    if (!has_m5)   add_ext(VK_KHR_MAINTENANCE_5_EXTENSION_NAME, 1);
    if (!has_m6)   add_ext(VK_KHR_MAINTENANCE_6_EXTENSION_NAME, 1);
    if (!has_drua) add_ext("VK_EXT_dynamic_rendering_unused_attachments", 1);
    if (!pProps) { *pCount = (uint32_t)exts.size(); return VK_SUCCESS; }
    uint32_t copy = std::min(*pCount, (uint32_t)exts.size());
    memcpy(pProps, exts.data(), copy * sizeof(VkExtensionProperties));
    *pCount = copy;
    return (copy < exts.size()) ? VK_INCOMPLETE : VK_SUCCESS;
}

// =============================================================================
// Device Creation (Maps ALL work to H20 Pure Compute Queue qf[2] + Int64 Atomics!)
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateDevice(
        VkPhysicalDevice pd, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks*, VkDevice* pDevice) {
    auto* spd = reinterpret_cast<SmPhysicalDevice*>(pd);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, spd->backend_compute_qf, 1, &prio};

    // Enable shaderInt64 + shaderBufferInt64Atomics on backend device for CuRast 64-bit Visibility Buffer!
    // Walk app's pNext chain (DXVK/vkd3d pass VkPhysicalDeviceFeatures2 + Vulkan11/12/13 + EXT features) safely!
    VkPhysicalDeviceRobustness2FeaturesEXT r2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    r2.robustBufferAccess2 = VK_TRUE;
    r2.robustImageAccess2 = VK_TRUE;
    r2.nullDescriptor = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &r2};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
    f12.shaderBufferInt64Atomics = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT vad{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    VkPhysicalDeviceDepthClipEnableFeaturesEXT dce{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT tfb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT};
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mdt{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
    VkPhysicalDeviceBorderColorSwizzleFeaturesEXT bcs{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT};
    VkPhysicalDeviceImage2DViewOf3DFeaturesEXT i2d3d{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT};
    VkPhysicalDeviceImageViewMinLodFeaturesEXT ivml{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT crf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT iu8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    void* tail_next = &r2;
    auto chain_tail = [&](void* s) {
        reinterpret_cast<VkBaseOutStructure*>(s)->pNext = nullptr;
        reinterpret_cast<VkBaseOutStructure*>(tail_next)->pNext = reinterpret_cast<VkBaseOutStructure*>(s);
        tail_next = s;
    };
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    if (pCreateInfo && pCreateInfo->pEnabledFeatures) f2.features = *pCreateInfo->pEnabledFeatures;
    if (pCreateInfo && pCreateInfo->pNext) {
        for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(pCreateInfo->pNext); n; n = n->pNext) {
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                f2.features = reinterpret_cast<const VkPhysicalDeviceFeatures2*>(n)->features;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES) {
                f11 = *reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(n);
                f11.pNext = &r2;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
                f12 = *reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(n);
                f12.pNext = &f11;
                f12.shaderBufferInt64Atomics = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
                f13 = *reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(n);
                f13.pNext = &f12;
                f13.dynamicRendering = VK_TRUE;
                f13.synchronization2 = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES) {
                const auto* di = reinterpret_cast<const VkPhysicalDeviceDescriptorIndexingFeatures*>(n);
                f12.descriptorIndexing = VK_TRUE;
                if (di->shaderSampledImageArrayNonUniformIndexing) f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
                if (di->shaderStorageBufferArrayNonUniformIndexing) f12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
                if (di->shaderStorageImageArrayNonUniformIndexing) f12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
                if (di->shaderUniformTexelBufferArrayNonUniformIndexing) f12.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE;
                if (di->shaderStorageTexelBufferArrayNonUniformIndexing) f12.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE;
                if (di->descriptorBindingSampledImageUpdateAfterBind) f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
                if (di->descriptorBindingStorageImageUpdateAfterBind) f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
                if (di->descriptorBindingStorageBufferUpdateAfterBind) f12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
                if (di->descriptorBindingUniformTexelBufferUpdateAfterBind) f12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
                if (di->descriptorBindingStorageTexelBufferUpdateAfterBind) f12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
                if (di->descriptorBindingUpdateUnusedWhilePending) f12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
                if (di->descriptorBindingPartiallyBound) f12.descriptorBindingPartiallyBound = VK_TRUE;
                if (di->descriptorBindingVariableDescriptorCount) f12.descriptorBindingVariableDescriptorCount = VK_TRUE;
                if (di->runtimeDescriptorArray) f12.runtimeDescriptorArray = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
                const auto* bda = reinterpret_cast<const VkPhysicalDeviceBufferDeviceAddressFeatures*>(n);
                if (bda->bufferDeviceAddress) f12.bufferDeviceAddress = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
                f12.timelineSemaphore = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES) {
                if (reinterpret_cast<const VkPhysicalDeviceScalarBlockLayoutFeatures*>(n)->scalarBlockLayout) f12.scalarBlockLayout = VK_TRUE;
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT) {
                cbc = *reinterpret_cast<const VkPhysicalDeviceCustomBorderColorFeaturesEXT*>(n); chain_tail(&cbc);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT) {
                vad = *reinterpret_cast<const VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT*>(n); chain_tail(&vad);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT) {
                dce = *reinterpret_cast<const VkPhysicalDeviceDepthClipEnableFeaturesEXT*>(n); chain_tail(&dce);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT) {
                tfb = *reinterpret_cast<const VkPhysicalDeviceTransformFeedbackFeaturesEXT*>(n); chain_tail(&tfb);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT) {
                mdt = *reinterpret_cast<const VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT*>(n); chain_tail(&mdt);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT) {
                bcs = *reinterpret_cast<const VkPhysicalDeviceBorderColorSwizzleFeaturesEXT*>(n); chain_tail(&bcs);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT) {
                i2d3d = *reinterpret_cast<const VkPhysicalDeviceImage2DViewOf3DFeaturesEXT*>(n); chain_tail(&i2d3d);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT) {
                ivml = *reinterpret_cast<const VkPhysicalDeviceImageViewMinLodFeaturesEXT*>(n); chain_tail(&ivml);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT) {
                crf = *reinterpret_cast<const VkPhysicalDeviceConditionalRenderingFeaturesEXT*>(n); chain_tail(&crf);
            } else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT) {
                iu8 = *reinterpret_cast<const VkPhysicalDeviceIndexTypeUint8FeaturesEXT*>(n); chain_tail(&iu8);
            }
        }
    }
    f2.features.shaderInt64 = VK_TRUE;

    std::vector<const char*> bexts;
    bexts.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
    bool has_r2 = false, has_pd = false;
    if (pCreateInfo) {
        for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i) {
            const char* e = pCreateInfo->ppEnabledExtensionNames[i];
            if (strcmp(e, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) continue;
            if (strcmp(e, "VK_KHR_swapchain_mutable_format") == 0) continue;
            if (strcmp(e, VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) == 0) continue;
            if (strcmp(e, "VK_KHR_load_store_op_none") == 0) continue;
            if (strcmp(e, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) continue;
            if (strcmp(e, VK_KHR_MAINTENANCE_6_EXTENSION_NAME) == 0) continue;
            if (strcmp(e, "VK_EXT_dynamic_rendering_unused_attachments") == 0) continue;
            if (strcmp(e, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) == 0) has_r2 = true;
            if (strcmp(e, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) has_pd = true;
            bexts.push_back(e);
        }
    }
    if (!has_r2) bexts.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    if (!has_pd) bexts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    VkDeviceCreateInfo bdci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, nullptr,
                            (uint32_t)bexts.size(), bexts.data(), nullptr};
    VkDevice rdev = VK_NULL_HANDLE;
    VkResult r = spd->inst->fn.CreateDevice(spd->real_pd, &bdci, nullptr, &rdev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[SmRast] ERROR: backend vkCreateDevice(qf=%u) failed: %d\n", spd->backend_compute_qf, r);
        return r;
    }

    auto* dev = new SmDevice();
    set_loader_magic_value(dev);
    dev->pd = spd;
    dev->real_dev = rdev;
    auto gdpa = spd->inst->fn.GetDeviceProcAddr;
#define LOAD_D(name) dev->fn.name = (PFN_vk##name)gdpa(rdev, "vk" #name)
    LOAD_D(DestroyDevice); LOAD_D(GetDeviceQueue); LOAD_D(DeviceWaitIdle);
    LOAD_D(QueueSubmit); LOAD_D(QueueSubmit2); LOAD_D(QueueWaitIdle);
    if (!dev->fn.QueueSubmit2) dev->fn.QueueSubmit2 = (PFN_vkQueueSubmit2)gdpa(rdev, "vkQueueSubmit2KHR");
    LOAD_D(AllocateMemory); LOAD_D(FreeMemory); LOAD_D(MapMemory); LOAD_D(UnmapMemory);
    LOAD_D(FlushMappedMemoryRanges); LOAD_D(InvalidateMappedMemoryRanges);
    LOAD_D(CreateBuffer); LOAD_D(DestroyBuffer); LOAD_D(GetBufferMemoryRequirements); LOAD_D(BindBufferMemory);
    LOAD_D(CreateImage); LOAD_D(DestroyImage); LOAD_D(GetImageMemoryRequirements); LOAD_D(BindImageMemory); LOAD_D(GetImageSubresourceLayout);
    LOAD_D(CreateImageView); LOAD_D(DestroyImageView); LOAD_D(CreateSampler); LOAD_D(DestroySampler);
    LOAD_D(CreateShaderModule); LOAD_D(DestroyShaderModule);
    LOAD_D(CreateDescriptorSetLayout); LOAD_D(DestroyDescriptorSetLayout);
    LOAD_D(CreatePipelineLayout); LOAD_D(DestroyPipelineLayout);
    LOAD_D(CreateComputePipelines); LOAD_D(DestroyPipeline);
    LOAD_D(CreatePipelineCache); LOAD_D(DestroyPipelineCache); LOAD_D(GetPipelineCacheData);
    LOAD_D(CreateDescriptorPool); LOAD_D(DestroyDescriptorPool); LOAD_D(ResetDescriptorPool);
    LOAD_D(AllocateDescriptorSets); LOAD_D(FreeDescriptorSets); LOAD_D(UpdateDescriptorSets);
    LOAD_D(CreateCommandPool); LOAD_D(DestroyCommandPool); LOAD_D(ResetCommandPool);
    LOAD_D(AllocateCommandBuffers); LOAD_D(FreeCommandBuffers);
    LOAD_D(BeginCommandBuffer); LOAD_D(EndCommandBuffer); LOAD_D(ResetCommandBuffer);
    LOAD_D(CreateFence); LOAD_D(DestroyFence); LOAD_D(ResetFences); LOAD_D(GetFenceStatus); LOAD_D(WaitForFences);
    LOAD_D(CreateSemaphore); LOAD_D(DestroySemaphore);
    LOAD_D(CmdBindPipeline); LOAD_D(CmdBindDescriptorSets); LOAD_D(CmdPushConstants);
    LOAD_D(CmdPushDescriptorSetKHR);
    LOAD_D(CmdDispatch); LOAD_D(CmdDispatchIndirect);
    LOAD_D(CmdCopyBuffer); LOAD_D(CmdCopyImage); LOAD_D(CmdCopyBufferToImage); LOAD_D(CmdCopyImageToBuffer);
    LOAD_D(CmdFillBuffer); LOAD_D(CmdUpdateBuffer); LOAD_D(CmdClearColorImage); LOAD_D(CmdPipelineBarrier);
#undef LOAD_D

    dev->main_queue = new SmQueue();
    set_loader_magic_value(dev->main_queue);
    dev->main_queue->dev = dev;
    dev->fn.GetDeviceQueue(rdev, spd->backend_compute_qf, 0, &dev->main_queue->real_q);

    if (!sm_init_raster_engine(dev)) {
        fprintf(stderr, "[SmRast] ERROR: sm_init_raster_engine failed!\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    *pDevice = reinterpret_cast<VkDevice>(dev);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL sm_DestroyDevice(VkDevice device, const VkAllocationCallbacks*) {
    auto* dev = reinterpret_cast<SmDevice*>(device);
    if (!dev) return;
    sm_destroy_raster_engine(dev);
    dev->fn.DestroyDevice(dev->real_dev, nullptr);
    delete dev->main_queue;
    delete dev;
}

static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceQueue(VkDevice device, uint32_t, uint32_t, VkQueue* pQueue) {
    auto* dev = reinterpret_cast<SmDevice*>(device);
    *pQueue = reinterpret_cast<VkQueue>(dev->main_queue);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_DeviceWaitIdle(VkDevice device) {
    auto* dev = reinterpret_cast<SmDevice*>(device);
    return dev->fn.DeviceWaitIdle(dev->real_dev);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_QueueWaitIdle(VkQueue queue) {
    auto* q = reinterpret_cast<SmQueue*>(queue);
    std::lock_guard<std::mutex> qlk(q->mtx);
    return q->dev->fn.QueueWaitIdle(q->real_q);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_QueueSubmit(
        VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
    auto* q = reinterpret_cast<SmQueue*>(queue);
    std::vector<VkSubmitInfo> mod(submitCount);
    std::vector<std::vector<VkCommandBuffer>> cbs(submitCount);
    std::vector<std::vector<VkPipelineStageFlags>> stages(submitCount);
    for (uint32_t i = 0; i < submitCount; ++i) {
        mod[i] = pSubmits[i];
        cbs[i].resize(pSubmits[i].commandBufferCount);
        for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; ++j)
            cbs[i][j] = reinterpret_cast<SmCommandBuffer*>(pSubmits[i].pCommandBuffers[j])->real_cb;
        mod[i].pCommandBuffers = cbs[i].data();
        if (pSubmits[i].waitSemaphoreCount && pSubmits[i].pWaitDstStageMask) {
            stages[i].resize(pSubmits[i].waitSemaphoreCount);
            for (uint32_t j = 0; j < pSubmits[i].waitSemaphoreCount; ++j)
                stages[i][j] = sanitize_stage_mask(pSubmits[i].pWaitDstStageMask[j]);
            mod[i].pWaitDstStageMask = stages[i].data();
        }
    }
    VkResult r;
    {
        std::lock_guard<std::mutex> qlk(q->mtx);
        r = q->dev->fn.QueueSubmit(q->real_q, submitCount, mod.data(), fence);
    }
    SM_TRACE("QueueSubmit(n=%u, fence=%p) -> %d", submitCount, (void*)fence, (int)r);
    if (r != VK_SUCCESS) fprintf(stderr, "[SmRast] ERROR: QueueSubmit returned %d\n", (int)r);
    return r;
}

// =============================================================================
// Memory, Buffer, Image, ImageView, Sampler (Native H20 HBM3 + TMUs!)
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_AllocateMemory(VkDevice d, const VkMemoryAllocateInfo* a, const VkAllocationCallbacks* c, VkDeviceMemory* m) {
    return reinterpret_cast<SmDevice*>(d)->fn.AllocateMemory(reinterpret_cast<SmDevice*>(d)->real_dev, a, c, m);
}
static VKAPI_ATTR void VKAPI_CALL sm_FreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* c) {
    reinterpret_cast<SmDevice*>(d)->fn.FreeMemory(reinterpret_cast<SmDevice*>(d)->real_dev, m, c);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_MapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize s, VkMemoryMapFlags f, void** p) {
    return reinterpret_cast<SmDevice*>(d)->fn.MapMemory(reinterpret_cast<SmDevice*>(d)->real_dev, m, o, s, f, p);
}
static VKAPI_ATTR void VKAPI_CALL sm_UnmapMemory(VkDevice d, VkDeviceMemory m) {
    reinterpret_cast<SmDevice*>(d)->fn.UnmapMemory(reinterpret_cast<SmDevice*>(d)->real_dev, m);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_FlushMappedMemoryRanges(VkDevice d, uint32_t n, const VkMappedMemoryRange* r) {
    return reinterpret_cast<SmDevice*>(d)->fn.FlushMappedMemoryRanges(reinterpret_cast<SmDevice*>(d)->real_dev, n, r);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_InvalidateMappedMemoryRanges(VkDevice d, uint32_t n, const VkMappedMemoryRange* r) {
    return reinterpret_cast<SmDevice*>(d)->fn.InvalidateMappedMemoryRanges(reinterpret_cast<SmDevice*>(d)->real_dev, n, r);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateBuffer(VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    VkBufferCreateInfo bci = *ci;
    for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(ci->pNext); n; n = n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR) {
            bci.usage |= (VkBufferUsageFlags)reinterpret_cast<const VkBufferUsageFlags2CreateInfoKHR*>(n)->usage;
            bci.pNext = nullptr; // strip maintenance5 pNext before passing to NVIDIA 535 backend
        }
    }
    // Every vertex/index buffer must also be readable as a STORAGE_BUFFER by our Compute Vertex Shader!
    bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.queueFamilyIndexCount = 0; bci.pQueueFamilyIndices = nullptr;
    VkBuffer rb = VK_NULL_HANDLE;
    VkResult r = dev->fn.CreateBuffer(dev->real_dev, &bci, nullptr, &rb);
    if (r != VK_SUCCESS) return r;
    auto* sb = new SmBuffer{rb, ci->size, bci.usage};
    *out = to_h<VkBuffer>(sb);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks*) {
    if (!b) return;
    auto* sb = from_h<SmBuffer>(b);
    reinterpret_cast<SmDevice*>(d)->fn.DestroyBuffer(reinterpret_cast<SmDevice*>(d)->real_dev, sb->real_buf, nullptr);
    delete sb;
}
static VKAPI_ATTR void VKAPI_CALL sm_GetBufferMemoryRequirements(VkDevice d, VkBuffer b, VkMemoryRequirements* mr) {
    reinterpret_cast<SmDevice*>(d)->fn.GetBufferMemoryRequirements(reinterpret_cast<SmDevice*>(d)->real_dev, from_h<SmBuffer>(b)->real_buf, mr);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_BindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o) {
    return reinterpret_cast<SmDevice*>(d)->fn.BindBufferMemory(reinterpret_cast<SmDevice*>(d)->real_dev, from_h<SmBuffer>(b)->real_buf, m, o);
}

static bool is_depth_format(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           f == VK_FORMAT_S8_UINT;
}
static bool is_stencil_format(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_S8_UINT;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateImage(VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    VkImageCreateInfo ici = *ci;
    bool depth = is_depth_format(ci->format);
    bool stencil = is_stencil_format(ci->format);
    // Strip GRAPHICS attachment bits & MSAA samples when creating real H20 image; add transfer bits for clean Compute interop!
    ici.usage = (ci->usage & ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT))
              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    if (depth) {
        // Store depth in real H20 image as R32_SFLOAT if sampled/transfer, and linear attach_buf for rasterizer
        ici.format = VK_FORMAT_R32_SFLOAT;
        ici.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        ici.pNext = nullptr; // avoid VkImageFormatListCreateInfo format mismatch with R32_SFLOAT
        ici.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.queueFamilyIndexCount = 0; ici.pQueueFamilyIndices = nullptr;
    VkImage ri = VK_NULL_HANDLE;
    VkResult r = dev->fn.CreateImage(dev->real_dev, &ici, nullptr, &ri);
    if (r != VK_SUCCESS) return r;
    auto* si = new SmImage();
    si->real_img = ri;
    si->format = ci->format;
    si->extent = ci->extent;
    si->mip_levels = ci->mipLevels;
    si->array_layers = ci->arrayLayers;
    si->usage = ci->usage;
    si->samples = ci->samples;
    si->is_depth = depth;
    si->has_stencil = stencil;
    if (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        std::lock_guard<std::mutex> lk(dev->mtx);
        dev->storage_images.push_back(si);
    }
    if (ci->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        sm_ensure_image_attach_buf(dev, si);
        if (stencil) sm_ensure_image_stencil_buf(dev, si);
    }
    SM_TRACE("CreateImage(%ux%u mips=%u lay=%u fmt=%d usage=0x%x til=%d)",
             ci->extent.width, ci->extent.height, ci->mipLevels, ci->arrayLayers,
             (int)ci->format, (unsigned)ci->usage, (int)ci->tiling);
    *out = to_h<VkImage>(si);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks*) {
    if (!i) return;
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* si = from_h<SmImage>(i);
    if (si->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        std::lock_guard<std::mutex> lk(dev->mtx);
        auto& v = dev->storage_images;
        v.erase(std::remove(v.begin(), v.end(), si), v.end());
    }
    if (si->attach_buf)  { dev->fn.DestroyBuffer(dev->real_dev, si->attach_buf, nullptr);  dev->fn.FreeMemory(dev->real_dev, si->attach_mem, nullptr); }
    if (si->stencil_buf) { dev->fn.DestroyBuffer(dev->real_dev, si->stencil_buf, nullptr); dev->fn.FreeMemory(dev->real_dev, si->stencil_mem, nullptr); }
    if (si->host_buf)    { dev->fn.DestroyBuffer(dev->real_dev, si->host_buf, nullptr);    dev->fn.FreeMemory(dev->real_dev, si->host_mem, nullptr); }
    if (si->real_img)    dev->fn.DestroyImage(dev->real_dev, si->real_img, nullptr);
    delete si;
}
static VKAPI_ATTR void VKAPI_CALL sm_GetImageMemoryRequirements(VkDevice d, VkImage i, VkMemoryRequirements* mr) {
    reinterpret_cast<SmDevice*>(d)->fn.GetImageMemoryRequirements(reinterpret_cast<SmDevice*>(d)->real_dev, from_h<SmImage>(i)->real_img, mr);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_BindImageMemory(VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o) {
    return reinterpret_cast<SmDevice*>(d)->fn.BindImageMemory(reinterpret_cast<SmDevice*>(d)->real_dev, from_h<SmImage>(i)->real_img, m, o);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetImageSubresourceLayout(VkDevice d, VkImage i, const VkImageSubresource* sr, VkSubresourceLayout* l) {
    reinterpret_cast<SmDevice*>(d)->fn.GetImageSubresourceLayout(reinterpret_cast<SmDevice*>(d)->real_dev, from_h<SmImage>(i)->real_img, sr, l);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateImageView(VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks*, VkImageView* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* si = from_h<SmImage>(ci->image);
    VkImageViewCreateInfo ivci = *ci;
    ivci.image = si->real_img;
    if (si->is_depth) {
        ivci.format = VK_FORMAT_R32_SFLOAT;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.pNext = nullptr;
    }
    VkImageView rv = VK_NULL_HANDLE;
    VkResult r = dev->fn.CreateImageView(dev->real_dev, &ivci, nullptr, &rv);
    if (r != VK_SUCCESS) return r;
    auto* sv = new SmImageView{rv, si, ci->format,
                               ci->subresourceRange.baseMipLevel, ci->subresourceRange.levelCount,
                               ci->subresourceRange.baseArrayLayer, ci->subresourceRange.layerCount};
    *out = to_h<VkImageView>(sv);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks*) {
    if (!v) return;
    auto* sv = from_h<SmImageView>(v);
    reinterpret_cast<SmDevice*>(d)->fn.DestroyImageView(reinterpret_cast<SmDevice*>(d)->real_dev, sv->real_view, nullptr);
    delete sv;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateSampler(VkDevice d, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* a, VkSampler* s) {
    return reinterpret_cast<SmDevice*>(d)->fn.CreateSampler(reinterpret_cast<SmDevice*>(d)->real_dev, ci, a, s);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroySampler(VkDevice d, VkSampler s, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroySampler(reinterpret_cast<SmDevice*>(d)->real_dev, s, a);
}

// =============================================================================
// Shaders, DescriptorSets, RenderPass, Framebuffer, PipelineLayouts, Pipelines
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks*, VkShaderModule* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    VkShaderModule rm = VK_NULL_HANDLE;
    dev->fn.CreateShaderModule(dev->real_dev, ci, nullptr, &rm);
    auto* sm = new SmShaderModule();
    sm->real_mod = rm;
    sm->spirv.assign(ci->pCode, ci->pCode + ci->codeSize / 4);
    *out = to_h<VkShaderModule>(sm);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyShaderModule(VkDevice d, VkShaderModule m, const VkAllocationCallbacks*) {
    if (!m) return;
    auto* sm = from_h<SmShaderModule>(m);
    if (sm->real_mod) reinterpret_cast<SmDevice*>(d)->fn.DestroyShaderModule(reinterpret_cast<SmDevice*>(d)->real_dev, sm->real_mod, nullptr);
    delete sm;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    // Allow every descriptor binding (UBO, SSBO, Sampler) to be accessed from COMPUTE stage too!
    std::vector<VkDescriptorSetLayoutBinding> b(ci->bindingCount);
    uint32_t dyn_cnt = 0;
    for (uint32_t i = 0; i < ci->bindingCount; ++i) {
        b[i] = ci->pBindings[i];
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        if (b[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            b[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            dyn_cnt += b[i].descriptorCount;
        }
    }
    VkDescriptorSetLayoutCreateInfo mci = *ci;
    mci.pBindings = b.data();
    VkResult r = dev->fn.CreateDescriptorSetLayout(dev->real_dev, &mci, nullptr, out);
    if (r == VK_SUCCESS && *out) {
        std::lock_guard<std::mutex> lk(dev->mtx);
        dev->dsl_dyn_counts[*out] = dyn_cnt;
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyDescriptorSetLayout(VkDevice d, VkDescriptorSetLayout l, const VkAllocationCallbacks* a) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    {
        std::lock_guard<std::mutex> lk(dev->mtx);
        dev->dsl_dyn_counts.erase(l);
    }
    dev->fn.DestroyDescriptorSetLayout(dev->real_dev, l, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorPool* p) {
    return reinterpret_cast<SmDevice*>(d)->fn.CreateDescriptorPool(reinterpret_cast<SmDevice*>(d)->real_dev, ci, a, p);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyDescriptorPool(VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroyDescriptorPool(reinterpret_cast<SmDevice*>(d)->real_dev, p, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_ResetDescriptorPool(VkDevice d, VkDescriptorPool p, VkDescriptorPoolResetFlags f) {
    return reinterpret_cast<SmDevice*>(d)->fn.ResetDescriptorPool(reinterpret_cast<SmDevice*>(d)->real_dev, p, f);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_AllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* s) {
    return reinterpret_cast<SmDevice*>(d)->fn.AllocateDescriptorSets(reinterpret_cast<SmDevice*>(d)->real_dev, ai, s);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_FreeDescriptorSets(VkDevice d, VkDescriptorPool p, uint32_t n, const VkDescriptorSet* s) {
    return reinterpret_cast<SmDevice*>(d)->fn.FreeDescriptorSets(reinterpret_cast<SmDevice*>(d)->real_dev, p, n, s);
}
static VKAPI_ATTR void VKAPI_CALL sm_UpdateDescriptorSets(VkDevice d, uint32_t wc, const VkWriteDescriptorSet* ws, uint32_t cc, const VkCopyDescriptorSet* cs) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    std::vector<VkWriteDescriptorSet> mws(ws, ws + wc);
    std::vector<std::vector<VkDescriptorBufferInfo>> bis(wc);
    std::vector<std::vector<VkDescriptorImageInfo>> iis(wc);
    for (uint32_t i = 0; i < wc; ++i) {
        VkDescriptorType dt = mws[i].descriptorType;
        bool is_buf = (dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                       dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
        bool is_img = (dt == VK_DESCRIPTOR_TYPE_SAMPLER || dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                       dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || dt == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
                       dt == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        if (is_buf && mws[i].pBufferInfo && mws[i].descriptorCount > 0) {
            bis[i].assign(mws[i].pBufferInfo, mws[i].pBufferInfo + mws[i].descriptorCount);
            for (auto& bi : bis[i]) {
                if (bi.buffer) bi.buffer = from_h<SmBuffer>(bi.buffer)->real_buf;
                else { bi.buffer = VK_NULL_HANDLE; bi.offset = 0; bi.range = VK_WHOLE_SIZE; }
            }
            mws[i].pBufferInfo = bis[i].data();
        }
        if (is_img && mws[i].pImageInfo && mws[i].descriptorCount > 0) {
            iis[i].assign(mws[i].pImageInfo, mws[i].pImageInfo + mws[i].descriptorCount);
            for (auto& ii : iis[i]) {
                if (dt != VK_DESCRIPTOR_TYPE_SAMPLER) {
                    if (ii.imageView) ii.imageView = from_h<SmImageView>(ii.imageView)->real_view;
                    ii.imageLayout = sanitize_layout(ii.imageLayout);
                }
            }
            mws[i].pImageInfo = iis[i].data();
        }
    }
    dev->fn.UpdateDescriptorSets(dev->real_dev, wc, mws.data(), cc, cs);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* pl = new SmPipelineLayout();
    pl->app_dsls.assign(ci->pSetLayouts, ci->pSetLayouts + ci->setLayoutCount);
    pl->sm_set_index = ci->setLayoutCount;
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i) {
        VkPushConstantRange pr = ci->pPushConstantRanges[i];
        pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pl->push_ranges.push_back(pr);
    }

    // 1. Real layout for native compute shaders
    VkPipelineLayoutCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        (uint32_t)pl->app_dsls.size(), pl->app_dsls.data(),
        (uint32_t)pl->push_ranges.size(), pl->push_ranges.data()};
    dev->fn.CreatePipelineLayout(dev->real_dev, &rci, nullptr, &pl->real_layout);

    // 2. Extended VS & FS layouts appending SmRast internal descriptor set at set = sm_set_index!
    auto vs_dsls = pl->app_dsls; vs_dsls.push_back(dev->rast.vs_int_dsl);
    VkPipelineLayoutCreateInfo vci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        (uint32_t)vs_dsls.size(), vs_dsls.data(),
        (uint32_t)pl->push_ranges.size(), pl->push_ranges.data()};
    dev->fn.CreatePipelineLayout(dev->real_dev, &vci, nullptr, &pl->sm_vs_layout);

    auto fs_dsls = pl->app_dsls; fs_dsls.push_back(dev->rast.fs_int_dsl);
    VkPipelineLayoutCreateInfo fci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        (uint32_t)fs_dsls.size(), fs_dsls.data(),
        (uint32_t)pl->push_ranges.size(), pl->push_ranges.data()};
    dev->fn.CreatePipelineLayout(dev->real_dev, &fci, nullptr, &pl->sm_fs_layout);

    *out = to_h<VkPipelineLayout>(pl);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyPipelineLayout(VkDevice d, VkPipelineLayout l, const VkAllocationCallbacks*) {
    if (!l) return;
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* pl = from_h<SmPipelineLayout>(l);
    if (pl->real_layout)  dev->fn.DestroyPipelineLayout(dev->real_dev, pl->real_layout, nullptr);
    if (pl->sm_vs_layout) dev->fn.DestroyPipelineLayout(dev->real_dev, pl->sm_vs_layout, nullptr);
    if (pl->sm_fs_layout) dev->fn.DestroyPipelineLayout(dev->real_dev, pl->sm_fs_layout, nullptr);
    delete pl;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateRenderPass(VkDevice, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks*, VkRenderPass* out) {
    auto* rp = new SmRenderPass();
    rp->attachments.assign(ci->pAttachments, ci->pAttachments + ci->attachmentCount);
    if (ci->subpassCount > 0) {
        const auto& sp = ci->pSubpasses[0];
        rp->color_attach_count = std::min(sp.colorAttachmentCount, SM_MAX_MRT);
        for (uint32_t i = 0; i < rp->color_attach_count; ++i) {
            if (sp.pColorAttachments[i].attachment != VK_ATTACHMENT_UNUSED)
                rp->color_attach_idxs[i] = (int)sp.pColorAttachments[i].attachment;
        }
        if (sp.pDepthStencilAttachment && sp.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED)
            rp->depth_attach_idx = (int)sp.pDepthStencilAttachment->attachment;
    }
    *out = to_h<VkRenderPass>(rp);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyRenderPass(VkDevice, VkRenderPass rp, const VkAllocationCallbacks*) {
    if (rp) delete from_h<SmRenderPass>(rp);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateFramebuffer(VkDevice, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks*, VkFramebuffer* out) {
    auto* fb = new SmFramebuffer();
    fb->rp = from_h<SmRenderPass>(ci->renderPass);
    fb->width = ci->width;
    fb->height = ci->height;
    if (ci->pAttachments) {
        for (uint32_t i = 0; i < ci->attachmentCount; ++i)
            fb->attachments.push_back(from_h<SmImageView>(ci->pAttachments[i]));
    }
    *out = to_h<VkFramebuffer>(fb);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyFramebuffer(VkDevice, VkFramebuffer fb, const VkAllocationCallbacks*) {
    if (fb) delete from_h<SmFramebuffer>(fb);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreatePipelineCache(VkDevice d, const VkPipelineCacheCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineCache* p) {
    return reinterpret_cast<SmDevice*>(d)->fn.CreatePipelineCache(reinterpret_cast<SmDevice*>(d)->real_dev, ci, a, p);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyPipelineCache(VkDevice d, VkPipelineCache p, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroyPipelineCache(reinterpret_cast<SmDevice*>(d)->real_dev, p, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPipelineCacheData(VkDevice d, VkPipelineCache p, size_t* s, void* data) {
    return reinterpret_cast<SmDevice*>(d)->fn.GetPipelineCacheData(reinterpret_cast<SmDevice*>(d)->real_dev, p, s, data);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateComputePipelines(
        VkDevice d, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo* cis,
        const VkAllocationCallbacks*, VkPipeline* outs) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    for (uint32_t i = 0; i < count; ++i) {
        VkComputePipelineCreateInfo m = cis[i];
        VkShaderModule tmp_sm = VK_NULL_HANDLE;
        if (cis[i].stage.module != VK_NULL_HANDLE) {
            m.stage.module = from_h<SmShaderModule>(cis[i].stage.module)->real_mod;
        } else {
            // VK_KHR_maintenance5 inline VkShaderModuleCreateInfo in stage.pNext!
            for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(cis[i].stage.pNext); n; n = n->pNext) {
                if (n->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
                    dev->fn.CreateShaderModule(dev->real_dev, reinterpret_cast<const VkShaderModuleCreateInfo*>(n), nullptr, &tmp_sm);
                    m.stage.module = tmp_sm;
                    m.stage.pNext = nullptr;
                    break;
                }
            }
        }
        m.layout = from_h<SmPipelineLayout>(cis[i].layout)->real_layout;
        VkPipeline rp = VK_NULL_HANDLE;
        VkResult r = dev->fn.CreateComputePipelines(dev->real_dev, cache, 1, &m, nullptr, &rp);
        if (tmp_sm) dev->fn.DestroyShaderModule(dev->real_dev, tmp_sm, nullptr);
        if (r != VK_SUCCESS) return r;
        auto* sp = new SmPipeline();
        sp->is_compute = true;
        sp->real_compute_pipe = rp;
        sp->layout = from_h<SmPipelineLayout>(cis[i].layout);
        outs[i] = to_h<VkPipeline>(sp);
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateGraphicsPipelines(
        VkDevice d, VkPipelineCache, uint32_t count, const VkGraphicsPipelineCreateInfo* cis,
        const VkAllocationCallbacks*, VkPipeline* outs) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& gci = cis[i];
        auto* sp = new SmPipeline();
        sp->is_compute = false;
        sp->layout = from_h<SmPipelineLayout>(gci.layout);

        if (gci.pVertexInputState) {
            sp->vtx_input.bindings.assign(gci.pVertexInputState->pVertexBindingDescriptions,
                                          gci.pVertexInputState->pVertexBindingDescriptions + gci.pVertexInputState->vertexBindingDescriptionCount);
            sp->vtx_input.attrs.assign(gci.pVertexInputState->pVertexAttributeDescriptions,
                                       gci.pVertexInputState->pVertexAttributeDescriptions + gci.pVertexInputState->vertexAttributeDescriptionCount);
        }
        if (gci.pInputAssemblyState) sp->topology = gci.pInputAssemblyState->topology;
        if (gci.pRasterizationState) {
            sp->cull_mode  = gci.pRasterizationState->cullMode;
            sp->front_face = gci.pRasterizationState->frontFace;
        }
        if (gci.pDepthStencilState) {
            sp->depth_test    = (gci.pDepthStencilState->depthTestEnable == VK_TRUE);
            sp->depth_write   = (gci.pDepthStencilState->depthWriteEnable == VK_TRUE);
            sp->depth_op      = gci.pDepthStencilState->depthCompareOp;
            sp->stencil_test  = (gci.pDepthStencilState->stencilTestEnable == VK_TRUE);
            sp->stencil_front = gci.pDepthStencilState->front;
            sp->stencil_back  = gci.pDepthStencilState->back;
        }
        bool has_dyn_stencil = false;
        if (gci.pDynamicState && gci.pDynamicState->pDynamicStates) {
            for (uint32_t ds_i = 0; ds_i < gci.pDynamicState->dynamicStateCount; ++ds_i) {
                VkDynamicState st = gci.pDynamicState->pDynamicStates[ds_i];
                if (st == VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE || st == VK_DYNAMIC_STATE_STENCIL_OP)
                    has_dyn_stencil = true;
            }
        }
        if (gci.pColorBlendState) {
            for (int k = 0; k < 4; ++k) sp->blend_constants[k] = gci.pColorBlendState->blendConstants[k];
            if (gci.pColorBlendState->attachmentCount > 0 && gci.pColorBlendState->pAttachments) {
                uint32_t nb = std::min(gci.pColorBlendState->attachmentCount, SM_MAX_MRT);
                for (uint32_t b = 0; b < nb; ++b) {
                    const auto& cb = gci.pColorBlendState->pAttachments[b];
                    sp->blend_enable[b]     = (cb.blendEnable == VK_TRUE);
                    sp->src_color_blend[b]  = cb.srcColorBlendFactor;
                    sp->dst_color_blend[b]  = cb.dstColorBlendFactor;
                    sp->color_blend_op[b]   = cb.colorBlendOp;
                    sp->src_alpha_blend[b]  = cb.srcAlphaBlendFactor;
                    sp->dst_alpha_blend[b]  = cb.dstAlphaBlendFactor;
                    sp->alpha_blend_op[b]   = cb.alphaBlendOp;
                    sp->color_write_mask[b] = cb.colorWriteMask;
                }
            }
        }
        if (gci.pViewportState) {
            if (gci.pViewportState->viewportCount > 0 && gci.pViewportState->pViewports) {
                sp->has_static_vp = true; sp->static_vp = gci.pViewportState->pViewports[0];
            }
            if (gci.pViewportState->scissorCount > 0 && gci.pViewportState->pScissors) {
                sp->has_static_scissor = true; sp->static_scissor = gci.pViewportState->pScissors[0];
            }
        }

        // Automatically lower Vertex & Fragment SPIR-V -> H20 Compute Shaders (supports VK_KHR_maintenance5 inline shaders!)
        SmShaderModule *vs_mod = nullptr, *fs_mod = nullptr;
        std::vector<uint32_t> vs_inline_spv, fs_inline_spv;
        const VkSpecializationInfo *vs_spec = nullptr, *fs_spec = nullptr;
        for (uint32_t s = 0; s < gci.stageCount; ++s) {
            auto extract_inline = [](const void* pNext, std::vector<uint32_t>& dst) {
                for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(pNext); n; n = n->pNext) {
                    if (n->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
                        const auto* smci = reinterpret_cast<const VkShaderModuleCreateInfo*>(n);
                        if (smci->pCode && smci->codeSize >= 4)
                            dst.assign(smci->pCode, smci->pCode + smci->codeSize / 4);
                        return;
                    }
                }
            };
            if (gci.pStages[s].stage == VK_SHADER_STAGE_VERTEX_BIT) {
                if (gci.pStages[s].module != VK_NULL_HANDLE) vs_mod = from_h<SmShaderModule>(gci.pStages[s].module);
                else extract_inline(gci.pStages[s].pNext, vs_inline_spv);
                vs_spec = gci.pStages[s].pSpecializationInfo;
            }
            if (gci.pStages[s].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                if (gci.pStages[s].module != VK_NULL_HANDLE) fs_mod = from_h<SmShaderModule>(gci.pStages[s].module);
                else extract_inline(gci.pStages[s].pNext, fs_inline_spv);
                fs_spec = gci.pStages[s].pSpecializationInfo;
            }
        }
        const std::vector<uint32_t>* vs_spv_ptr = vs_mod ? &vs_mod->spirv : (!vs_inline_spv.empty() ? &vs_inline_spv : nullptr);
        const std::vector<uint32_t>* fs_spv_ptr = fs_mod ? &fs_mod->spirv : (!fs_inline_spv.empty() ? &fs_inline_spv : nullptr);

        bool ok = true;
        if (vs_spv_ptr) {
            std::vector<uint32_t> vs_comp_spv;
            std::string err;
            if (!sm_lower_vertex_spirv(*vs_spv_ptr, sp->vtx_input, vs_spec, sp->layout->sm_set_index, vs_comp_spv, err)) {
                fprintf(stderr, "[SmRast] VS lowering failed: %s\n", err.c_str());
                ok = false;
            } else {
                VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, vs_comp_spv.size()*4, vs_comp_spv.data()};
                VkShaderModule csm = VK_NULL_HANDLE;
                dev->fn.CreateShaderModule(dev->real_dev, &smci, nullptr, &csm);
                VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, csm, "main", vs_spec},
                    sp->layout->sm_vs_layout, VK_NULL_HANDLE, -1};
                VkResult r = dev->fn.CreateComputePipelines(dev->real_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &sp->vs_comp_pipe);
                dev->fn.DestroyShaderModule(dev->real_dev, csm, nullptr);
                if (r != VK_SUCCESS) { fprintf(stderr, "[SmRast] VS CreateComputePipelines failed: %d\n", r); ok = false; }
            }
        }
        if (ok && fs_spv_ptr) {
            SmFsCompileHints hints;
            hints.stencil_en = sp->stencil_test || has_dyn_stencil;
            for (uint32_t b = 0; b < SM_MAX_MRT; ++b) {
                hints.blend_en[b]   = sp->blend_enable[b];
                hints.write_mask[b] = sp->color_write_mask[b];
            }
            if (gci.renderPass != VK_NULL_HANDLE) {
                auto* rp = from_h<SmRenderPass>(gci.renderPass);
                for (uint32_t r = 0; r < rp->color_attach_count && r < SM_MAX_MRT; ++r) {
                    int idx = rp->color_attach_idxs[r];
                    if (idx >= 0 && idx < (int)rp->attachments.size() && rp->attachments[idx].format != VK_FORMAT_UNDEFINED)
                        hints.rt_fmt[r] = (int)sm_format_code(rp->attachments[idx].format);
                }
            } else if (gci.pNext) {
                for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(gci.pNext); n; n = n->pNext) {
                    if (n->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                        const auto* prci = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(n);
                        for (uint32_t r = 0; r < prci->colorAttachmentCount && r < SM_MAX_MRT; ++r) {
                            if (prci->pColorAttachmentFormats && prci->pColorAttachmentFormats[r] != VK_FORMAT_UNDEFINED)
                                hints.rt_fmt[r] = (int)sm_format_code(prci->pColorAttachmentFormats[r]);
                        }
                    }
                }
            }
            std::vector<uint32_t> fs_comp_spv;
            std::string err;
            if (!sm_lower_fragment_spirv(*fs_spv_ptr, fs_spec, sp->layout->sm_set_index, hints, fs_comp_spv, err)) {
                fprintf(stderr, "[SmRast] FS lowering failed: %s\n", err.c_str());
                ok = false;
            } else {
                VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, fs_comp_spv.size()*4, fs_comp_spv.data()};
                VkShaderModule csm = VK_NULL_HANDLE;
                dev->fn.CreateShaderModule(dev->real_dev, &smci, nullptr, &csm);
                VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, csm, "main", fs_spec},
                    sp->layout->sm_fs_layout, VK_NULL_HANDLE, -1};
                VkResult r = dev->fn.CreateComputePipelines(dev->real_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &sp->fs_comp_pipe);
                dev->fn.DestroyShaderModule(dev->real_dev, csm, nullptr);
                if (r != VK_SUCCESS) { fprintf(stderr, "[SmRast] FS CreateComputePipelines failed: %d\n", r); ok = false; }
            }
        } else if (ok && !fs_spv_ptr && (sp->stencil_test || has_dyn_stencil)) {
            std::vector<uint32_t> stn_spv;
            std::string err;
            if (sm_build_null_fs_stencil_spv(sp->layout->sm_set_index, stn_spv, err)) {
                VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, stn_spv.size()*4, stn_spv.data()};
                VkShaderModule csm = VK_NULL_HANDLE;
                dev->fn.CreateShaderModule(dev->real_dev, &smci, nullptr, &csm);
                VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
                    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, csm, "main", nullptr},
                    sp->layout->sm_fs_layout, VK_NULL_HANDLE, -1};
                dev->fn.CreateComputePipelines(dev->real_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &sp->fs_comp_pipe);
                dev->fn.DestroyShaderModule(dev->real_dev, csm, nullptr);
            }
        }
        if (!ok) {
            if (sp->vs_comp_pipe) { dev->fn.DestroyPipeline(dev->real_dev, sp->vs_comp_pipe, nullptr); sp->vs_comp_pipe = VK_NULL_HANDLE; }
            if (sp->fs_comp_pipe) { dev->fn.DestroyPipeline(dev->real_dev, sp->fs_comp_pipe, nullptr); sp->fs_comp_pipe = VK_NULL_HANDLE; }
            std::lock_guard<std::mutex> lk(dev->mtx);
            dev->failed_gfx_pipes++;
        } else {
            uint32_t seq = 0;
            {
                std::lock_guard<std::mutex> lk(dev->mtx);
                seq = dev->compiled_gfx_pipes++;
            }
            if (seq < 15 || (seq % 10) == 0) {
                fprintf(stderr, "[SmRast] Compiled Graphics Pipeline #%u -> H20 Compute VS+Bin+Fine+%s OK!\n",
                        seq, fs_spv_ptr ? "QuadFS" : "DepthOnly");
            }
        }
        outs[i] = to_h<VkPipeline>(sp);
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL sm_DestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks*) {
    if (!p) return;
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* sp = from_h<SmPipeline>(p);
    if (sp->real_compute_pipe) dev->fn.DestroyPipeline(dev->real_dev, sp->real_compute_pipe, nullptr);
    if (sp->vs_comp_pipe)      dev->fn.DestroyPipeline(dev->real_dev, sp->vs_comp_pipe, nullptr);
    if (sp->fs_comp_pipe)      dev->fn.DestroyPipeline(dev->real_dev, sp->fs_comp_pipe, nullptr);
    delete sp;
}

// =============================================================================
// Command Pools, Command Buffers, Sync, Barriers & GPU Compute Rasterization!
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateCommandPool(VkDevice d, const VkCommandPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkCommandPool* p) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    VkCommandPoolCreateInfo m = *ci;
    m.queueFamilyIndex = dev->pd->backend_compute_qf;
    return dev->fn.CreateCommandPool(dev->real_dev, &m, a, p);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyCommandPool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroyCommandPool(reinterpret_cast<SmDevice*>(d)->real_dev, p, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_ResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) {
    return reinterpret_cast<SmDevice*>(d)->fn.ResetCommandPool(reinterpret_cast<SmDevice*>(d)->real_dev, p, f);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_AllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* outs) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    std::vector<VkCommandBuffer> rcbs(ai->commandBufferCount);
    VkResult r = dev->fn.AllocateCommandBuffers(dev->real_dev, ai, rcbs.data());
    if (r != VK_SUCCESS) return r;
    for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
        auto* scb = new SmCommandBuffer();
        set_loader_magic_value(scb);
        scb->dev = dev;
        scb->real_cb = rcbs[i];
        outs[i] = reinterpret_cast<VkCommandBuffer>(scb);
    }
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_FreeCommandBuffers(VkDevice d, VkCommandPool p, uint32_t n, const VkCommandBuffer* cbs) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    std::vector<VkCommandBuffer> rcbs(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto* scb = reinterpret_cast<SmCommandBuffer*>(cbs[i]);
        rcbs[i] = scb->real_cb;
        delete scb;
    }
    dev->fn.FreeCommandBuffers(dev->real_dev, p, n, rcbs.data());
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->in_render_pass = false;
    scb->in_dyn_rendering = false;
    for (uint32_t i = 0; i < SM_MAX_MRT; ++i) { scb->dyn_col_imgs[i] = nullptr; scb->dyn_col_views[i] = nullptr; scb->dyn_res_imgs[i] = nullptr; }
    scb->dyn_col_count = 0;
    scb->dyn_dep_img = nullptr;
    scb->dyn_dep_view = nullptr;
    scb->cur_gfx_pipe = nullptr;
    scb->cur_comp_pipe = nullptr;
    scb->has_vp = false;
    scb->has_scissor = false;
    scb->has_dyn_cull = false;
    scb->has_dyn_front_face = false;
    scb->has_dyn_topo = false;
    scb->has_dyn_depth_test = false;
    scb->has_dyn_depth_write = false;
    scb->has_dyn_depth_op = false;
    scb->has_dyn_blend_const = false;
    scb->has_dyn_stencil_test = false;
    for (int f = 0; f < 2; ++f) {
        scb->has_dyn_stencil_op[f] = false;
        scb->has_dyn_stencil_ref[f] = false;
        scb->has_dyn_stencil_cmask[f] = false;
        scb->has_dyn_stencil_wmask[f] = false;
    }
    scb->gfx_set_mask = 0;
    memset(scb->gfx_dyn_counts, 0, sizeof(scb->gfx_dyn_counts));
    memset(scb->has_dyn_vtx_stride, 0, sizeof(scb->has_dyn_vtx_stride));
    scb->push_size = 0;
    scb->comp_set_mask = 0;
    memset(scb->comp_dyn_counts, 0, sizeof(scb->comp_dyn_counts));
    scb->comp_push_size = 0;
    scb->comp_state_dirty = false;
    for (int s = 0; s < 16; ++s) {
        scb->gfx_push_descs[s].active = false;
        scb->gfx_push_descs[s].writes.clear(); scb->gfx_push_descs[s].bufs.clear();
        scb->gfx_push_descs[s].imgs.clear();   scb->gfx_push_descs[s].bvs.clear();
        scb->comp_push_descs[s].active = false;
        scb->comp_push_descs[s].writes.clear(); scb->comp_push_descs[s].bufs.clear();
        scb->comp_push_descs[s].imgs.clear();   scb->comp_push_descs[s].bvs.clear();
    }
    return scb->dev->fn.BeginCommandBuffer(scb->real_cb, bi);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_EndCommandBuffer(VkCommandBuffer cb) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    return scb->dev->fn.EndCommandBuffer(scb->real_cb);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags f) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    return scb->dev->fn.ResetCommandBuffer(scb->real_cb, f);
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateFence(VkDevice d, const VkFenceCreateInfo* ci, const VkAllocationCallbacks* a, VkFence* f) {
    return reinterpret_cast<SmDevice*>(d)->fn.CreateFence(reinterpret_cast<SmDevice*>(d)->real_dev, ci, a, f);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroyFence(reinterpret_cast<SmDevice*>(d)->real_dev, f, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_ResetFences(VkDevice d, uint32_t n, const VkFence* f) {
    return reinterpret_cast<SmDevice*>(d)->fn.ResetFences(reinterpret_cast<SmDevice*>(d)->real_dev, n, f);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetFenceStatus(VkDevice d, VkFence f) {
    return reinterpret_cast<SmDevice*>(d)->fn.GetFenceStatus(reinterpret_cast<SmDevice*>(d)->real_dev, f);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_WaitForFences(VkDevice d, uint32_t n, const VkFence* f, VkBool32 all, uint64_t t) {
    SM_TRACE("WaitForFences(n=%u, timeout=%llu) enter", n, (unsigned long long)t);
    VkResult r = reinterpret_cast<SmDevice*>(d)->fn.WaitForFences(reinterpret_cast<SmDevice*>(d)->real_dev, n, f, all, t);
    SM_TRACE("WaitForFences -> %d", (int)r);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateSemaphore(VkDevice d, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks* a, VkSemaphore* s) {
    return reinterpret_cast<SmDevice*>(d)->fn.CreateSemaphore(reinterpret_cast<SmDevice*>(d)->real_dev, ci, a, s);
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroySemaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a) {
    reinterpret_cast<SmDevice*>(d)->fn.DestroySemaphore(reinterpret_cast<SmDevice*>(d)->real_dev, s, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetSemaphoreCounterValue(VkDevice d, VkSemaphore s, uint64_t* v) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetSemaphoreCounterValue)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetSemaphoreCounterValue");
    return fn ? fn(dev->real_dev, s, v) : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_WaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo* wi, uint64_t t) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkWaitSemaphores)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkWaitSemaphores");
    SM_TRACE("WaitSemaphores(n=%u, val0=%llu, timeout=%llu) enter", wi ? wi->semaphoreCount : 0, (wi && wi->semaphoreCount && wi->pValues) ? (unsigned long long)wi->pValues[0] : 0ull, (unsigned long long)t);
    VkResult r = fn ? fn(dev->real_dev, wi, t) : VK_SUCCESS;
    SM_TRACE("WaitSemaphores -> %d", (int)r);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_SignalSemaphore(VkDevice d, const VkSemaphoreSignalInfo* si) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkSignalSemaphore)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkSignalSemaphore");
    return fn ? fn(dev->real_dev, si) : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateEvent(VkDevice d, const VkEventCreateInfo* ci, const VkAllocationCallbacks* a, VkEvent* e) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkCreateEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkCreateEvent");
    return fn ? fn(dev->real_dev, ci, a, e) : VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyEvent(VkDevice d, VkEvent e, const VkAllocationCallbacks* a) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkDestroyEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkDestroyEvent");
    if (fn) fn(dev->real_dev, e, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetEventStatus(VkDevice d, VkEvent e) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetEventStatus)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetEventStatus");
    return fn ? fn(dev->real_dev, e) : VK_EVENT_SET;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_SetEvent(VkDevice d, VkEvent e) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkSetEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkSetEvent");
    return fn ? fn(dev->real_dev, e) : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_ResetEvent(VkDevice d, VkEvent e) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkResetEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkResetEvent");
    return fn ? fn(dev->real_dev, e) : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateQueryPool(VkDevice d, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkQueryPool* qp) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkCreateQueryPool)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkCreateQueryPool");
    VkQueryPoolCreateInfo m = *ci;
    if (m.queryType != VK_QUERY_TYPE_TIMESTAMP) m.queryType = VK_QUERY_TYPE_TIMESTAMP;
    return fn ? fn(dev->real_dev, &m, a, qp) : VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyQueryPool(VkDevice d, VkQueryPool qp, const VkAllocationCallbacks* a) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkDestroyQueryPool)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkDestroyQueryPool");
    if (fn) fn(dev->real_dev, qp, a);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetQueryPoolResults(VkDevice, VkQueryPool, uint32_t, uint32_t count, size_t sz, void* data, VkDeviceSize stride, VkQueryResultFlags fl) {
    if (!data || sz == 0) return VK_SUCCESS;
    memset(data, 0, sz);
    static std::atomic<uint64_t> g_ts{1000000};
    bool is64 = (fl & VK_QUERY_RESULT_64_BIT) != 0;
    bool with_avail = (fl & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;
    size_t elem = is64 ? 8 : 4;
    size_t st = stride ? (size_t)stride : (with_avail ? elem * 2 : elem);
    uint8_t* dst = reinterpret_cast<uint8_t*>(data);
    for (uint32_t i = 0; i < count; ++i) {
        size_t off = (size_t)i * st;
        uint64_t v = g_ts.fetch_add(1000);
        if (off + elem <= sz) {
            if (is64) { uint64_t val = v; memcpy(dst + off, &val, 8); }
            else      { uint32_t val = (uint32_t)v; memcpy(dst + off, &val, 4); }
        }
        if (with_avail && off + elem * 2 <= sz) {
            if (is64) { uint64_t av = 1; memcpy(dst + off + 8, &av, 8); }
            else      { uint32_t av = 1; memcpy(dst + off + 4, &av, 4); }
        }
    }
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_ResetQueryPool(VkDevice d, VkQueryPool qp, uint32_t f, uint32_t c) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkResetQueryPool)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkResetQueryPool");
    if (fn) fn(dev->real_dev, qp, f, c);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateBufferView(VkDevice d, const VkBufferViewCreateInfo* ci, const VkAllocationCallbacks* a, VkBufferView* bv) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkCreateBufferView)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkCreateBufferView");
    VkBufferViewCreateInfo m = *ci;
    m.buffer = from_h<SmBuffer>(ci->buffer)->real_buf;
    return fn ? fn(dev->real_dev, &m, a, bv) : VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyBufferView(VkDevice d, VkBufferView bv, const VkAllocationCallbacks* a) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkDestroyBufferView)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkDestroyBufferView");
    if (fn) fn(dev->real_dev, bv, a);
}
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL sm_GetBufferDeviceAddress(VkDevice d, const VkBufferDeviceAddressInfo* info) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetBufferDeviceAddress)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetBufferDeviceAddress");
    VkBufferDeviceAddressInfo m = *info;
    m.buffer = from_h<SmBuffer>(info->buffer)->real_buf;
    return fn ? fn(dev->real_dev, &m) : 0;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateDescriptorUpdateTemplate(VkDevice, const VkDescriptorUpdateTemplateCreateInfo* ci, const VkAllocationCallbacks*, VkDescriptorUpdateTemplate* out) {
    auto* t = new SmDescriptorUpdateTemplate();
    if (ci) {
        t->bind_point = ci->pipelineBindPoint;
        if (ci->descriptorUpdateEntryCount > 0 && ci->pDescriptorUpdateEntries) {
            t->entries.assign(ci->pDescriptorUpdateEntries, ci->pDescriptorUpdateEntries + ci->descriptorUpdateEntryCount);
        }
    }
    *out = to_h<VkDescriptorUpdateTemplate>(t);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyDescriptorUpdateTemplate(VkDevice, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks*) {
    if (t) delete from_h<SmDescriptorUpdateTemplate>(t);
}
static VKAPI_ATTR void VKAPI_CALL sm_UpdateDescriptorSetWithTemplate(VkDevice d, VkDescriptorSet ds, VkDescriptorUpdateTemplate tmpl, const void* data) {
    if (!tmpl || !data) return;
    auto* t = from_h<SmDescriptorUpdateTemplate>(tmpl);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(data);
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs(t->entries.size());
    std::vector<std::vector<VkDescriptorImageInfo>>  imgs(t->entries.size());
    std::vector<std::vector<VkBufferView>>           bvs(t->entries.size());
    writes.reserve(t->entries.size());
    for (size_t i = 0; i < t->entries.size(); ++i) {
        const auto& e = t->entries[i];
        if (e.descriptorCount == 0) continue;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = ds;
        w.dstBinding = e.dstBinding;
        w.dstArrayElement = e.dstArrayElement;
        w.descriptorCount = e.descriptorCount;
        w.descriptorType = e.descriptorType;
        switch (e.descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
                bufs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k) {
                    const auto* src = reinterpret_cast<const VkDescriptorBufferInfo*>(base + e.offset + k * e.stride);
                    bufs[i][k] = *src;
                }
                w.pBufferInfo = bufs[i].data();
                writes.push_back(w);
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
                imgs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k) {
                    const auto* src = reinterpret_cast<const VkDescriptorImageInfo*>(base + e.offset + k * e.stride);
                    imgs[i][k] = *src;
                }
                w.pImageInfo = imgs[i].data();
                writes.push_back(w);
                break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
                bvs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k) {
                    const auto* src = reinterpret_cast<const VkBufferView*>(base + e.offset + k * e.stride);
                    bvs[i][k] = *src;
                }
                w.pTexelBufferView = bvs[i].data();
                writes.push_back(w);
                break;
            }
            default: break;
        }
    }
    if (!writes.empty()) sm_UpdateDescriptorSets(d, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginQuery(VkCommandBuffer, VkQueryPool, uint32_t, VkQueryControlFlags) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndQuery(VkCommandBuffer, VkQueryPool, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdWriteTimestamp2(VkCommandBuffer, VkPipelineStageFlags2, VkQueryPool, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyQueryPoolResults(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t, VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageCopy*);
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBufferToImage(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, uint32_t, const VkBufferImageCopy*);
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImageToBuffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy*);
static VKAPI_ATTR void VKAPI_CALL sm_CmdBlitImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageBlit*, VkFilter);
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* ci) {
    std::vector<VkBufferCopy> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i) r[i] = {ci->pRegions[i].srcOffset, ci->pRegions[i].dstOffset, ci->pRegions[i].size};
    sm_CmdCopyBuffer(cb, ci->srcBuffer, ci->dstBuffer, ci->regionCount, r.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* ci) {
    std::vector<VkImageCopy> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i)
        r[i] = {ci->pRegions[i].srcSubresource, ci->pRegions[i].srcOffset, ci->pRegions[i].dstSubresource, ci->pRegions[i].dstOffset, ci->pRegions[i].extent};
    sm_CmdCopyImage(cb, ci->srcImage, ci->srcImageLayout, ci->dstImage, ci->dstImageLayout, ci->regionCount, r.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* ci) {
    std::vector<VkBufferImageCopy> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i)
        r[i] = {ci->pRegions[i].bufferOffset, ci->pRegions[i].bufferRowLength, ci->pRegions[i].bufferImageHeight,
                ci->pRegions[i].imageSubresource, ci->pRegions[i].imageOffset, ci->pRegions[i].imageExtent};
    sm_CmdCopyBufferToImage(cb, ci->srcBuffer, ci->dstImage, ci->dstImageLayout, ci->regionCount, r.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* ci) {
    std::vector<VkBufferImageCopy> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i)
        r[i] = {ci->pRegions[i].bufferOffset, ci->pRegions[i].bufferRowLength, ci->pRegions[i].bufferImageHeight,
                ci->pRegions[i].imageSubresource, ci->pRegions[i].imageOffset, ci->pRegions[i].imageExtent};
    sm_CmdCopyImageToBuffer(cb, ci->srcImage, ci->srcImageLayout, ci->dstBuffer, ci->regionCount, r.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* ci) {
    std::vector<VkImageBlit> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i) {
        r[i].srcSubresource = ci->pRegions[i].srcSubresource;
        r[i].srcOffsets[0] = ci->pRegions[i].srcOffsets[0]; r[i].srcOffsets[1] = ci->pRegions[i].srcOffsets[1];
        r[i].dstSubresource = ci->pRegions[i].dstSubresource;
        r[i].dstOffsets[0] = ci->pRegions[i].dstOffsets[0]; r[i].dstOffsets[1] = ci->pRegions[i].dstOffsets[1];
    }
    sm_CmdBlitImage(cb, ci->srcImage, ci->srcImageLayout, ci->dstImage, ci->dstImageLayout, ci->regionCount, r.data(), ci->filter);
}
static void sm_begin_pass_clear_mrt(SmCommandBuffer* scb, SmImage* col_imgs[SM_MAX_MRT], SmImage* dep_img,
                                    uint32_t W, uint32_t H, uint32_t clear_mask,
                                    const uint32_t clear_words[SM_MAX_MRT][4], float clear_depth, uint32_t clear_stencil = 0u);
static void sm_sync_attach_to_image(SmCommandBuffer* scb, SmImage* img, VkImageLayout target_layout, uint32_t mip = 0, uint32_t layer = 0);
static void pack_clear_words(const VkClearColorValue& cv, VkFormat fmt, uint32_t out[4]);
static void sm_execute_draw_core(SmCommandBuffer* scb, uint32_t vert_or_idx_count, uint32_t instance_count,
                                 uint32_t first_vert_or_idx, int32_t vertex_offset, uint32_t first_instance,
                                 bool indexed, SmBuffer* indirect_buf, VkDeviceSize indirect_offset,
                                 SmBuffer* count_buf = nullptr, VkDeviceSize count_offset = 0, uint32_t draw_idx = 0);

static VKAPI_ATTR void VKAPI_CALL sm_CmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageResolve* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* si = from_h<SmImage>(s);
    auto* di = from_h<SmImage>(d);
    if (!si || !di) return;
    if (si->attach_buf) {
        sm_ensure_image_attach_buf(scb->dev, di);
        uint32_t bpp = si->is_depth ? 4u : sm_format_bpp(si->format);
        VkDeviceSize sz = (VkDeviceSize)std::min(si->extent.width, di->extent.width) * std::min(si->extent.height, di->extent.height) * bpp;
        if (sz > 0 && si->format == di->format && si->extent.width == di->extent.width && si->extent.height == di->extent.height) {
            VkBufferCopy bc{0, 0, sz};
            scb->dev->fn.CmdCopyBuffer(scb->real_cb, si->attach_buf, di->attach_buf, 1, &bc);
            sm_sync_attach_to_image(scb, di, dl);
            return;
        }
    }
    if (si->real_img && di->real_img && n > 0 && r) {
        VkImageCopy ic{r[0].srcSubresource, r[0].srcOffset, r[0].dstSubresource, r[0].dstOffset, r[0].extent};
        scb->dev->fn.CmdCopyImage(scb->real_cb, si->real_img, sanitize_layout(sl), di->real_img, sanitize_layout(dl), 1, &ic);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* ci) {
    if (!ci || ci->regionCount == 0) return;
    std::vector<VkImageResolve> r(ci->regionCount);
    for (uint32_t i = 0; i < ci->regionCount; ++i)
        r[i] = {ci->pRegions[i].srcSubresource, ci->pRegions[i].srcOffset, ci->pRegions[i].dstSubresource, ci->pRegions[i].dstOffset, ci->pRegions[i].extent};
    sm_CmdResolveImage(cb, ci->srcImage, ci->srcImageLayout, ci->dstImage, ci->dstImageLayout, ci->regionCount, r.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdClearAttachments(VkCommandBuffer cb, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t, const VkClearRect*) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (!pAttachments || attachmentCount == 0) return;
    SmImage* col_imgs[SM_MAX_MRT] = {nullptr};
    SmImage* dep_img = nullptr;
    uint32_t W = 0, H = 0;
    if (scb->in_dyn_rendering) {
        for (uint32_t i = 0; i < scb->dyn_col_count && i < SM_MAX_MRT; ++i) col_imgs[i] = scb->dyn_col_imgs[i];
        dep_img = scb->dyn_dep_img; W = scb->dyn_w; H = scb->dyn_h;
    } else if (scb->cur_rp && scb->cur_fb) {
        W = scb->cur_fb->width; H = scb->cur_fb->height;
        for (uint32_t i = 0; i < scb->cur_rp->color_attach_count && i < SM_MAX_MRT; ++i) {
            int idx = scb->cur_rp->color_attach_idxs[i];
            if (idx >= 0 && idx < (int)scb->cur_fb->attachments.size()) col_imgs[i] = scb->cur_fb->attachments[idx]->image;
        }
        if (scb->cur_rp->depth_attach_idx >= 0 && scb->cur_rp->depth_attach_idx < (int)scb->cur_fb->attachments.size())
            dep_img = scb->cur_fb->attachments[scb->cur_rp->depth_attach_idx]->image;
    } else return;
    if (W == 0 || H == 0) return;

    uint32_t clear_mask = 0;
    uint32_t clear_words[SM_MAX_MRT][4] = {{0}};
    float clear_depth = 1.0f;
    uint32_t clear_stencil = 0u;
    for (uint32_t i = 0; i < attachmentCount; ++i) {
        const auto& a = pAttachments[i];
        if ((a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) && a.colorAttachment < SM_MAX_MRT && col_imgs[a.colorAttachment]) {
            clear_mask |= (1u << a.colorAttachment);
            pack_clear_words(a.clearValue.color, col_imgs[a.colorAttachment]->format, clear_words[a.colorAttachment]);
        }
        if ((a.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) && dep_img) {
            clear_mask |= 256u;
            clear_depth = a.clearValue.depthStencil.depth;
        }
        if ((a.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) && dep_img && dep_img->has_stencil) {
            clear_mask |= 1024u;
            clear_stencil = a.clearValue.depthStencil.stencil & 0xFFu;
        }
    }
    if (clear_mask) sm_begin_pass_clear_mrt(scb, col_imgs, dep_img, W, H, clear_mask, clear_words, clear_depth, clear_stencil);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndexedIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkBuffer countBuf, VkDeviceSize countOff, uint32_t maxDrawCount, uint32_t stride) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* sb = from_h<SmBuffer>(buf);
    auto* ctb = from_h<SmBuffer>(countBuf);
    uint32_t st = stride ? stride : sizeof(VkDrawIndirectCommand);
    uint32_t n = std::min(maxDrawCount, 64u);
    for (uint32_t i = 0; i < n; ++i)
        sm_execute_draw_core(scb, 0, 1, 0, 0, 0, false, sb, off + (VkDeviceSize)i * st, ctb, countOff, i);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkBuffer countBuf, VkDeviceSize countOff, uint32_t maxDrawCount, uint32_t stride) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* sb = from_h<SmBuffer>(buf);
    auto* ctb = from_h<SmBuffer>(countBuf);
    uint32_t st = stride ? stride : sizeof(VkDrawIndexedIndirectCommand);
    uint32_t n = std::min(maxDrawCount, 64u);
    for (uint32_t i = 0; i < n; ++i)
        sm_execute_draw_core(scb, 0, 1, 0, 0, 0, true, sb, off + (VkDeviceSize)i * st, ctb, countOff, i);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetEvent(VkCommandBuffer cb, VkEvent ev, VkPipelineStageFlags stage) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* dev = scb->dev;
    auto fn = (PFN_vkCmdSetEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkCmdSetEvent");
    if (fn) fn(scb->real_cb, ev, sanitize_stage_mask(stage));
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdResetEvent(VkCommandBuffer cb, VkEvent ev, VkPipelineStageFlags stage) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* dev = scb->dev;
    auto fn = (PFN_vkCmdResetEvent)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkCmdResetEvent");
    if (fn) fn(scb->real_cb, ev, sanitize_stage_mask(stage));
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdWaitEvents(VkCommandBuffer cb, uint32_t, const VkEvent*, VkPipelineStageFlags, VkPipelineStageFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    scb->dev->fn.CmdPipelineBarrier(scb->real_cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetEvent2(VkCommandBuffer cb, VkEvent ev, const VkDependencyInfo*) {
    sm_CmdSetEvent(cb, ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdResetEvent2(VkCommandBuffer cb, VkEvent ev, VkPipelineStageFlags2) {
    sm_CmdResetEvent(cb, ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdWaitEvents2(VkCommandBuffer cb, uint32_t, const VkEvent*, const VkDependencyInfo* pDependencyInfos) {
    if (pDependencyInfos) {
        // Ensure memory/image barrier side-effects inside WaitEvents2 still execute cleanly!
        auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
        scb->dev->fn.CmdPipelineBarrier(scb->real_cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginQueryIndexedEXT(VkCommandBuffer, VkQueryPool, uint32_t, VkQueryControlFlags, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndQueryIndexedEXT(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {}
static VKAPI_ATTR VkResult VKAPI_CALL sm_MergePipelineCaches(VkDevice, VkPipelineCache, uint32_t, const VkPipelineCache*) { return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceMemoryCommitment(VkDevice, VkDeviceMemory, VkDeviceSize* pBytes) { if (pBytes) *pBytes = 0; }
static VKAPI_ATTR void VKAPI_CALL sm_GetRenderAreaGranularity(VkDevice, VkRenderPass, VkExtent2D* g) { if (g) *g = {16, 16}; }
static VKAPI_ATTR void VKAPI_CALL sm_GetImageSparseMemoryRequirements(VkDevice, VkImage, uint32_t* c, VkSparseImageMemoryRequirements*) { if (c) *c = 0; }
static VKAPI_ATTR void VKAPI_CALL sm_GetImageSparseMemoryRequirements2(VkDevice, const VkImageSparseMemoryRequirementsInfo2*, uint32_t* c, VkSparseImageMemoryRequirements2*) { if (c) *c = 0; }
static VKAPI_ATTR VkResult VKAPI_CALL sm_QueueBindSparse(VkQueue q, uint32_t, const VkBindSparseInfo*, VkFence fence) {
    if (fence != VK_NULL_HANDLE) {
        auto* sq = reinterpret_cast<SmQueue*>(q);
        sq->dev->fn.QueueSubmit(sq->real_q, 0, nullptr, fence);
    }
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_SetDebugUtilsObjectNameEXT(VkDevice, const VkDebugUtilsObjectNameInfoEXT*) { return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL sm_SetDebugUtilsObjectTagEXT(VkDevice, const VkDebugUtilsObjectTagInfoEXT*) { return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginDebugUtilsLabelEXT(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndDebugUtilsLabelEXT(VkCommandBuffer) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdInsertDebugUtilsLabelEXT(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
static VKAPI_ATTR void VKAPI_CALL sm_QueueBeginDebugUtilsLabelEXT(VkQueue, const VkDebugUtilsLabelEXT*) {}
static VKAPI_ATTR void VKAPI_CALL sm_QueueEndDebugUtilsLabelEXT(VkQueue) {}
static VKAPI_ATTR void VKAPI_CALL sm_QueueInsertDebugUtilsLabelEXT(VkQueue, const VkDebugUtilsLabelEXT*) {}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateDebugUtilsMessengerEXT(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT*, const VkAllocationCallbacks*, VkDebugUtilsMessengerEXT* m) {
    if (m) *m = (VkDebugUtilsMessengerEXT)(uintptr_t)0xDEADBEEF;
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyDebugUtilsMessengerEXT(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks*) {}
static VKAPI_ATTR void VKAPI_CALL sm_SubmitDebugUtilsMessageEXT(VkInstance, VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdExecuteCommands(VkCommandBuffer, uint32_t, const VkCommandBuffer*) {}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceBufferMemoryRequirements(VkDevice d, const VkDeviceBufferMemoryRequirements* info, VkMemoryRequirements2* mr) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetDeviceBufferMemoryRequirements)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetDeviceBufferMemoryRequirements");
    VkBufferCreateInfo bci = *info->pCreateInfo;
    for (const auto* n = reinterpret_cast<const VkBaseInStructure*>(bci.pNext); n; n = n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR) {
            bci.usage |= (VkBufferUsageFlags)reinterpret_cast<const VkBufferUsageFlags2CreateInfoKHR*>(n)->usage;
            bci.pNext = nullptr;
        }
    }
    bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; bci.queueFamilyIndexCount = 0; bci.pQueueFamilyIndices = nullptr;
    VkDeviceBufferMemoryRequirements m = *info; m.pCreateInfo = &bci;
    if (fn) fn(dev->real_dev, &m, mr);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceImageMemoryRequirements(VkDevice d, const VkDeviceImageMemoryRequirements* info, VkMemoryRequirements2* mr) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetDeviceImageMemoryRequirements)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetDeviceImageMemoryRequirements");
    VkImageCreateInfo ici = *info->pCreateInfo;
    ici.usage = (ici.usage & ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT))
              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    if (is_depth_format(ici.format)) { ici.format = VK_FORMAT_R32_SFLOAT; ici.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; ici.pNext = nullptr; ici.usage |= VK_IMAGE_USAGE_SAMPLED_BIT; }
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ici.queueFamilyIndexCount = 0; ici.pQueueFamilyIndices = nullptr;
    VkDeviceImageMemoryRequirements m = *info; m.pCreateInfo = &ici;
    if (fn) fn(dev->real_dev, &m, mr);
}

static void emit_compute_barrier(SmCommandBuffer* scb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    scb->dev->fn.CmdPipelineBarrier(scb->real_cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}
static uint32_t f32_to_uf11_host(float v) {
    if (!(v > 0.0f)) return 0u;
    uint32_t u; memcpy(&u, &v, 4);
    uint32_t e = (u >> 23) & 255u;
    if (e < 112u) return 0u;
    if (e >= 143u) return 0x7BFu;
    return ((e - 112u) << 6) | ((u >> 17) & 63u);
}
static uint32_t f32_to_uf10_host(float v) {
    if (!(v > 0.0f)) return 0u;
    uint32_t u; memcpy(&u, &v, 4);
    uint32_t e = (u >> 23) & 255u;
    if (e < 112u) return 0u;
    if (e >= 143u) return 0x3DFu;
    return ((e - 112u) << 5) | ((u >> 18) & 31u);
}
static void pack_clear_words(const VkClearColorValue& cv, VkFormat fmt, uint32_t out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0;
    uint32_t fc = sm_format_code(fmt);
    const float* f = cv.float32;
    auto cl01 = [](float x){ return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); };
    auto cl11 = [](float x){ return x < -1.f ? -1.f : (x > 1.f ? 1.f : x); };
    auto sn8  = [&](float x){ return (uint32_t)(uint8_t)(int8_t)(cl11(x) * 127.f); };
    auto un16 = [&](float x){ return (uint32_t)(uint16_t)(cl01(x) * 65535.f + 0.5f); };
    auto sn16 = [&](float x){ return (uint32_t)(uint16_t)(int16_t)(cl11(x) * 32767.f); };

    if (fc == 4u) { // RGBA16F
        out[0] = (uint32_t)f32_to_f16(f[0]) | ((uint32_t)f32_to_f16(f[1]) << 16);
        out[1] = (uint32_t)f32_to_f16(f[2]) | ((uint32_t)f32_to_f16(f[3]) << 16);
    } else if (fc == 5u) { // R11G11B10F
        out[0] = f32_to_uf11_host(f[0]) | (f32_to_uf11_host(f[1]) << 11) | (f32_to_uf10_host(f[2]) << 22);
    } else if (fc == 6u) { // R32F
        memcpy(&out[0], &f[0], 4);
    } else if (fc == 7u) { // RG16F
        out[0] = (uint32_t)f32_to_f16(f[0]) | ((uint32_t)f32_to_f16(f[1]) << 16);
    } else if (fc == 8u) { // RGBA32F
        memcpy(out, f, 16);
    } else if (fc == 10u) { // RG32F
        memcpy(out, f, 8);
    } else if (fc == 11u) { // R32_UINT / R32_SINT
        out[0] = cv.uint32[0];
    } else if (fc == 9u) { // RGB10A2
        out[0] = (uint32_t)(cl01(f[0])*1023.f+0.5f) | ((uint32_t)(cl01(f[1])*1023.f+0.5f)<<10)
               | ((uint32_t)(cl01(f[2])*1023.f+0.5f)<<20) | ((uint32_t)(cl01(f[3])*3.f+0.5f)<<30);
    } else if (fc == 12u) { // RGBA8_SNORM
        out[0] = sn8(f[0]) | (sn8(f[1]) << 8) | (sn8(f[2]) << 16) | (sn8(f[3]) << 24);
    } else if (fc == 13u) { // RG16_UNORM
        out[0] = un16(f[0]) | (un16(f[1]) << 16);
    } else if (fc == 14u) { // RG16_SNORM
        out[0] = sn16(f[0]) | (sn16(f[1]) << 16);
    } else if (fc == 15u) { // RGBA16_UNORM
        out[0] = un16(f[0]) | (un16(f[1]) << 16);
        out[1] = un16(f[2]) | (un16(f[3]) << 16);
    } else if (fc == 16u) { // RGBA16_SNORM
        out[0] = sn16(f[0]) | (sn16(f[1]) << 16);
        out[1] = sn16(f[2]) | (sn16(f[3]) << 16);
    } else if (fc == 17u) { // R8_UNORM / R8_UINT (replicate 1B x 4)
        uint32_t b = (fmt == VK_FORMAT_R8_UINT || fmt == VK_FORMAT_R8_SINT)
                     ? (cv.uint32[0] & 0xFFu) : ((uint32_t)(cl01(f[0]) * 255.f + 0.5f) & 0xFFu);
        out[0] = b | (b << 8) | (b << 16) | (b << 24);
    } else if (fc == 18u) { // RG8_UNORM / RG8_SNORM (replicate 2B x 2)
        uint32_t w = (fmt == VK_FORMAT_R8G8_SNORM)
                     ? (sn8(f[0]) | (sn8(f[1]) << 8))
                     : (((uint32_t)(cl01(f[0])*255.f+0.5f) & 0xFFu) | (((uint32_t)(cl01(f[1])*255.f+0.5f) & 0xFFu) << 8));
        out[0] = w | (w << 16);
    } else if (fc == 19u) { // R16_SFLOAT (replicate 2B x 2)
        uint32_t w = (uint32_t)f32_to_f16(f[0]);
        out[0] = w | (w << 16);
    } else if (fc == 20u) { // R16_UNORM / R16_UINT (replicate 2B x 2)
        uint32_t w = (fmt == VK_FORMAT_R16_UINT || fmt == VK_FORMAT_R16_SINT)
                     ? (cv.uint32[0] & 0xFFFFu) : un16(f[0]);
        out[0] = w | (w << 16);
    } else {
        out[0] = pack_clear_color(cv, fmt);
    }
}

// =============================================================================
// Core RenderPass / Dynamic Rendering (8x MRT + 21 Formats + 8-bit Stencil!)
// =============================================================================
static void sm_begin_pass_clear_mrt(SmCommandBuffer* scb, SmImage* col_imgs[SM_MAX_MRT], SmImage* dep_img,
                                    uint32_t W, uint32_t H, uint32_t clear_mask,
                                    const uint32_t clear_words[SM_MAX_MRT][4], float clear_depth, uint32_t clear_stencil) {
    auto* dev = scb->dev;
    sm_ensure_vis_size(dev, W, H);
    for (uint32_t i = 0; i < SM_MAX_MRT; ++i) {
        if (col_imgs[i]) {
            sm_ensure_image_attach_buf(dev, col_imgs[i]);
            if (clear_mask & (1u << i)) col_imgs[i]->attach_dirty = true;
        }
    }
    if (dep_img) {
        sm_ensure_image_attach_buf(dev, dep_img);
        if (dep_img->has_stencil) sm_ensure_image_stencil_buf(dev, dep_img);
        if (clear_mask & 256u) dep_img->attach_dirty = true;
    }

    uint32_t cslot = 0;
    {
        std::lock_guard<std::mutex> lk(dev->mtx);
        cslot = dev->rast.clear_cursor % SmRasterEngine::MAX_AUX_SLOTS;
        dev->rast.clear_cursor++;

        auto* cubo = reinterpret_cast<SmClearUBO*>((uint8_t*)dev->rast.clr_ubo_ptr + cslot * sizeof(SmClearUBO));
        cubo->width = W;
        cubo->height = H;
        cubo->clear_mask = clear_mask;
        cubo->clear_depth = clear_depth;
        cubo->clear_stencil = clear_stencil & 0xFFu;
        cubo->pad0 = cubo->pad1 = cubo->pad2 = 0u;
        for (uint32_t i = 0; i < SM_MAX_MRT; ++i) {
            cubo->rt_bpp_code[i] = col_imgs[i] ? sm_format_bpp_code(col_imgs[i]->format) : 0u;
            for (uint32_t k = 0; k < 4; ++k) cubo->rt_clear[i][k] = clear_words[i][k];
        }

        VkDescriptorBufferInfo bi[12];
        for (uint32_t i = 0; i < SM_MAX_MRT; ++i) {
            bi[i] = {col_imgs[i] ? col_imgs[i]->attach_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE};
        }
        bi[8]  = {dep_img ? dep_img->attach_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE};
        bi[9]  = {dev->rast.vis_buf, 0, VK_WHOLE_SIZE};
        bi[10] = {(dep_img && dep_img->stencil_buf) ? dep_img->stencil_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE};
        bi[11] = {dev->rast.clr_ubo_buf, cslot * sizeof(SmClearUBO), sizeof(SmClearUBO)};
        VkWriteDescriptorSet ws[12];
        for (uint32_t i = 0; i < 12; ++i) {
            ws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.clear_ds[cslot], i, 0, 1,
                     (i == 11) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i]};
        }
        dev->fn.UpdateDescriptorSets(dev->real_dev, 12, ws, 0, nullptr);
    }

    dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.clear_pipe);
    dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.clear_pl, 0, 1, &dev->rast.clear_ds[cslot], 0, nullptr);
    dev->fn.CmdDispatch(scb->real_cb, (W + 15) / 16, (H + 15) / 16, 1);
    scb->comp_state_dirty = true;
    emit_compute_barrier(scb);
}

static void sm_sync_image_to_attach(SmCommandBuffer* scb, SmImage* img, VkImageLayout cur_layout, uint32_t mip = 0, uint32_t layer = 0) {
    if (!img || !img->real_img || img->extent.width == 0 || img->extent.height == 0) return;
    sm_ensure_image_attach_buf(scb->dev, img);
    auto* dev = scb->dev;
    uint32_t mw = std::max(1u, img->extent.width >> mip);
    uint32_t mh = std::max(1u, img->extent.height >> mip);
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        sanitize_layout(cur_layout), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img->real_img,
        {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer, 1}};
    dev->fn.CmdPipelineBarrier(scb->real_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, nullptr, 0, nullptr, 1, &bar);
    VkBufferImageCopy bic{};
    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
    bic.imageExtent = {mw, mh, 1};
    dev->fn.CmdCopyImageToBuffer(scb->real_cb, img->real_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img->attach_buf, 1, &bic);
    bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; bar.newLayout = sanitize_layout(cur_layout);
    dev->fn.CmdPipelineBarrier(scb->real_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, nullptr, 0, nullptr, 1, &bar);
    emit_compute_barrier(scb);
    img->attach_dirty = true;
    img->attach_mip = (int32_t)mip;
    img->attach_layer = (int32_t)layer;
}

static void sm_sync_attach_to_image(SmCommandBuffer* scb, SmImage* img, VkImageLayout target_layout, uint32_t mip, uint32_t layer) {
    if (!img || img->is_swapchain || !img->real_img || !img->attach_buf) return;
    auto* dev = scb->dev;
    uint32_t mw = std::max(1u, img->extent.width >> mip);
    uint32_t mh = std::max(1u, img->extent.height >> mip);
    emit_compute_barrier(scb);
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img->real_img,
        {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer, 1}};
    dev->fn.CmdPipelineBarrier(scb->real_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, nullptr, 0, nullptr, 1, &bar);
    VkBufferImageCopy bic{};
    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
    bic.imageExtent = {mw, mh, 1};
    dev->fn.CmdCopyBufferToImage(scb->real_cb, img->attach_buf, img->real_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; bar.newLayout = sanitize_layout(target_layout);
    dev->fn.CmdPipelineBarrier(scb->real_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, nullptr, 0, nullptr, 1, &bar);
}

static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginRenderPass(
        VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, VkSubpassContents) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* rp  = from_h<SmRenderPass>(bi->renderPass);
    auto* fb  = from_h<SmFramebuffer>(bi->framebuffer);
    scb->in_render_pass = true;
    scb->in_dyn_rendering = false;
    scb->cur_rp = rp;
    scb->cur_fb = fb;

    SmImage* col_imgs[SM_MAX_MRT] = {nullptr};
    uint32_t clear_words[SM_MAX_MRT][4] = {{0}};
    uint32_t clear_mask = 512u; // always clear VisBuf
    float clear_depth = 1.0f;
    uint32_t clear_stencil = 0u;

    for (uint32_t i = 0; i < rp->color_attach_count && i < SM_MAX_MRT; ++i) {
        int idx = rp->color_attach_idxs[i];
        if (idx >= 0 && idx < (int)fb->attachments.size()) {
            col_imgs[i] = fb->attachments[idx]->image;
            if (col_imgs[i] && rp->attachments[idx].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR &&
                bi->clearValueCount > (uint32_t)idx) {
                clear_mask |= (1u << i);
                pack_clear_words(bi->pClearValues[idx].color, col_imgs[i]->format, clear_words[i]);
            }
        }
    }
    SmImage* dep_img = (rp->depth_attach_idx >= 0 && rp->depth_attach_idx < (int)fb->attachments.size())
                       ? fb->attachments[rp->depth_attach_idx]->image : nullptr;
    if (dep_img && bi->clearValueCount > (uint32_t)rp->depth_attach_idx) {
        if (rp->attachments[rp->depth_attach_idx].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
            clear_mask |= 256u;
            clear_depth = bi->pClearValues[rp->depth_attach_idx].depthStencil.depth;
        }
        if (dep_img->has_stencil && rp->attachments[rp->depth_attach_idx].stencilLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
            clear_mask |= 1024u;
            clear_stencil = bi->pClearValues[rp->depth_attach_idx].depthStencil.stencil & 0xFFu;
        }
    }
    SM_XRAY("BeginRenderPass(%ux%u, nc=%u, fmt0=%d, dep=%d, cmask=0x%x)",
            fb->width, fb->height, rp->color_attach_count,
            col_imgs[0] ? (int)col_imgs[0]->format : -1,
            dep_img ? (int)dep_img->format : -1, clear_mask);
    sm_begin_pass_clear_mrt(scb, col_imgs, dep_img, fb->width, fb->height, clear_mask, clear_words, clear_depth, clear_stencil);
}

static VKAPI_ATTR void VKAPI_CALL sm_CmdEndRenderPass(VkCommandBuffer cb) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (scb->cur_rp && scb->cur_fb) {
        for (uint32_t i = 0; i < scb->cur_rp->color_attach_count && i < SM_MAX_MRT; ++i) {
            int idx = scb->cur_rp->color_attach_idxs[i];
            if (idx >= 0 && idx < (int)scb->cur_fb->attachments.size()) {
                SmImage* col = scb->cur_fb->attachments[idx]->image;
                VkImageLayout fl = scb->cur_rp->attachments[idx].finalLayout;
                sm_sync_attach_to_image(scb, col, fl);
            }
        }
        if (scb->cur_rp->depth_attach_idx >= 0 && scb->cur_rp->depth_attach_idx < (int)scb->cur_fb->attachments.size()) {
            SmImage* dep = scb->cur_fb->attachments[scb->cur_rp->depth_attach_idx]->image;
            if (dep && (dep->usage & VK_IMAGE_USAGE_SAMPLED_BIT))
                sm_sync_attach_to_image(scb, dep, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    scb->in_render_pass = false;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateRenderPass2(
        VkDevice d, const VkRenderPassCreateInfo2* ci, const VkAllocationCallbacks* a, VkRenderPass* out) {
    auto* rp = new SmRenderPass();
    for (uint32_t i = 0; i < ci->attachmentCount; ++i) {
        VkAttachmentDescription ad{};
        ad.flags = ci->pAttachments[i].flags;
        ad.format = ci->pAttachments[i].format;
        ad.samples = ci->pAttachments[i].samples;
        ad.loadOp = ci->pAttachments[i].loadOp;
        ad.storeOp = ci->pAttachments[i].storeOp;
        ad.stencilLoadOp = ci->pAttachments[i].stencilLoadOp;
        ad.stencilStoreOp = ci->pAttachments[i].stencilStoreOp;
        ad.initialLayout = ci->pAttachments[i].initialLayout;
        ad.finalLayout = ci->pAttachments[i].finalLayout;
        rp->attachments.push_back(ad);
        if (is_depth_format(ad.format)) {
            if (rp->depth_attach_idx < 0) rp->depth_attach_idx = (int)i;
        } else if (rp->color_attach_count < SM_MAX_MRT) {
            rp->color_attach_idxs[rp->color_attach_count++] = (int)i;
        }
    }
    (void)d; (void)a;
    *out = to_h<VkRenderPass>(rp);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, const VkSubpassBeginInfo*) {
    sm_CmdBeginRenderPass(cb, bi, VK_SUBPASS_CONTENTS_INLINE);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo*) {
    sm_CmdEndRenderPass(cb);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdNextSubpass2(VkCommandBuffer, const VkSubpassBeginInfo*, const VkSubpassEndInfo*) {}

// VK_KHR_dynamic_rendering (supports up to 4 MRT Color Attachments + HDR + Mip/Layer Views!)
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->in_render_pass = false;
    scb->in_dyn_rendering = true;
    for (uint32_t i = 0; i < SM_MAX_MRT; ++i) { scb->dyn_col_imgs[i] = nullptr; scb->dyn_col_views[i] = nullptr; scb->dyn_res_imgs[i] = nullptr; }
    scb->dyn_col_count = 0;
    scb->dyn_dep_img = nullptr;
    scb->dyn_dep_view = nullptr;
    scb->dyn_w = ri->renderArea.offset.x + ri->renderArea.extent.width;
    scb->dyn_h = ri->renderArea.offset.y + ri->renderArea.extent.height;

    uint32_t clear_words[SM_MAX_MRT][4] = {{0}};
    uint32_t clear_mask = 512u; // always clear VisBuf
    float clear_depth = 1.0f;
    uint32_t clear_stencil = 0u;
    bool got_dim = false;

    uint32_t nc = std::min(ri->colorAttachmentCount, SM_MAX_MRT);
    scb->dyn_col_count = nc;
    for (uint32_t i = 0; i < nc; ++i) {
        if (ri->pColorAttachments && ri->pColorAttachments[i].imageView != VK_NULL_HANDLE) {
            auto* iv = from_h<SmImageView>(ri->pColorAttachments[i].imageView);
            scb->dyn_col_views[i] = iv;
            scb->dyn_col_imgs[i] = iv->image;
            if (ri->pColorAttachments[i].resolveImageView != VK_NULL_HANDLE) {
                scb->dyn_res_imgs[i] = from_h<SmImageView>(ri->pColorAttachments[i].resolveImageView)->image;
            }
            if (scb->dyn_col_imgs[i] && !got_dim) {
                scb->dyn_w = std::max(1u, scb->dyn_col_imgs[i]->extent.width >> iv->base_mip);
                scb->dyn_h = std::max(1u, scb->dyn_col_imgs[i]->extent.height >> iv->base_mip);
                got_dim = true;
            }
            VkFormat eff_fmt = (iv->format != VK_FORMAT_UNDEFINED) ? iv->format : (scb->dyn_col_imgs[i] ? scb->dyn_col_imgs[i]->format : VK_FORMAT_R8G8B8A8_UNORM);
            if (scb->dyn_col_imgs[i]) {
                bool sub_changed = (scb->dyn_col_imgs[i]->attach_mip != (int32_t)iv->base_mip ||
                                    scb->dyn_col_imgs[i]->attach_layer != (int32_t)iv->base_layer);
                if (ri->pColorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    clear_mask |= (1u << i);
                    pack_clear_words(ri->pColorAttachments[i].clearValue.color, eff_fmt, clear_words[i]);
                    scb->dyn_col_imgs[i]->attach_mip = (int32_t)iv->base_mip;
                    scb->dyn_col_imgs[i]->attach_layer = (int32_t)iv->base_layer;
                } else if (ri->pColorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD &&
                           !scb->dyn_col_imgs[i]->is_swapchain &&
                           (!scb->dyn_col_imgs[i]->attach_dirty || sub_changed) &&
                           scb->dyn_col_imgs[i]->real_img) {
                    sm_sync_image_to_attach(scb, scb->dyn_col_imgs[i], ri->pColorAttachments[i].imageLayout, iv->base_mip, iv->base_layer);
                } else {
                    scb->dyn_col_imgs[i]->attach_mip = (int32_t)iv->base_mip;
                    scb->dyn_col_imgs[i]->attach_layer = (int32_t)iv->base_layer;
                }
            }
        }
    }
    if (ri->pDepthAttachment && ri->pDepthAttachment->imageView != VK_NULL_HANDLE) {
        auto* iv = from_h<SmImageView>(ri->pDepthAttachment->imageView);
        scb->dyn_dep_view = iv;
        scb->dyn_dep_img = iv->image;
        if (!got_dim && scb->dyn_dep_img) {
            scb->dyn_w = std::max(1u, scb->dyn_dep_img->extent.width >> iv->base_mip);
            scb->dyn_h = std::max(1u, scb->dyn_dep_img->extent.height >> iv->base_mip);
            got_dim = true;
        }
        if (scb->dyn_dep_img) {
            bool sub_changed = (scb->dyn_dep_img->attach_mip != (int32_t)iv->base_mip ||
                                scb->dyn_dep_img->attach_layer != (int32_t)iv->base_layer);
            if (ri->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                clear_mask |= 256u;
                clear_depth = ri->pDepthAttachment->clearValue.depthStencil.depth;
                scb->dyn_dep_img->attach_mip = (int32_t)iv->base_mip;
                scb->dyn_dep_img->attach_layer = (int32_t)iv->base_layer;
            } else if (ri->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD &&
                       (!scb->dyn_dep_img->attach_dirty || sub_changed) &&
                       scb->dyn_dep_img->real_img) {
                sm_sync_image_to_attach(scb, scb->dyn_dep_img, ri->pDepthAttachment->imageLayout, iv->base_mip, iv->base_layer);
            } else {
                scb->dyn_dep_img->attach_mip = (int32_t)iv->base_mip;
                scb->dyn_dep_img->attach_layer = (int32_t)iv->base_layer;
            }
        }
    }
    if (ri->pStencilAttachment && ri->pStencilAttachment->imageView != VK_NULL_HANDLE) {
        auto* iv = from_h<SmImageView>(ri->pStencilAttachment->imageView);
        if (!scb->dyn_dep_img) {
            scb->dyn_dep_view = iv;
            scb->dyn_dep_img = iv->image;
            if (!got_dim && scb->dyn_dep_img) {
                scb->dyn_w = std::max(1u, scb->dyn_dep_img->extent.width >> iv->base_mip);
                scb->dyn_h = std::max(1u, scb->dyn_dep_img->extent.height >> iv->base_mip);
                got_dim = true;
            }
        }
        if (scb->dyn_dep_img) scb->dyn_dep_img->has_stencil = true;
        if (ri->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
            clear_mask |= 1024u;
            clear_stencil = ri->pStencilAttachment->clearValue.depthStencil.stencil & 0xFFu;
        }
    }
    if (scb->dyn_w == 0) scb->dyn_w = 1;
    if (scb->dyn_h == 0) scb->dyn_h = 1;
    SM_TRACE("BeginRendering(%ux%u, nc=%u, fmt0=%d, mip0=%u, sc0=%d, dep=%d, cmask=0x%x, area=%ux%u)",
             scb->dyn_w, scb->dyn_h, scb->dyn_col_count,
             scb->dyn_col_views[0] ? (int)scb->dyn_col_views[0]->format : -1,
             scb->dyn_col_views[0] ? scb->dyn_col_views[0]->base_mip : 0u,
             scb->dyn_col_imgs[0] ? (int)scb->dyn_col_imgs[0]->is_swapchain : 0,
             scb->dyn_dep_img ? (int)scb->dyn_dep_img->format : -1,
             clear_mask, ri->renderArea.extent.width, ri->renderArea.extent.height);
    SM_XRAY("BeginRendering(%ux%u nc=%u fmts=[%d,%d,%d,%d,%d,%d,%d,%d] dep=%d stn=%d sc0=%d cmask=0x%x)",
            scb->dyn_w, scb->dyn_h, scb->dyn_col_count,
            scb->dyn_col_imgs[0] ? (int)scb->dyn_col_imgs[0]->format : -1,
            scb->dyn_col_imgs[1] ? (int)scb->dyn_col_imgs[1]->format : -1,
            scb->dyn_col_imgs[2] ? (int)scb->dyn_col_imgs[2]->format : -1,
            scb->dyn_col_imgs[3] ? (int)scb->dyn_col_imgs[3]->format : -1,
            scb->dyn_col_imgs[4] ? (int)scb->dyn_col_imgs[4]->format : -1,
            scb->dyn_col_imgs[5] ? (int)scb->dyn_col_imgs[5]->format : -1,
            scb->dyn_col_imgs[6] ? (int)scb->dyn_col_imgs[6]->format : -1,
            scb->dyn_col_imgs[7] ? (int)scb->dyn_col_imgs[7]->format : -1,
            scb->dyn_dep_img ? (int)scb->dyn_dep_img->format : -1,
            (scb->dyn_dep_img && scb->dyn_dep_img->has_stencil) ? 1 : 0,
            scb->dyn_col_imgs[0] ? (int)scb->dyn_col_imgs[0]->is_swapchain : 0,
            clear_mask);
    sm_begin_pass_clear_mrt(scb, scb->dyn_col_imgs, scb->dyn_dep_img, scb->dyn_w, scb->dyn_h, clear_mask, clear_words, clear_depth, clear_stencil);
}

static VKAPI_ATTR void VKAPI_CALL sm_CmdEndRendering(VkCommandBuffer cb) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (scb->in_dyn_rendering) {
        for (uint32_t i = 0; i < scb->dyn_col_count && i < SM_MAX_MRT; ++i) {
            SmImage* col = scb->dyn_col_imgs[i];
            SmImageView* iv = scb->dyn_col_views[i];
            uint32_t mip = iv ? iv->base_mip : 0u;
            uint32_t lay = iv ? iv->base_layer : 0u;
            if (col) {
                sm_sync_attach_to_image(scb, col, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mip, lay);
                SmImage* res = scb->dyn_res_imgs[i];
                if (res && col->attach_buf) {
                    sm_ensure_image_attach_buf(scb->dev, res);
                    if (col->format == res->format && col->extent.width == res->extent.width && col->extent.height == res->extent.height) {
                        uint32_t bpp = sm_format_bpp(col->format);
                        VkBufferCopy bc{0, 0, (VkDeviceSize)col->extent.width * col->extent.height * bpp};
                        scb->dev->fn.CmdCopyBuffer(scb->real_cb, col->attach_buf, res->attach_buf, 1, &bc);
                        sm_sync_attach_to_image(scb, res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mip, lay);
                    }
                }
            }
        }
        if (scb->dyn_dep_img &&
            ((scb->dyn_dep_img->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) ||
             scb->dyn_dep_img->array_layers > 1 || scb->dyn_dep_img->mip_levels > 1)) {
            uint32_t dmip = scb->dyn_dep_view ? scb->dyn_dep_view->base_mip : 0u;
            uint32_t dlay = scb->dyn_dep_view ? scb->dyn_dep_view->base_layer : 0u;
            sm_sync_attach_to_image(scb, scb->dyn_dep_img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, dmip, dlay);
        }
    }
    scb->in_dyn_rendering = false;
}

static void sm_replay_push_descs(SmDevice* dev, VkCommandBuffer real_cb, VkPipelineLayout layout, uint32_t set, SmPushDescSet& pds) {
    if (!pds.active || pds.writes.empty() || !dev->fn.CmdPushDescriptorSetKHR) return;
    for (size_t i = 0; i < pds.writes.size(); ++i) {
        pds.writes[i].pBufferInfo      = pds.bufs[i].empty() ? nullptr : pds.bufs[i].data();
        pds.writes[i].pImageInfo       = pds.imgs[i].empty() ? nullptr : pds.imgs[i].data();
        pds.writes[i].pTexelBufferView = pds.bvs[i].empty()  ? nullptr : pds.bvs[i].data();
    }
    dev->fn.CmdPushDescriptorSetKHR(real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, set,
                                    (uint32_t)pds.writes.size(), pds.writes.data());
}

static uint32_t sm_pack_stencil_cfg(const VkStencilOpState& st) {
    return ((uint32_t)st.compareOp & 7u) |
           (((uint32_t)st.failOp & 7u) << 3u) |
           (((uint32_t)st.passOp & 7u) << 6u) |
           (((uint32_t)st.depthFailOp & 7u) << 9u) |
           ((st.compareMask & 0xFFu) << 16u) |
           ((st.writeMask & 0xFFu) << 24u);
}

static void sm_execute_draw_core(SmCommandBuffer* scb, uint32_t vert_or_idx_count, uint32_t instance_count,
                                 uint32_t first_vert_or_idx, int32_t vertex_offset, uint32_t first_instance,
                                 bool indexed, SmBuffer* indirect_buf, VkDeviceSize indirect_offset,
                                 SmBuffer* count_buf, VkDeviceSize count_offset, uint32_t draw_idx) {
    auto* dev = scb->dev;
    auto* pipe = scb->cur_gfx_pipe;
    bool is_indirect = (indirect_buf != nullptr);
    if (!pipe || !pipe->vs_comp_pipe) return;
    if (!is_indirect && (vert_or_idx_count < 3 || instance_count == 0)) return;

    SmImage* col_imgs[SM_MAX_MRT] = {nullptr};
    SmImage* dep_img = nullptr;
    uint32_t W = 0, H = 0;
    if (scb->in_dyn_rendering) {
        for (uint32_t i = 0; i < scb->dyn_col_count && i < SM_MAX_MRT; ++i) col_imgs[i] = scb->dyn_col_imgs[i];
        dep_img = scb->dyn_dep_img;
        W = scb->dyn_w; H = scb->dyn_h;
    } else if (scb->cur_rp && scb->cur_fb) {
        auto* rp = scb->cur_rp;
        auto* fb = scb->cur_fb;
        W = fb->width; H = fb->height;
        for (uint32_t i = 0; i < rp->color_attach_count && i < SM_MAX_MRT; ++i) {
            int idx = rp->color_attach_idxs[i];
            if (idx >= 0 && idx < (int)fb->attachments.size()) col_imgs[i] = fb->attachments[idx]->image;
        }
        dep_img = (rp->depth_attach_idx >= 0 && rp->depth_attach_idx < (int)fb->attachments.size())
                  ? fb->attachments[rp->depth_attach_idx]->image : nullptr;
    } else {
        return;
    }

    VkPrimitiveTopology eff_topo = scb->has_dyn_topo ? scb->dyn_topo : pipe->topology;
    uint32_t topo_code = 0;
    uint32_t tris_per_inst = 1, verts_per_inst = 3, total_verts = 0, total_tris = 0;
    if (!is_indirect) {
        tris_per_inst = vert_or_idx_count / 3;
        verts_per_inst = tris_per_inst * 3;
        if (eff_topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) {
            topo_code = 1; tris_per_inst = vert_or_idx_count - 2; verts_per_inst = vert_or_idx_count;
        } else if (eff_topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
            topo_code = 2; tris_per_inst = vert_or_idx_count - 2; verts_per_inst = vert_or_idx_count;
        }
        if (tris_per_inst == 0) return;
        uint32_t max_inst_v = SM_MAX_DRAW_VERTS / verts_per_inst;
        uint32_t max_inst_t = SM_MAX_DRAW_TRIS / tris_per_inst;
        uint32_t eff_inst = std::min(instance_count, std::max(1u, std::min(max_inst_v, max_inst_t)));
        total_verts = std::min(verts_per_inst * eff_inst, SM_MAX_DRAW_VERTS);
        total_tris  = std::min(tris_per_inst  * eff_inst, SM_MAX_DRAW_TRIS);
    } else {
        if (eff_topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) topo_code = 1;
        else if (eff_topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) topo_code = 2;
    }

    uint32_t tx = (W + 15) / 16, ty = (H + 15) / 16;
    VkBuffer dep_buf = dep_img ? dep_img->attach_buf : dev->rast.dummy_buf;
    if (dep_img && dep_img->has_stencil) sm_ensure_image_stencil_buf(dev, dep_img);
    VkBuffer stn_buf = (dep_img && dep_img->stencil_buf) ? dep_img->stencil_buf : dev->rast.dummy_buf;

    VkViewport vp = scb->has_vp ? scb->cur_vp : (pipe->has_static_vp ? pipe->static_vp : VkViewport{0, 0, (float)W, (float)H, 0.0f, 1.0f});
    VkRect2D sc   = scb->has_scissor ? scb->cur_scissor : (pipe->has_static_scissor ? pipe->static_scissor : VkRect2D{{0, 0}, {W, H}});
    SM_TRACE("Draw(v=%u inst=%u idx=%d itype=%d ioff=%llu vb0off=%llu vb0st=%u vp=%.0f,%.0f,%.0f,%.0f sc=%d,%d,%d,%d dt=%d dw=%d dop=%d cull=%u topo=%u pds4=%d pfnPD=%p)",
             vert_or_idx_count, instance_count, (int)indexed, (int)scb->idx_type, (unsigned long long)scb->idx_offset,
             (unsigned long long)scb->vtx_offsets[0], scb->has_dyn_vtx_stride[0] ? scb->dyn_vtx_strides[0] : (pipe->vtx_input.bindings.empty() ? 0u : pipe->vtx_input.bindings[0].stride),
             vp.x, vp.y, vp.width, vp.height, sc.offset.x, sc.offset.y, (int)sc.extent.width, (int)sc.extent.height,
             (int)(scb->has_dyn_depth_test ? scb->dyn_depth_test : pipe->depth_test),
             (int)(scb->has_dyn_depth_write ? scb->dyn_depth_write : pipe->depth_write),
             (int)(scb->has_dyn_depth_op ? scb->dyn_depth_op : pipe->depth_op),
             (unsigned)(scb->has_dyn_cull ? scb->dyn_cull : pipe->cull_mode),
             (unsigned)eff_topo, (int)scb->gfx_push_descs[4].active, (void*)dev->fn.CmdPushDescriptorSetKHR);
    int sc_x0 = std::max(0, sc.offset.x);
    int sc_y0 = std::max(0, sc.offset.y);
    int sc_x1 = std::min((int)W, sc.offset.x + (int)sc.extent.width);
    int sc_y1 = std::min((int)H, sc.offset.y + (int)sc.extent.height);

    VkCullModeFlags eff_cull = scb->has_dyn_cull ? scb->dyn_cull : pipe->cull_mode;
    VkFrontFace eff_ff       = scb->has_dyn_front_face ? scb->dyn_front_face : pipe->front_face;
    bool eff_dtest           = scb->has_dyn_depth_test ? scb->dyn_depth_test : pipe->depth_test;
    bool eff_dwrite          = scb->has_dyn_depth_write ? scb->dyn_depth_write : pipe->depth_write;
    VkCompareOp eff_dop      = scb->has_dyn_depth_op ? scb->dyn_depth_op : pipe->depth_op;
    bool eff_stest           = scb->has_dyn_stencil_test ? scb->dyn_stencil_test : pipe->stencil_test;
    VkStencilOpState eff_sf  = pipe->stencil_front;
    VkStencilOpState eff_sb  = pipe->stencil_back;
    if (scb->has_dyn_stencil_op[0])    { eff_sf.failOp = scb->dyn_stencil_fail[0]; eff_sf.passOp = scb->dyn_stencil_pass[0]; eff_sf.depthFailOp = scb->dyn_stencil_dfail[0]; eff_sf.compareOp = scb->dyn_stencil_cmp[0]; }
    if (scb->has_dyn_stencil_op[1])    { eff_sb.failOp = scb->dyn_stencil_fail[1]; eff_sb.passOp = scb->dyn_stencil_pass[1]; eff_sb.depthFailOp = scb->dyn_stencil_dfail[1]; eff_sb.compareOp = scb->dyn_stencil_cmp[1]; }
    if (scb->has_dyn_stencil_ref[0])   eff_sf.reference   = scb->dyn_stencil_ref[0];
    if (scb->has_dyn_stencil_ref[1])   eff_sb.reference   = scb->dyn_stencil_ref[1];
    if (scb->has_dyn_stencil_cmask[0]) eff_sf.compareMask = scb->dyn_stencil_cmask[0];
    if (scb->has_dyn_stencil_cmask[1]) eff_sb.compareMask = scb->dyn_stencil_cmask[1];
    if (scb->has_dyn_stencil_wmask[0]) eff_sf.writeMask   = scb->dyn_stencil_wmask[0];
    if (scb->has_dyn_stencil_wmask[1]) eff_sb.writeMask   = scb->dyn_stencil_wmask[1];

    SM_XRAY("  Draw(v=%u inst=%u idx=%d ind=%d fs=%d dt=%d dw=%d st=%d bl0=%d wm0=0x%x)",
            vert_or_idx_count, instance_count, (int)indexed, (int)is_indirect,
            pipe->fs_comp_pipe ? 1 : 0, (int)eff_dtest, (int)eff_dwrite, (int)eff_stest,
            (int)pipe->blend_enable[0], pipe->color_write_mask[0]);

    uint32_t slot = 0, islot = 0;
    uint32_t extra_v_base = 0, extra_t_base = 0;
    {
        std::lock_guard<std::mutex> lk(dev->mtx);
        // On-demand geometry buffer growth (keeps startup <15ms while supporting 2M+ tris!)
        uint32_t req_v = is_indirect ? 2097152u : (total_verts + 131072u);
        uint32_t req_t = is_indirect ? 1048576u : (total_tris  + 65536u);
        sm_ensure_geom_capacity(dev, req_v, req_t);
        extra_v_base = (dev->rast.cap_verts > 1000000u) ? (dev->rast.cap_verts - 900000u) : (dev->rast.cap_verts * 3u / 4u);
        extra_t_base = (dev->rast.cap_tris  > 500000u)  ? (dev->rast.cap_tris  - 390000u) : (dev->rast.cap_tris  * 3u / 4u);

        slot = dev->rast.draw_cursor % SmRasterEngine::MAX_DRAWS_IN_FLIGHT;
        dev->rast.draw_cursor++;
        if (is_indirect) {
            islot = dev->rast.ind_cursor % SmRasterEngine::MAX_IND_SLOTS;
            dev->rast.ind_cursor++;
        }

        auto* vs_ubo = reinterpret_cast<SmVsParamsUBO*>((uint8_t*)dev->rast.vs_ubo_ptr + slot * sizeof(SmVsParamsUBO));
        vs_ubo->vertex_count = total_verts;
        vs_ubo->first_vertex = indexed ? 0 : first_vert_or_idx;
        vs_ubo->instance_idx = first_instance;
        vs_ubo->indexed      = indexed ? (scb->idx_type == VK_INDEX_TYPE_UINT16 ? 1u : (scb->idx_type == VK_INDEX_TYPE_UINT8_EXT ? 3u : 2u)) : 0u;
        vs_ubo->first_index  = indexed ? first_vert_or_idx : 0;
        vs_ubo->vertex_offset = vertex_offset;
        vs_ubo->vert_stride_vec4 = SM_VERT_STRIDE_VEC4;
        vs_ubo->idx_buf_offset = (uint32_t)scb->idx_offset;
        vs_ubo->verts_per_instance = verts_per_inst;
        vs_ubo->vtx_inst_rate_mask = 0;
        for (uint32_t b = 0; b < SM_MAX_VTX_BINDINGS; ++b) {
            uint32_t off = (uint32_t)scb->vtx_offsets[b];
            uint32_t st  = 16;
            for (const auto& bd : pipe->vtx_input.bindings) {
                if (bd.binding == b) {
                    st = bd.stride;
                    if (bd.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE)
                        vs_ubo->vtx_inst_rate_mask |= (1u << b);
                }
            }
            if (scb->has_dyn_vtx_stride[b]) st = scb->dyn_vtx_strides[b];
            if (b < 4) { vs_ubo->vtx_offsets[b] = off; vs_ubo->vtx_strides[b] = st; }
            else       { vs_ubo->vtx_offsets_hi[b - 4] = off; vs_ubo->vtx_strides_hi[b - 4] = st; }
        }

        auto* fs_ubo = reinterpret_cast<SmFsParamsUBO*>((uint8_t*)dev->rast.fs_ubo_ptr + slot * sizeof(SmFsParamsUBO));
        fs_ubo->width = W;
        fs_ubo->height = H;
        fs_ubo->vert_stride_vec4 = SM_VERT_STRIDE_VEC4;
        fs_ubo->depth_write = (eff_dwrite && dep_img != nullptr) ? 1u : 0u;
        fs_ubo->rt_active_mask = 0;
        fs_ubo->stencil_enable = (eff_stest ? 1u : 0u) | ((dep_img && dep_img->stencil_buf) ? 2u : 0u);
        fs_ubo->stencil_front_cfg = sm_pack_stencil_cfg(eff_sf);
        fs_ubo->stencil_back_cfg  = sm_pack_stencil_cfg(eff_sb);
        fs_ubo->stencil_refs = (eff_sf.reference & 0xFFu) | ((eff_sb.reference & 0xFFu) << 8u);
        fs_ubo->pad0 = fs_ubo->pad1 = fs_ubo->pad2 = 0u;
        for (int k = 0; k < 4; ++k)
            fs_ubo->blend_const[k] = scb->has_dyn_blend_const ? scb->dyn_blend_const[k] : pipe->blend_constants[k];
        for (uint32_t r = 0; r < SM_MAX_MRT; ++r) {
            if (col_imgs[r] && pipe->color_write_mask[r] != 0u) {
                col_imgs[r]->attach_dirty = true;
                fs_ubo->rt_active_mask |= (1u << r);
                VkFormat rf = (scb->in_dyn_rendering && scb->dyn_col_views[r] && scb->dyn_col_views[r]->format != VK_FORMAT_UNDEFINED)
                              ? scb->dyn_col_views[r]->format : col_imgs[r]->format;
                fs_ubo->rt_fmt[r] = sm_format_code(rf);
            } else {
                fs_ubo->rt_fmt[r] = 0;
            }
            fs_ubo->rt_blend_en[r]   = pipe->blend_enable[r] ? 1u : 0u;
            fs_ubo->rt_src_blend[r]  = (uint32_t)pipe->src_color_blend[r];
            fs_ubo->rt_dst_blend[r]  = (uint32_t)pipe->dst_color_blend[r];
            fs_ubo->rt_blend_op[r]   = (uint32_t)pipe->color_blend_op[r];
            fs_ubo->rt_src_alpha[r]  = (uint32_t)pipe->src_alpha_blend[r];
            fs_ubo->rt_dst_alpha[r]  = (uint32_t)pipe->dst_alpha_blend[r];
            fs_ubo->rt_alpha_op[r]   = (uint32_t)pipe->alpha_blend_op[r];
            fs_ubo->rt_write_mask[r] = pipe->color_write_mask[r] & 0xFu;
        }

        VkDescriptorBufferInfo vbi[11] = {
            {dev->rast.clip_vert_buf, 0, VK_WHOLE_SIZE},
            {(indexed && scb->idx_buf) ? scb->idx_buf->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[0] ? scb->vtx_bufs[0]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[1] ? scb->vtx_bufs[1]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[2] ? scb->vtx_bufs[2]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[3] ? scb->vtx_bufs[3]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[4] ? scb->vtx_bufs[4]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[5] ? scb->vtx_bufs[5]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[6] ? scb->vtx_bufs[6]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {scb->vtx_bufs[7] ? scb->vtx_bufs[7]->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            {dev->rast.vs_ubo_buf, slot * sizeof(SmVsParamsUBO), sizeof(SmVsParamsUBO)},
        };
        VkWriteDescriptorSet vws[11];
        for (uint32_t i = 0; i < 11; ++i)
            vws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.vs_int_ds[slot], i, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vbi[i]};

        VkDescriptorBufferInfo fbi[14];
        fbi[0] = {dev->rast.clip_vert_buf, 0, VK_WHOLE_SIZE};
        fbi[1] = {dev->rast.tri_setup_buf, 0, VK_WHOLE_SIZE};
        fbi[2] = {dev->rast.vis_buf, 0, VK_WHOLE_SIZE};
        for (uint32_t r = 0; r < SM_MAX_MRT; ++r)
            fbi[3 + r] = {col_imgs[r] ? col_imgs[r]->attach_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE};
        fbi[11] = {dep_buf, 0, VK_WHOLE_SIZE};
        fbi[12] = {stn_buf, 0, VK_WHOLE_SIZE};
        fbi[13] = {dev->rast.fs_ubo_buf, slot * sizeof(SmFsParamsUBO), sizeof(SmFsParamsUBO)};
        VkWriteDescriptorSet fws[14];
        for (uint32_t i = 0; i < 14; ++i)
            fws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.fs_int_ds[slot], i, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fbi[i]};

        VkDescriptorBufferInfo bbi[6] = {
            {dev->rast.clip_vert_buf, 0, VK_WHOLE_SIZE},
            {dev->rast.tri_setup_buf, 0, VK_WHOLE_SIZE},
            {dev->rast.tile_count_buf, 0, VK_WHOLE_SIZE},
            {dev->rast.tile_bins_buf, 0, VK_WHOLE_SIZE},
            {dev->rast.vis_buf, 0, VK_WHOLE_SIZE},
            {dep_buf, 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet bws[6];
        for (uint32_t i = 0; i < 6; ++i)
            bws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.bin_ds[slot], i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bbi[i]};

        VkDescriptorBufferInfo fn_bi[5] = {bbi[1], bbi[2], bbi[3], bbi[4], bbi[5]};
        VkWriteDescriptorSet fn_ws[5];
        for (uint32_t i = 0; i < 5; ++i)
            fn_ws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.fine_ds[slot], i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fn_bi[i]};

        dev->fn.UpdateDescriptorSets(dev->real_dev, 11, vws, 0, nullptr);
        if (pipe->fs_comp_pipe) dev->fn.UpdateDescriptorSets(dev->real_dev, 14, fws, 0, nullptr);
        dev->fn.UpdateDescriptorSets(dev->real_dev, 6, bws, 0, nullptr);
        dev->fn.UpdateDescriptorSets(dev->real_dev, 5, fn_ws, 0, nullptr);

        if (!pipe->fs_comp_pipe && eff_dwrite && dep_img != nullptr) {
            VkDescriptorBufferInfo dbi[3] = {
                {dev->rast.tri_setup_buf, 0, VK_WHOLE_SIZE},
                {dev->rast.vis_buf, 0, VK_WHOLE_SIZE},
                {dep_buf, 0, VK_WHOLE_SIZE},
            };
            VkWriteDescriptorSet dws[3];
            for (uint32_t i = 0; i < 3; ++i)
                dws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.depth_only_ds[slot], i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi[i]};
            dev->fn.UpdateDescriptorSets(dev->real_dev, 3, dws, 0, nullptr);
        }

        if (is_indirect) {
            VkDescriptorBufferInfo ibi[5] = {
                {indirect_buf->real_buf, 0, VK_WHOLE_SIZE},
                {dev->rast.vs_ubo_buf, 0, VK_WHOLE_SIZE},
                {dev->rast.tile_count_buf, 0, VK_WHOLE_SIZE},
                {dev->rast.ind_disp_buf, 0, VK_WHOLE_SIZE},
                {count_buf ? count_buf->real_buf : dev->rast.dummy_buf, 0, VK_WHOLE_SIZE},
            };
            VkWriteDescriptorSet iws[5];
            for (uint32_t i = 0; i < 5; ++i)
                iws[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.ind_prep_ds[islot], i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ibi[i]};
            dev->fn.UpdateDescriptorSets(dev->real_dev, 5, iws, 0, nullptr);
        }
    }

    // --- PASS 0: Reset tile_counts (+ clipper & indirect metadata) & VisBuf ---
    dev->fn.CmdFillBuffer(scb->real_cb, dev->rast.tile_count_buf, 0, (VkDeviceSize)(tx * ty + 8) * 4, 0u);
    dev->fn.CmdFillBuffer(scb->real_cb, dev->rast.vis_buf, 0, (VkDeviceSize)W * H * 8, 0xFFFFFFFFu);

    if (is_indirect) {
        emit_compute_barrier(scb);
        struct { uint32_t ind_w_off, vs_w_off, meta_idx, indexed, topo, max_v, max_t, count_w_off, draw_idx, pad; } ipc = {
            (uint32_t)(indirect_offset >> 2),
            (uint32_t)((slot * sizeof(SmVsParamsUBO)) >> 2),
            tx * ty, indexed ? 1u : 0u, topo_code, extra_v_base, extra_t_base,
            count_buf ? (uint32_t)(count_offset >> 2) : 0xFFFFFFFFu, draw_idx, 0u
        };
        dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.ind_prep_pipe);
        dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.ind_prep_pl, 0, 1, &dev->rast.ind_prep_ds[islot], 0, nullptr);
        dev->fn.CmdPushConstants(scb->real_cb, dev->rast.ind_prep_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
        dev->fn.CmdDispatch(scb->real_cb, 1, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
        dev->fn.CmdPipelineBarrier(scb->real_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // --- PASS 1: Lowered Vertex Compute Shader on H20 78 SMs! ---
    dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->vs_comp_pipe);
    for (uint32_t s = 0; s < pipe->layout->sm_set_index && s < 16; ++s) {
        if (scb->gfx_set_mask & (1u << s)) {
            dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout->sm_vs_layout,
                                          s, 1, &scb->gfx_sets[s], scb->gfx_dyn_counts[s],
                                          scb->gfx_dyn_counts[s] ? scb->gfx_dyn_offs[s] : nullptr);
        } else if (scb->gfx_push_descs[s].active) {
            sm_replay_push_descs(dev, scb->real_cb, pipe->layout->sm_vs_layout, s, scb->gfx_push_descs[s]);
        }
    }
    dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout->sm_vs_layout,
                                  pipe->layout->sm_set_index, 1, &dev->rast.vs_int_ds[slot], 0, nullptr);
    if (scb->push_size > 0 && !pipe->layout->push_ranges.empty()) {
        for (const auto& pr : pipe->layout->push_ranges) {
            if (pr.offset < scb->push_size) {
                uint32_t sz = std::min(pr.size, scb->push_size - pr.offset);
                if (sz > 0)
                    dev->fn.CmdPushConstants(scb->real_cb, pipe->layout->sm_vs_layout, VK_SHADER_STAGE_COMPUTE_BIT, pr.offset, sz, scb->push_data + pr.offset);
            }
        }
    }
    if (is_indirect) dev->fn.CmdDispatchIndirect(scb->real_cb, dev->rast.ind_disp_buf, 0);
    else             dev->fn.CmdDispatch(scb->real_cb, (total_verts + 63) / 64, 1, 1);
    emit_compute_barrier(scb);

    // --- PASS 2: CUDARaster Binning + CuRast Micro-Triangle Fast-Path + Near-Plane Clipper ---
    uint32_t rflags = 0;
    if (eff_cull & VK_CULL_MODE_BACK_BIT)  rflags |= 1u;
    if (eff_cull & VK_CULL_MODE_FRONT_BIT) rflags |= 2u;
    if (eff_ff == VK_FRONT_FACE_CLOCKWISE) rflags |= 4u;
    if (eff_dtest && dep_img != nullptr)   rflags |= 8u;
    rflags |= (((uint32_t)eff_dop & 7u) << 4u);
    rflags |= ((topo_code & 3u) << 8u);

    struct {
        uint32_t tri_count, vstride, w, h, tx, ty, max_tpt, flags;
        float vp_x, vp_y, vp_w, vp_h, vp_min_d, vp_max_d;
        int32_t sx0, sy0, sx1, sy1;
        uint32_t tris_per_inst, verts_per_inst, extra_vert_base, extra_tri_base;
    } bpc = { is_indirect ? 0xFFFFFFFFu : total_tris, SM_VERT_STRIDE_VEC4, W, H, tx, ty, SM_MAX_TRIS_PER_TILE, rflags,
              vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth, sc_x0, sc_y0, sc_x1, sc_y1,
              tris_per_inst, verts_per_inst, extra_v_base, extra_t_base };

    dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.bin_pipe);
    dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.bin_pl, 0, 1, &dev->rast.bin_ds[slot], 0, nullptr);
    dev->fn.CmdPushConstants(scb->real_cb, dev->rast.bin_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bpc), &bpc);
    if (is_indirect) dev->fn.CmdDispatchIndirect(scb->real_cb, dev->rast.ind_disp_buf, 16);
    else             dev->fn.CmdDispatch(scb->real_cb, (total_tris + 63) / 64, 1, 1);
    emit_compute_barrier(scb);

    // --- PASS 3: RetroWarp / cuRE 16x16 Tiled Shared-Memory Fine Rasterizer ---
    struct { uint32_t w, h, tx, max_tpt, flags; } fpc = { W, H, tx, SM_MAX_TRIS_PER_TILE, rflags };
    dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.fine_pipe);
    dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.fine_pl, 0, 1, &dev->rast.fine_ds[slot], 0, nullptr);
    dev->fn.CmdPushConstants(scb->real_cb, dev->rast.fine_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fpc), &fpc);
    dev->fn.CmdDispatch(scb->real_cb, tx, ty, 1);
    emit_compute_barrier(scb);

    // --- PASS 4: Lowered Quad-Subgroup 8-MRT + Stencil Fragment Compute Shader (Zero Overdraw + H20 TMUs!) ---
    if (pipe->fs_comp_pipe) {
        dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->fs_comp_pipe);
        for (uint32_t s = 0; s < pipe->layout->sm_set_index && s < 16; ++s) {
            if (scb->gfx_set_mask & (1u << s)) {
                dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout->sm_fs_layout,
                                              s, 1, &scb->gfx_sets[s], scb->gfx_dyn_counts[s],
                                              scb->gfx_dyn_counts[s] ? scb->gfx_dyn_offs[s] : nullptr);
            } else if (scb->gfx_push_descs[s].active) {
                sm_replay_push_descs(dev, scb->real_cb, pipe->layout->sm_fs_layout, s, scb->gfx_push_descs[s]);
            }
        }
        dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout->sm_fs_layout,
                                      pipe->layout->sm_set_index, 1, &dev->rast.fs_int_ds[slot], 0, nullptr);
        if (scb->push_size > 0 && !pipe->layout->push_ranges.empty()) {
            for (const auto& pr : pipe->layout->push_ranges) {
                if (pr.offset < scb->push_size) {
                    uint32_t sz = std::min(pr.size, scb->push_size - pr.offset);
                    if (sz > 0)
                        dev->fn.CmdPushConstants(scb->real_cb, pipe->layout->sm_fs_layout, VK_SHADER_STAGE_COMPUTE_BIT, pr.offset, sz, scb->push_data + pr.offset);
                }
            }
        }
        dev->fn.CmdDispatch(scb->real_cb, tx, ty, 1);
        emit_compute_barrier(scb);
    } else if (eff_dwrite && dep_img != nullptr) {
        // Depth-Only Pass (Shadow Maps / Z-Prepass with fs_mod == nullptr)
        struct { uint32_t w, h, dwrite, pad; } dpc = { W, H, 1u, 0u };
        dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.depth_only_pipe);
        dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.depth_only_pl, 0, 1, &dev->rast.depth_only_ds[slot], 0, nullptr);
        dev->fn.CmdPushConstants(scb->real_cb, dev->rast.depth_only_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpc), &dpc);
        dev->fn.CmdDispatch(scb->real_cb, tx, ty, 1);
        emit_compute_barrier(scb);
    }
    scb->comp_state_dirty = true;
}

static void sm_execute_draw_compute(SmCommandBuffer* scb, uint32_t vert_or_idx_count, uint32_t instance_count,
                                    uint32_t first_vert_or_idx, int32_t vertex_offset, uint32_t first_instance, bool indexed) {
    sm_execute_draw_core(scb, vert_or_idx_count, instance_count, first_vert_or_idx, vertex_offset, first_instance, indexed, nullptr, 0);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* sb = from_h<SmBuffer>(buf);
    uint32_t st = stride ? stride : sizeof(VkDrawIndirectCommand);
    for (uint32_t i = 0; i < drawCount; ++i)
        sm_execute_draw_core(scb, 0, 1, 0, 0, 0, false, sb, offset + (VkDeviceSize)i * st);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* sb = from_h<SmBuffer>(buf);
    uint32_t st = stride ? stride : sizeof(VkDrawIndexedIndirectCommand);
    for (uint32_t i = 0; i < drawCount; ++i)
        sm_execute_draw_core(scb, 0, 1, 0, 0, 0, true, sb, offset + (VkDeviceSize)i * st);
}

static VKAPI_ATTR void VKAPI_CALL sm_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* sp = from_h<SmPipeline>(p);
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        scb->cur_gfx_pipe = sp;
    } else {
        scb->cur_comp_pipe = sp;
        scb->dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, sp->real_compute_pipe);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindDescriptorSets(
        VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t firstSet,
        uint32_t count, const VkDescriptorSet* sets, uint32_t dynCount, const uint32_t* dynOffs) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* pl = from_h<SmPipelineLayout>(layout);
    auto record_sets = [&](VkDescriptorSet dst_sets[16], uint32_t& dst_mask, uint32_t dst_dyn_counts[16],
                           uint32_t dst_dyn_offs[16][SmCommandBuffer::MAX_DYN_OFFS_PER_SET]) {
        uint32_t dyn_cursor = 0;
        for (uint32_t i = 0; i < count && (firstSet + i) < 16; ++i) {
            uint32_t s = firstSet + i;
            dst_sets[s] = sets[i];
            if (sets[i] != VK_NULL_HANDLE) dst_mask |= (1u << s);
            else dst_mask &= ~(1u << s);
            dst_dyn_counts[s] = 0;
            if (dynCount > 0 && dynOffs && pl && s < pl->app_dsls.size()) {
                uint32_t need = 0;
                {
                    std::lock_guard<std::mutex> lk(scb->dev->mtx);
                    auto it = scb->dev->dsl_dyn_counts.find(pl->app_dsls[s]);
                    if (it != scb->dev->dsl_dyn_counts.end()) need = it->second;
                }
                if (count == 1 && need == 0) need = dynCount;
                uint32_t take = std::min(need, std::min(SmCommandBuffer::MAX_DYN_OFFS_PER_SET,
                                                        (dyn_cursor < dynCount) ? (dynCount - dyn_cursor) : 0u));
                dst_dyn_counts[s] = take;
                for (uint32_t j = 0; j < take; ++j) dst_dyn_offs[s][j] = dynOffs[dyn_cursor + j];
                dyn_cursor += take;
            }
        }
    };
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        SM_TRACE("BindDescSets(GFX first=%u n=%u dyn=%u)", firstSet, count, dynCount);
        record_sets(scb->gfx_sets, scb->gfx_set_mask, scb->gfx_dyn_counts, scb->gfx_dyn_offs);
        for (uint32_t i = 0; i < count && (firstSet + i) < 16; ++i) {
            uint32_t s = firstSet + i;
            scb->gfx_push_descs[s].active = false;
            scb->gfx_push_descs[s].writes.clear(); scb->gfx_push_descs[s].bufs.clear();
            scb->gfx_push_descs[s].imgs.clear();   scb->gfx_push_descs[s].bvs.clear();
        }
    } else {
        record_sets(scb->comp_sets, scb->comp_set_mask, scb->comp_dyn_counts, scb->comp_dyn_offs);
        for (uint32_t i = 0; i < count && (firstSet + i) < 16; ++i) {
            uint32_t s = firstSet + i;
            scb->comp_push_descs[s].active = false;
            scb->comp_push_descs[s].writes.clear(); scb->comp_push_descs[s].bufs.clear();
            scb->comp_push_descs[s].imgs.clear();   scb->comp_push_descs[s].bvs.clear();
        }
        scb->dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE,
            pl->real_layout, firstSet, count, sets, dynCount, dynOffs);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushDescriptorSetKHR(
        VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set,
        uint32_t writeCount, const VkWriteDescriptorSet* pWrites) {
    if (!pWrites || writeCount == 0 || set >= 16) return;
    SM_TRACE("PushDescSet(bp=%d set=%u n=%u b0=%u type0=%d)", (int)bp, set, writeCount, pWrites[0].dstBinding, (int)pWrites[0].descriptorType);
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* pl = from_h<SmPipelineLayout>(layout);
    SmPushDescSet& dst = (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) ? scb->gfx_push_descs[set] : scb->comp_push_descs[set];
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) scb->gfx_set_mask &= ~(1u << set);
    else scb->comp_set_mask &= ~(1u << set);
    dst.active = true;

    std::vector<VkWriteDescriptorSet> batch_w;
    std::vector<std::vector<VkDescriptorBufferInfo>> batch_b(writeCount);
    std::vector<std::vector<VkDescriptorImageInfo>>  batch_i(writeCount);
    std::vector<std::vector<VkBufferView>>           batch_v(writeCount);
    batch_w.reserve(writeCount);

    for (uint32_t i = 0; i < writeCount; ++i) {
        VkWriteDescriptorSet w = pWrites[i];
        w.dstSet = VK_NULL_HANDLE;
        VkDescriptorType dt = w.descriptorType;
        bool is_buf = (dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                       dt == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || dt == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
        bool is_img = (dt == VK_DESCRIPTOR_TYPE_SAMPLER || dt == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                       dt == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || dt == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
                       dt == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        bool is_bv  = (dt == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || dt == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
        if (is_buf && w.pBufferInfo && w.descriptorCount > 0) {
            batch_b[i].assign(w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
            for (auto& bi : batch_b[i]) {
                if (bi.buffer) bi.buffer = from_h<SmBuffer>(bi.buffer)->real_buf;
                else { bi.buffer = VK_NULL_HANDLE; bi.offset = 0; bi.range = VK_WHOLE_SIZE; }
            }
            SM_TRACE("  PushDescBuf[%u] bind=%u buf=%p off=%llu range=%llu", i, w.dstBinding, (void*)batch_b[i][0].buffer, (unsigned long long)batch_b[i][0].offset, (unsigned long long)batch_b[i][0].range);
            w.pBufferInfo = batch_b[i].data(); w.pImageInfo = nullptr; w.pTexelBufferView = nullptr;
        } else if (is_img && w.pImageInfo && w.descriptorCount > 0) {
            batch_i[i].assign(w.pImageInfo, w.pImageInfo + w.descriptorCount);
            for (auto& ii : batch_i[i]) {
                if (dt != VK_DESCRIPTOR_TYPE_SAMPLER) {
                    if (ii.imageView) ii.imageView = from_h<SmImageView>(ii.imageView)->real_view;
                    ii.imageLayout = sanitize_layout(ii.imageLayout);
                }
            }
            w.pImageInfo = batch_i[i].data(); w.pBufferInfo = nullptr; w.pTexelBufferView = nullptr;
        } else if (is_bv && w.pTexelBufferView && w.descriptorCount > 0) {
            batch_v[i].assign(w.pTexelBufferView, w.pTexelBufferView + w.descriptorCount);
            w.pTexelBufferView = batch_v[i].data(); w.pBufferInfo = nullptr; w.pImageInfo = nullptr;
        }
        batch_w.push_back(w);

        // Update/replace matching (dstBinding, dstArrayElement, descriptorCount) entry in dst
        int found = -1;
        for (size_t k = 0; k < dst.writes.size(); ++k) {
            if (dst.writes[k].dstBinding == w.dstBinding && dst.writes[k].dstArrayElement == w.dstArrayElement &&
                dst.writes[k].descriptorCount == w.descriptorCount) { found = (int)k; break; }
        }
        if (found >= 0) {
            dst.writes[found] = w;
            dst.bufs[found] = batch_b[i]; dst.imgs[found] = batch_i[i]; dst.bvs[found] = batch_v[i];
        } else {
            dst.writes.push_back(w);
            dst.bufs.push_back(batch_b[i]); dst.imgs.push_back(batch_i[i]); dst.bvs.push_back(batch_v[i]);
        }
    }
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE && scb->dev->fn.CmdPushDescriptorSetKHR && pl && pl->real_layout) {
        scb->dev->fn.CmdPushDescriptorSetKHR(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->real_layout, set,
                                             (uint32_t)batch_w.size(), batch_w.data());
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushDescriptorSetWithTemplateKHR(
        VkCommandBuffer cb, VkDescriptorUpdateTemplate tmpl, VkPipelineLayout layout, uint32_t set, const void* data) {
    if (!tmpl || !data) return;
    auto* t = from_h<SmDescriptorUpdateTemplate>(tmpl);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(data);
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs(t->entries.size());
    std::vector<std::vector<VkDescriptorImageInfo>>  imgs(t->entries.size());
    std::vector<std::vector<VkBufferView>>           bvs(t->entries.size());
    writes.reserve(t->entries.size());
    for (size_t i = 0; i < t->entries.size(); ++i) {
        const auto& e = t->entries[i];
        if (e.descriptorCount == 0) continue;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = e.dstBinding; w.dstArrayElement = e.dstArrayElement;
        w.descriptorCount = e.descriptorCount; w.descriptorType = e.descriptorType;
        switch (e.descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
                bufs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k)
                    bufs[i][k] = *reinterpret_cast<const VkDescriptorBufferInfo*>(base + e.offset + k * e.stride);
                w.pBufferInfo = bufs[i].data(); writes.push_back(w); break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
                imgs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k)
                    imgs[i][k] = *reinterpret_cast<const VkDescriptorImageInfo*>(base + e.offset + k * e.stride);
                w.pImageInfo = imgs[i].data(); writes.push_back(w); break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
                bvs[i].resize(e.descriptorCount);
                for (uint32_t k = 0; k < e.descriptorCount; ++k)
                    bvs[i][k] = *reinterpret_cast<const VkBufferView*>(base + e.offset + k * e.stride);
                w.pTexelBufferView = bvs[i].data(); writes.push_back(w); break;
            }
            default: break;
        }
    }
    if (!writes.empty()) sm_CmdPushDescriptorSetKHR(cb, t->bind_point, layout, set, (uint32_t)writes.size(), writes.data());
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushConstants(
        VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* values) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    SM_TRACE("PushConst(stages=0x%x off=%u sz=%u)", (unsigned)stages, offset, size);
    if (offset + size <= sizeof(scb->push_data)) {
        memcpy(scb->push_data + offset, values, size);
        if (offset + size > scb->push_size) scb->push_size = offset + size;
    }
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT) {
        if (offset + size <= sizeof(scb->comp_push_data)) {
            memcpy(scb->comp_push_data + offset, values, size);
            if (offset + size > scb->comp_push_size) scb->comp_push_size = offset + size;
        }
        scb->dev->fn.CmdPushConstants(scb->real_cb, from_h<SmPipelineLayout>(layout)->real_layout,
                                      VK_SHADER_STAGE_COMPUTE_BIT, offset, size, values);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindDescriptorSets2KHR(VkCommandBuffer cb, const VkBindDescriptorSetsInfoKHR* info) {
    if (!info) return;
    if (info->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)) {
        sm_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, info->firstSet,
                                 info->descriptorSetCount, info->pDescriptorSets, info->dynamicOffsetCount, info->pDynamicOffsets);
    }
    if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
        sm_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, info->layout, info->firstSet,
                                 info->descriptorSetCount, info->pDescriptorSets, info->dynamicOffsetCount, info->pDynamicOffsets);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushConstants2KHR(VkCommandBuffer cb, const VkPushConstantsInfoKHR* info) {
    if (!info) return;
    sm_CmdPushConstants(cb, info->layout, info->stageFlags, info->offset, info->size, info->pValues);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetViewport(VkCommandBuffer cb, uint32_t, uint32_t count, const VkViewport* vps) {
    if (count > 0) {
        auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
        scb->cur_vp = vps[0]; scb->has_vp = true;
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetScissor(VkCommandBuffer cb, uint32_t, uint32_t count, const VkRect2D* scs) {
    if (count > 0) {
        auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
        scb->cur_scissor = scs[0]; scb->has_scissor = true;
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* bufs, const VkDeviceSize* offs) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    for (uint32_t i = 0; i < count && (first + i) < SM_MAX_VTX_BINDINGS; ++i) {
        scb->vtx_bufs[first + i] = from_h<SmBuffer>(bufs[i]);
        scb->vtx_offsets[first + i] = offs[i];
    }
    SM_TRACE("BindVB(first=%u n=%u b0=%p off0=%llu)", first, count, count ? (void*)bufs[0] : nullptr, count ? (unsigned long long)offs[0] : 0ull);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkIndexType type) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->idx_buf = from_h<SmBuffer>(buf);
    scb->idx_offset = off;
    scb->idx_type = type;
    SM_TRACE("BindIB(buf=%p off=%llu type=%d)", (void*)buf, (unsigned long long)off, (int)type);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindIndexBuffer2KHR(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkDeviceSize, VkIndexType type) {
    sm_CmdBindIndexBuffer(cb, buf, off, type);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetRenderingAreaGranularityKHR(VkDevice, const VkRenderingAreaInfoKHR*, VkExtent2D* pGranularity) {
    if (pGranularity) *pGranularity = {16, 16};
}
static VKAPI_ATTR void VKAPI_CALL sm_GetImageSubresourceLayout2KHR(VkDevice d, VkImage i, const VkImageSubresource2KHR* sr, VkSubresourceLayout2KHR* l) {
    if (l && sr) sm_GetImageSubresourceLayout(d, i, &sr->imageSubresource, &l->subresourceLayout);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceImageSubresourceLayoutKHR(VkDevice, const VkDeviceImageSubresourceInfoKHR* info, VkSubresourceLayout2KHR* l) {
    if (l) {
        memset(&l->subresourceLayout, 0, sizeof(l->subresourceLayout));
        if (info && info->pCreateInfo) {
            uint32_t bpp = sm_format_exact_bpp(info->pCreateInfo->format);
            l->subresourceLayout.rowPitch = (VkDeviceSize)info->pCreateInfo->extent.width * bpp;
            l->subresourceLayout.size = l->subresourceLayout.rowPitch * info->pCreateInfo->extent.height;
        }
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    sm_execute_draw_compute(reinterpret_cast<SmCommandBuffer*>(cb), vc, ic, fv, 0, fi, false);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndexed(VkCommandBuffer cb, uint32_t idxc, uint32_t ic, uint32_t fidx, int32_t voff, uint32_t fi) {
    sm_execute_draw_compute(reinterpret_cast<SmCommandBuffer*>(cb), idxc, ic, fidx, voff, fi, true);
}

static void sm_restore_app_compute_state(SmCommandBuffer* scb) {
    if (!scb->comp_state_dirty || !scb->cur_comp_pipe || !scb->cur_comp_pipe->real_compute_pipe) return;
    auto* dev = scb->dev;
    auto* pl = scb->cur_comp_pipe->layout;
    dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, scb->cur_comp_pipe->real_compute_pipe);
    if (pl && pl->real_layout) {
        for (uint32_t s = 0; s < pl->app_dsls.size() && s < 16; ++s) {
            if (scb->comp_set_mask & (1u << s)) {
                dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->real_layout,
                                              s, 1, &scb->comp_sets[s], scb->comp_dyn_counts[s],
                                              scb->comp_dyn_counts[s] ? scb->comp_dyn_offs[s] : nullptr);
            } else if (scb->comp_push_descs[s].active) {
                sm_replay_push_descs(dev, scb->real_cb, pl->real_layout, s, scb->comp_push_descs[s]);
            }
        }
        if (scb->comp_push_size > 0 && !pl->push_ranges.empty()) {
            for (const auto& pr : pl->push_ranges) {
                if (pr.offset < scb->comp_push_size) {
                    uint32_t sz = std::min(pr.size, scb->comp_push_size - pr.offset);
                    if (sz > 0)
                        dev->fn.CmdPushConstants(scb->real_cb, pl->real_layout, VK_SHADER_STAGE_COMPUTE_BIT, pr.offset, sz, scb->comp_push_data + pr.offset);
                }
            }
        }
    }
    scb->comp_state_dirty = false;
}
static void sm_invalidate_storage_image_attaches(SmDevice* dev) {
    std::lock_guard<std::mutex> lk(dev->mtx);
    for (SmImage* img : dev->storage_images) img->attach_dirty = false;
}

// Native H20 Compute & Transfer commands pass 100% straight through!
static VKAPI_ATTR void VKAPI_CALL sm_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    sm_restore_app_compute_state(scb);
    SM_TRACE("CmdDispatch(%u,%u,%u)", x, y, z);
    SM_XRAY("CmdDispatch(%u,%u,%u)", x, y, z);
    scb->dev->fn.CmdDispatch(scb->real_cb, x, y, z);
    sm_invalidate_storage_image_attaches(scb->dev);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    sm_restore_app_compute_state(scb);
    scb->dev->fn.CmdDispatchIndirect(scb->real_cb, from_h<SmBuffer>(b)->real_buf, o);
    sm_invalidate_storage_image_attaches(scb->dev);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->dev->fn.CmdCopyBuffer(scb->real_cb, from_h<SmBuffer>(s)->real_buf, from_h<SmBuffer>(d)->real_buf, n, r);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* si = from_h<SmImage>(s);
    auto* di = from_h<SmImage>(d);
    SM_TRACE("CopyImage(s=%ux%u f=%d sc=%d ad=%d -> d=%ux%u f=%d sc=%d)",
             si->extent.width, si->extent.height, (int)si->format, (int)si->is_swapchain, (int)si->attach_dirty,
             di->extent.width, di->extent.height, (int)di->format, (int)di->is_swapchain);
    if (si->attach_dirty && si->attach_buf && di->attach_buf && si->format == di->format &&
        si->extent.width == di->extent.width && si->extent.height == di->extent.height) {
        uint32_t bpp = si->is_depth ? 4u : sm_format_bpp(si->format);
        VkBufferCopy bc{0, 0, (VkDeviceSize)si->extent.width * si->extent.height * bpp};
        scb->dev->fn.CmdCopyBuffer(scb->real_cb, si->attach_buf, di->attach_buf, 1, &bc);
        di->attach_dirty = true;
        sm_sync_attach_to_image(scb, di, dl);
        return;
    }
    if (si->real_img && di->real_img) {
        scb->dev->fn.CmdCopyImage(scb->real_cb, si->real_img, sanitize_layout(sl),
                                  di->real_img, sanitize_layout(dl), n, r);
    }
    if (di->attach_buf && di->real_img) {
        sm_sync_image_to_attach(scb, di, dl);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl, uint32_t n, const VkBufferImageCopy* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* di = from_h<SmImage>(d);
    if (n > 0 && r) {
        SM_TRACE("CopyBuf2Img(dst=%ux%u fmt=%d n=%u rowLen=%u imgH=%u ext=%ux%u mip=%u)",
                 di->extent.width, di->extent.height, (int)di->format, n,
                 r[0].bufferRowLength, r[0].bufferImageHeight, r[0].imageExtent.width, r[0].imageExtent.height,
                 r[0].imageSubresource.mipLevel);
    }
    scb->dev->fn.CmdCopyBufferToImage(scb->real_cb, from_h<SmBuffer>(s)->real_buf,
                                      di->real_img, sanitize_layout(dl), n, r);
    di->attach_dirty = false;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d, uint32_t n, const VkBufferImageCopy* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* si = from_h<SmImage>(s);
    if (si->attach_dirty && si->attach_buf && n == 1 && r &&
        r[0].imageSubresource.mipLevel == 0 && r[0].imageSubresource.baseArrayLayer == 0 &&
        r[0].imageOffset.x == 0 && r[0].imageOffset.y == 0 &&
        r[0].imageExtent.width == si->extent.width && r[0].imageExtent.height == si->extent.height &&
        (r[0].bufferRowLength == 0 || r[0].bufferRowLength == si->extent.width)) {
        uint32_t bpp = si->is_depth ? 4u : sm_format_bpp(si->format);
        VkBufferCopy bc{0, r[0].bufferOffset, (VkDeviceSize)si->extent.width * si->extent.height * bpp};
        scb->dev->fn.CmdCopyBuffer(scb->real_cb, si->attach_buf, from_h<SmBuffer>(d)->real_buf, 1, &bc);
        return;
    }
    scb->dev->fn.CmdCopyImageToBuffer(scb->real_cb, si->real_img, sanitize_layout(sl), from_h<SmBuffer>(d)->real_buf, n, r);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t v) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->dev->fn.CmdFillBuffer(scb->real_cb, from_h<SmBuffer>(b)->real_buf, o, s, v);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void* d) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->dev->fn.CmdUpdateBuffer(scb->real_cb, from_h<SmBuffer>(b)->real_buf, o, s, d);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* img = from_h<SmImage>(i);
    if (img->attach_buf) {
        uint32_t cw[SM_MAX_MRT][4] = {{0}};
        pack_clear_words(*c, img->format, cw[0]);
        SmImage* cols[SM_MAX_MRT] = {img};
        sm_begin_pass_clear_mrt(scb, cols, nullptr, img->extent.width, img->extent.height, 1u, cw, 1.0f, 0u);
    }
    if (img->real_img) scb->dev->fn.CmdClearColorImage(scb->real_cb, img->real_img, sanitize_layout(l), c, n, r);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout, const VkClearDepthStencilValue* ds, uint32_t, const VkImageSubresourceRange*) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* img = from_h<SmImage>(i);
    sm_ensure_image_attach_buf(scb->dev, img);
    uint32_t dbits = 0; memcpy(&dbits, &ds->depth, 4);
    scb->dev->fn.CmdFillBuffer(scb->real_cb, img->attach_buf, 0, (VkDeviceSize)img->extent.width * img->extent.height * 4, dbits);
    if (img->has_stencil) {
        sm_ensure_image_stencil_buf(scb->dev, img);
        scb->dev->fn.CmdFillBuffer(scb->real_cb, img->stencil_buf, 0, (VkDeviceSize)img->extent.width * img->extent.height * 4, ds->stencil & 0xFFu);
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBlitImage(
        VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
        uint32_t regionCount, const VkImageBlit* regions, VkFilter filter) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    auto* dev = scb->dev;
    auto* si = from_h<SmImage>(s);
    auto* di = from_h<SmImage>(d);
    if (!si || !di || regionCount == 0 || !regions) return;
    SM_TRACE("BlitImage(s=%ux%u f=%d ad=%d -> d=%ux%u f=%d sc=%d)",
             si->extent.width, si->extent.height, (int)si->format, (int)si->attach_dirty,
             di->extent.width, di->extent.height, (int)di->format, (int)di->is_swapchain);
    if (!si->attach_dirty && si->real_img) {
        sm_sync_image_to_attach(scb, si, sl);
    }
    if (si->attach_buf && di->attach_buf && si->format == di->format &&
        si->extent.width == di->extent.width && si->extent.height == di->extent.height &&
        regions[0].srcOffsets[0].x == 0 && regions[0].srcOffsets[0].y == 0 &&
        regions[0].srcOffsets[1].x == (int32_t)si->extent.width && regions[0].srcOffsets[1].y == (int32_t)si->extent.height &&
        regions[0].dstOffsets[0].x == 0 && regions[0].dstOffsets[0].y == 0 &&
        regions[0].dstOffsets[1].x == (int32_t)di->extent.width && regions[0].dstOffsets[1].y == (int32_t)di->extent.height) {
        uint32_t bpp = si->is_depth ? 4u : sm_format_bpp(si->format);
        VkBufferCopy bc{0, 0, (VkDeviceSize)si->extent.width * si->extent.height * bpp};
        dev->fn.CmdCopyBuffer(scb->real_cb, si->attach_buf, di->attach_buf, 1, &bc);
        di->attach_dirty = true;
        sm_sync_attach_to_image(scb, di, dl);
        return;
    }
    if (si->attach_buf || di->attach_buf) {
        sm_ensure_image_attach_buf(dev, si);
        sm_ensure_image_attach_buf(dev, di);
        di->attach_dirty = true;
        for (uint32_t ri = 0; ri < regionCount; ++ri) {
            const auto& reg = regions[ri];
            int dw = std::abs(reg.dstOffsets[1].x - reg.dstOffsets[0].x);
            int dh = std::abs(reg.dstOffsets[1].y - reg.dstOffsets[0].y);
            if (dw <= 0 || dh <= 0) continue;
            uint32_t bslot = 0;
            {
                std::lock_guard<std::mutex> lk(dev->mtx);
                bslot = dev->rast.blit_cursor % SmRasterEngine::MAX_AUX_SLOTS;
                dev->rast.blit_cursor++;
                VkDescriptorBufferInfo bi[2] = {
                    {si->attach_buf, 0, VK_WHOLE_SIZE},
                    {di->attach_buf, 0, VK_WHOLE_SIZE},
                };
                VkWriteDescriptorSet ws[2] = {
                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.blit_ds[bslot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[0]},
                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dev->rast.blit_ds[bslot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[1]},
                };
                dev->fn.UpdateDescriptorSets(dev->real_dev, 2, ws, 0, nullptr);
            }
            struct {
                uint32_t sw, sh, dw, dh;
                int32_t sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1;
                uint32_t sfc, dfc, lin, pad;
            } pc = {
                si->extent.width, si->extent.height, di->extent.width, di->extent.height,
                reg.srcOffsets[0].x, reg.srcOffsets[0].y, reg.srcOffsets[1].x, reg.srcOffsets[1].y,
                reg.dstOffsets[0].x, reg.dstOffsets[0].y, reg.dstOffsets[1].x, reg.dstOffsets[1].y,
                sm_format_code(si->format), sm_format_code(di->format),
                (filter == VK_FILTER_LINEAR) ? 1u : 0u, 0u
            };
            emit_compute_barrier(scb);
            dev->fn.CmdBindPipeline(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.blit_pipe);
            dev->fn.CmdBindDescriptorSets(scb->real_cb, VK_PIPELINE_BIND_POINT_COMPUTE, dev->rast.blit_pl, 0, 1, &dev->rast.blit_ds[bslot], 0, nullptr);
            dev->fn.CmdPushConstants(scb->real_cb, dev->rast.blit_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            dev->fn.CmdDispatch(scb->real_cb, ((uint32_t)dw + 15) / 16, ((uint32_t)dh + 15) / 16, 1);
            scb->comp_state_dirty = true;
            emit_compute_barrier(scb);
        }
        sm_sync_attach_to_image(scb, di, dl);
        return;
    }
    if (si->real_img && di->real_img) {
        VkImageCopy ic{};
        ic.srcSubresource = regions[0].srcSubresource;
        ic.dstSubresource = regions[0].dstSubresource;
        ic.extent = { std::min(si->extent.width, di->extent.width), std::min(si->extent.height, di->extent.height), 1 };
        dev->fn.CmdCopyImage(scb->real_cb, si->real_img, sanitize_layout(sl), di->real_img, sanitize_layout(dl), 1, &ic);
    }
}

// VK_EXT_extended_dynamic_state / Vulkan 1.3 core
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetCullMode(VkCommandBuffer cb, VkCullModeFlags m) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_cull = true; scb->dyn_cull = m;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetFrontFace(VkCommandBuffer cb, VkFrontFace f) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_front_face = true; scb->dyn_front_face = f;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetPrimitiveTopology(VkCommandBuffer cb, VkPrimitiveTopology t) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_topo = true; scb->dyn_topo = t;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetViewportWithCount(VkCommandBuffer cb, uint32_t count, const VkViewport* vps) {
    sm_CmdSetViewport(cb, 0, count, vps);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetScissorWithCount(VkCommandBuffer cb, uint32_t count, const VkRect2D* scs) {
    sm_CmdSetScissor(cb, 0, count, scs);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindVertexBuffers2(
        VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* bufs, const VkDeviceSize* offs,
        const VkDeviceSize*, const VkDeviceSize* pStrides) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    sm_CmdBindVertexBuffers(cb, first, count, bufs, offs);
    if (pStrides) {
        for (uint32_t i = 0; i < count && (first + i) < SM_MAX_VTX_BINDINGS; ++i) {
            scb->dyn_vtx_strides[first + i] = (uint32_t)pStrides[i];
            scb->has_dyn_vtx_stride[first + i] = true;
        }
    }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthTestEnable(VkCommandBuffer cb, VkBool32 e) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_depth_test = true; scb->dyn_depth_test = (e != 0);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthWriteEnable(VkCommandBuffer cb, VkBool32 e) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_depth_write = true; scb->dyn_depth_write = (e != 0);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthCompareOp(VkCommandBuffer cb, VkCompareOp op) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_depth_op = true; scb->dyn_depth_op = op;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthBoundsTestEnable(VkCommandBuffer, VkBool32) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetStencilTestEnable(VkCommandBuffer cb, VkBool32 e) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb); scb->has_dyn_stencil_test = true; scb->dyn_stencil_test = (e != 0);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags mask, VkStencilOp failOp, VkStencilOp passOp, VkStencilOp depthFailOp, VkCompareOp cmpOp) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (mask & VK_STENCIL_FACE_FRONT_BIT) { scb->has_dyn_stencil_op[0] = true; scb->dyn_stencil_fail[0] = failOp; scb->dyn_stencil_pass[0] = passOp; scb->dyn_stencil_dfail[0] = depthFailOp; scb->dyn_stencil_cmp[0] = cmpOp; }
    if (mask & VK_STENCIL_FACE_BACK_BIT)  { scb->has_dyn_stencil_op[1] = true; scb->dyn_stencil_fail[1] = failOp; scb->dyn_stencil_pass[1] = passOp; scb->dyn_stencil_dfail[1] = depthFailOp; scb->dyn_stencil_cmp[1] = cmpOp; }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetRasterizerDiscardEnable(VkCommandBuffer, VkBool32) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthBiasEnable(VkCommandBuffer, VkBool32) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetPrimitiveRestartEnable(VkCommandBuffer, VkBool32) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginConditionalRenderingEXT(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndConditionalRenderingEXT(VkCommandBuffer) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBeginTransformFeedbackEXT(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer*, const VkDeviceSize*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdEndTransformFeedbackEXT(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer*, const VkDeviceSize*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer*, const VkDeviceSize*, const VkDeviceSize*) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDrawIndirectByteCountEXT(VkCommandBuffer, uint32_t, uint32_t, VkBuffer, VkDeviceSize, uint32_t, uint32_t) {}

// VK_KHR_synchronization2 / Vulkan 1.3 core (sanitizes all graphics stages -> COMPUTE on H20 qf[2]!)
static VKAPI_ATTR void VKAPI_CALL sm_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    std::vector<VkMemoryBarrier> mbs;
    VkPipelineStageFlags srcS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    for (uint32_t i = 0; i < di->memoryBarrierCount; ++i) {
        mbs.push_back({VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                       sanitize_access_mask((VkAccessFlags)di->pMemoryBarriers[i].srcAccessMask),
                       sanitize_access_mask((VkAccessFlags)di->pMemoryBarriers[i].dstAccessMask)});
    }
    std::vector<VkBufferMemoryBarrier> bbs;
    for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; ++i) {
        const auto& b = di->pBufferMemoryBarriers[i];
        bbs.push_back({VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                       sanitize_access_mask((VkAccessFlags)b.srcAccessMask),
                       sanitize_access_mask((VkAccessFlags)b.dstAccessMask),
                       VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                       from_h<SmBuffer>(b.buffer)->real_buf, b.offset, b.size});
    }
    std::vector<VkImageMemoryBarrier> ibs;
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount; ++i) {
        const auto& b = di->pImageMemoryBarriers[i];
        auto* si = from_h<SmImage>(b.image);
        VkImageSubresourceRange sr = b.subresourceRange;
        if (si->is_depth) sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ibs.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
                       sanitize_access_mask((VkAccessFlags)b.srcAccessMask),
                       sanitize_access_mask((VkAccessFlags)b.dstAccessMask),
                       sanitize_layout(b.oldLayout), sanitize_layout(b.newLayout),
                       VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, si->real_img, sr});
    }
    scb->dev->fn.CmdPipelineBarrier(scb->real_cb, srcS, dstS, di->dependencyFlags,
                                    (uint32_t)mbs.size(), mbs.data(),
                                    (uint32_t)bbs.size(), bbs.data(),
                                    (uint32_t)ibs.size(), ibs.data());
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_QueueSubmit2(
        VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
    auto* q = reinterpret_cast<SmQueue*>(queue);
    if (q->dev->fn.QueueSubmit2) {
        // Forward directly to native vkQueueSubmit2 so 64-bit Timeline Semaphores & values work 100% natively!
        std::vector<VkSubmitInfo2> mod(count);
        std::vector<std::vector<VkSemaphoreSubmitInfo>> wsems(count), ssems(count);
        std::vector<std::vector<VkCommandBufferSubmitInfo>> cbs(count);
        for (uint32_t i = 0; i < count; ++i) {
            mod[i] = submits[i];
            if (submits[i].waitSemaphoreInfoCount > 0 && submits[i].pWaitSemaphoreInfos) {
                wsems[i].assign(submits[i].pWaitSemaphoreInfos, submits[i].pWaitSemaphoreInfos + submits[i].waitSemaphoreInfoCount);
                for (auto& ws : wsems[i]) ws.stageMask = sanitize_stage_mask2(ws.stageMask);
                mod[i].pWaitSemaphoreInfos = wsems[i].data();
            }
            if (submits[i].commandBufferInfoCount > 0 && submits[i].pCommandBufferInfos) {
                cbs[i].assign(submits[i].pCommandBufferInfos, submits[i].pCommandBufferInfos + submits[i].commandBufferInfoCount);
                for (auto& cb : cbs[i]) cb.commandBuffer = reinterpret_cast<SmCommandBuffer*>(cb.commandBuffer)->real_cb;
                mod[i].pCommandBufferInfos = cbs[i].data();
            }
            if (submits[i].signalSemaphoreInfoCount > 0 && submits[i].pSignalSemaphoreInfos) {
                ssems[i].assign(submits[i].pSignalSemaphoreInfos, submits[i].pSignalSemaphoreInfos + submits[i].signalSemaphoreInfoCount);
                for (auto& ss : ssems[i]) ss.stageMask = sanitize_stage_mask2(ss.stageMask);
                mod[i].pSignalSemaphoreInfos = ssems[i].data();
            }
        }
        VkResult r;
        {
            std::lock_guard<std::mutex> qlk(q->mtx);
            r = q->dev->fn.QueueSubmit2(q->real_q, count, mod.data(), fence);
        }
        SM_TRACE("QueueSubmit2(n=%u, cb0=%u, w0=%u, s0=%u, sval0=%llu, fence=%p) -> %d",
                 count, count ? submits[0].commandBufferInfoCount : 0,
                 count ? submits[0].waitSemaphoreInfoCount : 0,
                 count ? submits[0].signalSemaphoreInfoCount : 0,
                 (count && submits[0].signalSemaphoreInfoCount && submits[0].pSignalSemaphoreInfos) ? (unsigned long long)submits[0].pSignalSemaphoreInfos[0].value : 0ull,
                 (void*)fence, (int)r);
        if (r != VK_SUCCESS) fprintf(stderr, "[SmRast] ERROR: QueueSubmit2 returned %d\n", (int)r);
        return r;
    }
    // Fallback preserving Timeline Semaphore values via VkTimelineSemaphoreSubmitInfo
    std::vector<VkSubmitInfo> sis(count);
    std::vector<VkTimelineSemaphoreSubmitInfo> tsis(count);
    std::vector<std::vector<VkCommandBuffer>> cbs(count);
    std::vector<std::vector<VkSemaphore>> wsems(count), ssems(count);
    std::vector<std::vector<uint64_t>> wvals(count), svals(count);
    std::vector<std::vector<VkPipelineStageFlags>> wstages(count);
    for (uint32_t i = 0; i < count; ++i) {
        sis[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        tsis[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        for (uint32_t j = 0; j < submits[i].waitSemaphoreInfoCount; ++j) {
            wsems[i].push_back(submits[i].pWaitSemaphoreInfos[j].semaphore);
            wvals[i].push_back(submits[i].pWaitSemaphoreInfos[j].value);
            wstages[i].push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; ++j) {
            cbs[i].push_back(reinterpret_cast<SmCommandBuffer*>(submits[i].pCommandBufferInfos[j].commandBuffer)->real_cb);
        }
        for (uint32_t j = 0; j < submits[i].signalSemaphoreInfoCount; ++j) {
            ssems[i].push_back(submits[i].pSignalSemaphoreInfos[j].semaphore);
            svals[i].push_back(submits[i].pSignalSemaphoreInfos[j].value);
        }
        tsis[i].waitSemaphoreValueCount = (uint32_t)wvals[i].size();
        tsis[i].pWaitSemaphoreValues = wvals[i].data();
        tsis[i].signalSemaphoreValueCount = (uint32_t)svals[i].size();
        tsis[i].pSignalSemaphoreValues = svals[i].data();
        sis[i].pNext = &tsis[i];
        sis[i].waitSemaphoreCount = (uint32_t)wsems[i].size();
        sis[i].pWaitSemaphores = wsems[i].data();
        sis[i].pWaitDstStageMask = wstages[i].data();
        sis[i].commandBufferCount = (uint32_t)cbs[i].size();
        sis[i].pCommandBuffers = cbs[i].data();
        sis[i].signalSemaphoreCount = (uint32_t)ssems[i].size();
        sis[i].pSignalSemaphores = ssems[i].data();
    }
    return q->dev->fn.QueueSubmit(q->real_q, count, sis.data(), fence);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPipelineBarrier(
        VkCommandBuffer cb, VkPipelineStageFlags srcS, VkPipelineStageFlags dstS, VkDependencyFlags dep,
        uint32_t mbc, const VkMemoryBarrier* mbs, uint32_t bbc, const VkBufferMemoryBarrier* bbs,
        uint32_t ibc, const VkImageMemoryBarrier* ibs) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    std::vector<VkMemoryBarrier> mmb(mbs, mbs + mbc);
    for (auto& b : mmb) { b.srcAccessMask = sanitize_access_mask(b.srcAccessMask); b.dstAccessMask = sanitize_access_mask(b.dstAccessMask); }
    std::vector<VkBufferMemoryBarrier> mbb(bbs, bbs + bbc);
    for (auto& b : mbb) {
        b.buffer = from_h<SmBuffer>(b.buffer)->real_buf;
        b.srcAccessMask = sanitize_access_mask(b.srcAccessMask);
        b.dstAccessMask = sanitize_access_mask(b.dstAccessMask);
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    std::vector<VkImageMemoryBarrier> mib(ibs, ibs + ibc);
    for (auto& b : mib) {
        auto* si = from_h<SmImage>(b.image);
        b.image = si->real_img;
        b.srcAccessMask = sanitize_access_mask(b.srcAccessMask);
        b.dstAccessMask = sanitize_access_mask(b.dstAccessMask);
        b.oldLayout = sanitize_layout(b.oldLayout);
        b.newLayout = sanitize_layout(b.newLayout);
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        if (si->is_depth) b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    scb->dev->fn.CmdPipelineBarrier(scb->real_cb, sanitize_stage_mask(srcS), sanitize_stage_mask(dstS), dep,
                                    mbc, mmb.data(), bbc, mbb.data(), ibc, mib.data());
}

// =============================================================================
// WSI Surface & Swapchain Implementation (XCB + Xlib -> Xvfb!)
// =============================================================================
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateXcbSurfaceKHR(VkInstance, const VkXcbSurfaceCreateInfoKHR* ci, const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    auto* s = new SmSurface();
    s->is_xcb = true; s->xcb_conn = ci->connection; s->xcb_win = ci->window;
    *out = to_h<VkSurfaceKHR>(s);
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateXlibSurfaceKHR(VkInstance, const VkXlibSurfaceCreateInfoKHR* ci, const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    auto* s = new SmSurface();
    s->is_xcb = false; s->xlib_dpy = ci->dpy; s->xlib_win = ci->window;
    *out = to_h<VkSurfaceKHR>(s);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroySurfaceKHR(VkInstance, VkSurfaceKHR s, const VkAllocationCallbacks*) {
    if (s) delete from_h<SmSurface>(s);
}
static VKAPI_ATTR VkBool32 VKAPI_CALL sm_GetPhysicalDeviceXcbPresentationSupportKHR(VkPhysicalDevice, uint32_t, xcb_connection_t*, xcb_visualid_t) {
    return VK_TRUE;
}
static VKAPI_ATTR VkBool32 VKAPI_CALL sm_GetPhysicalDeviceXlibPresentationSupportKHR(VkPhysicalDevice, uint32_t, Display*, VisualID) {
    return VK_TRUE;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32* sup) {
    *sup = VK_TRUE; return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice, VkSurfaceKHR s, VkSurfaceCapabilitiesKHR* cap) {
    auto* surf = from_h<SmSurface>(s);
    uint32_t w = 800, h = 600;
    if (surf && surf->is_xcb && surf->xcb_conn) {
        auto cookie = xcb_get_geometry(surf->xcb_conn, surf->xcb_win);
        if (auto* g = xcb_get_geometry_reply(surf->xcb_conn, cookie, nullptr)) {
            w = g->width; h = g->height; free(g);
        }
    } else if (surf && !surf->is_xcb && surf->xlib_dpy) {
        XWindowAttributes wa{};
        if (XGetWindowAttributes(surf->xlib_dpy, surf->xlib_win, &wa)) { w = wa.width; h = wa.height; }
    }
    memset(cap, 0, sizeof(*cap));
    cap->minImageCount = 2;
    cap->maxImageCount = 4;
    cap->currentExtent = {w, h};
    cap->minImageExtent = {1, 1};
    cap->maxImageExtent = {4096, 4096};
    cap->maxImageArrayLayers = 1;
    cap->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    cap->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    cap->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    // Do NOT advertise STORAGE_BIT on swapchain so DXVK Presenter uses Graphics/Transfer path -> SmRast attach_buf!
    cap->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t* count, VkSurfaceFormatKHR* fmts) {
    static const VkSurfaceFormatKHR list[] = {
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_B8G8R8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_SRGB,  VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    };
    uint32_t n = 4;
    if (!fmts) { *count = n; return VK_SUCCESS; }
    uint32_t c = std::min(*count, n);
    memcpy(fmts, list, c * sizeof(VkSurfaceFormatKHR));
    *count = c;
    return (c < n) ? VK_INCOMPLETE : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t* count, VkPresentModeKHR* modes) {
    static const VkPresentModeKHR list[] = { VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR };
    uint32_t n = 3;
    if (!modes) { *count = n; return VK_SUCCESS; }
    uint32_t c = std::min(*count, n);
    memcpy(modes, list, c * sizeof(VkPresentModeKHR));
    *count = c;
    return (c < n) ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_CreateSwapchainKHR(VkDevice d, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* sc = new SmSwapchain();
    sc->dev = dev;
    sc->surface = from_h<SmSurface>(ci->surface);
    sc->format = ci->imageFormat;
    sc->extent = ci->imageExtent;

    uint32_t n = std::max(2u, ci->minImageCount);
    VkDeviceSize sz = (VkDeviceSize)sc->extent.width * sc->extent.height * 4;
    for (uint32_t i = 0; i < n; ++i) {
        auto* img = new SmImage();
        img->format = sc->format;
        img->extent = {sc->extent.width, sc->extent.height, 1};
        img->usage = ci->imageUsage;
        img->is_swapchain = true;

        // Backing VkImage + linear attach_buf + host-coherent staging readback buffer
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, sc->format,
                              img->extent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
        dev->fn.CreateImage(dev->real_dev, &ici, nullptr, &img->real_img);
        VkMemoryRequirements mr{};
        dev->fn.GetImageMemoryRequirements(dev->real_dev, img->real_img, &mr);
        VkDeviceMemory imem = VK_NULL_HANDLE;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
                                 sm_find_mem_type(dev->pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        dev->fn.AllocateMemory(dev->real_dev, &mai, nullptr, &imem);
        dev->fn.BindImageMemory(dev->real_dev, img->real_img, imem, 0);

        sm_ensure_image_attach_buf(dev, img);
        VkBufferCreateInfo hbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        dev->fn.CreateBuffer(dev->real_dev, &hbci, nullptr, &img->host_buf);
        VkMemoryRequirements hmr{};
        dev->fn.GetBufferMemoryRequirements(dev->real_dev, img->host_buf, &hmr);
        VkMemoryAllocateInfo hmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, hmr.size,
                                  sm_find_mem_type(dev->pd, hmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        dev->fn.AllocateMemory(dev->real_dev, &hmai, nullptr, &img->host_mem);
        dev->fn.BindBufferMemory(dev->real_dev, img->host_buf, img->host_mem, 0);
        dev->fn.MapMemory(dev->real_dev, img->host_mem, 0, sz, 0, &img->host_ptr);

        sc->images.push_back(img);
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, dev->pd->backend_compute_qf};
    dev->fn.CreateCommandPool(dev->real_dev, &cpci, nullptr, &sc->present_pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, sc->present_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    dev->fn.AllocateCommandBuffers(dev->real_dev, &cbai, &sc->present_cmd);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    dev->fn.CreateFence(dev->real_dev, &fci, nullptr, &sc->present_fence);

    // Zero host staging buffers on CPU (never QueueWaitIdle during CreateSwapchain so vkd3d-proton timeline submits never deadlock!)
    for (auto* img : sc->images) {
        if (img->host_ptr) memset(img->host_ptr, 0, (size_t)sz);
    }

    fprintf(stderr, "[SmRast-WSI] Created swapchain %ux%u fmt=%d imgs=%u\n",
            sc->extent.width, sc->extent.height, (int)sc->format, n);
    fflush(stderr);
    *out = to_h<VkSwapchainKHR>(sc);
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroySwapchainKHR(VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks*) {
    if (!s) return;
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* sc = from_h<SmSwapchain>(s);
    dev->fn.DeviceWaitIdle(dev->real_dev);
    sm_wsi_cleanup_swapchain(sc);
    if (sc->present_fence) dev->fn.DestroyFence(dev->real_dev, sc->present_fence, nullptr);
    if (sc->present_pool) dev->fn.DestroyCommandPool(dev->real_dev, sc->present_pool, nullptr);
    for (auto* img : sc->images) sm_DestroyImage(d, to_h<VkImage>(img), nullptr);
    delete sc;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetSwapchainImagesKHR(VkDevice, VkSwapchainKHR s, uint32_t* count, VkImage* imgs) {
    auto* sc = from_h<SmSwapchain>(s);
    uint32_t n = (uint32_t)sc->images.size();
    if (!imgs) { *count = n; return VK_SUCCESS; }
    uint32_t c = std::min(*count, n);
    for (uint32_t i = 0; i < c; ++i) imgs[i] = to_h<VkImage>(sc->images[i]);
    *count = c;
    return (c < n) ? VK_INCOMPLETE : VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_AcquireNextImageKHR(
        VkDevice d, VkSwapchainKHR s, uint64_t, VkSemaphore sem, VkFence fence, uint32_t* idx) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto* sc = from_h<SmSwapchain>(s);
    *idx = sc->next_image % (uint32_t)sc->images.size();
    sc->next_image++;
    // Signal semaphore/fence immediately via empty submit on compute queue so app sync never blocks!
    if (sem != VK_NULL_HANDLE || fence != VK_NULL_HANDLE) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 0, nullptr,
                        (sem != VK_NULL_HANDLE) ? 1u : 0u, (sem != VK_NULL_HANDLE) ? &sem : nullptr};
        std::lock_guard<std::mutex> qlk(dev->main_queue->mtx);
        dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, fence);
    }
    SM_TRACE("AcquireNextImageKHR -> idx=%u sem=%p fence=%p", *idx, (void*)sem, (void*)fence);
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi) {
    auto* q = reinterpret_cast<SmQueue*>(queue);
    // Wait for waitSemaphores on compute queue before reading back frame
    if (pi->waitSemaphoreCount > 0) {
        std::vector<VkPipelineStageFlags> st(pi->waitSemaphoreCount, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, pi->waitSemaphoreCount, pi->pWaitSemaphores, st.data()};
        std::lock_guard<std::mutex> qlk(q->mtx);
        q->dev->fn.QueueSubmit(q->real_q, 1, &si, VK_NULL_HANDLE);
    }
    for (uint32_t i = 0; i < pi->swapchainCount; ++i) {
        auto* sc = from_h<SmSwapchain>(pi->pSwapchains[i]);
        uint32_t idx = pi->pImageIndices[i];
        if (idx < sc->images.size()) sm_wsi_blit_to_surface(sc, sc->images[idx]);
        if (g_sm_xray_env) {
            if (g_sm_xray_frames.load(std::memory_order_relaxed) > 0) {
                int rem = g_sm_xray_frames.fetch_sub(1, std::memory_order_relaxed) - 1;
                fprintf(stderr, "[SmXRay] === PRESENT FRAME #%llu (rem=%d) ===\n", (unsigned long long)sc->present_frames, rem);
                fflush(stderr);
            } else if (access("/tmp/smrast_xray.trigger", F_OK) == 0) {
                unlink("/tmp/smrast_xray.trigger");
                g_sm_xray_frames.store(2, std::memory_order_relaxed);
            }
        }
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfaceCapabilities2KHR(
        VkPhysicalDevice pd, const VkPhysicalDeviceSurfaceInfo2KHR* info, VkSurfaceCapabilities2KHR* caps) {
    return sm_GetPhysicalDeviceSurfaceCapabilitiesKHR(pd, info->surface, &caps->surfaceCapabilities);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_GetPhysicalDeviceSurfaceFormats2KHR(
        VkPhysicalDevice pd, const VkPhysicalDeviceSurfaceInfo2KHR* info, uint32_t* count, VkSurfaceFormat2KHR* fmts) {
    if (!fmts) return sm_GetPhysicalDeviceSurfaceFormatsKHR(pd, info->surface, count, nullptr);
    std::vector<VkSurfaceFormatKHR> tmp(*count);
    VkResult r = sm_GetPhysicalDeviceSurfaceFormatsKHR(pd, info->surface, count, tmp.data());
    for (uint32_t i = 0; i < *count; ++i) {
        fmts[i].sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR;
        fmts[i].surfaceFormat = tmp[i];
    }
    return r;
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceQueue2(VkDevice d, const VkDeviceQueueInfo2* info, VkQueue* q) {
    sm_GetDeviceQueue(d, info->queueFamilyIndex, info->queueIndex, q);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetBufferMemoryRequirements2(VkDevice d, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* mr) {
    sm_GetBufferMemoryRequirements(d, info->buffer, &mr->memoryRequirements);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetImageMemoryRequirements2(VkDevice d, const VkImageMemoryRequirementsInfo2* info, VkMemoryRequirements2* mr) {
    sm_GetImageMemoryRequirements(d, info->image, &mr->memoryRequirements);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_BindBufferMemory2(VkDevice d, uint32_t count, const VkBindBufferMemoryInfo* infos) {
    for (uint32_t i = 0; i < count; ++i) {
        VkResult r = sm_BindBufferMemory(d, infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_BindImageMemory2(VkDevice d, uint32_t count, const VkBindImageMemoryInfo* infos) {
    for (uint32_t i = 0; i < count; ++i) {
        VkResult r = sm_BindImageMemory(d, infos[i].image, infos[i].memory, infos[i].memoryOffset);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetLineWidth(VkCommandBuffer, float) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthBias(VkCommandBuffer, float, float, float) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->has_dyn_blend_const = true;
    for (int k = 0; k < 4; ++k) scb->dyn_blend_const[k] = c[k];
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDepthBounds(VkCommandBuffer, float, float) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags mask, uint32_t cmpMask) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (mask & VK_STENCIL_FACE_FRONT_BIT) { scb->has_dyn_stencil_cmask[0] = true; scb->dyn_stencil_cmask[0] = cmpMask & 0xFFu; }
    if (mask & VK_STENCIL_FACE_BACK_BIT)  { scb->has_dyn_stencil_cmask[1] = true; scb->dyn_stencil_cmask[1] = cmpMask & 0xFFu; }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags mask, uint32_t wMask) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (mask & VK_STENCIL_FACE_FRONT_BIT) { scb->has_dyn_stencil_wmask[0] = true; scb->dyn_stencil_wmask[0] = wMask & 0xFFu; }
    if (mask & VK_STENCIL_FACE_BACK_BIT)  { scb->has_dyn_stencil_wmask[1] = true; scb->dyn_stencil_wmask[1] = wMask & 0xFFu; }
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags mask, uint32_t ref) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    if (mask & VK_STENCIL_FACE_FRONT_BIT) { scb->has_dyn_stencil_ref[0] = true; scb->dyn_stencil_ref[0] = ref & 0xFFu; }
    if (mask & VK_STENCIL_FACE_BACK_BIT)  { scb->has_dyn_stencil_ref[1] = true; scb->dyn_stencil_ref[1] = ref & 0xFFu; }
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDescriptorSetLayoutSupport(
        VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, VkDescriptorSetLayoutSupport* sup) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetDescriptorSetLayoutSupport)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetDescriptorSetLayoutSupport");
    if (!fn || !ci) { if (sup) sup->supported = VK_TRUE; return; }
    std::vector<VkDescriptorSetLayoutBinding> b(ci->bindingCount);
    for (uint32_t i = 0; i < ci->bindingCount; ++i) {
        b[i] = ci->pBindings[i];
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo mci = *ci;
    mci.pBindings = b.data();
    fn(dev->real_dev, &mci, sup);
}
static VKAPI_ATTR void VKAPI_CALL sm_TrimCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolTrimFlags f) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkTrimCommandPool)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkTrimCommandPool");
    if (fn) fn(dev->real_dev, p, f);
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceGroupPeerMemoryFeatures(
        VkDevice, uint32_t, uint32_t, uint32_t, VkPeerMemoryFeatureFlags* f) {
    if (f) *f = VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT | VK_PEER_MEMORY_FEATURE_COPY_DST_BIT |
                VK_PEER_MEMORY_FEATURE_GENERIC_SRC_BIT | VK_PEER_MEMORY_FEATURE_GENERIC_DST_BIT;
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdSetDeviceMask(VkCommandBuffer, uint32_t) {}
static VKAPI_ATTR void VKAPI_CALL sm_CmdDispatchBase(VkCommandBuffer cb, uint32_t, uint32_t, uint32_t, uint32_t x, uint32_t y, uint32_t z) {
    auto* scb = reinterpret_cast<SmCommandBuffer*>(cb);
    scb->dev->fn.CmdDispatch(scb->real_cb, x, y, z);
}
static VKAPI_ATTR uint64_t VKAPI_CALL sm_GetBufferOpaqueCaptureAddress(VkDevice d, const VkBufferDeviceAddressInfo* info) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetBufferOpaqueCaptureAddress)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetBufferOpaqueCaptureAddress");
    if (!fn || !info) return 0;
    VkBufferDeviceAddressInfo m = *info; m.buffer = from_h<SmBuffer>(info->buffer)->real_buf;
    return fn(dev->real_dev, &m);
}
static VKAPI_ATTR uint64_t VKAPI_CALL sm_GetDeviceMemoryOpaqueCaptureAddress(VkDevice d, const VkDeviceMemoryOpaqueCaptureAddressInfo* info) {
    auto* dev = reinterpret_cast<SmDevice*>(d);
    auto fn = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)dev->pd->inst->fn.GetDeviceProcAddr(dev->real_dev, "vkGetDeviceMemoryOpaqueCaptureAddress");
    return (fn && info) ? fn(dev->real_dev, info) : 0;
}
static VKAPI_ATTR void VKAPI_CALL sm_GetDeviceImageSparseMemoryRequirements(
        VkDevice, const VkDeviceImageMemoryRequirements*, uint32_t* count, VkSparseImageMemoryRequirements2*) {
    if (count) *count = 0;
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_CreatePrivateDataSlot(VkDevice, const VkPrivateDataSlotCreateInfo*, const VkAllocationCallbacks*, VkPrivateDataSlot* out) {
    if (out) *out = (VkPrivateDataSlot)(uintptr_t)(new std::unordered_map<uint64_t, uint64_t>());
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_DestroyPrivateDataSlot(VkDevice, VkPrivateDataSlot s, const VkAllocationCallbacks*) {
    if (s) delete reinterpret_cast<std::unordered_map<uint64_t, uint64_t>*>((uintptr_t)s);
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_SetPrivateData(VkDevice, VkObjectType, uint64_t obj, VkPrivateDataSlot s, uint64_t data) {
    if (s) (*reinterpret_cast<std::unordered_map<uint64_t, uint64_t>*>((uintptr_t)s))[obj] = data;
    return VK_SUCCESS;
}
static VKAPI_ATTR void VKAPI_CALL sm_GetPrivateData(VkDevice, VkObjectType, uint64_t obj, VkPrivateDataSlot s, uint64_t* out) {
    if (!out) return;
    *out = 0;
    if (s) {
        auto* m = reinterpret_cast<std::unordered_map<uint64_t, uint64_t>*>((uintptr_t)s);
        auto it = m->find(obj);
        if (it != m->end()) *out = it->second;
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL sm_AcquireNextImage2KHR(VkDevice d, const VkAcquireNextImageInfoKHR* info, uint32_t* idx) {
    return sm_AcquireNextImageKHR(d, info->swapchain, info->timeout, info->semaphore, info->fence, idx);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushDescriptorSet2KHR(VkCommandBuffer cb, const VkPushDescriptorSetInfoKHR* info) {
    if (!info) return;
    if (info->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT))
        sm_CmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, info->set, info->descriptorWriteCount, info->pDescriptorWrites);
    if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
        sm_CmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, info->layout, info->set, info->descriptorWriteCount, info->pDescriptorWrites);
}
static VKAPI_ATTR void VKAPI_CALL sm_CmdPushDescriptorSetWithTemplate2KHR(VkCommandBuffer cb, const VkPushDescriptorSetWithTemplateInfoKHR* info) {
    if (!info) return;
    sm_CmdPushDescriptorSetWithTemplateKHR(cb, info->descriptorUpdateTemplate, info->layout, info->set, info->pData);
}

static VKAPI_ATTR void VKAPI_CALL sm_CmdNextSubpass(VkCommandBuffer, VkSubpassContents) {}

// =============================================================================
// Dispatch Table & Official Khronos Vulkan ICD Entry Points
// =============================================================================
static PFN_vkVoidFunction sm_lookup_fn(const char* name) {
    if (!name) return nullptr;
#define MAP(n) if (strcmp(name, "vk" #n) == 0) return (PFN_vkVoidFunction)sm_##n
    MAP(EnumerateInstanceExtensionProperties);
    MAP(EnumerateInstanceLayerProperties);
    MAP(EnumerateInstanceVersion);
    MAP(CreateInstance);
    MAP(DestroyInstance);
    MAP(EnumeratePhysicalDevices);
    MAP(EnumeratePhysicalDeviceGroups);
    MAP(GetPhysicalDeviceProperties);
    MAP(GetPhysicalDeviceProperties2);
    MAP(GetPhysicalDeviceFeatures);
    MAP(GetPhysicalDeviceFeatures2);
    MAP(GetPhysicalDeviceQueueFamilyProperties);
    MAP(GetPhysicalDeviceQueueFamilyProperties2);
    MAP(GetPhysicalDeviceMemoryProperties);
    MAP(GetPhysicalDeviceMemoryProperties2);
    MAP(GetPhysicalDeviceFormatProperties);
    MAP(GetPhysicalDeviceFormatProperties2);
    MAP(GetPhysicalDeviceImageFormatProperties);
    MAP(GetPhysicalDeviceImageFormatProperties2);
    MAP(GetPhysicalDeviceSparseImageFormatProperties);
    MAP(GetPhysicalDeviceSparseImageFormatProperties2);
    MAP(GetPhysicalDeviceExternalBufferProperties);
    MAP(GetPhysicalDeviceExternalFenceProperties);
    MAP(GetPhysicalDeviceExternalSemaphoreProperties);
    MAP(GetPhysicalDeviceToolProperties);
    MAP(EnumerateDeviceExtensionProperties);
    MAP(EnumerateDeviceLayerProperties);
    MAP(CreateDevice);
    MAP(DestroyDevice);
    MAP(GetDeviceQueue);
    MAP(GetDeviceQueue2);
    MAP(DeviceWaitIdle);
    MAP(QueueWaitIdle);
    MAP(QueueSubmit);
    MAP(AllocateMemory);
    MAP(FreeMemory);
    MAP(MapMemory);
    MAP(UnmapMemory);
    MAP(FlushMappedMemoryRanges);
    MAP(InvalidateMappedMemoryRanges);
    MAP(CreateBuffer);
    MAP(DestroyBuffer);
    MAP(GetBufferMemoryRequirements);
    MAP(GetBufferMemoryRequirements2);
    MAP(BindBufferMemory);
    MAP(BindBufferMemory2);
    MAP(CreateImage);
    MAP(DestroyImage);
    MAP(GetImageMemoryRequirements);
    MAP(GetImageMemoryRequirements2);
    MAP(BindImageMemory);
    MAP(BindImageMemory2);
    MAP(GetImageSubresourceLayout);
    MAP(CreateImageView);
    MAP(DestroyImageView);
    MAP(CreateSampler);
    MAP(DestroySampler);
    MAP(CreateShaderModule);
    MAP(DestroyShaderModule);
    MAP(CreateDescriptorSetLayout);
    MAP(DestroyDescriptorSetLayout);
    MAP(CreateDescriptorPool);
    MAP(DestroyDescriptorPool);
    MAP(ResetDescriptorPool);
    MAP(AllocateDescriptorSets);
    MAP(FreeDescriptorSets);
    MAP(UpdateDescriptorSets);
    MAP(CreatePipelineLayout);
    MAP(DestroyPipelineLayout);
    MAP(CreateRenderPass);
    MAP(CreateRenderPass2);
    MAP(DestroyRenderPass);
    MAP(CreateFramebuffer);
    MAP(DestroyFramebuffer);
    MAP(CreatePipelineCache);
    MAP(DestroyPipelineCache);
    MAP(GetPipelineCacheData);
    MAP(CreateComputePipelines);
    MAP(CreateGraphicsPipelines);
    MAP(DestroyPipeline);
    MAP(CreateCommandPool);
    MAP(DestroyCommandPool);
    MAP(ResetCommandPool);
    MAP(AllocateCommandBuffers);
    MAP(FreeCommandBuffers);
    MAP(BeginCommandBuffer);
    MAP(EndCommandBuffer);
    MAP(ResetCommandBuffer);
    MAP(CreateFence);
    MAP(DestroyFence);
    MAP(ResetFences);
    MAP(GetFenceStatus);
    MAP(WaitForFences);
    MAP(CreateSemaphore);
    MAP(DestroySemaphore);
    MAP(GetSemaphoreCounterValue);
    MAP(WaitSemaphores);
    MAP(SignalSemaphore);
    MAP(CreateEvent);
    MAP(DestroyEvent);
    MAP(GetEventStatus);
    MAP(SetEvent);
    MAP(ResetEvent);
    MAP(CreateQueryPool);
    MAP(DestroyQueryPool);
    MAP(GetQueryPoolResults);
    MAP(ResetQueryPool);
    MAP(CreateBufferView);
    MAP(DestroyBufferView);
    MAP(GetBufferDeviceAddress);
    MAP(CreateDescriptorUpdateTemplate);
    MAP(DestroyDescriptorUpdateTemplate);
    MAP(UpdateDescriptorSetWithTemplate);
    MAP(GetDeviceBufferMemoryRequirements);
    MAP(GetDeviceImageMemoryRequirements);
    MAP(QueueSubmit2);
    MAP(CmdResetQueryPool);
    MAP(CmdBeginQuery);
    MAP(CmdEndQuery);
    MAP(CmdWriteTimestamp);
    MAP(CmdWriteTimestamp2);
    MAP(CmdCopyQueryPoolResults);
    MAP(CmdCopyBuffer2);
    MAP(CmdCopyImage2);
    MAP(CmdCopyBufferToImage2);
    MAP(CmdCopyImageToBuffer2);
    MAP(CmdBlitImage2);
    MAP(CmdResolveImage);
    MAP(CmdResolveImage2);
    MAP(CmdClearAttachments);
    MAP(CmdDrawIndirect);
    MAP(CmdDrawIndexedIndirect);
    MAP(CmdDrawIndirectCount);
    MAP(CmdDrawIndexedIndirectCount);
    MAP(CmdSetEvent);
    MAP(CmdResetEvent);
    MAP(CmdWaitEvents);
    MAP(CmdSetEvent2);
    MAP(CmdResetEvent2);
    MAP(CmdWaitEvents2);
    MAP(CmdBeginQueryIndexedEXT);
    MAP(CmdEndQueryIndexedEXT);
    MAP(MergePipelineCaches);
    MAP(GetDeviceMemoryCommitment);
    MAP(GetRenderAreaGranularity);
    MAP(GetImageSparseMemoryRequirements);
    MAP(GetImageSparseMemoryRequirements2);
    MAP(QueueBindSparse);
    MAP(SetDebugUtilsObjectNameEXT);
    MAP(SetDebugUtilsObjectTagEXT);
    MAP(CmdBeginDebugUtilsLabelEXT);
    MAP(CmdEndDebugUtilsLabelEXT);
    MAP(CmdInsertDebugUtilsLabelEXT);
    MAP(QueueBeginDebugUtilsLabelEXT);
    MAP(QueueEndDebugUtilsLabelEXT);
    MAP(QueueInsertDebugUtilsLabelEXT);
    MAP(CreateDebugUtilsMessengerEXT);
    MAP(DestroyDebugUtilsMessengerEXT);
    MAP(SubmitDebugUtilsMessageEXT);
    MAP(CmdExecuteCommands);
    MAP(CmdBeginRenderPass);
    MAP(CmdBeginRenderPass2);
    MAP(CmdEndRenderPass);
    MAP(CmdEndRenderPass2);
    MAP(CmdNextSubpass);
    MAP(CmdNextSubpass2);
    MAP(CmdBeginRendering);
    MAP(CmdEndRendering);
    MAP(GetDescriptorSetLayoutSupport);
    MAP(TrimCommandPool);
    MAP(GetDeviceGroupPeerMemoryFeatures);
    MAP(CmdSetDeviceMask);
    MAP(CmdDispatchBase);
    MAP(GetBufferOpaqueCaptureAddress);
    MAP(GetDeviceMemoryOpaqueCaptureAddress);
    MAP(GetDeviceImageSparseMemoryRequirements);
    MAP(CreatePrivateDataSlot);
    MAP(DestroyPrivateDataSlot);
    MAP(SetPrivateData);
    MAP(GetPrivateData);
    MAP(AcquireNextImage2KHR);
    MAP(CmdPushDescriptorSet2KHR);
    MAP(CmdPushDescriptorSetWithTemplate2KHR);
    MAP(CmdBindPipeline);
    MAP(CmdBindDescriptorSets);
    MAP(CmdPushConstants);
    MAP(CmdPushDescriptorSetKHR);
    MAP(CmdPushDescriptorSetWithTemplateKHR);
    MAP(CmdBeginConditionalRenderingEXT);
    MAP(CmdEndConditionalRenderingEXT);
    MAP(CmdSetViewport);
    MAP(CmdSetViewportWithCount);
    MAP(CmdSetScissor);
    MAP(CmdSetScissorWithCount);
    MAP(CmdSetCullMode);
    MAP(CmdSetFrontFace);
    MAP(CmdSetPrimitiveTopology);
    MAP(CmdSetDepthTestEnable);
    MAP(CmdSetDepthWriteEnable);
    MAP(CmdSetDepthCompareOp);
    MAP(CmdSetDepthBoundsTestEnable);
    MAP(CmdSetStencilTestEnable);
    MAP(CmdSetStencilOp);
    MAP(CmdSetLineWidth);
    MAP(CmdSetDepthBias);
    MAP(CmdSetBlendConstants);
    MAP(CmdSetDepthBounds);
    MAP(CmdSetStencilCompareMask);
    MAP(CmdSetStencilWriteMask);
    MAP(CmdSetStencilReference);
    MAP(CmdBindVertexBuffers);
    MAP(CmdBindVertexBuffers2);
    MAP(CmdBindIndexBuffer);
    MAP(CmdDraw);
    MAP(CmdDrawIndexed);
    MAP(CmdDispatch);
    MAP(CmdDispatchIndirect);
    MAP(CmdCopyBuffer);
    MAP(CmdCopyImage);
    MAP(CmdCopyBufferToImage);
    MAP(CmdCopyImageToBuffer);
    MAP(CmdBlitImage);
    MAP(CmdFillBuffer);
    MAP(CmdUpdateBuffer);
    MAP(CmdClearColorImage);
    MAP(CmdClearDepthStencilImage);
    MAP(CmdPipelineBarrier);
    MAP(CmdPipelineBarrier2);
    MAP(CreateXcbSurfaceKHR);
    MAP(CreateXlibSurfaceKHR);
    MAP(DestroySurfaceKHR);
    MAP(GetPhysicalDeviceXcbPresentationSupportKHR);
    MAP(GetPhysicalDeviceXlibPresentationSupportKHR);
    MAP(GetPhysicalDeviceSurfaceSupportKHR);
    MAP(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    MAP(GetPhysicalDeviceSurfaceCapabilities2KHR);
    MAP(GetPhysicalDeviceSurfaceFormatsKHR);
    MAP(GetPhysicalDeviceSurfaceFormats2KHR);
    MAP(GetPhysicalDeviceSurfacePresentModesKHR);
    MAP(CreateSwapchainKHR);
    MAP(DestroySwapchainKHR);
    MAP(GetSwapchainImagesKHR);
    MAP(AcquireNextImageKHR);
    MAP(QueuePresentKHR);
#undef MAP
    // Aliases for KHR physical device & device queries
    if (strcmp(name, "vkGetPhysicalDeviceProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceProperties2;
    if (strcmp(name, "vkGetPhysicalDeviceFeatures2KHR") == 0)   return (PFN_vkVoidFunction)sm_GetPhysicalDeviceFeatures2;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceMemoryProperties2;
    if (strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceQueueFamilyProperties2;
    if (strcmp(name, "vkGetPhysicalDeviceFormatProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceFormatProperties2;
    if (strcmp(name, "vkGetPhysicalDeviceImageFormatProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceImageFormatProperties2;
    if (strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceSparseImageFormatProperties2;
    if (strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR") == 0) return (PFN_vkVoidFunction)sm_EnumeratePhysicalDeviceGroups;
    if (strcmp(name, "vkGetPhysicalDeviceExternalBufferPropertiesKHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceExternalBufferProperties;
    if (strcmp(name, "vkGetPhysicalDeviceExternalFencePropertiesKHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceExternalFenceProperties;
    if (strcmp(name, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceExternalSemaphoreProperties;
    if (strcmp(name, "vkGetPhysicalDeviceToolPropertiesEXT") == 0) return (PFN_vkVoidFunction)sm_GetPhysicalDeviceToolProperties;
    if (strcmp(name, "vkGetBufferMemoryRequirements2KHR") == 0) return (PFN_vkVoidFunction)sm_GetBufferMemoryRequirements2;
    if (strcmp(name, "vkGetImageMemoryRequirements2KHR") == 0) return (PFN_vkVoidFunction)sm_GetImageMemoryRequirements2;
    if (strcmp(name, "vkBindBufferMemory2KHR") == 0) return (PFN_vkVoidFunction)sm_BindBufferMemory2;
    if (strcmp(name, "vkBindImageMemory2KHR") == 0) return (PFN_vkVoidFunction)sm_BindImageMemory2;
    if (strcmp(name, "vkCreateRenderPass2KHR") == 0) return (PFN_vkVoidFunction)sm_CreateRenderPass2;
    if (strcmp(name, "vkCmdBeginRenderPass2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdBeginRenderPass2;
    if (strcmp(name, "vkCmdEndRenderPass2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdEndRenderPass2;
    if (strcmp(name, "vkCmdNextSubpass2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdNextSubpass2;
    if (strcmp(name, "vkCmdBeginRenderingKHR") == 0) return (PFN_vkVoidFunction)sm_CmdBeginRendering;
    if (strcmp(name, "vkCmdEndRenderingKHR") == 0) return (PFN_vkVoidFunction)sm_CmdEndRendering;
    if (strcmp(name, "vkCmdPipelineBarrier2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdPipelineBarrier2;
    if (strcmp(name, "vkQueueSubmit2KHR") == 0) return (PFN_vkVoidFunction)sm_QueueSubmit2;
    if (strcmp(name, "vkCmdSetCullModeEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetCullMode;
    if (strcmp(name, "vkCmdSetFrontFaceEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetFrontFace;
    if (strcmp(name, "vkCmdSetPrimitiveTopologyEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetPrimitiveTopology;
    if (strcmp(name, "vkCmdSetViewportWithCountEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetViewportWithCount;
    if (strcmp(name, "vkCmdSetScissorWithCountEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetScissorWithCount;
    if (strcmp(name, "vkCmdBindVertexBuffers2EXT") == 0) return (PFN_vkVoidFunction)sm_CmdBindVertexBuffers2;
    if (strcmp(name, "vkCmdSetDepthTestEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetDepthTestEnable;
    if (strcmp(name, "vkCmdSetDepthWriteEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetDepthWriteEnable;
    if (strcmp(name, "vkCmdSetDepthCompareOpEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetDepthCompareOp;
    if (strcmp(name, "vkCmdSetDepthBoundsTestEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetDepthBoundsTestEnable;
    if (strcmp(name, "vkCmdSetStencilTestEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetStencilTestEnable;
    if (strcmp(name, "vkCmdSetStencilOpEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetStencilOp;
    if (strcmp(name, "vkCmdSetRasterizerDiscardEnable") == 0 || strcmp(name, "vkCmdSetRasterizerDiscardEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetRasterizerDiscardEnable;
    if (strcmp(name, "vkCmdSetDepthBiasEnable") == 0 || strcmp(name, "vkCmdSetDepthBiasEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetDepthBiasEnable;
    if (strcmp(name, "vkCmdSetPrimitiveRestartEnable") == 0 || strcmp(name, "vkCmdSetPrimitiveRestartEnableEXT") == 0) return (PFN_vkVoidFunction)sm_CmdSetPrimitiveRestartEnable;
    if (strcmp(name, "vkCmdBeginTransformFeedbackEXT") == 0) return (PFN_vkVoidFunction)sm_CmdBeginTransformFeedbackEXT;
    if (strcmp(name, "vkCmdEndTransformFeedbackEXT") == 0) return (PFN_vkVoidFunction)sm_CmdEndTransformFeedbackEXT;
    if (strcmp(name, "vkCmdBindTransformFeedbackBuffersEXT") == 0) return (PFN_vkVoidFunction)sm_CmdBindTransformFeedbackBuffersEXT;
    if (strcmp(name, "vkCmdDrawIndirectByteCountEXT") == 0) return (PFN_vkVoidFunction)sm_CmdDrawIndirectByteCountEXT;
    if (strcmp(name, "vkGetSemaphoreCounterValueKHR") == 0) return (PFN_vkVoidFunction)sm_GetSemaphoreCounterValue;
    if (strcmp(name, "vkWaitSemaphoresKHR") == 0) return (PFN_vkVoidFunction)sm_WaitSemaphores;
    if (strcmp(name, "vkSignalSemaphoreKHR") == 0) return (PFN_vkVoidFunction)sm_SignalSemaphore;
    if (strcmp(name, "vkGetBufferDeviceAddressKHR") == 0) return (PFN_vkVoidFunction)sm_GetBufferDeviceAddress;
    if (strcmp(name, "vkResetQueryPoolEXT") == 0) return (PFN_vkVoidFunction)sm_ResetQueryPool;
    if (strcmp(name, "vkCreateDescriptorUpdateTemplateKHR") == 0) return (PFN_vkVoidFunction)sm_CreateDescriptorUpdateTemplate;
    if (strcmp(name, "vkDestroyDescriptorUpdateTemplateKHR") == 0) return (PFN_vkVoidFunction)sm_DestroyDescriptorUpdateTemplate;
    if (strcmp(name, "vkUpdateDescriptorSetWithTemplateKHR") == 0) return (PFN_vkVoidFunction)sm_UpdateDescriptorSetWithTemplate;
    if (strcmp(name, "vkCmdCopyBuffer2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdCopyBuffer2;
    if (strcmp(name, "vkCmdCopyImage2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdCopyImage2;
    if (strcmp(name, "vkCmdCopyBufferToImage2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdCopyBufferToImage2;
    if (strcmp(name, "vkCmdCopyImageToBuffer2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdCopyImageToBuffer2;
    if (strcmp(name, "vkCmdBlitImage2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdBlitImage2;
    if (strcmp(name, "vkCmdResolveImage2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdResolveImage2;
    if (strcmp(name, "vkCmdWriteTimestamp2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdWriteTimestamp2;
    if (strcmp(name, "vkCmdDrawIndirectCountKHR") == 0) return (PFN_vkVoidFunction)sm_CmdDrawIndirectCount;
    if (strcmp(name, "vkCmdDrawIndexedIndirectCountKHR") == 0) return (PFN_vkVoidFunction)sm_CmdDrawIndexedIndirectCount;
    if (strcmp(name, "vkGetDeviceBufferMemoryRequirementsKHR") == 0) return (PFN_vkVoidFunction)sm_GetDeviceBufferMemoryRequirements;
    if (strcmp(name, "vkGetDeviceImageMemoryRequirementsKHR") == 0) return (PFN_vkVoidFunction)sm_GetDeviceImageMemoryRequirements;
    if (strcmp(name, "vkCmdSetEvent2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdSetEvent2;
    if (strcmp(name, "vkCmdResetEvent2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdResetEvent2;
    if (strcmp(name, "vkCmdWaitEvents2KHR") == 0) return (PFN_vkVoidFunction)sm_CmdWaitEvents2;
    if (strcmp(name, "vkGetImageSparseMemoryRequirements2KHR") == 0) return (PFN_vkVoidFunction)sm_GetImageSparseMemoryRequirements2;
    if (strcmp(name, "vkCmdBindIndexBuffer2KHR") == 0 || strcmp(name, "vkCmdBindIndexBuffer2") == 0) return (PFN_vkVoidFunction)sm_CmdBindIndexBuffer2KHR;
    if (strcmp(name, "vkCmdBindDescriptorSets2KHR") == 0 || strcmp(name, "vkCmdBindDescriptorSets2") == 0) return (PFN_vkVoidFunction)sm_CmdBindDescriptorSets2KHR;
    if (strcmp(name, "vkCmdPushConstants2KHR") == 0 || strcmp(name, "vkCmdPushConstants2") == 0) return (PFN_vkVoidFunction)sm_CmdPushConstants2KHR;
    if (strcmp(name, "vkGetRenderingAreaGranularityKHR") == 0 || strcmp(name, "vkGetRenderingAreaGranularity") == 0) return (PFN_vkVoidFunction)sm_GetRenderingAreaGranularityKHR;
    if (strcmp(name, "vkGetImageSubresourceLayout2KHR") == 0 || strcmp(name, "vkGetImageSubresourceLayout2EXT") == 0 || strcmp(name, "vkGetImageSubresourceLayout2") == 0) return (PFN_vkVoidFunction)sm_GetImageSubresourceLayout2KHR;
    if (strcmp(name, "vkGetDeviceImageSubresourceLayoutKHR") == 0 || strcmp(name, "vkGetDeviceImageSubresourceLayout") == 0) return (PFN_vkVoidFunction)sm_GetDeviceImageSubresourceLayoutKHR;
    if (strcmp(name, "vkGetDescriptorSetLayoutSupportKHR") == 0) return (PFN_vkVoidFunction)sm_GetDescriptorSetLayoutSupport;
    if (strcmp(name, "vkTrimCommandPoolKHR") == 0) return (PFN_vkVoidFunction)sm_TrimCommandPool;
    if (strcmp(name, "vkGetDeviceGroupPeerMemoryFeaturesKHR") == 0) return (PFN_vkVoidFunction)sm_GetDeviceGroupPeerMemoryFeatures;
    if (strcmp(name, "vkCmdSetDeviceMaskKHR") == 0) return (PFN_vkVoidFunction)sm_CmdSetDeviceMask;
    if (strcmp(name, "vkCmdDispatchBaseKHR") == 0) return (PFN_vkVoidFunction)sm_CmdDispatchBase;
    if (strcmp(name, "vkGetBufferOpaqueCaptureAddressKHR") == 0) return (PFN_vkVoidFunction)sm_GetBufferOpaqueCaptureAddress;
    if (strcmp(name, "vkGetDeviceMemoryOpaqueCaptureAddressKHR") == 0) return (PFN_vkVoidFunction)sm_GetDeviceMemoryOpaqueCaptureAddress;
    if (strcmp(name, "vkGetDeviceImageSparseMemoryRequirementsKHR") == 0) return (PFN_vkVoidFunction)sm_GetDeviceImageSparseMemoryRequirements;
    if (strcmp(name, "vkCreatePrivateDataSlotEXT") == 0) return (PFN_vkVoidFunction)sm_CreatePrivateDataSlot;
    if (strcmp(name, "vkDestroyPrivateDataSlotEXT") == 0) return (PFN_vkVoidFunction)sm_DestroyPrivateDataSlot;
    if (strcmp(name, "vkSetPrivateDataEXT") == 0) return (PFN_vkVoidFunction)sm_SetPrivateData;
    if (strcmp(name, "vkGetPrivateDataEXT") == 0) return (PFN_vkVoidFunction)sm_GetPrivateData;
    if (strcmp(name, "vkCmdPushDescriptorSet") == 0) return (PFN_vkVoidFunction)sm_CmdPushDescriptorSetKHR;
    if (strcmp(name, "vkCmdPushDescriptorSetWithTemplate") == 0) return (PFN_vkVoidFunction)sm_CmdPushDescriptorSetWithTemplateKHR;
    if (strcmp(name, "vkCmdPushDescriptorSet2") == 0) return (PFN_vkVoidFunction)sm_CmdPushDescriptorSet2KHR;
    if (strcmp(name, "vkCmdPushDescriptorSetWithTemplate2") == 0) return (PFN_vkVoidFunction)sm_CmdPushDescriptorSetWithTemplate2KHR;
    if (getenv("SMRAST_LOG_MISSING")) fprintf(stderr, "[SmRast] MISSING FN: %s\n", name);
    return nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL sm_GetDeviceProcAddr(VkDevice, const char* name) {
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)sm_GetDeviceProcAddr;
    return sm_lookup_fn(name);
}

SM_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance, const char* name) {
    if (!name) return nullptr;
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vk_icdGetInstanceProcAddr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0)   return (PFN_vkVoidFunction)sm_GetDeviceProcAddr;
    return sm_lookup_fn(name);
}

SM_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetPhysicalDeviceProcAddr(VkInstance, const char* name) {
    return sm_lookup_fn(name);
}

SM_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t* pVersion) {
    if (*pVersion > 5) *pVersion = 5;
    return VK_SUCCESS;
}
