// smrast_stress_bench.cpp — Standard Unmodified Vulkan 1.3 Graphics PBR Benchmark
// Runs identically on NVIDIA H20 (SmRast-VK 78 SMs), CPU Lavapipe, or any hardware Vulkan GPU!
// Tests: 1080p offscreen RenderPass, 100K..2M+ indexed triangles, Hardware sampler2D + dFdx/dFdy,
//        Cook-Torrance PBR shading, Near-Plane Clipping, Depth Testing, and Alpha-Blended Glass!

#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>

#define VK_CHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "VK ERR %d at %s:%d\n", _r, __FILE__, __LINE__); exit(1); } } while(0)

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
    float r, g, b, a;
};

struct SceneUBO {
    float viewProj[16];
    float camPos[4];
    float sunDir[4];
    float pointPos[4];
    float pointColor[4];
    float timeParams[4];
};

static void mat4_mul(const float a[16], const float b[16], float o[16]) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
}
static void mat4_persp_vk(float fovy_rad, float aspect, float zn, float zf, float m[16]) {
    memset(m, 0, 64);
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    m[0] = f / aspect;
    m[5] = -f; // Vulkan Y-down NDC
    m[10] = zf / (zn - zf);
    m[11] = -1.0f;
    m[14] = (zf * zn) / (zn - zf);
}
static void mat4_lookat(float ex, float ey, float ez, float cx, float cy, float cz, float m[16]) {
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz); fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0.0f - fz * 1.0f, sy = fz * 0.0f - fx * 0.0f, sz = fx * 1.0f - fy * 0.0f;
    // up = (0, 1, 0): s = f x up
    sx = -fz; sy = 0.0f; sz = fx;
    float sl = sqrtf(sx*sx + sy*sy + sz*sz); sx /= sl; sy /= sl; sz /= sl;
    float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    m[0] = sx;  m[4] = sy;  m[8]  = sz;  m[12] = -(sx*ex + sy*ey + sz*ez);
    m[1] = ux;  m[5] = uy;  m[9]  = uz;  m[13] = -(ux*ex + uy*ey + uz*ez);
    m[2] = -fx; m[6] = -fy; m[10] = -fz; m[14] =  (fx*ex + fy*ey + fz*ez);
    m[3] = 0;   m[7] = 0;   m[11] = 0;   m[15] = 1.0f;
}

static void add_sculpted_sphere(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs,
                                float cx, float cy, float cz, float rad, int seg_u, int seg_v,
                                float cr, float cg, float cb, float freq) {
    uint32_t base = (uint32_t)verts.size();
    for (int iv = 0; iv <= seg_v; ++iv) {
        float v = (float)iv / seg_v;
        float phi = v * 3.14159265f;
        for (int iu = 0; iu <= seg_u; ++iu) {
            float u = (float)iu / seg_u;
            float theta = u * 6.2831853f;
            float dx = sinf(phi) * cosf(theta);
            float dy = cosf(phi);
            float dz = sinf(phi) * sinf(theta);
            float disp = 1.0f + 0.14f * sinf(theta * freq) * sinf(phi * freq);
            float r = rad * disp;
            // Perturbed normal for crisp PBR micro-facets
            float nx = dx + 0.18f * cosf(theta * freq) * sinf(phi * freq);
            float ny = dy + 0.18f * sinf(theta * freq) * cosf(phi * freq);
            float nz = dz;
            float nl = sqrtf(nx*nx + ny*ny + nz*nz);
            verts.push_back({cx + dx * r, cy + dy * r, cz + dz * r,
                             nx/nl, ny/nl, nz/nl, u * 4.0f, v * 2.0f, cr, cg, cb, 1.0f});
        }
    }
    int stride = seg_u + 1;
    for (int iv = 0; iv < seg_v; ++iv) {
        for (int iu = 0; iu < seg_u; ++iu) {
            uint32_t a = base + iv * stride + iu;
            uint32_t b = a + 1;
            uint32_t c = a + stride;
            uint32_t d = c + 1;
            idxs.push_back(a); idxs.push_back(b); idxs.push_back(c);
            idxs.push_back(b); idxs.push_back(d); idxs.push_back(c);
        }
    }
}

static void add_torus_knot(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs,
                           float cx, float cy, float cz, float scale, int seg_p, int seg_t,
                           float cr, float cg, float cb) {
    uint32_t base = (uint32_t)verts.size();
    const float P = 3.0f, Q = 2.0f;
    for (int i = 0; i <= seg_p; ++i) {
        float u = (float)i / seg_p;
        float t = u * 6.2831853f;
        float r = (2.0f + cosf(P * t)) * 0.45f * scale;
        float px = r * cosf(Q * t), py = sinf(P * t) * 0.45f * scale, pz = r * sinf(Q * t);
        float t2 = t + 0.005f;
        float r2 = (2.0f + cosf(P * t2)) * 0.45f * scale;
        float tx = r2 * cosf(Q * t2) - px, ty = sinf(P * t2) * 0.45f * scale - py, tz = r2 * sinf(Q * t2) - pz;
        float tl = sqrtf(tx*tx + ty*ty + tz*tz); tx /= tl; ty /= tl; tz /= tl;
        float bx = -tz, by = 0.0f, bz = tx;
        float bl = sqrtf(bx*bx + bz*bz) + 1e-6f; bx /= bl; bz /= bl;
        float nx0 = ty * bz - tz * by, ny0 = tz * bx - tx * bz, nz0 = tx * by - ty * bx;
        float tube = 0.24f * scale;
        for (int j = 0; j <= seg_t; ++j) {
            float v = (float)j / seg_t;
            float a = v * 6.2831853f;
            float ca = cosf(a), sa = sinf(a);
            float nrmx = nx0 * ca + bx * sa, nrmy = ny0 * ca + by * sa, nrmz = nz0 * ca + bz * sa;
            verts.push_back({cx + px + nrmx * tube, cy + py + nrmy * tube, cz + pz + nrmz * tube,
                             nrmx, nrmy, nrmz, u * 16.0f, v * 3.0f, cr, cg, cb, 1.0f});
        }
    }
    int stride = seg_t + 1;
    for (int i = 0; i < seg_p; ++i) {
        for (int j = 0; j < seg_t; ++j) {
            uint32_t a = base + i * stride + j;
            uint32_t b = a + 1;
            uint32_t c = a + stride;
            uint32_t d = c + 1;
            idxs.push_back(a); idxs.push_back(c); idxs.push_back(b);
            idxs.push_back(b); idxs.push_back(c); idxs.push_back(d);
        }
    }
}

static void add_quad(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs,
                     Vertex a, Vertex b, Vertex c, Vertex d) {
    uint32_t base = (uint32_t)verts.size();
    verts.push_back(a); verts.push_back(b); verts.push_back(c); verts.push_back(d);
    idxs.push_back(base+0); idxs.push_back(base+1); idxs.push_back(base+2);
    idxs.push_back(base+0); idxs.push_back(base+2); idxs.push_back(base+3);
}

static std::vector<uint32_t> load_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> buf(sz / 4);
    f.read((char*)buf.data(), sz);
    return buf;
}

static uint32_t find_mem(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & req) == req)) return i;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if (bits & (1u << i)) return i;
    return 0;
}

int main(int argc, char** argv) {
    uint32_t W = 1920, H = 1080;
    uint32_t target_tris = 1000000;
    int warmup = 5, frames = 25;
    const char* out_ppm = nullptr;
    const char* shader_dir = "./bench";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--tris") && i + 1 < argc) target_tris = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) W = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) H = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_ppm = argv[++i];
        else if (!strcmp(argv[i], "--shaders") && i + 1 < argc) shader_dir = argv[++i];
    }

    // 1. Build Procedural Multi-Object 3D PBR Scene (Ground + Hero Torus Knot + Sculpted Sphere Field + Glass Panels)
    std::vector<Vertex> opaque_v, glass_v;
    std::vector<uint32_t> opaque_i, glass_i;

    // Large ground plane extending BEHIND the camera to exercise Near-Plane Clipping!
    int gN = 24;
    float gS = 18.0f;
    for (int gz = 0; gz < gN; ++gz) {
        for (int gx = 0; gx < gN; ++gx) {
            float x0 = -gS + (2.0f * gS * gx) / gN, x1 = -gS + (2.0f * gS * (gx + 1)) / gN;
            float z0 = -gS + (2.0f * gS * gz) / gN, z1 = -gS + (2.0f * gS * (gz + 1)) / gN;
            float y = -2.2f;
            float u0 = x0 * 0.45f, u1 = x1 * 0.45f, v0 = z0 * 0.45f, v1 = z1 * 0.45f;
            add_quad(opaque_v, opaque_i,
                {x0, y, z0, 0,1,0, u0,v0, 0.72f,0.78f,0.86f,1.0f},
                {x0, y, z1, 0,1,0, u0,v1, 0.72f,0.78f,0.86f,1.0f},
                {x1, y, z1, 0,1,0, u1,v1, 0.72f,0.78f,0.86f,1.0f},
                {x1, y, z0, 0,1,0, u1,v0, 0.72f,0.78f,0.86f,1.0f});
        }
    }

    // Distribute target_tris across 1 Hero Torus Knot + 12 Sculpted PBR Spheres!
    uint32_t hero_tris = target_tris / 3;
    int kp = std::max(64, (int)sqrtf((float)hero_tris * 1.5f));
    int kt = std::max(24, (int)(hero_tris / (2 * kp)));
    add_torus_knot(opaque_v, opaque_i, 0.0f, 0.4f, 0.0f, 1.55f, kp, kt, 0.95f, 0.72f, 0.28f);

    int num_spheres = 12;
    uint32_t per_sphere = std::max(2000u, (target_tris - (uint32_t)(opaque_i.size() / 3)) / num_spheres);
    int su = std::max(16, (int)sqrtf((float)per_sphere * 0.7f));
    int sv = std::max(12, (int)(per_sphere / (2 * su)));
    const float palette[6][3] = {
        {0.18f, 0.78f, 0.92f}, {0.92f, 0.30f, 0.36f}, {0.32f, 0.88f, 0.54f},
        {0.82f, 0.42f, 0.96f}, {0.96f, 0.62f, 0.20f}, {0.55f, 0.75f, 0.98f}
    };
    for (int s = 0; s < num_spheres; ++s) {
        float ang = (float)s / num_spheres * 6.2831853f;
        float ring = (s % 2 == 0) ? 4.3f : 6.4f;
        float sx = cosf(ang) * ring;
        float sz = sinf(ang) * ring;
        float sy = -0.8f + 0.65f * sinf(ang * 3.0f);
        const float* c = palette[s % 6];
        add_sculpted_sphere(opaque_v, opaque_i, sx, sy, sz, 0.95f, su, sv, c[0], c[1], c[2], 5.0f + (s % 3) * 2.0f);
    }

    // Two translucent tinted glass shields (tests Depth Read + No Depth Write + Alpha Blending!)
    add_quad(glass_v, glass_i,
        {-3.6f, -1.5f, 3.2f, 0,0,1, 0,0, 0.20f,0.85f,1.00f,0.34f},
        {-1.4f, -1.5f, 3.8f, 0,0,1, 1,0, 0.20f,0.85f,1.00f,0.34f},
        {-1.4f,  1.6f, 3.8f, 0,0,1, 1,1, 0.20f,0.85f,1.00f,0.34f},
        {-3.6f,  1.6f, 3.2f, 0,0,1, 0,1, 0.20f,0.85f,1.00f,0.34f});
    add_quad(glass_v, glass_i,
        { 1.5f, -1.5f, 3.8f, 0,0,1, 0,0, 1.00f,0.48f,0.18f,0.34f},
        { 3.7f, -1.5f, 3.1f, 0,0,1, 1,0, 1.00f,0.48f,0.18f,0.34f},
        { 3.7f,  1.6f, 3.1f, 0,0,1, 1,1, 1.00f,0.48f,0.18f,0.34f},
        { 1.5f,  1.6f, 3.8f, 0,0,1, 0,1, 1.00f,0.48f,0.18f,0.34f});

    uint32_t actual_tris = (uint32_t)((opaque_i.size() + glass_i.size()) / 3);

    // 2. Vulkan Init
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "SmRastBench", 1, "Bench", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai};
    VkInstance inst; VK_CHK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < nqf; ++i) if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qf, 1, &prio};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &qci};
    VkDevice dev; VK_CHK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags m, VkBuffer& b, VkDeviceMemory& mem, const void* init = nullptr) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, u, VK_SHARING_MODE_EXCLUSIVE};
        VK_CHK(vkCreateBuffer(dev, &bci, nullptr, &b));
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(mp, mr.memoryTypeBits, m)};
        VK_CHK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        VK_CHK(vkBindBufferMemory(dev, b, mem, 0));
        if (init) { void* p = nullptr; VK_CHK(vkMapMemory(dev, mem, 0, sz, 0, &p)); memcpy(p, init, sz); vkUnmapMemory(dev, mem); }
    };

    VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Upload Vertex & Index buffers to DEVICE_LOCAL HBM3 via staging!
    VkBuffer ovb, oib, gvb, gib, stg_ovb, stg_oib, ubo_buf, readback_buf;
    VkDeviceMemory ovb_m, oib_m, gvb_m, gib_m, stg_ovb_m, stg_oib_m, ubo_m, readback_m;
    size_t ov_sz = opaque_v.size() * sizeof(Vertex), oi_sz = opaque_i.size() * sizeof(uint32_t);
    size_t gv_sz = glass_v.size() * sizeof(Vertex),  gi_sz = glass_i.size() * sizeof(uint32_t);
    make_buf(ov_sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HV, stg_ovb, stg_ovb_m, opaque_v.data());
    make_buf(oi_sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HV, stg_oib, stg_oib_m, opaque_i.data());
    make_buf(ov_sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DL, ovb, ovb_m);
    make_buf(oi_sz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT  | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DL, oib, oib_m);
    make_buf(gv_sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, HV, gvb, gvb_m, glass_v.data());
    make_buf(gi_sz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  HV, gib, gib_m, glass_i.data());
    make_buf(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HV, ubo_buf, ubo_m);
    make_buf((VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HV, readback_buf, readback_m);
    SceneUBO* ubo_ptr = nullptr; VK_CHK(vkMapMemory(dev, ubo_m, 0, sizeof(SceneUBO), 0, (void**)&ubo_ptr));

    // Procedural 512x512 PBR Albedo+Smoothness Texture (checkerboard + subtle brushed grain)
    const uint32_t TW = 512, TH = 512;
    std::vector<uint8_t> tex_px(TW * TH * 4);
    for (uint32_t y = 0; y < TH; ++y) {
        for (uint32_t x = 0; x < TW; ++x) {
            bool chk = (((x >> 5) ^ (y >> 5)) & 1) != 0;
            bool border = ((x & 31) < 2) || ((y & 31) < 2);
            float grain = 0.92f + 0.08f * sinf(x * 0.25f) * cosf(y * 0.25f);
            float base = border ? 0.28f : (chk ? 0.94f : 0.62f);
            uint8_t v = (uint8_t)(std::min(1.0f, base * grain) * 255.0f);
            size_t o = ((size_t)y * TW + x) * 4;
            tex_px[o+0] = v; tex_px[o+1] = v; tex_px[o+2] = (uint8_t)(v * 0.97f);
            tex_px[o+3] = chk ? 220 : 120; // smoothness in alpha
        }
    }
    VkBuffer tex_stg; VkDeviceMemory tex_stg_m;
    make_buf(tex_px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HV, tex_stg, tex_stg_m, tex_px.data());

    auto make_img = [&](uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags u, VkImageAspectFlags asp,
                        VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkImageCreateInfo ici2{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, fmt, {w, h, 1},
            1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
        VK_CHK(vkCreateImage(dev, &ici2, nullptr, &img));
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(dev, img, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, find_mem(mp, mr.memoryTypeBits, DL)};
        VK_CHK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        VK_CHK(vkBindImageMemory(dev, img, mem, 0));
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, img, VK_IMAGE_VIEW_TYPE_2D, fmt, {}, {asp, 0, 1, 0, 1}};
        VK_CHK(vkCreateImageView(dev, &ivci, nullptr, &view));
    };

    VkImage col_img, dep_img, tex_img; VkDeviceMemory col_m, dep_m, tex_m; VkImageView col_v, dep_v, tex_v;
    make_img(W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, col_img, col_m, col_v);
    make_img(W, H, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, dep_img, dep_m, dep_v);
    make_img(TW, TH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, tex_img, tex_m, tex_v);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
        VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT};
    VkSampler sampler; VK_CHK(vkCreateSampler(dev, &sci, nullptr, &sampler));

    // Command pool + upload staging buffers & texture to GPU once
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf};
    VkCommandPool pool; VK_CHK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
    VkBufferCopy cp1{0, 0, ov_sz}; vkCmdCopyBuffer(cb, stg_ovb, ovb, 1, &cp1);
    VkBufferCopy cp2{0, 0, oi_sz}; vkCmdCopyBuffer(cb, stg_oib, oib, 1, &cp2);
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        tex_img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    VkBufferImageCopy bic{}; bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {TW, TH, 1};
    vkCmdCopyBufferToImage(cb, tex_stg, tex_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    VK_CHK(vkEndCommandBuffer(cb));
    VkSubmitInfo si0{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
    VK_CHK(vkQueueSubmit(queue, 1, &si0, VK_NULL_HANDLE));
    VK_CHK(vkQueueWaitIdle(queue));

    // Descriptors
    VkDescriptorSetLayoutBinding dbs[2] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, dbs};
    VkDescriptorSetLayout dsl; VK_CHK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl};
    VkPipelineLayout pl; VK_CHK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkDescriptorPoolSize dps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, dps};
    VkDescriptorPool dpool; VK_CHK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dpool, 1, &dsl};
    VkDescriptorSet ds; VK_CHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi{ubo_buf, 0, sizeof(SceneUBO)};
    VkDescriptorImageInfo dii{sampler, tex_v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wds[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &dbi},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dii},
    };
    vkUpdateDescriptorSets(dev, 2, wds, 0, nullptr);

    // RenderPass & Framebuffer
    VkAttachmentDescription atts[2] = {
        {0, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
        {0, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
    };
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, nullptr, 1, &cref, nullptr, &dref};
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0, 2, atts, 1, &sub};
    VkRenderPass rp; VK_CHK(vkCreateRenderPass(dev, &rpci, nullptr, &rp));
    VkImageView fba[2] = {col_v, dep_v};
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0, rp, 2, fba, W, H, 1};
    VkFramebuffer fb; VK_CHK(vkCreateFramebuffer(dev, &fbci, nullptr, &fb));

    // Load SPIR-V shaders & create Opaque + Translucent Graphics Pipelines
    char vp_path[512], fp_path[512];
    snprintf(vp_path, sizeof(vp_path), "%s/pbr.vert.spv", shader_dir);
    snprintf(fp_path, sizeof(fp_path), "%s/pbr.frag.spv", shader_dir);
    auto vspv = load_spv(vp_path), fspv = load_spv(fp_path);
    VkShaderModuleCreateInfo vsmci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, vspv.size()*4, vspv.data()};
    VkShaderModuleCreateInfo fsmci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, fspv.size()*4, fspv.data()};
    VkShaderModule vsm, fsm; VK_CHK(vkCreateShaderModule(dev, &vsmci, nullptr, &vsm)); VK_CHK(vkCreateShaderModule(dev, &fsmci, nullptr, &fsm));

    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vsm, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fsm, "main"},
    };
    VkVertexInputBindingDescription vbd{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vad[4] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
        {2, 0, VK_FORMAT_R32G32_SFLOAT,    24},
        {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &vbd, 4, vad};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport vp{0, 0, (float)W, (float)H, 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS};
    VkPipelineColorBlendAttachmentState cba{VK_FALSE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
                                            VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF};
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba};

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stages, &vi, &ia, nullptr, &vps, &rs, &ms, &dss, &cbs, nullptr, pl, rp, 0};
    VkPipeline opaque_pipe, glass_pipe;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &opaque_pipe));

    rs.cullMode = VK_CULL_MODE_NONE;
    dss.depthWriteEnable = VK_FALSE;
    cba.blendEnable = VK_TRUE;
    VK_CHK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &glass_pipe));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VK_CHK(vkCreateFence(dev, &fci, nullptr, &fence));

    auto record_and_submit = [&](float t, bool do_readback) {
        float proj[16], view[16];
        mat4_persp_vk(52.0f * 3.14159265f / 180.0f, (float)W / (float)H, 0.25f, 120.0f, proj);
        float cam_ang = 0.45f + t * 0.25f;
        float ex = sinf(cam_ang) * 8.6f, ey = 3.2f, ez = cosf(cam_ang) * 8.6f;
        mat4_lookat(ex, ey, ez, 0.0f, 0.0f, 0.0f, view);
        mat4_mul(proj, view, ubo_ptr->viewProj);
        ubo_ptr->camPos[0] = ex; ubo_ptr->camPos[1] = ey; ubo_ptr->camPos[2] = ez; ubo_ptr->camPos[3] = 1.0f;
        ubo_ptr->sunDir[0] = 0.55f; ubo_ptr->sunDir[1] = 0.78f; ubo_ptr->sunDir[2] = 0.42f; ubo_ptr->sunDir[3] = 0.0f;
        ubo_ptr->pointPos[0] = cosf(t * 2.0f) * 3.8f; ubo_ptr->pointPos[1] = 2.0f; ubo_ptr->pointPos[2] = sinf(t * 2.0f) * 3.8f; ubo_ptr->pointPos[3] = 1.0f;
        ubo_ptr->pointColor[0] = 0.35f; ubo_ptr->pointColor[1] = 0.85f; ubo_ptr->pointColor[2] = 1.0f; ubo_ptr->pointColor[3] = 1.0f;
        ubo_ptr->timeParams[0] = t;

        VK_CHK(vkResetCommandBuffer(cb, 0));
        VK_CHK(vkBeginCommandBuffer(cb, &cbbi));
        VkClearValue cv[2];
        cv[0].color = {{0.04f, 0.06f, 0.11f, 1.0f}};
        cv[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, rp, fb, {{0, 0}, {W, H}}, 2, cv};
        vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
        VkDeviceSize off0 = 0;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque_pipe);
        vkCmdBindVertexBuffers(cb, 0, 1, &ovb, &off0);
        vkCmdBindIndexBuffer(cb, oib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, (uint32_t)opaque_i.size(), 1, 0, 0, 0);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, glass_pipe);
        vkCmdBindVertexBuffers(cb, 0, 1, &gvb, &off0);
        vkCmdBindIndexBuffer(cb, gib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, (uint32_t)glass_i.size(), 1, 0, 0, 0);

        vkCmdEndRenderPass(cb);

        if (do_readback) {
            VkBufferImageCopy rbic{};
            rbic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            rbic.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(cb, col_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_buf, 1, &rbic);
        }
        VK_CHK(vkEndCommandBuffer(cb));
        VK_CHK(vkResetFences(dev, 1, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb};
        VK_CHK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    };

    for (int i = 0; i < warmup; ++i) record_and_submit(0.0f, false);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) record_and_submit(0.0f, false);
    auto t1 = std::chrono::steady_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
    double fps = 1000.0 / avg_ms;
    double mtris = (double)actual_tris * fps / 1e6;

    printf("RESULT | device=\"%s\" | res=%ux%u | tris=%u | ms=%.2f | fps=%.1f | Mtris/s=%.2f\n",
           props.deviceName, W, H, actual_tris, avg_ms, fps, mtris);
    fflush(stdout);

    if (out_ppm) {
        record_and_submit(0.0f, true);
        uint8_t* px = nullptr;
        VK_CHK(vkMapMemory(dev, readback_m, 0, (VkDeviceSize)W * H * 4, 0, (void**)&px));
        FILE* f = fopen(out_ppm, "wb");
        fprintf(f, "P6\n%u %u\n255\n", W, H);
        std::vector<uint8_t> rgb((size_t)W * H * 3);
        for (size_t i = 0; i < (size_t)W * H; ++i) {
            rgb[i*3+0] = px[i*4+0]; rgb[i*3+1] = px[i*4+1]; rgb[i*3+2] = px[i*4+2];
        }
        fwrite(rgb.data(), 1, rgb.size(), f);
        fclose(f);
        vkUnmapMemory(dev, readback_m);
    }

    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
