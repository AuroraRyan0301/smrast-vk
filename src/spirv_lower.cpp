// spirv_lower.cpp — Automated SPIR-V Vertex & Fragment -> Vulkan Compute Shader Lowerer
// Uses libspirv-cross-c-shared for full SPIR-V reflection & GLSL 460 decompilation,
// then rewrites VS/FS into H20/H100/MI300X Compute Shaders:
//   - Preserves ALL original app Descriptor Sets (UBO, SSBO, Hardware sampler2D, PushConstants)
//   - VS -> GLCompute (local_size_x=64): pulls gl_VertexIndex or VkBuffer vertex attributes -> ClipVertBuf
//   - FS -> Quad-Swizzled GLCompute (local_size_x=256 = 64 2x2 quads):
//           * 1-cycle subgroupQuadSwapHorizontal/Vertical exact dFdx/dFdy!
//           * Hardware H20 TMU textureGrad() sampling!
//           * Perspective-correct varying interpolation + discard + Depth Write + Alpha Blending!

#include "smrast_internal.h"
#include <spirv_cross/spirv_cross_c.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

static uint64_t fnv1a_64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static bool compile_glsl_to_spv(const std::string& glsl, const char* tag,
                                std::vector<uint32_t>& out_spv, std::string& err) {
    uint64_t h = fnv1a_64(glsl.data(), glsl.size());
    char src_path[256], spv_path[256], log_path[256];
    snprintf(src_path, sizeof(src_path), "/tmp/smrast_cache_%s_%016llx.comp", tag, (unsigned long long)h);
    snprintf(spv_path, sizeof(spv_path), "/tmp/smrast_cache_%s_%016llx.spv", tag, (unsigned long long)h);
    snprintf(log_path, sizeof(log_path), "/tmp/smrast_cache_%s_%016llx.log", tag, (unsigned long long)h);

    // Check warm disk cache first (<0.1ms!)
    {
        std::ifstream cached(spv_path, std::ios::binary | std::ios::ate);
        if (cached.good()) {
            size_t sz = (size_t)cached.tellg();
            if (sz >= 20 && (sz % 4) == 0) {
                cached.seekg(0);
                out_spv.resize(sz / 4);
                cached.read((char*)out_spv.data(), sz);
                return true;
            }
        }
    }

    {
        std::ofstream out(src_path);
        out << glsl;
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "glslangValidator --target-env vulkan1.3 -V \"%s\" -o \"%s\" >\"%s\" 2>&1",
             src_path, spv_path, log_path);
    int rc = system(cmd);
    if (rc != 0) {
        std::ifstream lf(log_path);
        std::stringstream ss;
        ss << "glslangValidator failed (rc=" << rc << ") for " << src_path << ":\n" << lf.rdbuf();
        err = ss.str();
        fprintf(stderr, "[SmRast] %s\n", err.c_str());
        return false;
    }

    std::ifstream spvf(spv_path, std::ios::binary | std::ios::ate);
    if (!spvf.good()) { err = "Failed to open compiled SPV"; return false; }
    size_t sz = (size_t)spvf.tellg();
    spvf.seekg(0);
    out_spv.resize(sz / 4);
    spvf.read((char*)out_spv.data(), sz);
    return true;
}

struct ReflectedIO {
    std::string name;
    uint32_t location = 0;
    uint32_t vecsize = 4;
    spvc_basetype basetype = SPVC_BASETYPE_FP32;
    std::string glsl_type;
};

static std::string glsl_type_str(spvc_basetype bt, uint32_t vecsize) {
    if (bt == SPVC_BASETYPE_INT32) {
        if (vecsize == 1) return "int";
        return "ivec" + std::to_string(vecsize);
    }
    if (bt == SPVC_BASETYPE_UINT32) {
        if (vecsize == 1) return "uint";
        return "uvec" + std::to_string(vecsize);
    }
    if (vecsize == 1) return "float";
    return "vec" + std::to_string(vecsize);
}

static bool decompile_spirv(const std::vector<uint32_t>& spv,
                            std::string& out_glsl,
                            std::vector<ReflectedIO>& inputs,
                            std::vector<ReflectedIO>& outputs,
                            std::string& err) {
    spvc_context ctx = nullptr;
    if (spvc_context_create(&ctx) != SPVC_SUCCESS) { err = "spvc_context_create failed"; return false; }
    spvc_parsed_ir ir = nullptr;
    if (spvc_context_parse_spirv(ctx, spv.data(), spv.size(), &ir) != SPVC_SUCCESS) {
        err = "spvc_context_parse_spirv failed";
        spvc_context_destroy(ctx);
        return false;
    }
    spvc_compiler comp = nullptr;
    if (spvc_context_create_compiler(ctx, SPVC_BACKEND_GLSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS) {
        err = "spvc_context_create_compiler failed";
        spvc_context_destroy(ctx);
        return false;
    }

    spvc_compiler_options opts = nullptr;
    spvc_compiler_create_compiler_options(comp, &opts);
    spvc_compiler_options_set_uint(opts, SPVC_COMPILER_OPTION_GLSL_VERSION, 460);
    spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_FALSE);
    spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS, SPVC_TRUE);
    spvc_compiler_install_compiler_options(comp, opts);

    spvc_resources res = nullptr;
    spvc_compiler_create_shader_resources(comp, &res);

    auto collect = [&](spvc_resource_type rtype, std::vector<ReflectedIO>& dst) {
        const spvc_reflected_resource* list = nullptr;
        size_t count = 0;
        spvc_resources_get_resource_list_for_type(res, rtype, &list, &count);
        for (size_t i = 0; i < count; ++i) {
            ReflectedIO io;
            const char* nm = spvc_compiler_get_name(comp, list[i].id);
            io.name = (nm && nm[0]) ? nm : ("_loc_" + std::to_string(i));
            io.location = spvc_compiler_get_decoration(comp, list[i].id, SpvDecorationLocation);
            spvc_type t = spvc_compiler_get_type_handle(comp, list[i].type_id);
            io.vecsize = spvc_type_get_vector_size(t);
            io.basetype = spvc_type_get_basetype(t);
            io.glsl_type = glsl_type_str(io.basetype, io.vecsize);
            dst.push_back(io);
        }
    };
    collect(SPVC_RESOURCE_TYPE_STAGE_INPUT, inputs);
    collect(SPVC_RESOURCE_TYPE_STAGE_OUTPUT, outputs);

    const char* compiled = nullptr;
    if (spvc_compiler_compile(comp, &compiled) != SPVC_SUCCESS || !compiled) {
        err = "spvc_compiler_compile failed";
        spvc_context_destroy(ctx);
        return false;
    }
    out_glsl = compiled;
    spvc_context_destroy(ctx);
    return true;
}

// Replace whole-word identifier occurrences in GLSL source
static std::string replace_ident(const std::string& src, const std::string& from, const std::string& to) {
    std::string out;
    out.reserve(src.size() + 64);
    auto is_id = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; };
    size_t pos = 0;
    while (pos < src.size()) {
        size_t f = src.find(from, pos);
        if (f == std::string::npos) { out.append(src, pos, std::string::npos); break; }
        bool left_ok = (f == 0) || !is_id(src[f - 1]);
        bool right_ok = (f + from.size() == src.size()) || !is_id(src[f + from.size()]);
        if (left_ok && right_ok) {
            out.append(src, pos, f - pos);
            out.append(to);
            pos = f + from.size();
        } else {
            out.append(src, pos, f - pos + 1);
            pos = f + 1;
        }
    }
    return out;
}

// Strip #version line and stage in/out layout declarations, turning them into global variables
static std::string strip_version_and_stage_io(const std::string& glsl,
                                              const std::vector<ReflectedIO>& inputs,
                                              const std::vector<ReflectedIO>& outputs) {
    std::istringstream iss(glsl);
    std::ostringstream oss;
    std::string line;
    std::regex re_ver(R"(^\s*#version\b.*)");
    std::regex re_in(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*in\s+(.+);)");
    std::regex re_out(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*out\s+(.+);)");
    std::regex re_qual_line(R"(^\s*(?:invariant|precise)\s+[A-Za-z_][A-Za-z0-9_]*\s*;)");
    std::regex re_fs_layout(R"(^\s*layout\s*\([^)]*(?:early_fragment_tests|origin_upper_left|pixel_center_integer|depth_any|depth_greater|depth_less|depth_unchanged)[^)]*\)\s*(?:in|out)\b.*;)");
    std::regex re_perv(R"(^\s*out\s+gl_PerVertex\b.*)");
    bool skip_perv_block = false;

    while (std::getline(iss, line)) {
        if (skip_perv_block) {
            if (line.find("};") != std::string::npos) skip_perv_block = false;
            continue;
        }
        if (std::regex_match(line, re_ver)) continue;
        if (std::regex_match(line, re_qual_line)) continue;
        if (std::regex_match(line, re_fs_layout)) continue;
        if (std::regex_match(line, re_perv)) {
            if (line.find("};") == std::string::npos) skip_perv_block = true;
            continue;
        }
        std::smatch m;
        if (std::regex_match(line, m, re_in)) {
            oss << m[1].str() << "; // lowered stage input\n";
            continue;
        }
        if (std::regex_match(line, m, re_out)) {
            oss << m[1].str() << "; // lowered stage output\n";
            continue;
        }
        oss << line << "\n";
    }
    (void)inputs; (void)outputs;
    return oss.str();
}

// Generate vertex attribute unpacking code for a given VkFormat (supports VERTEX and INSTANCE input rates!)
static std::string emit_attr_unpack(const ReflectedIO& io, const VkVertexInputAttributeDescription* attr) {
    std::ostringstream ss;
    uint32_t b = attr->binding;
    if (b >= 4) b = 0;
    ss << "    {\n";
    ss << "        uint elem_i = ((sm_vs.vtx_inst_rate_mask & (1u << " << b << "u)) != 0u) ? uint(sm_iidx) : uint(sm_vidx);\n";
    ss << "        uint byte_off = sm_vs.vtx_offsets[" << b << "] + elem_i * sm_vs.vtx_strides[" << b << "] + " << attr->offset << "u;\n";
    ss << "        uint w0 = byte_off >> 2u;\n";
    auto load_f = [&](int k) {
        return "uintBitsToFloat(sm_vb" + std::to_string(b) + "[w0 + " + std::to_string(k) + "u])";
    };
    auto load_u = [&](int k) {
        return "sm_vb" + std::to_string(b) + "[w0 + " + std::to_string(k) + "u]";
    };
    switch (attr->format) {
        case VK_FORMAT_R32_SFLOAT:
            ss << "        " << io.name << " = " << load_f(0) << ";\n"; break;
        case VK_FORMAT_R32G32_SFLOAT:
            if (io.vecsize == 2) ss << "        " << io.name << " = vec2(" << load_f(0) << ", " << load_f(1) << ");\n";
            else if (io.vecsize == 4) ss << "        " << io.name << " = vec4(" << load_f(0) << ", " << load_f(1) << ", 0.0, 1.0);\n";
            break;
        case VK_FORMAT_R32G32B32_SFLOAT:
            if (io.vecsize == 3) ss << "        " << io.name << " = vec3(" << load_f(0) << ", " << load_f(1) << ", " << load_f(2) << ");\n";
            else if (io.vecsize == 4) ss << "        " << io.name << " = vec4(" << load_f(0) << ", " << load_f(1) << ", " << load_f(2) << ", 1.0);\n";
            break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            ss << "        " << io.name << " = vec4(" << load_f(0) << ", " << load_f(1) << ", " << load_f(2) << ", " << load_f(3) << ");\n"; break;
        case VK_FORMAT_R16G16_SFLOAT:
            ss << "        vec2 hf = unpackHalf2x16(" << load_u(0) << ");\n";
            if (io.vecsize == 2) ss << "        " << io.name << " = hf;\n";
            else if (io.vecsize == 4) ss << "        " << io.name << " = vec4(hf, 0.0, 1.0);\n";
            break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            ss << "        vec4 hf4 = vec4(unpackHalf2x16(" << load_u(0) << "), unpackHalf2x16(" << load_u(1) << "));\n";
            if (io.vecsize == 4) ss << "        " << io.name << " = hf4;\n";
            else if (io.vecsize == 3) ss << "        " << io.name << " = hf4.xyz;\n";
            break;
        case VK_FORMAT_R8G8B8A8_UNORM:
            ss << "        vec4 un = unpackUnorm4x8(" << load_u(0) << ");\n";
            if (io.vecsize == 4) ss << "        " << io.name << " = un;\n";
            else if (io.vecsize == 3) ss << "        " << io.name << " = un.xyz;\n";
            break;
        case VK_FORMAT_B8G8R8A8_UNORM:
            ss << "        vec4 un = unpackUnorm4x8(" << load_u(0) << ").bgra;\n";
            if (io.vecsize == 4) ss << "        " << io.name << " = un;\n";
            else if (io.vecsize == 3) ss << "        " << io.name << " = un.xyz;\n";
            break;
        case VK_FORMAT_R8G8B8A8_SNORM:
            ss << "        vec4 sn = unpackSnorm4x8(" << load_u(0) << ");\n";
            if (io.vecsize == 4) ss << "        " << io.name << " = sn;\n";
            else if (io.vecsize == 3) ss << "        " << io.name << " = sn.xyz;\n";
            break;
        case VK_FORMAT_R32_UINT:
            ss << "        " << io.name << " = " << load_u(0) << ";\n"; break;
        case VK_FORMAT_R32G32B32A32_UINT:
            ss << "        " << io.name << " = uvec4(" << load_u(0) << ", " << load_u(1) << ", " << load_u(2) << ", " << load_u(3) << ");\n"; break;
        default:
            if (io.vecsize == 4) ss << "        " << io.name << " = vec4(" << load_f(0) << ", " << load_f(1) << ", " << load_f(2) << ", " << load_f(3) << ");\n";
            else if (io.vecsize == 3) ss << "        " << io.name << " = vec3(" << load_f(0) << ", " << load_f(1) << ", " << load_f(2) << ");\n";
            else if (io.vecsize == 2) ss << "        " << io.name << " = vec2(" << load_f(0) << ", " << load_f(1) << ");\n";
            else ss << "        " << io.name << " = " << load_f(0) << ";\n";
            break;
    }
    ss << "    }\n";
    return ss.str();
}

bool sm_lower_vertex_spirv(const std::vector<uint32_t>& in_spv,
                           const SmVertexInputState& vtx_in,
                           uint32_t sm_set_idx,
                           std::vector<uint32_t>& out_comp_spv,
                           std::string& err) {
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err)) return false;

    std::string body = strip_version_and_stage_io(glsl, inputs, outputs);
    body = replace_ident(body, "gl_Position", "sm_gl_Position");
    body = replace_ident(body, "gl_PointSize", "sm_gl_PointSize");
    body = replace_ident(body, "gl_ClipDistance", "sm_gl_ClipDistance");
    body = replace_ident(body, "gl_CullDistance", "sm_gl_CullDistance");
    body = replace_ident(body, "gl_Layer", "sm_gl_Layer");
    body = replace_ident(body, "gl_ViewportIndex", "sm_gl_ViewportIndex");
    body = replace_ident(body, "gl_VertexIndex", "sm_vidx");
    body = replace_ident(body, "gl_InstanceIndex", "sm_iidx");
    // Rename void main() -> void sm_user_vs_main()
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_vs_main()");

    std::ostringstream ss;
    ss << "#version 460\n";
    ss << "layout(local_size_x = 64) in;\n\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0) writeonly buffer SmClipVertBuf { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1) readonly buffer SmIdxBuf { uint sm_idx_raw[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 2) readonly buffer SmVB0 { uint sm_vb0[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 3) readonly buffer SmVB1 { uint sm_vb1[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 4) readonly buffer SmVB2 { uint sm_vb2[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 5) readonly buffer SmVB3 { uint sm_vb3[]; };\n";
    ss << "layout(std140, set = " << sm_set_idx << ", binding = 6) uniform SmVsParams {\n"
       << "    uint vertex_count;\n"
       << "    uint first_vertex;\n"
       << "    uint instance_idx;\n"
       << "    uint indexed;\n"
       << "    uint first_index;\n"
       << "    int  vertex_offset;\n"
       << "    uint vert_stride_vec4;\n"
       << "    uint idx_buf_offset;\n"
       << "    uvec4 vtx_offsets;\n"
       << "    uvec4 vtx_strides;\n"
       << "    uint verts_per_instance;\n"
       << "    uint vtx_inst_rate_mask;\n"
       << "    uint pad0;\n"
       << "    uint pad1;\n"
       << "} sm_vs;\n\n";

    ss << "vec4 sm_gl_Position;\n";
    ss << "float sm_gl_PointSize = 1.0;\n";
    ss << "float sm_gl_ClipDistance[8];\n";
    ss << "float sm_gl_CullDistance[8];\n";
    ss << "int sm_gl_Layer = 0;\n";
    ss << "int sm_gl_ViewportIndex = 0;\n";
    ss << "int sm_vidx;\n";
    ss << "int sm_iidx;\n\n";
    ss << body << "\n\n";

    ss << "void main() {\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    if (gid >= sm_vs.vertex_count) return;\n";
    ss << "    uint inst_local = gid / sm_vs.verts_per_instance;\n";
    ss << "    uint vert_local = gid - inst_local * sm_vs.verts_per_instance;\n";
    ss << "    sm_iidx = int(sm_vs.instance_idx + inst_local);\n";
    ss << "    if (sm_vs.indexed == 0u) {\n";
    ss << "        sm_vidx = int(sm_vs.first_vertex + vert_local);\n";
    ss << "    } else if (sm_vs.indexed == 1u) {\n";
    ss << "        uint ii = (sm_vs.idx_buf_offset >> 1u) + sm_vs.first_index + vert_local;\n";
    ss << "        uint w = sm_idx_raw[ii >> 1u];\n";
    ss << "        uint idx16 = ((ii & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);\n";
    ss << "        sm_vidx = int(idx16) + sm_vs.vertex_offset;\n";
    ss << "    } else {\n";
    ss << "        uint ii = (sm_vs.idx_buf_offset >> 2u) + sm_vs.first_index + vert_local;\n";
    ss << "        sm_vidx = int(sm_idx_raw[ii]) + sm_vs.vertex_offset;\n";
    ss << "    }\n";

    // Unpack reflected vertex attributes from bound VkBuffers
    for (const auto& io : inputs) {
        const VkVertexInputAttributeDescription* found = nullptr;
        for (const auto& a : vtx_in.attrs) {
            if (a.location == io.location) { found = &a; break; }
        }
        if (found) ss << emit_attr_unpack(io, found);
    }

    ss << "    sm_user_vs_main();\n";
    ss << "    uint base_slot = gid * sm_vs.vert_stride_vec4;\n";
    ss << "    sm_clip_verts[base_slot] = sm_gl_Position;\n";
    for (const auto& io : outputs) {
        if (io.location >= SM_MAX_VARYINGS) continue;
        ss << "    sm_clip_verts[base_slot + " << (1u + io.location) << "u] = ";
        if (io.vecsize == 4) ss << "vec4(" << io.name << ");\n";
        else if (io.vecsize == 3) ss << "vec4(" << io.name << ", 0.0);\n";
        else if (io.vecsize == 2) ss << "vec4(" << io.name << ", 0.0, 0.0);\n";
        else ss << "vec4(float(" << io.name << "), 0.0, 0.0, 0.0);\n";
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "vs", out_comp_spv, err);
}

bool sm_lower_fragment_spirv(const std::vector<uint32_t>& in_spv,
                             uint32_t sm_set_idx,
                             std::vector<uint32_t>& out_comp_spv,
                             std::string& err) {
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err)) return false;

    std::string body = strip_version_and_stage_io(glsl, inputs, outputs);
    body = replace_ident(body, "gl_FragCoord", "sm_gl_FragCoord");
    body = replace_ident(body, "gl_FrontFacing", "sm_gl_FrontFacing");
    body = replace_ident(body, "gl_FragDepth", "sm_gl_FragDepth");
    body = replace_ident(body, "gl_SampleMask", "sm_gl_SampleMask");
    body = replace_ident(body, "gl_SampleID", "sm_gl_SampleID");
    body = replace_ident(body, "gl_Layer", "sm_gl_Layer");
    body = replace_ident(body, "gl_ViewportIndex", "sm_gl_ViewportIndex");
    body = replace_ident(body, "gl_PrimitiveID", "sm_gl_PrimitiveID");
    body = replace_ident(body, "dFdx", "sm_dFdx");
    body = replace_ident(body, "dFdy", "sm_dFdy");
    body = replace_ident(body, "dFdxFine", "sm_dFdx");
    body = replace_ident(body, "dFdyFine", "sm_dFdy");
    body = replace_ident(body, "dFdxCoarse", "sm_dFdx");
    body = replace_ident(body, "dFdyCoarse", "sm_dFdy");
    body = replace_ident(body, "texture", "sm_texture");
    body = std::regex_replace(body, std::regex(R"(\bdiscard\s*;)"), "{ sm_discarded = true; return; }");
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_fs_main()");

    std::ostringstream ss;
    ss << "#version 460\n";
    ss << "#extension GL_ARB_gpu_shader_int64 : require\n";
    ss << "#extension GL_KHR_shader_subgroup_quad : require\n";
    ss << "#extension GL_KHR_shader_subgroup_vote : require\n";
    // 256 threads = 64 2x2 pixel quads per 16x16 screen tile!
    ss << "layout(local_size_x = 256) in;\n\n";

    ss << "struct SmTriSetup {\n"
       << "    vec4 v0;\n"
       << "    vec4 v1;\n"
       << "    vec4 v2;\n"
       << "    vec4 area;\n"
       << "    uvec4 idx;\n"
       << "};\n";

    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0) readonly buffer SmClipVertBuf { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1) readonly buffer SmTriSetupBuf { SmTriSetup sm_tris[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 2) readonly buffer SmVisBuf      { uint64_t sm_vis_px[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 3) buffer SmColorBuf             { uint sm_color_px[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 4) buffer SmDepthBuf             { float sm_depth_px[]; };\n";
    ss << "layout(std140, set = " << sm_set_idx << ", binding = 5) uniform SmFsParams {\n"
       << "    uint width;\n"
       << "    uint height;\n"
       << "    uint vert_stride_vec4;\n"
       << "    uint format_flags;\n"
       << "    uint depth_write;\n"
       << "    uint blend_enable;\n"
       << "    uint src_blend;\n"
       << "    uint dst_blend;\n"
       << "    uint blend_op;\n"
       << "    uvec3 pad;\n"
       << "} sm_fs;\n\n";

    // Hardware 1-cycle Quad derivative helpers (exact 2x2 quad parity via lane 0,1,2,3!)
    ss << "uint sm_quad_lane;\n";
    ss << "vec4 sm_gl_FragCoord;\n";
    ss << "bool sm_gl_FrontFacing = true;\n";
    ss << "float sm_gl_FragDepth = 0.0;\n";
    ss << "int sm_gl_SampleMask[1];\n";
    ss << "int sm_gl_SampleID = 0;\n";
    ss << "int sm_gl_Layer = 0;\n";
    ss << "int sm_gl_ViewportIndex = 0;\n";
    ss << "int sm_gl_PrimitiveID = 0;\n";
    ss << "bool sm_discarded = false;\n\n";

    ss << "float sm_dFdx(float v) { float h = subgroupQuadSwapHorizontal(v); return ((sm_quad_lane & 1u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec2  sm_dFdx(vec2  v) { vec2  h = subgroupQuadSwapHorizontal(v); return ((sm_quad_lane & 1u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec3  sm_dFdx(vec3  v) { vec3  h = subgroupQuadSwapHorizontal(v); return ((sm_quad_lane & 1u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec4  sm_dFdx(vec4  v) { vec4  h = subgroupQuadSwapHorizontal(v); return ((sm_quad_lane & 1u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "float sm_dFdy(float v) { float h = subgroupQuadSwapVertical(v);   return ((sm_quad_lane & 2u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec2  sm_dFdy(vec2  v) { vec2  h = subgroupQuadSwapVertical(v);   return ((sm_quad_lane & 2u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec3  sm_dFdy(vec3  v) { vec3  h = subgroupQuadSwapVertical(v);   return ((sm_quad_lane & 2u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec4  sm_dFdy(vec4  v) { vec4  h = subgroupQuadSwapVertical(v);   return ((sm_quad_lane & 2u) == 0u) ? (h - v) : (v - h); }\n";
    ss << "vec4  sm_texture(sampler2D s, vec2 uv) { return textureGrad(s, uv, sm_dFdx(uv), sm_dFdy(uv)); }\n";
    ss << "vec4  sm_texture(sampler2D s, vec2 uv, float bias) { return textureGrad(s, uv, sm_dFdx(uv) * exp2(bias), sm_dFdy(uv) * exp2(bias)); }\n";
    ss << "vec4  sm_texture(samplerCube s, vec3 dir) { return textureGrad(s, dir, sm_dFdx(dir), sm_dFdy(dir)); }\n\n";

    // Color pack/unpack + blending helpers
    ss << "vec4 sm_unpack_px(uint pk, bool bgra) {\n"
       << "    vec4 c = vec4(float(pk & 255u), float((pk>>8u)&255u), float((pk>>16u)&255u), float((pk>>24u)&255u)) / 255.0;\n"
       << "    return bgra ? c.bgra : c;\n"
       << "}\n";
    ss << "float sm_lin_to_srgb(float x) { return (x <= 0.0031308) ? (x * 12.92) : (1.055 * pow(max(x, 0.0), 1.0/2.4) - 0.055); }\n";
    ss << "uint sm_pack_px(vec4 c, uint fmt_flags) {\n"
       << "    c = clamp(c, 0.0, 1.0);\n"
       << "    if ((fmt_flags & 2u) != 0u) c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b));\n"
       << "    if ((fmt_flags & 1u) != 0u) c = c.bgra;\n"
       << "    uvec4 u = uvec4(round(c * 255.0));\n"
       << "    return u.r | (u.g << 8u) | (u.b << 16u) | (u.a << 24u);\n"
       << "}\n";
    ss << "vec4 sm_blend_factor(uint f, vec4 src, vec4 dst) {\n"
       << "    if (f == 0u) return vec4(0.0);\n"
       << "    if (f == 1u) return vec4(1.0);\n"
       << "    if (f == 2u) return src;\n"
       << "    if (f == 3u) return vec4(1.0) - src;\n"
       << "    if (f == 4u) return dst;\n"
       << "    if (f == 5u) return vec4(1.0) - dst;\n"
       << "    if (f == 6u) return vec4(src.a);\n"
       << "    if (f == 7u) return vec4(1.0 - src.a);\n"
       << "    if (f == 8u) return vec4(dst.a);\n"
       << "    if (f == 9u) return vec4(1.0 - dst.a);\n"
       << "    return vec4(1.0);\n"
       << "}\n"
       << "vec4 sm_blend_combine(uint op, vec4 s, vec4 d) {\n"
       << "    if (op == 1u) return s - d;\n"
       << "    if (op == 2u) return d - s;\n"
       << "    if (op == 3u) return min(s, d);\n"
       << "    if (op == 4u) return max(s, d);\n"
       << "    return s + d;\n"
       << "}\n\n";

    ss << body << "\n\n";

    ss << "void main() {\n";
    ss << "    uint tid = gl_LocalInvocationIndex;\n";
    ss << "    uint quad_id = tid >> 2u;\n";
    ss << "    sm_quad_lane = tid & 3u;\n";
    ss << "    uint lx = ((quad_id & 7u) << 1u) | (sm_quad_lane & 1u);\n";
    ss << "    uint ly = ((quad_id >> 3u) << 1u) | ((sm_quad_lane >> 1u) & 1u);\n";
    ss << "    uvec2 p = gl_WorkGroupID.xy * 16u + uvec2(lx, ly);\n";
    ss << "    bool in_bounds = (p.x < sm_fs.width && p.y < sm_fs.height);\n";
    ss << "    uint pidx = in_bounds ? (p.y * sm_fs.width + p.x) : 0u;\n";
    ss << "    uint64_t vis = in_bounds ? sm_vis_px[pidx] : 0xFFFFFFFFFFFFFFFFul;\n";
    ss << "    bool active_px = in_bounds && (vis != 0xFFFFFFFFFFFFFFFFul);\n";
    ss << "    // Quad helper lane emulation: keep all 4 lanes of a covered quad alive for exact dFdx/dFdy!\n";
    ss << "    uint my_tri = active_px ? uint(vis & 0xFFFFFFFFul) : 0xFFFFFFFFu;\n";
    ss << "    uint t0 = subgroupQuadBroadcast(my_tri, 0u);\n";
    ss << "    uint t1 = subgroupQuadBroadcast(my_tri, 1u);\n";
    ss << "    uint t2 = subgroupQuadBroadcast(my_tri, 2u);\n";
    ss << "    uint t3 = subgroupQuadBroadcast(my_tri, 3u);\n";
    ss << "    uint eff_tri = active_px ? my_tri : min(min(t0, t1), min(t2, t3));\n";
    ss << "    if (eff_tri == 0xFFFFFFFFu) return;\n\n";

    ss << "    SmTriSetup ts = sm_tris[eff_tri];\n";
    ss << "    float fx = float(p.x) + 0.5, fy = float(p.y) + 0.5;\n";
    ss << "    vec2 s0 = ts.v0.xy, s1 = ts.v1.xy, s2 = ts.v2.xy;\n";
    ss << "    float l0 = ((s2.x - s1.x) * (fy - s1.y) - (s2.y - s1.y) * (fx - s1.x)) * ts.area.x;\n";
    ss << "    float l1 = ((s0.x - s2.x) * (fy - s2.y) - (s0.y - s2.y) * (fx - s2.x)) * ts.area.x;\n";
    ss << "    float l2 = 1.0 - l0 - l1;\n";
    ss << "    float pz = clamp(l0 * ts.v0.z + l1 * ts.v1.z + l2 * ts.v2.z, 0.0, 1.0);\n";
    ss << "    float w0 = l0 * ts.v0.w, w1 = l1 * ts.v1.w, w2 = l2 * ts.v2.w;\n";
    ss << "    float inv_wsum = 1.0 / max(1e-20, w0 + w1 + w2);\n";
    ss << "    float b0 = w0 * inv_wsum, b1 = w1 * inv_wsum, b2 = w2 * inv_wsum;\n";
    ss << "    sm_gl_FragCoord = vec4(fx, fy, pz, inv_wsum);\n";
    ss << "    uint vi0 = ts.idx.x * sm_fs.vert_stride_vec4;\n";
    ss << "    uint vi1 = ts.idx.y * sm_fs.vert_stride_vec4;\n";
    ss << "    uint vi2 = ts.idx.z * sm_fs.vert_stride_vec4;\n";

    // Perspective-correct interpolation for every reflected Fragment Input varying!
    for (const auto& io : inputs) {
        if (io.location >= SM_MAX_VARYINGS) continue;
        uint32_t off = 1u + io.location;
        ss << "    {\n";
        ss << "        vec4 interp = sm_clip_verts[vi0 + " << off << "u] * b0 + "
           << "sm_clip_verts[vi1 + " << off << "u] * b1 + "
           << "sm_clip_verts[vi2 + " << off << "u] * b2;\n";
        if (io.vecsize == 4) ss << "        " << io.name << " = interp;\n";
        else if (io.vecsize == 3) ss << "        " << io.name << " = interp.xyz;\n";
        else if (io.vecsize == 2) ss << "        " << io.name << " = interp.xy;\n";
        else ss << "        " << io.name << " = interp.x;\n";
        ss << "    }\n";
    }

    ss << "    sm_discarded = false;\n";
    ss << "    sm_user_fs_main();\n";
    ss << "    if (!active_px || sm_discarded) return;\n";
    ss << "    if (sm_fs.depth_write != 0u) sm_depth_px[pidx] = pz;\n";

    // Write Location 0 color output (with optional blending)
    std::string out0_name;
    uint32_t out0_vec = 4;
    for (const auto& io : outputs) {
        if (io.location == 0) { out0_name = io.name; out0_vec = io.vecsize; break; }
    }
    if (!out0_name.empty()) {
        ss << "    vec4 out_c = ";
        if (out0_vec == 4) ss << out0_name << ";\n";
        else if (out0_vec == 3) ss << "vec4(" << out0_name << ", 1.0);\n";
        else if (out0_vec == 2) ss << "vec4(" << out0_name << ", 0.0, 1.0);\n";
        else ss << "vec4(" << out0_name << ", 0.0, 0.0, 1.0);\n";
        ss << "    if (sm_fs.blend_enable != 0u) {\n";
        ss << "        vec4 dst_c = sm_unpack_px(sm_color_px[pidx], (sm_fs.format_flags & 1u) != 0u);\n";
        ss << "        vec4 sf = sm_blend_factor(sm_fs.src_blend, out_c, dst_c);\n";
        ss << "        vec4 df = sm_blend_factor(sm_fs.dst_blend, out_c, dst_c);\n";
        ss << "        out_c = sm_blend_combine(sm_fs.blend_op, out_c * sf, dst_c * df);\n";
        ss << "    }\n";
        ss << "    sm_color_px[pidx] = sm_pack_px(out_c, sm_fs.format_flags);\n";
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "fs", out_comp_spv, err);
}
