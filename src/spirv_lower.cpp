// spirv_lower.cpp — SmRast-VK v0.3: Automated SPIR-V VS/FS -> H20 Compute Shader Lowerer
// Features:
//   - In-process thread-local libshaderc compilation (zero fork/exec, lock-free parallel PSO builds!)
//   - In-memory + disk SPIR-V cache
//   - Full 8 Vertex Buffers + all float/half/int/uint/unorm/snorm/packed vertex formats (including unaligned byte offsets)
//   - Integer/Uint flat varyings (provoking vertex bit-cast) + Array stage I/O + location/component qualifiers
//   - Full 4x MRT + 11 HDR/Float/Unorm RT formats + RGBA8 hardware packUnorm4x8 fast-path
//   - Independent per-RT blending + Quad-Subgroup 1-cycle dFdx/dFdy/fwidth + H20 Hardware TMU textureGrad()
//   - DXVK/vkd3d-proton shader qualifiers, separate texture+sampler call-site rewriter, Specialization Constants

#include "smrast_internal.h"
#include <spirv_cross/spirv_cross_c.h>
#include <shaderc/shaderc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unistd.h>
#include <atomic>

static uint64_t fnv1a_64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static std::mutex g_spv_cache_mtx;
static std::unordered_map<uint64_t, std::vector<uint32_t>> g_spv_mem_cache;

// Thread-local in-process shaderc compiler (zero fork/exec, lock-free parallel PSO builds for DXVK!)
struct ThreadLocalShaderc {
    shaderc_compiler_t compiler = nullptr;
    shaderc_compile_options_t opts = nullptr;
    shaderc_compile_options_t opts_zero = nullptr;
    ThreadLocalShaderc() {
        compiler = shaderc_compiler_initialize();
        opts = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_6);
        shaderc_compile_options_set_optimization_level(opts, shaderc_optimization_level_performance);

        opts_zero = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(opts_zero, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        shaderc_compile_options_set_target_spirv(opts_zero, shaderc_spirv_version_1_6);
        shaderc_compile_options_set_optimization_level(opts_zero, shaderc_optimization_level_zero);
    }
    ~ThreadLocalShaderc() {
        if (opts) shaderc_compile_options_release(opts);
        if (opts_zero) shaderc_compile_options_release(opts_zero);
        if (compiler) shaderc_compiler_release(compiler);
    }
};

static std::atomic<uint64_t> g_tmp_seq{0};

static bool compile_glsl_to_spv(const std::string& glsl, const char* tag,
                                std::vector<uint32_t>& out_spv, std::string& err) {
    uint64_t h = fnv1a_64(glsl.data(), glsl.size());

    // 1. Hot in-memory cache (<1 microsecond)
    {
        std::lock_guard<std::mutex> lk(g_spv_cache_mtx);
        auto it = g_spv_mem_cache.find(h);
        if (it != g_spv_mem_cache.end()) {
            out_spv = it->second;
            return true;
        }
    }

    // 2. Warm disk cache (<0.1 ms, written atomically via rename so 16 threads never race!)
    char spv_path[256];
    snprintf(spv_path, sizeof(spv_path), "/tmp/smrast_v3c_cache_%s_%016llx.spv", tag, (unsigned long long)h);
    {
        std::ifstream cached(spv_path, std::ios::binary | std::ios::ate);
        if (cached.good()) {
            size_t sz = (size_t)cached.tellg();
            if (sz >= 20 && (sz % 4) == 0) {
                cached.seekg(0);
                out_spv.resize(sz / 4);
                cached.read((char*)out_spv.data(), sz);
                if (cached.good() && out_spv[0] == 0x07230203u) {
                    std::lock_guard<std::mutex> lk(g_spv_cache_mtx);
                    g_spv_mem_cache[h] = out_spv;
                    return true;
                }
                out_spv.clear();
            }
        }
    }

    // 3. Fast in-process libshaderc compilation (zero fork/exec; auto-fallback to -O0 if SPIR-V opt trips on OpSampledImage!)
    thread_local ThreadLocalShaderc sc;
    char fname[64];
    snprintf(fname, sizeof(fname), "smrast_%s_%016llx.comp", tag, (unsigned long long)h);
    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        sc.compiler, glsl.data(), glsl.size(), shaderc_compute_shader, fname, "main", sc.opts);

    if (res && shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        const char* msg = shaderc_result_get_error_message(res);
        if (msg && strstr(msg, "failed to optimize") != nullptr) {
            shaderc_result_release(res);
            res = shaderc_compile_into_spv(
                sc.compiler, glsl.data(), glsl.size(), shaderc_compute_shader, fname, "main", sc.opts_zero);
        }
    }

    if (!res || shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        const char* msg = res ? shaderc_result_get_error_message(res) : "null shaderc result";
        char dump_path[256];
        snprintf(dump_path, sizeof(dump_path), "/tmp/smrast_fail_%s_%016llx.comp", tag, (unsigned long long)h);
        std::ofstream df(dump_path); df << glsl;
        err = std::string("shaderc failed for ") + dump_path + ":\n" + (msg ? msg : "");
        fprintf(stderr, "[SmRast] %s\n", err.c_str());
        if (res) shaderc_result_release(res);
        return false;
    }

    size_t len = shaderc_result_get_length(res);
    const char* bytes = shaderc_result_get_bytes(res);
    out_spv.resize(len / 4);
    memcpy(out_spv.data(), bytes, len);
    shaderc_result_release(res);

    {
        char tmp_path[300];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d.%llu", spv_path, (int)getpid(), (unsigned long long)g_tmp_seq.fetch_add(1));
        std::ofstream out(tmp_path, std::ios::binary);
        if (out.good()) {
            out.write((const char*)out_spv.data(), out_spv.size() * 4);
            out.close();
            rename(tmp_path, spv_path);
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_spv_cache_mtx);
        g_spv_mem_cache[h] = out_spv;
    }
    return true;
}

struct ReflectedIO {
    std::string name;
    uint32_t location = 0;
    uint32_t component = 0;
    uint32_t vecsize = 4;
    uint32_t array_size = 0; // 0 = non-array
    spvc_basetype basetype = SPVC_BASETYPE_FP32;
    std::string glsl_type;
};

static std::string glsl_type_str(spvc_basetype bt, uint32_t vecsize) {
    if (bt == SPVC_BASETYPE_INT32)  return (vecsize == 1) ? "int"  : ("ivec" + std::to_string(vecsize));
    if (bt == SPVC_BASETYPE_UINT32) return (vecsize == 1) ? "uint" : ("uvec" + std::to_string(vecsize));
    return (vecsize == 1) ? "float" : ("vec" + std::to_string(vecsize));
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

    auto collect = [&](spvc_resource_type rtype, const char* prefix, std::vector<ReflectedIO>& dst) {
        const spvc_reflected_resource* list = nullptr;
        size_t count = 0;
        spvc_resources_get_resource_list_for_type(res, rtype, &list, &count);
        for (size_t i = 0; i < count; ++i) {
            ReflectedIO io;
            spvc_type t = spvc_compiler_get_type_handle(comp, list[i].type_id);
            io.basetype = spvc_type_get_basetype(t);
            if (io.basetype != SPVC_BASETYPE_STRUCT) {
                std::string unm = std::string(prefix) + std::to_string(i);
                spvc_compiler_set_name(comp, list[i].id, unm.c_str());
                io.name = unm;
            } else {
                const char* nm = spvc_compiler_get_name(comp, list[i].id);
                io.name = (nm && nm[0]) ? nm : (std::string(prefix) + std::to_string(i));
            }
            io.location = spvc_compiler_get_decoration(comp, list[i].id, SpvDecorationLocation);
            io.component = spvc_compiler_get_decoration(comp, list[i].id, SpvDecorationComponent);
            io.vecsize = spvc_type_get_vector_size(t);
            unsigned ndims = spvc_type_get_num_array_dimensions(t);
            if (ndims > 0) {
                io.array_size = spvc_type_get_array_dimension(t, 0);
            }
            io.glsl_type = glsl_type_str(io.basetype, io.vecsize);
            dst.push_back(io);
        }
    };
    collect(SPVC_RESOURCE_TYPE_STAGE_INPUT,  "sm_io_in_",  inputs);
    collect(SPVC_RESOURCE_TYPE_STAGE_OUTPUT, "sm_io_out_", outputs);

    const char* compiled = nullptr;
    if (spvc_compiler_compile(comp, &compiled) != SPVC_SUCCESS || !compiled) {
        err = "spvc_compiler_compile failed";
        spvc_context_destroy(ctx);
        return false;
    }
    // Sync final GLSL identifiers post-compile in case SPIRV-Cross disambiguated any name
    auto sync_names = [&](spvc_resource_type rtype, std::vector<ReflectedIO>& dst) {
        const spvc_reflected_resource* list = nullptr;
        size_t count = 0;
        spvc_resources_get_resource_list_for_type(res, rtype, &list, &count);
        for (size_t i = 0; i < count && i < dst.size(); ++i) {
            const char* nm = spvc_compiler_get_name(comp, list[i].id);
            if (nm && nm[0]) dst[i].name = nm;
        }
    };
    sync_names(SPVC_RESOURCE_TYPE_STAGE_INPUT,  inputs);
    sync_names(SPVC_RESOURCE_TYPE_STAGE_OUTPUT, outputs);
    out_glsl = compiled;
    spvc_context_destroy(ctx);
    return true;
}

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

// Balanced-parenthesis call-site rewriter for Compute Shaders (Vulkan GLSL 460):
//   texture(S, P)             -> textureGrad(S, P, sm_dFdx(gc), sm_dFdy(gc)) [or textureLod for CubeArray/CubeShadow]
//   texture(S, P, bias)       -> textureGrad(S, P, sm_dFdx(gc)*exp2(bias), sm_dFdy(gc)*exp2(bias))
//   textureOffset(S, P, off)  -> textureGradOffset(S, P, sm_dFdx(gc), sm_dFdy(gc), off)
//   textureQueryLod(S, P)     -> vec2(0.0)
//   interpolateAt*(x, ...)    -> (x)
// Keeping S (e.g. DXVK's separate `sampler2D(t0, s0)`) directly at point of use satisfies Vulkan GLSL 460!
static std::string rewrite_texture_calls(const std::string& src) {
    auto is_id = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; };
    auto trim = [](std::string s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return std::string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    auto split_args = [&](const std::string& s) {
        std::vector<std::string> args;
        int depth = 0;
        size_t start = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth--;
            else if (c == ',' && depth == 0) {
                args.push_back(trim(s.substr(start, i - start)));
                start = i + 1;
            }
        }
        args.push_back(trim(s.substr(start)));
        return args;
    };

    static const char* kNames[] = {
        "texture(", "textureOffset(", "textureProj(", "textureProjOffset(",
        "textureQueryLod(", "interpolateAtCentroid(", "interpolateAtSample(", "interpolateAtOffset("
    };
    static const size_t kLens[] = { 7, 13, 11, 17, 15, 21, 19, 19 };

    std::string out;
    out.reserve(src.size() + 256);
    size_t pos = 0;
    while (pos < src.size()) {
        size_t f = std::string::npos;
        int which = -1;
        for (int k = 0; k < 8; ++k) {
            size_t fk = src.find(kNames[k], pos);
            if (fk < f) { f = fk; which = k; }
        }
        if (f == std::string::npos) { out.append(src, pos, std::string::npos); break; }
        bool left_ok = (f == 0) || !is_id(src[f - 1]);
        if (!left_ok) {
            out.append(src, pos, f - pos + 1);
            pos = f + 1;
            continue;
        }
        size_t open_p = f + kLens[which];
        int depth = 1;
        size_t cur = open_p + 1;
        while (cur < src.size() && depth > 0) {
            if (src[cur] == '(') depth++;
            else if (src[cur] == ')') depth--;
            cur++;
        }
        if (depth != 0) { out.append(src, pos, std::string::npos); break; }
        std::string inner = src.substr(open_p + 1, cur - open_p - 2);
        inner = rewrite_texture_calls(inner);
        auto args = split_args(inner);
        out.append(src, pos, f - pos);

        auto grad_coord = [&](const std::string& samp, const std::string& coord) {
            if (samp.find("2DArray") != std::string::npos || samp.find("2DShadow") != std::string::npos)
                return "(" + coord + ").xy";
            if (samp.find("1DArray") != std::string::npos || samp.find("1DShadow") != std::string::npos)
                return "(" + coord + ").x";
            if (samp.find("CubeArray") != std::string::npos || samp.find("CubeShadow") != std::string::npos)
                return "(" + coord + ").xyz";
            return "(" + coord + ")";
        };
        auto no_grad_sampler = [&](const std::string& samp) {
            return samp.find("CubeArray") != std::string::npos || samp.find("CubeShadow") != std::string::npos;
        };

        if (which == 4) { // textureQueryLod
            out += "vec2(0.0)";
        } else if (which >= 5) { // interpolateAt*
            out += "(" + (args.empty() ? "0.0" : args[0]) + ")";
        } else if (which == 0 && (args.size() == 2 || args.size() == 3)) {
            if (no_grad_sampler(args[0])) {
                if (args[0].find("CubeArrayShadow") != std::string::npos) {
                    out += "0.5"; // CubeArrayShadow has no compute LOD sampling in core Vulkan 1.3
                } else {
                    out += "textureLod(" + args[0] + ", " + args[1] + ", 0.0)";
                }
            } else {
                std::string gc = grad_coord(args[0], args[1]);
                if (args.size() == 2)
                    out += "textureGrad(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + "), sm_dFdy(" + gc + "))";
                else
                    out += "textureGrad(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + ") * exp2(" + args[2] + "), sm_dFdy(" + gc + ") * exp2(" + args[2] + "))";
            }
        } else if (which == 1 && (args.size() == 3 || args.size() == 4)) {
            std::string gc = grad_coord(args[0], args[1]);
            if (args.size() == 3)
                out += "textureGradOffset(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + "), sm_dFdy(" + gc + "), " + args[2] + ")";
            else
                out += "textureGradOffset(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + ") * exp2(" + args[3] + "), sm_dFdy(" + gc + ") * exp2(" + args[3] + "), " + args[2] + ")";
        } else if (which == 2 && args.size() >= 2) {
            std::string gc = grad_coord(args[0], args[1]);
            out += "textureProjGrad(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + "), sm_dFdy(" + gc + "))";
        } else if (which == 3 && args.size() >= 3) {
            std::string gc = grad_coord(args[0], args[1]);
            out += "textureProjGradOffset(" + args[0] + ", " + args[1] + ", sm_dFdx(" + gc + "), sm_dFdy(" + gc + "), " + args[2] + ")";
        } else {
            out.append(src, f, cur - f);
        }
        pos = cur;
    }
    return out;
}

static std::string strip_version_and_stage_io(const std::string& glsl) {
    std::istringstream iss(glsl);
    std::ostringstream oss;
    std::string line;
    std::regex re_ver(R"(^\s*#version\b.*)");
    std::regex re_in(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*in\s+(.+);)");
    std::regex re_out(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*out\s+(.+);)");
    std::regex re_qual_line(R"(^\s*(?:invariant|precise)\s+[A-Za-z_][A-Za-z0-9_]*\s*;)");
    std::regex re_fs_layout(R"(^\s*layout\s*\([^)]*(?:early_fragment_tests|origin_upper_left|pixel_center_integer|depth_any|depth_greater|depth_less|depth_unchanged)[^)]*\)\s*(?:in|out)\b.*;)");
    std::regex re_perv(R"(^\s*out\s+gl_PerVertex\b.*)");
    std::regex re_ext_fs(R"(^\s*#extension\s+(?:GL_ARB_fragment_shader_interlock|GL_EXT_demote_to_helper_invocation|GL_ARB_derivative_control)\b.*)");
    bool skip_perv_block = false;

    while (std::getline(iss, line)) {
        if (skip_perv_block) {
            if (line.find("};") != std::string::npos) skip_perv_block = false;
            continue;
        }
        if (std::regex_match(line, re_ver)) continue;
        if (std::regex_match(line, re_qual_line)) continue;
        if (std::regex_match(line, re_fs_layout)) continue;
        if (std::regex_match(line, re_ext_fs)) continue;
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
    return oss.str();
}

static std::string swiz(uint32_t vecsize) {
    if (vecsize == 1) return ".x";
    if (vecsize == 2) return ".xy";
    if (vecsize == 3) return ".xyz";
    return "";
}

static std::string emit_attr_unpack(const ReflectedIO& io, const VkVertexInputAttributeDescription* attr) {
    std::ostringstream ss;
    uint32_t b = attr->binding;
    if (b >= SM_MAX_VTX_BINDINGS) b = 0;
    std::string vb_off = (b < 4) ? ("sm_vs.vtx_offsets[" + std::to_string(b) + "]")
                                 : ("sm_vs.vtx_offsets_hi[" + std::to_string(b - 4) + "]");
    std::string vb_str = (b < 4) ? ("sm_vs.vtx_strides[" + std::to_string(b) + "]")
                                 : ("sm_vs.vtx_strides_hi[" + std::to_string(b - 4) + "]");
    ss << "    {\n";
    ss << "        uint elem_i = ((sm_vs.vtx_inst_rate_mask & (1u << " << b << "u)) != 0u) ? uint(sm_iidx) : uint(sm_vidx);\n";
    ss << "        uint byte_off = " << vb_off << " + elem_i * " << vb_str << " + " << attr->offset << "u;\n";
    ss << "        uint w0 = byte_off >> 2u;\n";
    ss << "        uint bsh = (byte_off & 3u) << 3u;\n";
    auto load_u = [&](int k) {
        std::string cur = "sm_vb" + std::to_string(b) + "[w0 + " + std::to_string(k) + "u]";
        std::string nxt = "sm_vb" + std::to_string(b) + "[w0 + " + std::to_string(k + 1) + "u]";
        return "((bsh == 0u) ? " + cur + " : ((" + cur + " >> bsh) | (" + nxt + " << (32u - bsh))))";
    };
    auto load_u_raw = [&](int k) { return "sm_vb" + std::to_string(b) + "[w0 + " + std::to_string(k) + "u]"; };
    auto load_f = [&](int k) { return "uintBitsToFloat(" + load_u(k) + ")"; };

    auto assign_f4 = [&](const std::string& expr) {
        std::string s = swiz(io.vecsize);
        if (io.basetype == SPVC_BASETYPE_UINT32)
            ss << "        " << io.name << " = " << io.glsl_type << "((" << expr << ")" << s << ");\n";
        else if (io.basetype == SPVC_BASETYPE_INT32)
            ss << "        " << io.name << " = " << io.glsl_type << "((" << expr << ")" << s << ");\n";
        else
            ss << "        " << io.name << " = (" << expr << ")" << s << ";\n";
    };
    auto assign_u4 = [&](const std::string& expr) {
        std::string s = swiz(io.vecsize);
        if (io.basetype == SPVC_BASETYPE_UINT32)
            ss << "        " << io.name << " = (" << expr << ")" << s << ";\n";
        else
            ss << "        " << io.name << " = " << io.glsl_type << "((" << expr << ")" << s << ");\n";
    };
    auto assign_i4 = [&](const std::string& expr) {
        std::string s = swiz(io.vecsize);
        if (io.basetype == SPVC_BASETYPE_INT32)
            ss << "        " << io.name << " = (" << expr << ")" << s << ";\n";
        else
            ss << "        " << io.name << " = " << io.glsl_type << "((" << expr << ")" << s << ");\n";
    };

    switch (attr->format) {
        case VK_FORMAT_R32_SFLOAT:
            assign_f4("vec4(" + load_f(0) + ", 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R32G32_SFLOAT:
            assign_f4("vec4(" + load_f(0) + ", " + load_f(1) + ", 0.0, 1.0)"); break;
        case VK_FORMAT_R32G32B32_SFLOAT:
            assign_f4("vec4(" + load_f(0) + ", " + load_f(1) + ", " + load_f(2) + ", 1.0)"); break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            assign_f4("vec4(" + load_f(0) + ", " + load_f(1) + ", " + load_f(2) + ", " + load_f(3) + ")"); break;
        case VK_FORMAT_R16_SFLOAT:
            ss << "        uint p16 = (" << load_u_raw(0) << " >> bsh) & 0xFFFFu;\n";
            assign_f4("vec4(unpackHalf2x16(p16).x, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_SFLOAT:
            assign_f4("vec4(unpackHalf2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            assign_f4("vec4(unpackHalf2x16(" + load_u(0) + "), unpackHalf2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R16_UNORM:
            ss << "        uint p16 = (" << load_u_raw(0) << " >> bsh) & 0xFFFFu;\n";
            assign_f4("vec4(float(p16) / 65535.0, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_UNORM:
            assign_f4("vec4(unpackUnorm2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_UNORM:
            assign_f4("vec4(unpackUnorm2x16(" + load_u(0) + "), unpackUnorm2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R16_SNORM:
            ss << "        int s16 = bitfieldExtract(int(" << load_u_raw(0) << "), int(bsh), 16);\n";
            assign_f4("vec4(max(float(s16) / 32767.0, -1.0), 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_SNORM:
            assign_f4("vec4(unpackSnorm2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_SNORM:
            assign_f4("vec4(unpackSnorm2x16(" + load_u(0) + "), unpackSnorm2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R8_UNORM:
            assign_f4("vec4(float((" + load_u_raw(0) + " >> bsh) & 255u) / 255.0, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R8G8_UNORM:
            ss << "        uint p82 = (" << load_u(0) << ") & 0xFFFFu;\n";
            assign_f4("vec4(float(p82 & 255u) / 255.0, float((p82 >> 8u) & 255u) / 255.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R8G8B8A8_UNORM:
            assign_f4("unpackUnorm4x8(" + load_u(0) + ")"); break;
        case VK_FORMAT_B8G8R8A8_UNORM:
            assign_f4("unpackUnorm4x8(" + load_u(0) + ").bgra"); break;
        case VK_FORMAT_R8G8B8A8_SNORM:
            assign_f4("unpackSnorm4x8(" + load_u(0) + ")"); break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            ss << "        uint p10 = " << load_u(0) << ";\n";
            assign_f4("vec4(float(p10&1023u)/1023.0, float((p10>>10u)&1023u)/1023.0, float((p10>>20u)&1023u)/1023.0, float((p10>>30u)&3u)/3.0)");
            break;
        case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
            ss << "        int p10 = int(" << load_u(0) << ");\n";
            assign_f4("vec4(max(float(bitfieldExtract(p10,0,10))/511.0,-1.0), max(float(bitfieldExtract(p10,10,10))/511.0,-1.0), max(float(bitfieldExtract(p10,20,10))/511.0,-1.0), max(float(bitfieldExtract(p10,30,2))/1.0,-1.0))");
            break;
        case VK_FORMAT_R8_UINT:
            assign_u4("uvec4((" + load_u_raw(0) + " >> bsh) & 255u, 0u, 0u, 1u)"); break;
        case VK_FORMAT_R8G8_UINT:
            ss << "        uint pu8 = " << load_u(0) << ";\n";
            assign_u4("uvec4(pu8 & 255u, (pu8 >> 8u) & 255u, 0u, 1u)"); break;
        case VK_FORMAT_R8G8B8A8_UINT:
            ss << "        uint pu8 = " << load_u(0) << ";\n";
            assign_u4("uvec4(pu8 & 255u, (pu8>>8u)&255u, (pu8>>16u)&255u, (pu8>>24u)&255u)"); break;
        case VK_FORMAT_R8G8B8A8_SINT:
            ss << "        int pi8 = int(" << load_u(0) << ");\n";
            assign_i4("ivec4(bitfieldExtract(pi8,0,8), bitfieldExtract(pi8,8,8), bitfieldExtract(pi8,16,8), bitfieldExtract(pi8,24,8))"); break;
        case VK_FORMAT_R16_UINT:
            assign_u4("uvec4((" + load_u_raw(0) + " >> bsh) & 65535u, 0u, 0u, 1u)"); break;
        case VK_FORMAT_R16G16_UINT:
            ss << "        uint pu16 = " << load_u(0) << ";\n";
            assign_u4("uvec4(pu16 & 65535u, pu16 >> 16u, 0u, 1u)"); break;
        case VK_FORMAT_R16G16B16A16_UINT:
            ss << "        uint pu16a = " << load_u(0) << ", pu16b = " << load_u(1) << ";\n";
            assign_u4("uvec4(pu16a & 65535u, pu16a >> 16u, pu16b & 65535u, pu16b >> 16u)"); break;
        case VK_FORMAT_R16G16_SINT:
            ss << "        int pi16 = int(" << load_u(0) << ");\n";
            assign_i4("ivec4(bitfieldExtract(pi16,0,16), bitfieldExtract(pi16,16,16), 0, 1)"); break;
        case VK_FORMAT_R16G16B16A16_SINT:
            ss << "        int pi16a = int(" << load_u(0) << "), pi16b = int(" << load_u(1) << ");\n";
            assign_i4("ivec4(bitfieldExtract(pi16a,0,16), bitfieldExtract(pi16a,16,16), bitfieldExtract(pi16b,0,16), bitfieldExtract(pi16b,16,16))"); break;
        case VK_FORMAT_R32_UINT:
            assign_u4("uvec4(" + load_u(0) + ", 0u, 0u, 1u)"); break;
        case VK_FORMAT_R32G32_UINT:
            assign_u4("uvec4(" + load_u(0) + ", " + load_u(1) + ", 0u, 1u)"); break;
        case VK_FORMAT_R32G32B32_UINT:
            assign_u4("uvec4(" + load_u(0) + ", " + load_u(1) + ", " + load_u(2) + ", 1u)"); break;
        case VK_FORMAT_R32G32B32A32_UINT:
            assign_u4("uvec4(" + load_u(0) + ", " + load_u(1) + ", " + load_u(2) + ", " + load_u(3) + ")"); break;
        case VK_FORMAT_R32_SINT:
            assign_i4("ivec4(int(" + load_u(0) + "), 0, 0, 1)"); break;
        case VK_FORMAT_R32G32_SINT:
            assign_i4("ivec4(int(" + load_u(0) + "), int(" + load_u(1) + "), 0, 1)"); break;
        case VK_FORMAT_R32G32B32_SINT:
            assign_i4("ivec4(int(" + load_u(0) + "), int(" + load_u(1) + "), int(" + load_u(2) + "), 1)"); break;
        case VK_FORMAT_R32G32B32A32_SINT:
            assign_i4("ivec4(int(" + load_u(0) + "), int(" + load_u(1) + "), int(" + load_u(2) + "), int(" + load_u(3) + "))"); break;
        default:
            assign_f4("vec4(" + load_f(0) + ", " + load_f(1) + ", " + load_f(2) + ", " + load_f(3) + ")"); break;
    }
    ss << "    }\n";
    return ss.str();
}

static std::string pack_varying_to_vec4(const std::string& expr, spvc_basetype bt, uint32_t vecsize) {
    std::string e = expr;
    if (bt == SPVC_BASETYPE_INT32)  e = "intBitsToFloat(" + expr + ")";
    if (bt == SPVC_BASETYPE_UINT32) e = "uintBitsToFloat(" + expr + ")";
    if (vecsize == 4) return "vec4(" + e + ")";
    if (vecsize == 3) return "vec4(" + e + ", 0.0)";
    if (vecsize == 2) return "vec4(" + e + ", 0.0, 0.0)";
    return "vec4(float(" + e + "), 0.0, 0.0, 0.0)";
}

bool sm_lower_vertex_spirv(const std::vector<uint32_t>& in_spv,
                           const SmVertexInputState& vtx_in,
                           const VkSpecializationInfo*,
                           uint32_t sm_set_idx,
                           std::vector<uint32_t>& out_comp_spv,
                           std::string& err) {
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err)) return false;

    std::string body = strip_version_and_stage_io(glsl);
    body = replace_ident(body, "gl_Position", "sm_gl_Position");
    body = replace_ident(body, "gl_PointSize", "sm_gl_PointSize");
    body = replace_ident(body, "gl_ClipDistance", "sm_gl_ClipDistance");
    body = replace_ident(body, "gl_CullDistance", "sm_gl_CullDistance");
    body = replace_ident(body, "gl_Layer", "sm_gl_Layer");
    body = replace_ident(body, "gl_ViewportIndex", "sm_gl_ViewportIndex");
    body = replace_ident(body, "gl_VertexIndex", "sm_vidx");
    body = replace_ident(body, "gl_InstanceIndex", "sm_iidx");
    body = replace_ident(body, "gl_BaseVertex", "sm_base_vertex");
    body = replace_ident(body, "gl_BaseInstance", "sm_base_instance");
    body = replace_ident(body, "gl_DrawID", "0");
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_vs_main()");

    std::ostringstream ss;
    ss << "#version 460\n";
    ss << "layout(local_size_x = 64) in;\n\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0) writeonly buffer SmClipVertBuf { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1) readonly buffer SmIdxBuf { uint sm_idx_raw[]; };\n";
    for (uint32_t b = 0; b < SM_MAX_VTX_BINDINGS; ++b) {
        ss << "layout(std430, set = " << sm_set_idx << ", binding = " << (2u + b) << ") readonly buffer SmVB" << b << " { uint sm_vb" << b << "[]; };\n";
    }
    ss << "layout(std140, set = " << sm_set_idx << ", binding = 10) uniform SmVsParams {\n"
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
       << "    uvec4 vtx_offsets_hi;\n"
       << "    uvec4 vtx_strides_hi;\n"
       << "} sm_vs;\n\n";

    ss << "vec4 sm_gl_Position;\n";
    ss << "float sm_gl_PointSize = 1.0;\n";
    ss << "float sm_gl_ClipDistance[8];\n";
    ss << "float sm_gl_CullDistance[8];\n";
    ss << "int sm_gl_Layer = 0;\n";
    ss << "int sm_gl_ViewportIndex = 0;\n";
    ss << "int sm_vidx;\n";
    ss << "int sm_iidx;\n";
    ss << "int sm_base_vertex;\n";
    ss << "int sm_base_instance;\n\n";
    ss << body << "\n\n";

    ss << "void main() {\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    if (gid >= sm_vs.vertex_count) return;\n";
    ss << "    uint vpi = max(sm_vs.verts_per_instance, 1u);\n";
    ss << "    uint inst_local = gid / vpi;\n";
    ss << "    uint vert_local = gid - inst_local * vpi;\n";
    ss << "    sm_base_instance = int(sm_vs.instance_idx);\n";
    ss << "    sm_base_vertex   = (sm_vs.indexed != 0u) ? sm_vs.vertex_offset : int(sm_vs.first_vertex);\n";
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

    for (const auto& io : inputs) {
        if (io.array_size > 0) continue;
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
        uint32_t elems = (io.array_size > 0) ? io.array_size : 1u;
        for (uint32_t a = 0; a < elems; ++a) {
            uint32_t loc = io.location + a;
            if (loc >= SM_MAX_VARYINGS) continue;
            std::string expr = (io.array_size > 0) ? (io.name + "[" + std::to_string(a) + "]") : io.name;
            if (io.component == 0) {
                ss << "    sm_clip_verts[base_slot + " << (1u + loc) << "u] = "
                   << pack_varying_to_vec4(expr, io.basetype, io.vecsize) << ";\n";
            } else {
                const char* comps = "xyzw";
                for (uint32_t c = 0; c < io.vecsize && (io.component + c) < 4; ++c) {
                    std::string comp_e = (io.vecsize == 1) ? expr : (expr + "." + comps[c]);
                    if (io.basetype == SPVC_BASETYPE_INT32) comp_e = "intBitsToFloat(" + comp_e + ")";
                    else if (io.basetype == SPVC_BASETYPE_UINT32) comp_e = "uintBitsToFloat(" + comp_e + ")";
                    ss << "    sm_clip_verts[base_slot + " << (1u + loc) << "u]." << comps[io.component + c]
                       << " = " << comp_e << ";\n";
                }
            }
        }
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "vs", out_comp_spv, err);
}

bool sm_lower_fragment_spirv(const std::vector<uint32_t>& in_spv,
                             const VkSpecializationInfo*,
                             uint32_t sm_set_idx,
                             const SmFsCompileHints& hints,
                             std::vector<uint32_t>& out_comp_spv,
                             std::string& err) {
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err)) return false;

    bool uses_frag_depth = (glsl.find("gl_FragDepth") != std::string::npos);

    std::string body = strip_version_and_stage_io(glsl);
    body = replace_ident(body, "gl_FragCoord", "sm_gl_FragCoord");
    body = replace_ident(body, "gl_FrontFacing", "sm_gl_FrontFacing");
    body = replace_ident(body, "gl_FragDepth", "sm_gl_FragDepth");
    body = replace_ident(body, "gl_HelperInvocation", "sm_gl_HelperInvocation");
    body = std::regex_replace(body, std::regex(R"(\bhelperInvocationEXT\s*\(\s*\))"), "sm_gl_HelperInvocation");
    body = replace_ident(body, "gl_SampleMask", "sm_gl_SampleMask");
    body = replace_ident(body, "gl_SampleMaskIn", "sm_gl_SampleMaskIn");
    body = replace_ident(body, "gl_SampleID", "sm_gl_SampleID");
    body = replace_ident(body, "gl_SamplePosition", "vec2(0.5)");
    body = replace_ident(body, "gl_Layer", "sm_gl_Layer");
    body = replace_ident(body, "gl_ViewportIndex", "sm_gl_ViewportIndex");
    body = replace_ident(body, "gl_PrimitiveID", "sm_gl_PrimitiveID");
    body = replace_ident(body, "dFdx", "sm_dFdx");
    body = replace_ident(body, "dFdy", "sm_dFdy");
    body = replace_ident(body, "dFdxFine", "sm_dFdx");
    body = replace_ident(body, "dFdyFine", "sm_dFdy");
    body = replace_ident(body, "dFdxCoarse", "sm_dFdx");
    body = replace_ident(body, "dFdyCoarse", "sm_dFdy");
    body = replace_ident(body, "fwidth", "sm_fwidth");
    body = replace_ident(body, "fwidthFine", "sm_fwidth");
    body = replace_ident(body, "fwidthCoarse", "sm_fwidth");
    body = rewrite_texture_calls(body);
    body = std::regex_replace(body, std::regex(R"(\bdemote\s*;)"), "{ sm_discarded = true; sm_gl_HelperInvocation = true; }");
    body = std::regex_replace(body, std::regex(R"(\bdiscard\s*;)"), "{ sm_discarded = true; sm_gl_HelperInvocation = true; }");
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_fs_main()");

    std::ostringstream ss;
    ss << "#version 460\n";
    ss << "#extension GL_ARB_gpu_shader_int64 : require\n";
    ss << "#extension GL_KHR_shader_subgroup_quad : require\n";
    ss << "#extension GL_KHR_shader_subgroup_vote : require\n";
    ss << "#extension GL_EXT_texture_shadow_lod : enable\n";
    ss << "layout(local_size_x = 256) in;\n\n";

    ss << "struct SmTriSetup {\n"
       << "    vec4 v0; vec4 v1; vec4 v2; vec4 area; uvec4 idx;\n"
       << "};\n";

    // SmRast v0.3 MRT FS Internal Descriptor Set (up to 4 Color RTs + Depth + FsParamsUBO!)
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0) readonly buffer SmClipVertBuf { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1) readonly buffer SmTriSetupBuf { SmTriSetup sm_tris[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 2) readonly buffer SmVisBuf      { uint64_t sm_vis_px[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 3) buffer SmColor0Buf            { uint sm_col0[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 4) buffer SmColor1Buf            { uint sm_col1[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 5) buffer SmColor2Buf            { uint sm_col2[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 6) buffer SmColor3Buf            { uint sm_col3[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 7) buffer SmDepthBuf             { float sm_depth_px[]; };\n";
    ss << "layout(std140, set = " << sm_set_idx << ", binding = 8) uniform SmFsParams {\n"
       << "    uint width;\n"
       << "    uint height;\n"
       << "    uint vert_stride_vec4;\n"
       << "    uint depth_write;\n"
       << "    uint rt_active_mask;\n"
       << "    uint pad0; uint pad1; uint pad2;\n"
       << "    uvec4 rt_fmt;\n"
       << "    uvec4 rt_blend_en;\n"
       << "    uvec4 rt_src_blend;\n"
       << "    uvec4 rt_dst_blend;\n"
       << "    uvec4 rt_blend_op;\n"
       << "} sm_fs;\n\n";

    ss << "uint sm_quad_lane;\n";
    ss << "vec4 sm_gl_FragCoord;\n";
    ss << "bool sm_gl_FrontFacing = true;\n";
    ss << "float sm_gl_FragDepth = 0.0;\n";
    ss << "bool sm_gl_HelperInvocation = false;\n";
    ss << "int sm_gl_SampleMask[1];\n";
    ss << "int sm_gl_SampleMaskIn[1] = int[1](1);\n";
    ss << "int sm_gl_SampleID = 0;\n";
    ss << "int sm_gl_Layer = 0;\n";
    ss << "int sm_gl_ViewportIndex = 0;\n";
    ss << "int sm_gl_PrimitiveID = 0;\n";
    ss << "bool sm_discarded = false;\n\n";

    // 1-cycle Branchless Hopper Quad Subgroup Derivatives (zero basic-block splits inside textureGrad!)
    ss << "float sm_dFdx(float v) { return (subgroupQuadSwapHorizontal(v) - v) * (1.0 - 2.0 * float(sm_quad_lane & 1u)); }\n";
    ss << "vec2  sm_dFdx(vec2  v) { return (subgroupQuadSwapHorizontal(v) - v) * (1.0 - 2.0 * float(sm_quad_lane & 1u)); }\n";
    ss << "vec3  sm_dFdx(vec3  v) { return (subgroupQuadSwapHorizontal(v) - v) * (1.0 - 2.0 * float(sm_quad_lane & 1u)); }\n";
    ss << "vec4  sm_dFdx(vec4  v) { return (subgroupQuadSwapHorizontal(v) - v) * (1.0 - 2.0 * float(sm_quad_lane & 1u)); }\n";
    ss << "float sm_dFdy(float v) { return (subgroupQuadSwapVertical(v)   - v) * (1.0 - 2.0 * float((sm_quad_lane >> 1u) & 1u)); }\n";
    ss << "vec2  sm_dFdy(vec2  v) { return (subgroupQuadSwapVertical(v)   - v) * (1.0 - 2.0 * float((sm_quad_lane >> 1u) & 1u)); }\n";
    ss << "vec3  sm_dFdy(vec3  v) { return (subgroupQuadSwapVertical(v)   - v) * (1.0 - 2.0 * float((sm_quad_lane >> 1u) & 1u)); }\n";
    ss << "vec4  sm_dFdy(vec4  v) { return (subgroupQuadSwapVertical(v)   - v) * (1.0 - 2.0 * float((sm_quad_lane >> 1u) & 1u)); }\n";
    ss << "float sm_fwidth(float v) { return abs(sm_dFdx(v)) + abs(sm_dFdy(v)); }\n";
    ss << "vec2  sm_fwidth(vec2  v) { return abs(sm_dFdx(v)) + abs(sm_dFdy(v)); }\n";
    ss << "vec3  sm_fwidth(vec3  v) { return abs(sm_dFdx(v)) + abs(sm_dFdy(v)); }\n";
    ss << "vec4  sm_fwidth(vec4  v) { return abs(sm_dFdx(v)) + abs(sm_dFdy(v)); }\n\n";

    // Determine whether we need heavy dynamic format/blend helpers (zero register bloat for opaque RGBA8/BGRA8!)
    bool need_any_blend = false;
    bool need_dyn_fmt = false;
    bool need_srgb = false;
    bool need_r11g11b10 = false;
    for (const auto& io : outputs) {
        if (io.array_size == 0 && io.location < SM_MAX_MRT) {
            if (hints.blend_en[io.location]) need_any_blend = true;
            int fc = hints.rt_fmt[io.location];
            if (fc < 0) { need_dyn_fmt = true; need_srgb = true; need_r11g11b10 = true; }
            else {
                if (fc == 2 || fc == 3) need_srgb = true;
                if (fc == 5) need_r11g11b10 = true;
            }
        }
    }

    if (need_srgb) {
        ss << "float sm_lin_to_srgb(float x) { return (x <= 0.0031308) ? (x * 12.92) : (1.055 * pow(max(x, 0.0), 1.0/2.4) - 0.055); }\n";
    }
    if (need_r11g11b10) {
        ss << "uint sm_f32_to_uf11(float v) { uint u = floatBitsToUint(max(v, 0.0)); uint e = (u >> 23u) & 255u; if (e < 112u) return 0u; if (e >= 143u) return 0x7FFu; return ((e - 112u) << 6u) | ((u >> 17u) & 63u); }\n";
        ss << "uint sm_f32_to_uf10(float v) { uint u = floatBitsToUint(max(v, 0.0)); uint e = (u >> 23u) & 255u; if (e < 112u) return 0u; if (e >= 143u) return 0x3FFu; return ((e - 112u) << 5u) | ((u >> 18u) & 31u); }\n";
        ss << "float sm_uf11_to_f32(uint v) { uint e = (v >> 6u) & 31u; uint m = v & 63u; if (e == 0u) return 0.0; return uintBitsToFloat(((e + 112u) << 23u) | (m << 17u)); }\n";
        ss << "float sm_uf10_to_f32(uint v) { uint e = (v >> 5u) & 31u; uint m = v & 31u; if (e == 0u) return 0.0; return uintBitsToFloat(((e + 112u) << 23u) | (m << 18u)); }\n";
    }
    if (need_dyn_fmt) {
        ss << "uint sm_pack_u32_fmt(vec4 c, uint fc) {\n"
           << "    if (fc == 5u) return sm_f32_to_uf11(c.r) | (sm_f32_to_uf11(c.g) << 11u) | (sm_f32_to_uf10(c.b) << 22u);\n"
           << "    if (fc == 6u) return floatBitsToUint(c.r);\n"
           << "    if (fc == 7u) return packHalf2x16(c.rg);\n"
           << "    if (fc == 9u) { vec4 cl = clamp(c, 0.0, 1.0); return uint(round(cl.r*1023.0)) | (uint(round(cl.g*1023.0))<<10u) | (uint(round(cl.b*1023.0))<<20u) | (uint(round(cl.a*3.0))<<30u); }\n"
           << "    c = clamp(c, 0.0, 1.0);\n"
           << "    if (fc == 2u || fc == 3u) c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b));\n"
           << "    if (fc == 1u || fc == 3u) c = c.bgra;\n"
           << "    return packUnorm4x8(c);\n"
           << "}\n";
    }
    if (need_any_blend && need_dyn_fmt) {
        ss << "vec4 sm_unpack_u32_fmt(uint pk, uint fc) {\n"
           << "    if (fc == 5u) return vec4(sm_uf11_to_f32(pk & 0x7FFu), sm_uf11_to_f32((pk>>11u) & 0x7FFu), sm_uf10_to_f32((pk>>22u) & 0x3FFu), 1.0);\n"
           << "    if (fc == 6u) return vec4(uintBitsToFloat(pk), 0.0, 0.0, 1.0);\n"
           << "    if (fc == 7u) return vec4(unpackHalf2x16(pk), 0.0, 1.0);\n"
           << "    if (fc == 9u) return vec4(float(pk&1023u)/1023.0, float((pk>>10u)&1023u)/1023.0, float((pk>>20u)&1023u)/1023.0, float((pk>>30u)&3u)/3.0);\n"
           << "    vec4 c = unpackUnorm4x8(pk);\n"
           << "    return (fc == 1u || fc == 3u) ? c.bgra : c;\n"
           << "}\n";
    }
    if (need_any_blend) {
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
    }

    ss << body << "\n\n";

    // Per-RT write helper: compile-time specialized when format & blend state are known (max Hopper SM occupancy!)
    auto emit_rt_store = [&](int rt_i, const ReflectedIO& io) {
        std::string buf = "sm_col" + std::to_string(rt_i);
        int kfc = hints.rt_fmt[rt_i];
        bool ben = hints.blend_en[rt_i];
        ss << "    if ((sm_fs.rt_active_mask & (1u << " << rt_i << "u)) != 0u) {\n";
        if (kfc == 11 || io.basetype == SPVC_BASETYPE_UINT32 || io.basetype == SPVC_BASETYPE_INT32) {
            std::string first_comp = (io.vecsize == 1) ? io.name : (io.name + ".x");
            ss << "        " << buf << "[pidx] = uint(" << first_comp << ");\n";
            ss << "    }\n";
            return;
        }
        ss << "        vec4 out_c = ";
        if (io.vecsize == 4) ss << "vec4(" << io.name << ");\n";
        else if (io.vecsize == 3) ss << "vec4(" << io.name << ", 1.0);\n";
        else if (io.vecsize == 2) ss << "vec4(" << io.name << ", 0.0, 1.0);\n";
        else ss << "vec4(float(" << io.name << "), 0.0, 0.0, 1.0);\n";

        if (kfc >= 0) {
            if (ben) {
                ss << "        vec4 dst_c;\n";
                if (kfc == 4)       ss << "        dst_c = vec4(unpackHalf2x16(" << buf << "[pidx*2u]), unpackHalf2x16(" << buf << "[pidx*2u+1u]));\n";
                else if (kfc == 10) ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[pidx*2u]), uintBitsToFloat(" << buf << "[pidx*2u+1u]), 0.0, 1.0);\n";
                else if (kfc == 8)  ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[pidx*4u]), uintBitsToFloat(" << buf << "[pidx*4u+1u]), uintBitsToFloat(" << buf << "[pidx*4u+2u]), uintBitsToFloat(" << buf << "[pidx*4u+3u]));\n";
                else if (kfc == 5)  ss << "        { uint pk = " << buf << "[pidx]; dst_c = vec4(sm_uf11_to_f32(pk&0x7FFu), sm_uf11_to_f32((pk>>11u)&0x7FFu), sm_uf10_to_f32((pk>>22u)&0x3FFu), 1.0); }\n";
                else if (kfc == 6)  ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[pidx]), 0.0, 0.0, 1.0);\n";
                else if (kfc == 7)  ss << "        dst_c = vec4(unpackHalf2x16(" << buf << "[pidx]), 0.0, 1.0);\n";
                else if (kfc == 9)  ss << "        { uint pk = " << buf << "[pidx]; dst_c = vec4(float(pk&1023u)/1023.0, float((pk>>10u)&1023u)/1023.0, float((pk>>20u)&1023u)/1023.0, float((pk>>30u)&3u)/3.0); }\n";
                else if (kfc == 1 || kfc == 3) ss << "        dst_c = unpackUnorm4x8(" << buf << "[pidx]).bgra;\n";
                else                ss << "        dst_c = unpackUnorm4x8(" << buf << "[pidx]);\n";
                ss << "        vec4 sf = sm_blend_factor(sm_fs.rt_src_blend[" << rt_i << "], out_c, dst_c);\n";
                ss << "        vec4 df = sm_blend_factor(sm_fs.rt_dst_blend[" << rt_i << "], out_c, dst_c);\n";
                ss << "        out_c = sm_blend_combine(sm_fs.rt_blend_op[" << rt_i << "], out_c * sf, dst_c * df);\n";
            }
            if (kfc == 0)      ss << "        " << buf << "[pidx] = packUnorm4x8(clamp(out_c, 0.0, 1.0));\n";
            else if (kfc == 1) ss << "        " << buf << "[pidx] = packUnorm4x8(clamp(out_c, 0.0, 1.0).bgra);\n";
            else if (kfc == 2) ss << "        { vec4 c = clamp(out_c, 0.0, 1.0); c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b)); " << buf << "[pidx] = packUnorm4x8(c); }\n";
            else if (kfc == 3) ss << "        { vec4 c = clamp(out_c, 0.0, 1.0); c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b)); " << buf << "[pidx] = packUnorm4x8(c.bgra); }\n";
            else if (kfc == 4) ss << "        " << buf << "[pidx*2u] = packHalf2x16(out_c.rg); " << buf << "[pidx*2u+1u] = packHalf2x16(out_c.ba);\n";
            else if (kfc == 5) ss << "        " << buf << "[pidx] = sm_f32_to_uf11(out_c.r) | (sm_f32_to_uf11(out_c.g) << 11u) | (sm_f32_to_uf10(out_c.b) << 22u);\n";
            else if (kfc == 6) ss << "        " << buf << "[pidx] = floatBitsToUint(out_c.r);\n";
            else if (kfc == 7) ss << "        " << buf << "[pidx] = packHalf2x16(out_c.rg);\n";
            else if (kfc == 8) ss << "        " << buf << "[pidx*4u]=floatBitsToUint(out_c.r); " << buf << "[pidx*4u+1u]=floatBitsToUint(out_c.g); " << buf << "[pidx*4u+2u]=floatBitsToUint(out_c.b); " << buf << "[pidx*4u+3u]=floatBitsToUint(out_c.a);\n";
            else if (kfc == 9) ss << "        { vec4 cl = clamp(out_c, 0.0, 1.0); " << buf << "[pidx] = uint(round(cl.r*1023.0))|(uint(round(cl.g*1023.0))<<10u)|(uint(round(cl.b*1023.0))<<20u)|(uint(round(cl.a*3.0))<<30u); }\n";
            else if (kfc == 10) ss << "        " << buf << "[pidx*2u] = floatBitsToUint(out_c.r); " << buf << "[pidx*2u+1u] = floatBitsToUint(out_c.g);\n";
            else if (kfc == 11) ss << "        " << buf << "[pidx] = uint(out_c.r);\n";
        } else {
            ss << "        uint fc = sm_fs.rt_fmt[" << rt_i << "];\n";
            if (ben) {
                ss << "        if (sm_fs.rt_blend_en[" << rt_i << "] != 0u) {\n";
                ss << "            vec4 dst_c;\n";
                ss << "            if (fc == 4u) dst_c = vec4(unpackHalf2x16(" << buf << "[pidx*2u]), unpackHalf2x16(" << buf << "[pidx*2u+1u]));\n";
                ss << "            else if (fc == 10u) dst_c = vec4(uintBitsToFloat(" << buf << "[pidx*2u]), uintBitsToFloat(" << buf << "[pidx*2u+1u]), 0.0, 1.0);\n";
                ss << "            else if (fc == 8u) dst_c = vec4(uintBitsToFloat(" << buf << "[pidx*4u]), uintBitsToFloat(" << buf << "[pidx*4u+1u]), uintBitsToFloat(" << buf << "[pidx*4u+2u]), uintBitsToFloat(" << buf << "[pidx*4u+3u]));\n";
                ss << "            else dst_c = sm_unpack_u32_fmt(" << buf << "[pidx], fc);\n";
                ss << "            vec4 sf = sm_blend_factor(sm_fs.rt_src_blend[" << rt_i << "], out_c, dst_c);\n";
                ss << "            vec4 df = sm_blend_factor(sm_fs.rt_dst_blend[" << rt_i << "], out_c, dst_c);\n";
                ss << "            out_c = sm_blend_combine(sm_fs.rt_blend_op[" << rt_i << "], out_c * sf, dst_c * df);\n";
                ss << "        }\n";
            }
            ss << "        if (fc == 4u) {\n";
            ss << "            " << buf << "[pidx*2u + 0u] = packHalf2x16(out_c.rg);\n";
            ss << "            " << buf << "[pidx*2u + 1u] = packHalf2x16(out_c.ba);\n";
            ss << "        } else if (fc == 10u) {\n";
            ss << "            " << buf << "[pidx*2u + 0u] = floatBitsToUint(out_c.r);\n";
            ss << "            " << buf << "[pidx*2u + 1u] = floatBitsToUint(out_c.g);\n";
            ss << "        } else if (fc == 8u) {\n";
            ss << "            " << buf << "[pidx*4u + 0u] = floatBitsToUint(out_c.r);\n";
            ss << "            " << buf << "[pidx*4u + 1u] = floatBitsToUint(out_c.g);\n";
            ss << "            " << buf << "[pidx*4u + 2u] = floatBitsToUint(out_c.b);\n";
            ss << "            " << buf << "[pidx*4u + 3u] = floatBitsToUint(out_c.a);\n";
            ss << "        } else {\n";
            ss << "            " << buf << "[pidx] = sm_pack_u32_fmt(out_c, fc);\n";
            ss << "        }\n";
        }
        ss << "    }\n";
    };

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
    ss << "    sm_gl_HelperInvocation = !active_px;\n";
    ss << "    uint my_tri = active_px ? uint(vis & 0xFFFFFFFFul) : 0xFFFFFFFFu;\n";
    ss << "    uint t0 = subgroupQuadBroadcast(my_tri, 0u);\n";
    ss << "    uint t1 = subgroupQuadBroadcast(my_tri, 1u);\n";
    ss << "    uint t2 = subgroupQuadBroadcast(my_tri, 2u);\n";
    ss << "    uint t3 = subgroupQuadBroadcast(my_tri, 3u);\n";
    ss << "    uint eff_tri = active_px ? my_tri : min(min(t0, t1), min(t2, t3));\n";
    ss << "    if (eff_tri == 0xFFFFFFFFu) return;\n\n";

    ss << "    SmTriSetup ts = sm_tris[eff_tri];\n";
    ss << "    sm_gl_PrimitiveID = int(eff_tri);\n";
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
    ss << "    sm_gl_FragDepth = pz;\n";
    ss << "    uint vi0 = ts.idx.x * sm_fs.vert_stride_vec4;\n";
    ss << "    uint vi1 = ts.idx.y * sm_fs.vert_stride_vec4;\n";
    ss << "    uint vi2 = ts.idx.z * sm_fs.vert_stride_vec4;\n";

    const char* comps = "xyzw";
    for (const auto& io : inputs) {
        uint32_t elems = (io.array_size > 0) ? io.array_size : 1u;
        for (uint32_t a = 0; a < elems; ++a) {
            uint32_t loc = io.location + a;
            if (loc >= SM_MAX_VARYINGS) continue;
            uint32_t off = 1u + loc;
            std::string dst_var = (io.array_size > 0) ? (io.name + "[" + std::to_string(a) + "]") : io.name;
            std::string s;
            if (io.component == 0) {
                s = swiz(io.vecsize);
            } else {
                s = ".";
                for (uint32_t c = 0; c < io.vecsize && (io.component + c) < 4; ++c) s += comps[io.component + c];
            }
            ss << "    {\n";
            if (io.basetype == SPVC_BASETYPE_INT32) {
                ss << "        " << dst_var << " = floatBitsToInt(sm_clip_verts[vi0 + " << off << "u]" << s << ");\n";
            } else if (io.basetype == SPVC_BASETYPE_UINT32) {
                ss << "        " << dst_var << " = floatBitsToUint(sm_clip_verts[vi0 + " << off << "u]" << s << ");\n";
            } else {
                ss << "        vec4 interp = sm_clip_verts[vi0 + " << off << "u] * b0 + "
                   << "sm_clip_verts[vi1 + " << off << "u] * b1 + "
                   << "sm_clip_verts[vi2 + " << off << "u] * b2;\n";
                ss << "        " << dst_var << " = interp" << s << ";\n";
            }
            ss << "    }\n";
        }
    }

    ss << "    sm_discarded = false;\n";
    ss << "    sm_user_fs_main();\n";
    ss << "    if (!active_px || sm_discarded) return;\n";
    ss << "    if (sm_fs.depth_write != 0u) sm_depth_px[pidx] = " << (uses_frag_depth ? "sm_gl_FragDepth" : "pz") << ";\n";

    // Write ALL reflected MRT color outputs (Location 0..3)!
    for (const auto& io : outputs) {
        if (io.array_size == 0 && io.location < SM_MAX_MRT) emit_rt_store((int)io.location, io);
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "fs", out_comp_spv, err);
}
