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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <cerrno>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
extern char** environ;

// Prevent Wine thread teardown (pthread_cancel -> __forced_unwind) from firing inside
// libspirv-cross-c-shared / libshaderc C++ catch(...) blocks (which would trigger glibc
// "FATAL: exception not rethrown" during multi-threaded PSO compilation!).
struct SmCancelDisableGuard {
    int old_state = PTHREAD_CANCEL_ENABLE;
    SmCancelDisableGuard()  { pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_state); }
    ~SmCancelDisableGuard() { pthread_setcancelstate(old_state, nullptr); }
};

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

// Vulkan SPIR-V spec rule (VUID-StandaloneSpirv-OpSampledImage-04651 / 06671):
//   "All OpSampledImage instructions must be in the same basic block in which their Result <id> are consumed."
// Complex ternary/conditional coordinate expressions inside `textureGrad(sampler2D(tex, smp), cond ? a : b, ...)`
// cause glslang to emit OpSampledImage before the branch and OpImageSample* after the merge block, which crashes
// NVIDIA's libnvidia-glvkspirv.so (_nv002nvvm, si_addr=0x4) when compiled at -O0!
// Sinking each cross-block OpSampledImage immediately before its consumer instruction(s) guarantees 100% valid SPIR-V.
static void sm_sink_op_sampled_image(std::vector<uint32_t>& spv) {
    if (spv.size() < 5 || spv[0] != 0x07230203u) return;
    struct Inst { uint32_t pos, wc, op; };
    std::vector<Inst> insts;
    insts.reserve(spv.size() / 4);
    for (uint32_t p = 5; p < (uint32_t)spv.size(); ) {
        uint32_t w0 = spv[p];
        uint32_t wc = w0 >> 16;
        uint32_t op = w0 & 0xFFFFu;
        if (wc == 0 || p + wc > spv.size()) return; // malformed
        insts.push_back({p, wc, op});
        p += wc;
    }

    constexpr uint32_t kOpFunction     = 54;
    constexpr uint32_t kOpFunctionEnd  = 56;
    constexpr uint32_t kOpDecorate     = 71;
    constexpr uint32_t kOpSampledImage = 86;
    constexpr uint32_t kOpLabel        = 248;

    struct SDef {
        uint32_t inst_idx;
        uint32_t block_id;
        uint32_t res_type, res_id, img_id, smp_id;
    };
    struct SUse {
        uint32_t inst_idx;
        uint32_t word_k;
        uint32_t block_id;
    };

    uint32_t id_bound = spv[3];
    std::vector<uint8_t> skip_inst(insts.size(), 0);
    std::unordered_map<uint32_t, std::vector<std::array<uint32_t, 5>>> insert_before;
    std::unordered_map<uint64_t, uint32_t> rewrite_word; // (inst_idx<<32 | k) -> new_id
    std::vector<std::pair<uint32_t, uint32_t>> cloned_ids; // (orig_id, new_id) for OpDecorate cloning
    bool modified = false;

    for (size_t i = 0; i < insts.size(); ++i) {
        if (insts[i].op != kOpFunction) continue;
        size_t fn_end = i + 1;
        while (fn_end < insts.size() && insts[fn_end].op != kOpFunctionEnd) ++fn_end;
        if (fn_end >= insts.size()) break;

        std::unordered_map<uint32_t, SDef> defs;
        uint32_t blk = 0;
        for (size_t j = i + 1; j < fn_end; ++j) {
            if (insts[j].op == kOpLabel) ++blk;
            else if (insts[j].op == kOpSampledImage && insts[j].wc == 5) {
                uint32_t p = insts[j].pos;
                SDef d{(uint32_t)j, blk, spv[p+1], spv[p+2], spv[p+3], spv[p+4]};
                defs[d.res_id] = d;
            }
        }
        if (defs.empty()) { i = fn_end; continue; }

        std::unordered_map<uint32_t, std::vector<SUse>> uses;
        blk = 0;
        for (size_t j = i + 1; j < fn_end; ++j) {
            if (insts[j].op == kOpLabel) { ++blk; continue; }
            if (insts[j].op == kOpSampledImage) continue;
            uint32_t p = insts[j].pos, wc = insts[j].wc;
            for (uint32_t k = 1; k < wc; ++k) {
                uint32_t w = spv[p + k];
                if (defs.find(w) != defs.end()) {
                    uses[w].push_back({(uint32_t)j, k, blk});
                }
            }
        }

        for (const auto& kv : defs) {
            const SDef& d = kv.second;
            auto uit = uses.find(d.res_id);
            if (uit == uses.end() || uit->second.empty()) continue;
            bool cross_block = false;
            for (const auto& u : uit->second) {
                if (u.block_id != d.block_id) { cross_block = true; break; }
            }
            if (!cross_block) continue;

            modified = true;
            skip_inst[d.inst_idx] = 1;
            // Group uses by consumer instruction index so each consumer gets its OpSampledImage right before it
            std::unordered_map<uint32_t, uint32_t> inst_to_id;
            bool reused_orig = false;
            for (const auto& u : uit->second) {
                auto it_id = inst_to_id.find(u.inst_idx);
                uint32_t use_id = 0;
                if (it_id != inst_to_id.end()) {
                    use_id = it_id->second;
                } else {
                    if (!reused_orig) {
                        use_id = d.res_id;
                        reused_orig = true;
                    } else {
                        use_id = id_bound++;
                        cloned_ids.push_back({d.res_id, use_id});
                    }
                    inst_to_id[u.inst_idx] = use_id;
                    insert_before[u.inst_idx].push_back({(5u << 16) | kOpSampledImage, d.res_type, use_id, d.img_id, d.smp_id});
                }
                if (use_id != d.res_id) {
                    rewrite_word[((uint64_t)u.inst_idx << 32) | u.word_k] = use_id;
                }
            }
        }
        i = fn_end;
    }

    if (!modified) return;

    // Collect any OpDecorate instructions targeting cloned original IDs
    std::vector<std::vector<uint32_t>> extra_decs;
    if (!cloned_ids.empty()) {
        for (size_t i = 0; i < insts.size(); ++i) {
            if (insts[i].op == kOpDecorate && insts[i].wc >= 3) {
                uint32_t tgt = spv[insts[i].pos + 1];
                for (const auto& pr : cloned_ids) {
                    if (pr.first == tgt) {
                        std::vector<uint32_t> dec(spv.begin() + insts[i].pos, spv.begin() + insts[i].pos + insts[i].wc);
                        dec[1] = pr.second;
                        extra_decs.push_back(std::move(dec));
                    }
                }
            }
        }
    }

    // Emit cloned OpDecorate instructions inside the SPIR-V Annotations section (right before types/globals!)
    size_t insert_decs_before = insts.size();
    if (!extra_decs.empty()) {
        for (size_t i = 0; i < insts.size(); ++i) {
            uint32_t op = insts[i].op;
            if ((op >= 19 && op <= 39) || op == kOpFunction) { // first OpType* or OpFunction
                insert_decs_before = i;
                break;
            }
        }
    }

    std::vector<uint32_t> out;
    out.reserve(spv.size() + insert_before.size() * 5 + extra_decs.size() * 4);
    for (uint32_t h = 0; h < 5; ++h) out.push_back(spv[h]);
    out[3] = id_bound;
    bool emitted_extra_decs = extra_decs.empty();

    for (size_t i = 0; i < insts.size(); ++i) {
        if (!emitted_extra_decs && i == insert_decs_before) {
            for (const auto& dec : extra_decs) out.insert(out.end(), dec.begin(), dec.end());
            emitted_extra_decs = true;
        }
        if (skip_inst[i]) continue;
        auto iit = insert_before.find((uint32_t)i);
        if (iit != insert_before.end()) {
            for (const auto& ins5 : iit->second) out.insert(out.end(), ins5.begin(), ins5.end());
        }
        uint32_t p = insts[i].pos, wc = insts[i].wc;
        for (uint32_t k = 0; k < wc; ++k) {
            uint32_t w = spv[p + k];
            if (k >= 1) {
                auto rit = rewrite_word.find(((uint64_t)i << 32) | k);
                if (rit != rewrite_word.end()) w = rit->second;
            }
            out.push_back(w);
        }
    }
    spv.swap(out);
}

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
    snprintf(spv_path, sizeof(spv_path), "/tmp/smrast_v10c_cache_%s_%016llx.spv", tag, (unsigned long long)h);
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

    // 3. Fast in-process libshaderc compilation (zero fork/exec; auto-fallback to -O0 + sink + spirv-opt -O if
    //    glslang emitted a cross-block OpSampledImage before shaderc's internal optimizer could run!)
    thread_local ThreadLocalShaderc sc;
    char fname[64];
    snprintf(fname, sizeof(fname), "smrast_%s_%016llx.comp", tag, (unsigned long long)h);
    bool used_o0_fallback = false;
    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        sc.compiler, glsl.data(), glsl.size(), shaderc_compute_shader, fname, "main", sc.opts);

    if (res && shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        const char* msg = shaderc_result_get_error_message(res);
        if (msg && strstr(msg, "failed to optimize") != nullptr) {
            shaderc_result_release(res);
            used_o0_fallback = true;
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

    // Ensure every OpSampledImage resides in the exact same basic block as its consumer(s)
    sm_sink_op_sampled_image(out_spv);

    // For the ~1% of shaders that fell back to -O0 due to pre-sink cross-block OpSampledImage,
    // run spirv-opt -O AFTER sinking so all helper functions (sm_user_fs_main) are inlined and
    // SSA-rewritten (avoids NVIDIA 535 libnvidia-glvkspirv.so _nv002nvvm crash on un-inlined bindless -O0 SPIR-V!).
    if (used_o0_fallback) {
        uint64_t seq = g_tmp_seq.fetch_add(1);
        char in_p[128], out_p[128];
        snprintf(in_p, sizeof(in_p), "/tmp/smrast_sopt_%d_%llu_in.spv", (int)getpid(), (unsigned long long)seq);
        snprintf(out_p, sizeof(out_p), "/tmp/smrast_sopt_%d_%llu_out.spv", (int)getpid(), (unsigned long long)seq);
        {
            std::ofstream fin(in_p, std::ios::binary);
            if (fin.good()) fin.write((const char*)out_spv.data(), out_spv.size() * 4);
        }
        char arg0[] = "spirv-opt";
        char arg1[] = "-O";
        char arg2[] = "-o";
        char* argv[] = { arg0, arg1, in_p, arg2, out_p, nullptr };
        pid_t child = -1;
        if (posix_spawnp(&child, "spirv-opt", nullptr, nullptr, argv, environ) == 0 && child > 0) {
            int st = 0;
            while (waitpid(child, &st, 0) < 0 && errno == EINTR) {}
            if (WIFEXITED(st) && WEXITSTATUS(st) == 0) {
                std::ifstream fout(out_p, std::ios::binary | std::ios::ate);
                if (fout.good()) {
                    size_t osz = (size_t)fout.tellg();
                    if (osz >= 20 && (osz % 4) == 0) {
                        fout.seekg(0);
                        std::vector<uint32_t> opt_spv(osz / 4);
                        fout.read((char*)opt_spv.data(), osz);
                        if (fout.good() && opt_spv[0] == 0x07230203u) {
                            sm_sink_op_sampled_image(opt_spv);
                            out_spv.swap(opt_spv);
                        }
                    }
                }
            }
        }
        unlink(in_p);
        unlink(out_p);
    }

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
    bool is_flat = false;
    bool is_noperspective = false;
    std::string glsl_type;
};

static std::string glsl_type_str(spvc_basetype bt, uint32_t vecsize) {
    if (bt == SPVC_BASETYPE_INT32)  return (vecsize == 1) ? "int"  : ("ivec" + std::to_string(vecsize));
    if (bt == SPVC_BASETYPE_UINT32) return (vecsize == 1) ? "uint" : ("uvec" + std::to_string(vecsize));
    return (vecsize == 1) ? "float" : ("vec" + std::to_string(vecsize));
}

struct ReflectedBlockMember {
    spvc_basetype basetype = SPVC_BASETYPE_FP32;
    uint32_t vecsize = 1;
    uint32_t cols = 1;
    uint32_t array_size = 0; // 0 = non-array
    uint32_t offset = 0;
    uint32_t array_stride = 0;
    uint32_t matrix_stride = 0;
};

struct ReflectedBlock {
    bool valid = false;
    bool batchable = false;
    uint32_t set = 0;
    uint32_t binding = 0;
    uint32_t dyn_slot = 0xFFFFFFFFu;
    size_t byte_size = 0;
    std::vector<ReflectedBlockMember> members;
};

static bool decompile_spirv(const std::vector<uint32_t>& spv,
                            std::string& out_glsl,
                            std::vector<ReflectedIO>& inputs,
                            std::vector<ReflectedIO>& outputs,
                            std::string& err,
                            const std::vector<SmDynUboDesc>* dyn_ubos = nullptr,
                            ReflectedBlock* out_push = nullptr,
                            ReflectedBlock* out_dyn_ubo = nullptr) {
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
            io.is_flat = (spvc_compiler_has_decoration(comp, list[i].id, SpvDecorationFlat) == SPVC_TRUE);
            io.is_noperspective = (spvc_compiler_has_decoration(comp, list[i].id, SpvDecorationNoPerspective) == SPVC_TRUE);
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

    auto reflect_block = [&](const spvc_reflected_resource& r, size_t max_bytes, ReflectedBlock& out) {
        spvc_type bt = spvc_compiler_get_type_handle(comp, r.base_type_id);
        if (!bt) return;
        size_t sz = 0;
        if (spvc_compiler_get_declared_struct_size(comp, bt, &sz) != SPVC_SUCCESS) return;
        unsigned nm = spvc_type_get_num_member_types(bt);
        out.valid = (nm > 0);
        out.byte_size = sz;
        out.batchable = (nm > 0 && sz > 0 && sz <= max_bytes);
        out.members.clear();
        for (unsigned m = 0; m < nm && out.batchable; ++m) {
            spvc_type_id mtid = spvc_type_get_member_type(bt, m);
            spvc_type mt = spvc_compiler_get_type_handle(comp, mtid);
            if (!mt) { out.batchable = false; break; }
            ReflectedBlockMember bm;
            bm.basetype = spvc_type_get_basetype(mt);
            bm.vecsize  = spvc_type_get_vector_size(mt);
            bm.cols     = spvc_type_get_columns(mt);
            unsigned ndims = spvc_type_get_num_array_dimensions(mt);
            if ((bm.basetype != SPVC_BASETYPE_FP32 && bm.basetype != SPVC_BASETYPE_INT32 && bm.basetype != SPVC_BASETYPE_UINT32) ||
                bm.vecsize < 1 || bm.vecsize > 4 || bm.cols < 1 || bm.cols > 4 || ndims > 1) {
                out.batchable = false; break;
            }
            if (ndims == 1) {
                if (spvc_type_array_dimension_is_literal(mt, 0) != SPVC_TRUE) { out.batchable = false; break; }
                bm.array_size = spvc_type_get_array_dimension(mt, 0);
                if (bm.array_size == 0 || bm.array_size > 32) { out.batchable = false; break; }
                bm.array_stride = spvc_compiler_get_decoration(comp, mtid, SpvDecorationArrayStride);
                if (bm.array_stride == 0 || (bm.array_stride % 4) != 0) { out.batchable = false; break; }
            }
            bm.offset = spvc_compiler_get_member_decoration(comp, r.base_type_id, m, SpvDecorationOffset);
            if ((bm.offset % 4) != 0) { out.batchable = false; break; }
            if (spvc_compiler_has_member_decoration(comp, r.base_type_id, m, SpvDecorationRowMajor) == SPVC_TRUE) {
                out.batchable = false; break;
            }
            if (bm.cols > 1) {
                bm.matrix_stride = spvc_compiler_get_member_decoration(comp, r.base_type_id, m, SpvDecorationMatrixStride);
                if (bm.matrix_stride == 0 || (bm.matrix_stride % 4) != 0) { out.batchable = false; break; }
            }
            out.members.push_back(bm);
        }
    };

    if (out_push) {
        const spvc_reflected_resource* plist = nullptr;
        size_t pcnt = 0;
        spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_PUSH_CONSTANT, &plist, &pcnt);
        if (pcnt == 1) reflect_block(plist[0], 128, *out_push);
    }
    if (dyn_ubos && !dyn_ubos->empty() && out_dyn_ubo) {
        const spvc_reflected_resource* ulist = nullptr;
        size_t ucnt = 0;
        spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, &ulist, &ucnt);
        int match_idx = -1, match_cnt = 0;
        SmDynUboDesc matched_desc{};
        for (size_t i = 0; i < ucnt; ++i) {
            uint32_t uset = spvc_compiler_get_decoration(comp, ulist[i].id, SpvDecorationDescriptorSet);
            uint32_t ubind = spvc_compiler_get_decoration(comp, ulist[i].id, SpvDecorationBinding);
            for (const auto& du : *dyn_ubos) {
                if (du.set == uset && du.binding == ubind) {
                    match_idx = (int)i;
                    matched_desc = du;
                    match_cnt++;
                    break;
                }
            }
        }
        if (match_cnt == 1 && match_idx >= 0) {
            reflect_block(ulist[match_idx], 4096, *out_dyn_ubo);
            out_dyn_ubo->set = matched_desc.set;
            out_dyn_ubo->binding = matched_desc.binding;
            out_dyn_ubo->dyn_slot = matched_desc.dyn_slot;
        }
    }

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

static bool rewrite_block_to_ssbo_unpack(std::string& body,
                                         const ReflectedBlock& blk,
                                         bool is_push,
                                         const std::string& struct_name,
                                         const std::string& buf_var,
                                         const std::string& base_word_expr,
                                         std::string& out_init_code) {
    if (!blk.batchable || blk.members.empty()) return false;
    size_t main_pos = body.find("void sm_user_fs_main()");
    size_t search_end = (main_pos == std::string::npos) ? body.size() : main_pos;

    auto has_exact_kv = [](const std::string& hdr, const std::string& key, uint32_t val) -> bool {
        std::regex re("\\b" + key + "\\s*=\\s*" + std::to_string(val) + "\\b");
        return std::regex_search(hdr, re);
    };

    size_t pos = 0;
    while (pos < search_end) {
        size_t lp = body.find("layout(", pos);
        if (lp == std::string::npos || lp >= search_end) return false;
        size_t rp = body.find(')', lp + 7);
        if (rp == std::string::npos || rp >= search_end) return false;
        std::string hdr = body.substr(lp + 7, rp - (lp + 7));
        bool match = is_push
            ? (hdr.find("push_constant") != std::string::npos)
            : (has_exact_kv(hdr, "set", blk.set) && has_exact_kv(hdr, "binding", blk.binding));
        if (!match) { pos = rp + 1; continue; }

        // Confirm `uniform` follows layout(...) before `{`
        size_t lb = body.find('{', rp + 1);
        if (lb == std::string::npos || lb >= search_end) { pos = rp + 1; continue; }
        std::string between = body.substr(rp + 1, lb - (rp + 1));
        if (between.find("uniform") == std::string::npos || between.find(';') != std::string::npos) {
            pos = rp + 1; continue;
        }
        size_t rb = body.find('}', lb + 1);
        if (rb == std::string::npos || rb >= search_end) return false;
        std::string inner = body.substr(lb + 1, rb - (lb + 1));
        if (inner.find('{') != std::string::npos) return false; // nested block/struct -> safe fallback

        size_t semi = body.find(';', rb + 1);
        if (semi == std::string::npos || semi >= search_end) return false;
        std::string inst = body.substr(rb + 1, semi - (rb + 1));
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            size_t b = s.find_last_not_of(" \t\r\n");
            return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
        };
        inst = trim(inst);
        // Reject array of uniform blocks (e.g., `ubo[2]`)
        if (inst.find('[') != std::string::npos) return false;

        // Parse member statements inside `{ ... }` and strip `layout(...)` member qualifiers
        std::vector<std::string> member_names;
        std::ostringstream clean_inner;
        size_t mp = 0;
        while (mp < inner.size()) {
            size_t ms = inner.find(';', mp);
            if (ms == std::string::npos) break;
            std::string stmt = trim(inner.substr(mp, ms - mp));
            mp = ms + 1;
            if (stmt.empty()) continue;
            // Strip any layout(...) qualifiers on member
            while (true) {
                size_t mlp = stmt.find("layout(");
                if (mlp == std::string::npos) break;
                size_t mrp = stmt.find(')', mlp + 7);
                if (mrp == std::string::npos) return false;
                stmt.erase(mlp, mrp - mlp + 1);
                stmt = trim(stmt);
            }
            // Extract member identifier (token immediately before `[` or end of stmt)
            size_t br = stmt.find('[');
            std::string before_br = trim((br == std::string::npos) ? stmt : stmt.substr(0, br));
            size_t sp = before_br.find_last_of(" \t");
            if (sp == std::string::npos) return false;
            std::string mname = trim(before_br.substr(sp + 1));
            if (mname.empty()) return false;
            member_names.push_back(mname);
            clean_inner << "    " << stmt << ";\n";
        }
        if (member_names.size() != blk.members.size()) return false;

        // Replace `layout(...) uniform ... { ... } inst;` with local struct + global variable
        std::ostringstream repl;
        if (!inst.empty()) {
            repl << "struct " << struct_name << " {\n" << clean_inner.str() << "};\n"
                 << struct_name << " " << inst << ";";
        } else {
            repl << clean_inner.str();
        }
        body.replace(lp, semi - lp + 1, repl.str());

        // Emit exact word-level SSBO unpack code for main()
        auto emit_vec = [&](spvc_basetype bt, uint32_t vs, uint32_t woff) -> std::string {
            auto rd = [&](uint32_t w) {
                std::string raw = buf_var + "[_b + " + std::to_string(w) + "u]";
                if (bt == SPVC_BASETYPE_INT32)  return "int(" + raw + ")";
                if (bt == SPVC_BASETYPE_UINT32) return raw;
                return "uintBitsToFloat(" + raw + ")";
            };
            if (vs == 1) return rd(woff);
            std::string r = glsl_type_str(bt, vs) + "(";
            for (uint32_t c = 0; c < vs; ++c) { if (c) r += ", "; r += rd(woff + c); }
            return r + ")";
        };

        std::ostringstream init;
        init << "    {\n        uint _b = " << base_word_expr << ";\n";
        for (size_t m = 0; m < blk.members.size(); ++m) {
            const auto& bm = blk.members[m];
            std::string lhs = inst.empty() ? member_names[m] : (inst + "." + member_names[m]);
            uint32_t elems = (bm.array_size > 0) ? bm.array_size : 1u;
            for (uint32_t a = 0; a < elems; ++a) {
                uint32_t eo = bm.offset + a * bm.array_stride;
                std::string alhs = (bm.array_size > 0) ? (lhs + "[" + std::to_string(a) + "]") : lhs;
                if (bm.cols == 1) {
                    init << "        " << alhs << " = " << emit_vec(bm.basetype, bm.vecsize, eo >> 2) << ";\n";
                } else {
                    for (uint32_t c = 0; c < bm.cols; ++c) {
                        init << "        " << alhs << "[" << c << "] = "
                             << emit_vec(bm.basetype, bm.vecsize, (eo + c * bm.matrix_stride) >> 2) << ";\n";
                    }
                }
            }
        }
        init << "    }\n";
        out_init_code = init.str();
        return true;
    }
    return false;
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
            return samp.find("CubeArray") != std::string::npos || samp.find("CubeShadow") != std::string::npos ||
                   samp.find("2DArrayShadow") != std::string::npos || samp.find("1DArrayShadow") != std::string::npos;
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

static std::string strip_version_and_stage_io(const std::string& glsl, std::string& out_exts) {
    std::istringstream iss(glsl);
    std::ostringstream oss;
    std::ostringstream ext_ss;
    std::string line;
    std::regex re_ver(R"(^\s*#version\b.*)");
    std::regex re_in(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*in\s+(.+);)");
    std::regex re_out(R"(^\s*layout\s*\([^)]*location\s*=\s*\d+[^)]*\)\s*(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*out\s+(.+);)");
    std::regex re_qual_line(R"(^\s*(?:invariant|precise)\s+[A-Za-z_][A-Za-z0-9_]*\s*;)");
    std::regex re_fs_layout(R"(^\s*layout\s*\([^)]*(?:early_fragment_tests|origin_upper_left|pixel_center_integer|depth_any|depth_greater|depth_less|depth_unchanged)[^)]*\)\s*(?:in|out)\b.*;)");
    std::regex re_perv(R"(^\s*(?:in|out)\s+gl_PerVertex\b.*)");
    // Strip bare built-in stage I/O redeclarations (e.g. `out float gl_ClipDistance[8];`, `in float gl_CullDistance[4];`)
    std::regex re_builtin_io(R"(^\s*(?:layout\s*\([^)]*\)\s*)?(?:(?:flat|noperspective|smooth|centroid|sample|invariant|precise)\s+)*(?:in|out)\s+[^;{]*\bgl_[A-Za-z0-9_]+\b[^;{]*;)");
    std::regex re_ext_fs(R"(^\s*#extension\s+(?:GL_ARB_fragment_shader_interlock|GL_EXT_demote_to_helper_invocation|GL_ARB_derivative_control|GL_EXT_fragment_shader_barycentric|GL_NV_fragment_shader_barycentric|GL_EXT_post_depth_coverage)\b.*)");
    std::regex re_ext_any(R"(^\s*#extension\b.*)");
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
        if (std::regex_match(line, re_ext_any)) {
            ext_ss << line << "\n";
            continue;
        }
        if (std::regex_match(line, re_perv)) {
            if (line.find("};") == std::string::npos) skip_perv_block = true;
            continue;
        }
        if (std::regex_match(line, re_builtin_io)) continue;
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
    out_exts = ext_ss.str();
    return oss.str();
}

static std::string swiz(uint32_t vecsize) {
    if (vecsize == 1) return ".x";
    if (vecsize == 2) return ".xy";
    if (vecsize == 3) return ".xyz";
    return "";
}

static std::string emit_attr_unpack(const ReflectedIO& io, const VkVertexInputAttributeDescription* attr, uint32_t divisor) {
    std::ostringstream ss;
    uint32_t b = attr->binding;
    if (b >= SM_MAX_VTX_BINDINGS) b = 0;
    std::string vb_off = (b < 4) ? ("sm_vs.vtx_offsets[" + std::to_string(b) + "]")
                                 : ("sm_vs.vtx_offsets_hi[" + std::to_string(b - 4) + "]");
    std::string vb_str = (b < 4) ? ("sm_vs.vtx_strides[" + std::to_string(b) + "]")
                                 : ("sm_vs.vtx_strides_hi[" + std::to_string(b - 4) + "]");
    std::string inst_expr = (divisor == 0u) ? "sm_vs.instance_idx"
                          : ((divisor > 1u) ? ("(sm_vs.instance_idx + inst_local / " + std::to_string(divisor) + "u)")
                                            : "uint(sm_iidx)");
    ss << "    {\n";
    ss << "        uint elem_i = ((sm_vs.vtx_inst_rate_mask & (1u << " << b << "u)) != 0u) ? " << inst_expr << " : uint(sm_vidx);\n";
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
            ss << "        uint p16 = (" << load_u(0) << ") & 0xFFFFu;\n";
            assign_f4("vec4(unpackHalf2x16(p16).x, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_SFLOAT:
            assign_f4("vec4(unpackHalf2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            assign_f4("vec4(unpackHalf2x16(" + load_u(0) + "), unpackHalf2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R16_UNORM:
            ss << "        uint p16 = (" << load_u(0) << ") & 0xFFFFu;\n";
            assign_f4("vec4(float(p16) / 65535.0, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_UNORM:
            assign_f4("vec4(unpackUnorm2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_UNORM:
            assign_f4("vec4(unpackUnorm2x16(" + load_u(0) + "), unpackUnorm2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R16_SNORM:
            ss << "        int s16 = bitfieldExtract(int(" << load_u(0) << "), 0, 16);\n";
            assign_f4("vec4(max(float(s16) / 32767.0, -1.0), 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_SNORM:
            assign_f4("vec4(unpackSnorm2x16(" + load_u(0) + "), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_SNORM:
            assign_f4("vec4(unpackSnorm2x16(" + load_u(0) + "), unpackSnorm2x16(" + load_u(1) + "))"); break;
        case VK_FORMAT_R8_UNORM:
            assign_f4("vec4(float((" + load_u_raw(0) + " >> bsh) & 255u) / 255.0, 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R8_SNORM:
            ss << "        int s8 = bitfieldExtract(int(" << load_u_raw(0) << "), int(bsh), 8);\n";
            assign_f4("vec4(max(float(s8) / 127.0, -1.0), 0.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R8G8_UNORM:
            ss << "        uint p82 = (" << load_u(0) << ") & 0xFFFFu;\n";
            assign_f4("vec4(float(p82 & 255u) / 255.0, float((p82 >> 8u) & 255u) / 255.0, 0.0, 1.0)"); break;
        case VK_FORMAT_R8G8_SNORM:
            ss << "        int s82 = int(" << load_u(0) << ");\n";
            assign_f4("vec4(max(float(bitfieldExtract(s82,0,8))/127.0,-1.0), max(float(bitfieldExtract(s82,8,8))/127.0,-1.0), 0.0, 1.0)"); break;
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
        case VK_FORMAT_R8_SINT:
            assign_i4("ivec4(bitfieldExtract(int(" + load_u_raw(0) + "), int(bsh), 8), 0, 0, 1)"); break;
        case VK_FORMAT_R8G8_UINT:
            ss << "        uint pu8 = " << load_u(0) << ";\n";
            assign_u4("uvec4(pu8 & 255u, (pu8 >> 8u) & 255u, 0u, 1u)"); break;
        case VK_FORMAT_R8G8_SINT:
            ss << "        int pi82 = int(" << load_u(0) << ");\n";
            assign_i4("ivec4(bitfieldExtract(pi82,0,8), bitfieldExtract(pi82,8,8), 0, 1)"); break;
        case VK_FORMAT_R8G8B8A8_UINT:
            ss << "        uint pu8 = " << load_u(0) << ";\n";
            assign_u4("uvec4(pu8 & 255u, (pu8>>8u)&255u, (pu8>>16u)&255u, (pu8>>24u)&255u)"); break;
        case VK_FORMAT_R8G8B8A8_SINT:
            ss << "        int pi8 = int(" << load_u(0) << ");\n";
            assign_i4("ivec4(bitfieldExtract(pi8,0,8), bitfieldExtract(pi8,8,8), bitfieldExtract(pi8,16,8), bitfieldExtract(pi8,24,8))"); break;
        case VK_FORMAT_R16_UINT:
            assign_u4("uvec4((" + load_u(0) + ") & 65535u, 0u, 0u, 1u)"); break;
        case VK_FORMAT_R16_SINT:
            assign_i4("ivec4(bitfieldExtract(int(" + load_u(0) + "), 0, 16), 0, 0, 1)"); break;
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
        case VK_FORMAT_R8G8B8A8_USCALED:
            ss << "        uint pu8s = " << load_u(0) << ";\n";
            assign_f4("vec4(float(pu8s&255u), float((pu8s>>8u)&255u), float((pu8s>>16u)&255u), float((pu8s>>24u)&255u))"); break;
        case VK_FORMAT_R8G8B8A8_SSCALED:
            ss << "        int pi8s = int(" << load_u(0) << ");\n";
            assign_f4("vec4(float(bitfieldExtract(pi8s,0,8)), float(bitfieldExtract(pi8s,8,8)), float(bitfieldExtract(pi8s,16,8)), float(bitfieldExtract(pi8s,24,8)))"); break;
        case VK_FORMAT_R16G16_USCALED:
            ss << "        uint pu16s = " << load_u(0) << ";\n";
            assign_f4("vec4(float(pu16s&65535u), float(pu16s>>16u), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16_SSCALED:
            ss << "        int pi16s = int(" << load_u(0) << ");\n";
            assign_f4("vec4(float(bitfieldExtract(pi16s,0,16)), float(bitfieldExtract(pi16s,16,16)), 0.0, 1.0)"); break;
        case VK_FORMAT_R16G16B16A16_USCALED:
            ss << "        uint pu16sa = " << load_u(0) << ", pu16sb = " << load_u(1) << ";\n";
            assign_f4("vec4(float(pu16sa&65535u), float(pu16sa>>16u), float(pu16sb&65535u), float(pu16sb>>16u))"); break;
        case VK_FORMAT_R16G16B16A16_SSCALED:
            ss << "        int pi16sa = int(" << load_u(0) << "), pi16sb = int(" << load_u(1) << ");\n";
            assign_f4("vec4(float(bitfieldExtract(pi16sa,0,16)), float(bitfieldExtract(pi16sa,16,16)), float(bitfieldExtract(pi16sb,0,16)), float(bitfieldExtract(pi16sb,16,16)))"); break;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            ss << "        uint p10r = " << load_u(0) << ";\n";
            assign_f4("vec4(float((p10r>>20u)&1023u)/1023.0, float((p10r>>10u)&1023u)/1023.0, float(p10r&1023u)/1023.0, float((p10r>>30u)&3u)/3.0)"); break;
        case VK_FORMAT_A2R10G10B10_SNORM_PACK32:
            ss << "        int p10rs = int(" << load_u(0) << ");\n";
            assign_f4("vec4(max(float(bitfieldExtract(p10rs,20,10))/511.0,-1.0), max(float(bitfieldExtract(p10rs,10,10))/511.0,-1.0), max(float(bitfieldExtract(p10rs,0,10))/511.0,-1.0), max(float(bitfieldExtract(p10rs,30,2))/1.0,-1.0))"); break;
        default:
            fprintf(stderr, "[SmRast] WARN: unhandled vertex format=%d (binding=%u off=%u loc=%u)\n",
                    (int)attr->format, attr->binding, attr->offset, io.location);
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
                           std::string& err,
                           bool* out_writes_point_size,
                           uint32_t* out_stride_vec4) {
    SmCancelDisableGuard _cancel_guard;
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err)) return false;

    bool writes_ps = (glsl.find("gl_PointSize") != std::string::npos);
    if (out_writes_point_size) *out_writes_point_size = writes_ps;
    if (out_stride_vec4) {
        if (writes_ps) {
            *out_stride_vec4 = SM_VERT_STRIDE_VEC4;
        } else {
            uint32_t max_slot = 1u; // slot 0 = gl_Position
            for (const auto& io : outputs) {
                uint32_t elems = (io.array_size > 0) ? io.array_size : 1u;
                for (uint32_t a = 0; a < elems; ++a) {
                    uint32_t loc = io.location + a;
                    if (loc < SM_MAX_VARYINGS) max_slot = std::max(max_slot, 2u + loc);
                }
            }
            *out_stride_vec4 = std::min<uint32_t>(SM_VERT_STRIDE_VEC4, std::max<uint32_t>(2u, max_slot));
        }
    }

    std::string user_exts;
    std::string body = strip_version_and_stage_io(glsl, user_exts);
    body = replace_ident(body, "gl_Position", "sm_gl_Position");
    body = replace_ident(body, "gl_PointSize", "sm_gl_PointSize");
    body = replace_ident(body, "gl_ClipDistance", "sm_gl_ClipDistance");
    body = replace_ident(body, "gl_CullDistance", "sm_gl_CullDistance");
    body = replace_ident(body, "gl_Layer", "sm_gl_Layer");
    body = replace_ident(body, "gl_ViewportIndex", "sm_gl_ViewportIndex");
    body = replace_ident(body, "gl_VertexIndex", "sm_vidx");
    body = replace_ident(body, "gl_VertexID", "sm_vidx");
    body = replace_ident(body, "gl_InstanceIndex", "sm_iidx");
    body = replace_ident(body, "gl_InstanceID", "sm_iidx");
    body = replace_ident(body, "gl_BaseVertex", "sm_base_vertex");
    body = replace_ident(body, "gl_BaseVertexARB", "sm_base_vertex");
    body = replace_ident(body, "gl_BaseInstance", "sm_base_instance");
    body = replace_ident(body, "gl_BaseInstanceARB", "sm_base_instance");
    body = replace_ident(body, "gl_DrawID", "0");
    body = replace_ident(body, "gl_DrawIDARB", "0");
    body = replace_ident(body, "gl_ViewIndex", "0");
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_vs_main()");

    std::ostringstream ss;
    ss << "#version 460\n";
    if (!user_exts.empty()) ss << user_exts;
    ss << "layout(local_size_x = 64) in;\n\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0) writeonly buffer SmClipVertBuf { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1) readonly buffer SmIdxBuf { uint sm_idx_raw[]; };\n";
    for (uint32_t b = 0; b < SM_MAX_VTX_BINDINGS; ++b) {
        ss << "layout(std430, set = " << sm_set_idx << ", binding = " << (2u + b) << ") readonly buffer SmVB" << b << " { uint sm_vb" << b << "[]; };\n";
    }
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 10) readonly buffer SmVsParams {\n"
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
       << "    uint prim_restart_en;\n"
       << "    uint clip_vert_base;\n"
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
    ss << "    uint base_slot = (sm_vs.clip_vert_base + gid) * sm_vs.vert_stride_vec4;\n";
    ss << "    if (sm_vs.indexed == 0u) {\n";
    ss << "        sm_vidx = int(sm_vs.first_vertex + vert_local);\n";
    ss << "    } else if (sm_vs.indexed == 1u) {\n";
    ss << "        uint ii = (sm_vs.idx_buf_offset >> 1u) + sm_vs.first_index + vert_local;\n";
    ss << "        uint w = sm_idx_raw[ii >> 1u];\n";
    ss << "        uint idx16 = ((ii & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);\n";
    ss << "        if (sm_vs.prim_restart_en != 0u && idx16 == 0xFFFFu) { sm_clip_verts[base_slot] = vec4(0.0, 0.0, 0.0, uintBitsToFloat(0x7FC0DEADu)); return; }\n";
    ss << "        sm_vidx = int(idx16) + sm_vs.vertex_offset;\n";
    ss << "    } else if (sm_vs.indexed == 3u) {\n";
    ss << "        uint ii = sm_vs.idx_buf_offset + sm_vs.first_index + vert_local;\n";
    ss << "        uint w = sm_idx_raw[ii >> 2u];\n";
    ss << "        uint idx8 = (w >> ((ii & 3u) << 3u)) & 0xFFu;\n";
    ss << "        if (sm_vs.prim_restart_en != 0u && idx8 == 0xFFu) { sm_clip_verts[base_slot] = vec4(0.0, 0.0, 0.0, uintBitsToFloat(0x7FC0DEADu)); return; }\n";
    ss << "        sm_vidx = int(idx8) + sm_vs.vertex_offset;\n";
    ss << "    } else {\n";
    ss << "        uint ii = (sm_vs.idx_buf_offset >> 2u) + sm_vs.first_index + vert_local;\n";
    ss << "        uint idx32 = sm_idx_raw[ii];\n";
    ss << "        if (sm_vs.prim_restart_en != 0u && idx32 == 0xFFFFFFFFu) { sm_clip_verts[base_slot] = vec4(0.0, 0.0, 0.0, uintBitsToFloat(0x7FC0DEADu)); return; }\n";
    ss << "        sm_vidx = int(idx32) + sm_vs.vertex_offset;\n";
    ss << "    }\n";

    for (const auto& io : inputs) {
        if (io.array_size > 0) continue;
        const VkVertexInputAttributeDescription* found = nullptr;
        for (const auto& a : vtx_in.attrs) {
            if (a.location == io.location) { found = &a; break; }
        }
        if (found) {
            uint32_t div = (found->binding < SM_MAX_VTX_BINDINGS) ? vtx_in.divisors[found->binding] : 1u;
            ss << emit_attr_unpack(io, found, div);
        }
    }

    ss << "    sm_user_vs_main();\n";
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
    if (writes_ps) {
        ss << "    sm_clip_verts[base_slot + 15u].w = clamp(sm_gl_PointSize, 1.0, 64.0);\n";
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "vs", out_comp_spv, err);
}

bool sm_lower_fragment_spirv(const std::vector<uint32_t>& in_spv,
                             const VkSpecializationInfo*,
                             uint32_t sm_set_idx,
                             const SmFsCompileHints& hints,
                             std::vector<uint32_t>& out_comp_spv,
                             std::string& err,
                             bool* out_fs_uses_push,
                             bool* out_fs_has_discard_or_depth,
                             const std::vector<SmDynUboDesc>* dyn_ubos,
                             bool* out_fs_batched_push_ok,
                             SmDynUboDesc* out_batched_dyn_ubo,
                             uint32_t* out_stride_vec4) {
    SmCancelDisableGuard _cancel_guard;
    std::string glsl;
    std::vector<ReflectedIO> inputs, outputs;
    ReflectedBlock push_blk, dyn_ubo_blk;
    if (!decompile_spirv(in_spv, glsl, inputs, outputs, err, dyn_ubos, &push_blk, &dyn_ubo_blk)) return false;

    if (out_stride_vec4) {
        uint32_t max_slot = 1u;
        for (const auto& io : inputs) {
            uint32_t elems = (io.array_size > 0) ? io.array_size : 1u;
            for (uint32_t a = 0; a < elems; ++a) {
                uint32_t loc = io.location + a;
                if (loc < SM_MAX_VARYINGS) max_slot = std::max(max_slot, 2u + loc);
            }
        }
        *out_stride_vec4 = std::min<uint32_t>(SM_VERT_STRIDE_VEC4, std::max<uint32_t>(2u, max_slot));
    }

    if (out_fs_uses_push) *out_fs_uses_push = (glsl.find("push_constant") != std::string::npos);
    if (out_fs_batched_push_ok) *out_fs_batched_push_ok = false;
    if (out_batched_dyn_ubo) out_batched_dyn_ubo->dyn_slot = 0xFFFFFFFFu;
    bool uses_frag_depth = (glsl.find("gl_FragDepth") != std::string::npos);

    std::string user_exts;
    std::string body = strip_version_and_stage_io(glsl, user_exts);
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
    body = replace_ident(body, "gl_ClipDistance", "sm_gl_ClipDistance");
    body = replace_ident(body, "gl_CullDistance", "sm_gl_CullDistance");
    body = replace_ident(body, "gl_PointCoord", "vec2(0.5)");
    body = replace_ident(body, "gl_ViewIndex", "0");
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
    if (out_fs_has_discard_or_depth)
        *out_fs_has_discard_or_depth = (uses_frag_depth || body.find("sm_discarded = true;") != std::string::npos);
    body = std::regex_replace(body, std::regex(R"(\bvoid\s+main\s*\(\s*\))"), "void sm_user_fs_main()");

    // v0.8 Per-Draw Push-Constant & Dynamic-UBO SSBO Cross-Material Uber-Batching Lowering!
    std::string push_init_code, dyn_ubo_init_code;
    bool batched_push = rewrite_block_to_ssbo_unpack(
        body, push_blk, true, "SmBatchPushStruct", "sm_bp_raw",
        "(sm_fs.batch_push_base + sm_draw_id) * 32u", push_init_code);
    bool batched_dyn_ubo = rewrite_block_to_ssbo_unpack(
        body, dyn_ubo_blk, false, "SmBatchDynUboStruct", "sm_dyn_raw",
        "sm_dyn_off[sm_fs.batch_push_base + sm_draw_id] >> 2u", dyn_ubo_init_code);
    if (out_fs_batched_push_ok) *out_fs_batched_push_ok = batched_push;
    if (batched_dyn_ubo && out_batched_dyn_ubo) {
        out_batched_dyn_ubo->set      = dyn_ubo_blk.set;
        out_batched_dyn_ubo->binding  = dyn_ubo_blk.binding;
        out_batched_dyn_ubo->dyn_slot = dyn_ubo_blk.dyn_slot;
    }

    std::ostringstream ss;
    ss << "#version 460\n";
    ss << "#extension GL_ARB_gpu_shader_int64 : require\n";
    ss << "#extension GL_KHR_shader_subgroup_quad : require\n";
    ss << "#extension GL_KHR_shader_subgroup_vote : require\n";
    ss << "#extension GL_EXT_texture_shadow_lod : enable\n";
    if (!user_exts.empty()) ss << user_exts;
    ss << "layout(local_size_x = 256) in;\n\n";

    ss << "struct SmTriSetup {\n"
       << "    vec4 v0; vec4 v1; vec4 v2; vec4 area; uvec4 idx;\n"
       << "};\n";

    // SmRast v0.6 Full 8-MRT + Depth + 8-bit Stencil + Active-Tile + Self-Clearing VisBuf FS Internal Set!
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 0)  readonly buffer SmClipVertBuf    { vec4 sm_clip_verts[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 1)  readonly buffer SmTriSetupBuf    { SmTriSetup sm_tris[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 2)  buffer SmVisBuf                  { uint64_t sm_vis_px[]; };\n";
    for (uint32_t m = 0; m < SM_MAX_MRT; ++m) {
        ss << "layout(std430, set = " << sm_set_idx << ", binding = " << (3u + m) << ") buffer SmColor" << m << "Buf { uint sm_col" << m << "[]; };\n";
    }
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 11) buffer SmDepthBuf                { float sm_depth_px[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 12) buffer SmStencilBuf              { uint sm_stencil_px[]; };\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 13) readonly buffer SmFsParams {\n"
       << "    uint width;\n"
       << "    uint height;\n"
       << "    uint vert_stride_vec4;\n"
       << "    uint depth_write;\n"
       << "    uint rt_active_mask;\n"
       << "    uint stencil_enable;\n"
       << "    uint stencil_front_cfg;\n"
       << "    uint stencil_back_cfg;\n"
       << "    uint stencil_refs;\n"
       << "    uint tiles_x; uint direct_2d; uint batch_push_base;\n"
       << "    vec4 blend_const;\n"
       << "    uvec4 rt_fmt[2];\n"
       << "    uvec4 rt_blend_en[2];\n"
       << "    uvec4 rt_src_blend[2];\n"
       << "    uvec4 rt_dst_blend[2];\n"
       << "    uvec4 rt_blend_op[2];\n"
       << "    uvec4 rt_src_alpha[2];\n"
       << "    uvec4 rt_dst_alpha[2];\n"
       << "    uvec4 rt_alpha_op[2];\n"
       << "    uvec4 rt_write_mask[2];\n"
       << "    uvec4 rt_stride[2];\n"
       << "    vec4  quad_vp;\n"
       << "    ivec4 quad_sc;\n"
       << "    vec2  quad_z;\n"
       << "    uint  quad_meta;\n"
       << "    uint  quad_pad;\n"
       << "} sm_fs;\n";
    ss << "shared SmTriSetup sm_fq_tris[16];\n";
    ss << "shared vec3 sm_fq_u0[16];\n";
    ss << "shared vec3 sm_fq_u1[16];\n";
    ss << "shared vec3 sm_fq_u2[16];\n";
    ss << "shared uint sm_fq_cnt;\n";
    ss << "shared uint sm_fq_clipped;\n";
    ss << "layout(std430, set = " << sm_set_idx << ", binding = 14) readonly buffer SmActiveTilesBuf { uint sm_active_tiles[]; };\n";
    if (batched_push)
        ss << "layout(std430, set = " << sm_set_idx << ", binding = 15) readonly buffer SmBatchPushBuf   { uint sm_bp_raw[]; };\n";
    if (batched_dyn_ubo) {
        ss << "layout(std430, set = " << sm_set_idx << ", binding = 16) readonly buffer SmBatchDynUboBuf { uint sm_dyn_raw[]; };\n";
        ss << "layout(std430, set = " << sm_set_idx << ", binding = 17) readonly buffer SmBatchDynOffBuf { uint sm_dyn_off[]; };\n";
    }
    ss << "\n";

    ss << "uint sm_u4(uvec4 a[2], uint i) { return a[i >> 2u][i & 3u]; }\n";
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
    ss << "float sm_gl_ClipDistance[8];\n";
    ss << "float sm_gl_CullDistance[8];\n";
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

    // Determine whether we need heavy dynamic format/blend/stencil helpers (zero register bloat for opaque RGBA8/BGRA8!)
    bool need_any_blend = false;
    bool need_any_rmw = false;
    bool need_dyn_fmt = false;
    bool need_srgb = false;
    bool need_r11g11b10 = false;
    for (const auto& io : outputs) {
        if (io.array_size == 0 && io.location < SM_MAX_MRT) {
            if (hints.blend_en[io.location]) { need_any_blend = true; need_any_rmw = true; }
            if ((hints.write_mask[io.location] & 0xFu) != 0xFu) need_any_rmw = true;
            int fc = hints.rt_fmt[io.location];
            if (fc < 0) { need_dyn_fmt = true; need_any_rmw = true; need_srgb = true; need_r11g11b10 = true; }
            else {
                if (fc == 2 || fc == 3) need_srgb = true;
                if (fc == 5) need_r11g11b10 = true;
            }
        }
    }

    ss << "vec4 sm_safe_f16(vec4 v) { return clamp(mix(v, vec4(0.0), isnan(v)), vec4(-65504.0), vec4(65504.0)); }\n";
    if (need_srgb) {
        ss << "float sm_lin_to_srgb(float x) { return (x <= 0.0031308) ? (x * 12.92) : (1.055 * pow(max(x, 0.0), 1.0/2.4) - 0.055); }\n";
        ss << "float sm_srgb_to_lin(float x) { return (x <= 0.04045) ? (x / 12.92) : pow((max(x, 0.0) + 0.055) / 1.055, 2.4); }\n";
    }
    if (need_r11g11b10) {
        ss << "uint sm_f32_to_uf11(float v) { if (isnan(v) || v <= 0.0) return 0u; uint u = floatBitsToUint(v); uint e = (u >> 23u) & 255u; if (e < 112u) return 0u; if (e >= 143u) return 0x7BFu; return ((e - 112u) << 6u) | ((u >> 17u) & 63u); }\n";
        ss << "uint sm_f32_to_uf10(float v) { if (isnan(v) || v <= 0.0) return 0u; uint u = floatBitsToUint(v); uint e = (u >> 23u) & 255u; if (e < 112u) return 0u; if (e >= 143u) return 0x3DFu; return ((e - 112u) << 5u) | ((u >> 18u) & 31u); }\n";
        ss << "float sm_uf11_to_f32(uint v) { uint e = (v >> 6u) & 31u; uint m = v & 63u; if (e == 0u) return float(m) * exp2(-20.0); if (e == 31u) return 65024.0; return uintBitsToFloat(((e + 112u) << 23u) | (m << 17u)); }\n";
        ss << "float sm_uf10_to_f32(uint v) { uint e = (v >> 5u) & 31u; uint m = v & 31u; if (e == 0u) return float(m) * exp2(-19.0); if (e == 31u) return 64512.0; return uintBitsToFloat(((e + 112u) << 23u) | (m << 18u)); }\n";
    }
    if (need_dyn_fmt) {
        ss << "uint sm_pack_u32_fmt(vec4 c, uint fc) {\n"
           << "    if (fc == 5u)  return sm_f32_to_uf11(c.r) | (sm_f32_to_uf11(c.g) << 11u) | (sm_f32_to_uf10(c.b) << 22u);\n"
           << "    if (fc == 6u)  return floatBitsToUint(isnan(c.r) ? 0.0 : c.r);\n"
           << "    if (fc == 7u)  return packHalf2x16(sm_safe_f16(c).rg);\n"
           << "    if (fc == 9u)  { vec4 cl = clamp(c, 0.0, 1.0); return uint(round(cl.r*1023.0)) | (uint(round(cl.g*1023.0))<<10u) | (uint(round(cl.b*1023.0))<<20u) | (uint(round(cl.a*3.0))<<30u); }\n"
           << "    if (fc == 11u) return uint(c.r);\n"
           << "    if (fc == 12u) return packSnorm4x8(clamp(c, -1.0, 1.0));\n"
           << "    if (fc == 13u) return packUnorm2x16(clamp(c.rg, 0.0, 1.0));\n"
           << "    if (fc == 14u) return packSnorm2x16(clamp(c.rg, -1.0, 1.0));\n"
           << "    c = clamp(c, 0.0, 1.0);\n"
           << "    if (fc == 2u || fc == 3u) c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b));\n"
           << "    if (fc == 1u || fc == 3u) c = c.bgra;\n"
           << "    return packUnorm4x8(c);\n"
           << "}\n";
    }
    if (need_any_rmw && need_dyn_fmt) {
        ss << "vec4 sm_unpack_u32_fmt(uint pk, uint fc) {\n"
           << "    if (fc == 5u)  return vec4(sm_uf11_to_f32(pk & 0x7FFu), sm_uf11_to_f32((pk>>11u) & 0x7FFu), sm_uf10_to_f32((pk>>22u) & 0x3FFu), 1.0);\n"
           << "    if (fc == 6u)  return vec4(uintBitsToFloat(pk), 0.0, 0.0, 1.0);\n"
           << "    if (fc == 7u)  return vec4(unpackHalf2x16(pk), 0.0, 1.0);\n"
           << "    if (fc == 9u)  return vec4(float(pk&1023u)/1023.0, float((pk>>10u)&1023u)/1023.0, float((pk>>20u)&1023u)/1023.0, float((pk>>30u)&3u)/3.0);\n"
           << "    if (fc == 12u) return unpackSnorm4x8(pk);\n"
           << "    if (fc == 13u) return vec4(unpackUnorm2x16(pk), 0.0, 1.0);\n"
           << "    if (fc == 14u) return vec4(unpackSnorm2x16(pk), 0.0, 1.0);\n"
           << "    vec4 c = unpackUnorm4x8(pk);\n"
           << "    if (fc == 1u || fc == 3u) c = c.bgra;\n"
           << "    if (fc == 2u || fc == 3u) c.rgb = vec3(sm_srgb_to_lin(c.r), sm_srgb_to_lin(c.g), sm_srgb_to_lin(c.b));\n"
           << "    return c;\n"
           << "}\n";
    }
    if (need_any_blend) {
        ss << "vec4 sm_blend_factor(uint f, vec4 src, vec4 dst) {\n"
           << "    if (f == 0u)  return vec4(0.0);\n"
           << "    if (f == 1u)  return vec4(1.0);\n"
           << "    if (f == 2u)  return src;\n"
           << "    if (f == 3u)  return vec4(1.0) - src;\n"
           << "    if (f == 4u)  return dst;\n"
           << "    if (f == 5u)  return vec4(1.0) - dst;\n"
           << "    if (f == 6u)  return vec4(src.a);\n"
           << "    if (f == 7u)  return vec4(1.0 - src.a);\n"
           << "    if (f == 8u)  return vec4(dst.a);\n"
           << "    if (f == 9u)  return vec4(1.0 - dst.a);\n"
           << "    if (f == 10u) return sm_fs.blend_const;\n"
           << "    if (f == 11u) return vec4(1.0) - sm_fs.blend_const;\n"
           << "    if (f == 12u) return vec4(sm_fs.blend_const.a);\n"
           << "    if (f == 13u) return vec4(1.0 - sm_fs.blend_const.a);\n"
           << "    if (f == 14u) return vec4(vec3(min(src.a, 1.0 - dst.a)), 1.0);\n"
           << "    return vec4(1.0);\n"
           << "}\n"
           << "vec3 sm_blend_combine3(uint op, vec3 rs, vec3 rd, vec3 sf, vec3 df) {\n"
           << "    if (op == 1u) return rs * sf - rd * df;\n"
           << "    if (op == 2u) return rd * df - rs * sf;\n"
           << "    if (op == 3u) return min(rs, rd);\n"
           << "    if (op == 4u) return max(rs, rd);\n"
           << "    return rs * sf + rd * df;\n"
           << "}\n"
           << "float sm_blend_combine1(uint op, float rs, float rd, float sf, float df) {\n"
           << "    if (op == 1u) return rs * sf - rd * df;\n"
           << "    if (op == 2u) return rd * df - rs * sf;\n"
           << "    if (op == 3u) return min(rs, rd);\n"
           << "    if (op == 4u) return max(rs, rd);\n"
           << "    return rs * sf + rd * df;\n"
           << "}\n\n";
    }
    if (hints.stencil_en) {
        ss << "bool sm_stencil_cmp(uint op, uint ref_m, uint val_m) {\n"
           << "    if (op == 0u) return false;\n"
           << "    if (op == 1u) return ref_m <  val_m;\n"
           << "    if (op == 2u) return ref_m == val_m;\n"
           << "    if (op == 3u) return ref_m <= val_m;\n"
           << "    if (op == 4u) return ref_m >  val_m;\n"
           << "    if (op == 5u) return ref_m != val_m;\n"
           << "    if (op == 6u) return ref_m >= val_m;\n"
           << "    return true;\n"
           << "}\n"
           << "uint sm_stencil_apply(uint op, uint cur, uint ref) {\n"
           << "    if (op == 1u) return 0u;\n"
           << "    if (op == 2u) return ref;\n"
           << "    if (op == 3u) return min(cur + 1u, 255u);\n"
           << "    if (op == 4u) return (cur > 0u) ? (cur - 1u) : 0u;\n"
           << "    if (op == 5u) return (~cur) & 0xFFu;\n"
           << "    if (op == 6u) return (cur + 1u) & 0xFFu;\n"
           << "    if (op == 7u) return (cur - 1u) & 0xFFu;\n"
           << "    return cur;\n"
           << "}\n\n";
    }

    ss << body << "\n\n";

    // Per-RT write helper: compile-time specialized when format/blend/writeMask are known (max Hopper SM occupancy!)
    auto emit_rt_store = [&](int rt_i, const ReflectedIO& io) {
        std::string buf = "sm_col" + std::to_string(rt_i);
        std::string rts = std::to_string(rt_i) + "u";
        int kfc = hints.rt_fmt[rt_i];
        bool ben = hints.blend_en[rt_i];
        uint32_t kwm = hints.write_mask[rt_i] & 0xFu;
        if (kfc >= 0 && kwm == 0u) return;

        ss << "    if ((sm_fs.rt_active_mask & (1u << " << rts << ")) != 0u) {\n";
        ss << "        uint rpidx = p.y * sm_u4(sm_fs.rt_stride, " << rts << ") + p.x;\n";
        if (kfc == 11 || io.basetype == SPVC_BASETYPE_UINT32 || io.basetype == SPVC_BASETYPE_INT32) {
            std::string first_comp = (io.vecsize == 1) ? io.name : (io.name + ".x");
            if (kfc == 17) {
                ss << "        { uint sh = (rpidx & 3u) << 3u; uint v8 = uint(" << first_comp << ") & 0xFFu; "
                   << "atomicAnd(" << buf << "[rpidx >> 2u], ~(0xFFu << sh)); atomicOr(" << buf << "[rpidx >> 2u], v8 << sh); }\n";
            } else if (kfc == 20) {
                ss << "        { uint sh = (rpidx & 1u) << 4u; uint v16 = uint(" << first_comp << ") & 0xFFFFu; "
                   << "atomicAnd(" << buf << "[rpidx >> 1u], ~(0xFFFFu << sh)); atomicOr(" << buf << "[rpidx >> 1u], v16 << sh); }\n";
            } else {
                ss << "        " << buf << "[rpidx] = uint(" << first_comp << ");\n";
            }
            ss << "    }\n";
            return;
        }
        ss << "        vec4 out_c = ";
        if (io.vecsize == 4) ss << "vec4(" << io.name << ");\n";
        else if (io.vecsize == 3) ss << "vec4(" << io.name << ", 1.0);\n";
        else if (io.vecsize == 2) ss << "vec4(" << io.name << ", 0.0, 1.0);\n";
        else ss << "vec4(float(" << io.name << "), 0.0, 0.0, 1.0);\n";

        if (kfc >= 0) {
            bool need_wm = (kwm != 0xFu);
            bool need_rmw = ben || need_wm;
            if (need_rmw) {
                ss << "        vec4 dst_c;\n";
                if (kfc == 4)       ss << "        dst_c = vec4(unpackHalf2x16(" << buf << "[rpidx*2u]), unpackHalf2x16(" << buf << "[rpidx*2u+1u]));\n";
                else if (kfc == 10) ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[rpidx*2u]), uintBitsToFloat(" << buf << "[rpidx*2u+1u]), 0.0, 1.0);\n";
                else if (kfc == 15) ss << "        dst_c = vec4(unpackUnorm2x16(" << buf << "[rpidx*2u]), unpackUnorm2x16(" << buf << "[rpidx*2u+1u]));\n";
                else if (kfc == 16) ss << "        dst_c = vec4(unpackSnorm2x16(" << buf << "[rpidx*2u]), unpackSnorm2x16(" << buf << "[rpidx*2u+1u]));\n";
                else if (kfc == 8)  ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[rpidx*4u]), uintBitsToFloat(" << buf << "[rpidx*4u+1u]), uintBitsToFloat(" << buf << "[rpidx*4u+2u]), uintBitsToFloat(" << buf << "[rpidx*4u+3u]));\n";
                else if (kfc == 5)  ss << "        { uint pk = " << buf << "[rpidx]; dst_c = vec4(sm_uf11_to_f32(pk&0x7FFu), sm_uf11_to_f32((pk>>11u)&0x7FFu), sm_uf10_to_f32((pk>>22u)&0x3FFu), 1.0); }\n";
                else if (kfc == 6)  ss << "        dst_c = vec4(uintBitsToFloat(" << buf << "[rpidx]), 0.0, 0.0, 1.0);\n";
                else if (kfc == 7)  ss << "        dst_c = vec4(unpackHalf2x16(" << buf << "[rpidx]), 0.0, 1.0);\n";
                else if (kfc == 9)  ss << "        { uint pk = " << buf << "[rpidx]; dst_c = vec4(float(pk&1023u)/1023.0, float((pk>>10u)&1023u)/1023.0, float((pk>>20u)&1023u)/1023.0, float((pk>>30u)&3u)/3.0); }\n";
                else if (kfc == 12) ss << "        dst_c = unpackSnorm4x8(" << buf << "[rpidx]);\n";
                else if (kfc == 13) ss << "        dst_c = vec4(unpackUnorm2x16(" << buf << "[rpidx]), 0.0, 1.0);\n";
                else if (kfc == 14) ss << "        dst_c = vec4(unpackSnorm2x16(" << buf << "[rpidx]), 0.0, 1.0);\n";
                else if (kfc == 17) ss << "        { uint b8 = (" << buf << "[rpidx >> 2u] >> ((rpidx & 3u) << 3u)) & 0xFFu; dst_c = vec4(float(b8)/255.0, 0.0, 0.0, 1.0); }\n";
                else if (kfc == 18) ss << "        { uint w16 = (" << buf << "[rpidx >> 1u] >> ((rpidx & 1u) << 4u)) & 0xFFFFu; dst_c = vec4(float(w16&255u)/255.0, float((w16>>8u)&255u)/255.0, 0.0, 1.0); }\n";
                else if (kfc == 19) ss << "        { uint w16 = (" << buf << "[rpidx >> 1u] >> ((rpidx & 1u) << 4u)) & 0xFFFFu; dst_c = vec4(unpackHalf2x16(w16).x, 0.0, 0.0, 1.0); }\n";
                else if (kfc == 20) ss << "        { uint w16 = (" << buf << "[rpidx >> 1u] >> ((rpidx & 1u) << 4u)) & 0xFFFFu; dst_c = vec4(float(w16)/65535.0, 0.0, 0.0, 1.0); }\n";
                else if (kfc == 1) ss << "        dst_c = unpackUnorm4x8(" << buf << "[rpidx]).bgra;\n";
                else if (kfc == 2) ss << "        { dst_c = unpackUnorm4x8(" << buf << "[rpidx]); dst_c.rgb = vec3(sm_srgb_to_lin(dst_c.r), sm_srgb_to_lin(dst_c.g), sm_srgb_to_lin(dst_c.b)); }\n";
                else if (kfc == 3) ss << "        { dst_c = unpackUnorm4x8(" << buf << "[rpidx]).bgra; dst_c.rgb = vec3(sm_srgb_to_lin(dst_c.r), sm_srgb_to_lin(dst_c.g), sm_srgb_to_lin(dst_c.b)); }\n";
                else                ss << "        dst_c = unpackUnorm4x8(" << buf << "[rpidx]);\n";
                if (ben) {
                    ss << "        vec3 sf_rgb = sm_blend_factor(sm_u4(sm_fs.rt_src_blend, " << rts << "), out_c, dst_c).rgb;\n";
                    ss << "        vec3 df_rgb = sm_blend_factor(sm_u4(sm_fs.rt_dst_blend, " << rts << "), out_c, dst_c).rgb;\n";
                    ss << "        float sf_a  = sm_blend_factor(sm_u4(sm_fs.rt_src_alpha, " << rts << "), out_c, dst_c).a;\n";
                    ss << "        float df_a  = sm_blend_factor(sm_u4(sm_fs.rt_dst_alpha, " << rts << "), out_c, dst_c).a;\n";
                    ss << "        out_c = vec4(sm_blend_combine3(sm_u4(sm_fs.rt_blend_op, " << rts << "), out_c.rgb, dst_c.rgb, sf_rgb, df_rgb), "
                       << "sm_blend_combine1(sm_u4(sm_fs.rt_alpha_op, " << rts << "), out_c.a, dst_c.a, sf_a, df_a));\n";
                }
                if (need_wm) {
                    ss << "        out_c = mix(dst_c, out_c, bvec4("
                       << ((kwm & 1u) ? "true" : "false") << ", "
                       << ((kwm & 2u) ? "true" : "false") << ", "
                       << ((kwm & 4u) ? "true" : "false") << ", "
                       << ((kwm & 8u) ? "true" : "false") << "));\n";
                }
            }
            if (kfc == 0)      ss << "        " << buf << "[rpidx] = packUnorm4x8(clamp(out_c, 0.0, 1.0));\n";
            else if (kfc == 1) ss << "        " << buf << "[rpidx] = packUnorm4x8(clamp(out_c, 0.0, 1.0).bgra);\n";
            else if (kfc == 2) ss << "        { vec4 c = clamp(out_c, 0.0, 1.0); c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b)); " << buf << "[rpidx] = packUnorm4x8(c); }\n";
            else if (kfc == 3) ss << "        { vec4 c = clamp(out_c, 0.0, 1.0); c.rgb = vec3(sm_lin_to_srgb(c.r), sm_lin_to_srgb(c.g), sm_lin_to_srgb(c.b)); " << buf << "[rpidx] = packUnorm4x8(c.bgra); }\n";
            else if (kfc == 4) ss << "        { vec4 s16 = sm_safe_f16(out_c); " << buf << "[rpidx*2u] = packHalf2x16(s16.rg); " << buf << "[rpidx*2u+1u] = packHalf2x16(s16.ba); }\n";
            else if (kfc == 5) ss << "        " << buf << "[rpidx] = sm_f32_to_uf11(out_c.r) | (sm_f32_to_uf11(out_c.g) << 11u) | (sm_f32_to_uf10(out_c.b) << 22u);\n";
            else if (kfc == 6) ss << "        " << buf << "[rpidx] = floatBitsToUint(isnan(out_c.r) ? 0.0 : out_c.r);\n";
            else if (kfc == 7) ss << "        " << buf << "[rpidx] = packHalf2x16(sm_safe_f16(out_c).rg);\n";
            else if (kfc == 8) ss << "        " << buf << "[rpidx*4u]=floatBitsToUint(out_c.r); " << buf << "[rpidx*4u+1u]=floatBitsToUint(out_c.g); " << buf << "[rpidx*4u+2u]=floatBitsToUint(out_c.b); " << buf << "[rpidx*4u+3u]=floatBitsToUint(out_c.a);\n";
            else if (kfc == 9) ss << "        { vec4 cl = clamp(out_c, 0.0, 1.0); " << buf << "[rpidx] = uint(round(cl.r*1023.0))|(uint(round(cl.g*1023.0))<<10u)|(uint(round(cl.b*1023.0))<<20u)|(uint(round(cl.a*3.0))<<30u); }\n";
            else if (kfc == 10) ss << "        " << buf << "[rpidx*2u] = floatBitsToUint(out_c.r); " << buf << "[rpidx*2u+1u] = floatBitsToUint(out_c.g);\n";
            else if (kfc == 11) ss << "        " << buf << "[rpidx] = uint(out_c.r);\n";
            else if (kfc == 12) ss << "        " << buf << "[rpidx] = packSnorm4x8(clamp(out_c, -1.0, 1.0));\n";
            else if (kfc == 13) ss << "        " << buf << "[rpidx] = packUnorm2x16(clamp(out_c.rg, 0.0, 1.0));\n";
            else if (kfc == 14) ss << "        " << buf << "[rpidx] = packSnorm2x16(clamp(out_c.rg, -1.0, 1.0));\n";
            else if (kfc == 15) ss << "        " << buf << "[rpidx*2u] = packUnorm2x16(clamp(out_c.rg, 0.0, 1.0)); " << buf << "[rpidx*2u+1u] = packUnorm2x16(clamp(out_c.ba, 0.0, 1.0));\n";
            else if (kfc == 16) ss << "        " << buf << "[rpidx*2u] = packSnorm2x16(clamp(out_c.rg, -1.0, 1.0)); " << buf << "[rpidx*2u+1u] = packSnorm2x16(clamp(out_c.ba, -1.0, 1.0));\n";
            else if (kfc == 17) ss << "        { uint sh = (rpidx & 3u) << 3u; uint v8 = uint(round(clamp(out_c.r, 0.0, 1.0) * 255.0)) & 0xFFu; atomicAnd(" << buf << "[rpidx >> 2u], ~(0xFFu << sh)); atomicOr(" << buf << "[rpidx >> 2u], v8 << sh); }\n";
            else if (kfc == 18) ss << "        { uint sh = (rpidx & 1u) << 4u; uint v16 = (uint(round(clamp(out_c.r, 0.0, 1.0)*255.0))&0xFFu) | ((uint(round(clamp(out_c.g, 0.0, 1.0)*255.0))&0xFFu)<<8u); atomicAnd(" << buf << "[rpidx >> 1u], ~(0xFFFFu << sh)); atomicOr(" << buf << "[rpidx >> 1u], v16 << sh); }\n";
            else if (kfc == 19) ss << "        { uint sh = (rpidx & 1u) << 4u; uint v16 = packHalf2x16(vec2(sm_safe_f16(out_c).r, 0.0)) & 0xFFFFu; atomicAnd(" << buf << "[rpidx >> 1u], ~(0xFFFFu << sh)); atomicOr(" << buf << "[rpidx >> 1u], v16 << sh); }\n";
            else if (kfc == 20) ss << "        { uint sh = (rpidx & 1u) << 4u; uint v16 = uint(round(clamp(out_c.r, 0.0, 1.0) * 65535.0)) & 0xFFFFu; atomicAnd(" << buf << "[rpidx >> 1u], ~(0xFFFFu << sh)); atomicOr(" << buf << "[rpidx >> 1u], v16 << sh); }\n";
        } else {
            ss << "        uint fc = sm_u4(sm_fs.rt_fmt, " << rts << ");\n";
            ss << "        uint wm = sm_u4(sm_fs.rt_write_mask, " << rts << ") & 0xFu;\n";
            ss << "        if (wm != 0u) {\n";
            ss << "            bool do_bl = (sm_u4(sm_fs.rt_blend_en, " << rts << ") != 0u);\n";
            ss << "            if (do_bl || wm != 0xFu) {\n";
            ss << "                vec4 dst_c;\n";
            ss << "                if (fc == 4u)       dst_c = vec4(unpackHalf2x16(" << buf << "[rpidx*2u]), unpackHalf2x16(" << buf << "[rpidx*2u+1u]));\n";
            ss << "                else if (fc == 10u) dst_c = vec4(uintBitsToFloat(" << buf << "[rpidx*2u]), uintBitsToFloat(" << buf << "[rpidx*2u+1u]), 0.0, 1.0);\n";
            ss << "                else if (fc == 15u) dst_c = vec4(unpackUnorm2x16(" << buf << "[rpidx*2u]), unpackUnorm2x16(" << buf << "[rpidx*2u+1u]));\n";
            ss << "                else if (fc == 16u) dst_c = vec4(unpackSnorm2x16(" << buf << "[rpidx*2u]), unpackSnorm2x16(" << buf << "[rpidx*2u+1u]));\n";
            ss << "                else if (fc == 8u)  dst_c = vec4(uintBitsToFloat(" << buf << "[rpidx*4u]), uintBitsToFloat(" << buf << "[rpidx*4u+1u]), uintBitsToFloat(" << buf << "[rpidx*4u+2u]), uintBitsToFloat(" << buf << "[rpidx*4u+3u]));\n";
            ss << "                else if (fc == 17u) { uint b8 = (" << buf << "[rpidx>>2u] >> ((rpidx&3u)<<3u)) & 0xFFu; dst_c = vec4(float(b8)/255.0, 0.0, 0.0, 1.0); }\n";
            ss << "                else if (fc == 18u) { uint w16 = (" << buf << "[rpidx>>1u] >> ((rpidx&1u)<<4u)) & 0xFFFFu; dst_c = vec4(float(w16&255u)/255.0, float((w16>>8u)&255u)/255.0, 0.0, 1.0); }\n";
            ss << "                else if (fc == 19u) { uint w16 = (" << buf << "[rpidx>>1u] >> ((rpidx&1u)<<4u)) & 0xFFFFu; dst_c = vec4(unpackHalf2x16(w16).x, 0.0, 0.0, 1.0); }\n";
            ss << "                else if (fc == 20u) { uint w16 = (" << buf << "[rpidx>>1u] >> ((rpidx&1u)<<4u)) & 0xFFFFu; dst_c = vec4(float(w16)/65535.0, 0.0, 0.0, 1.0); }\n";
            ss << "                else dst_c = sm_unpack_u32_fmt(" << buf << "[rpidx], fc);\n";
            if (ben) {
                ss << "                if (do_bl) {\n";
                ss << "                    vec3 sf_rgb = sm_blend_factor(sm_u4(sm_fs.rt_src_blend, " << rts << "), out_c, dst_c).rgb;\n";
                ss << "                    vec3 df_rgb = sm_blend_factor(sm_u4(sm_fs.rt_dst_blend, " << rts << "), out_c, dst_c).rgb;\n";
                ss << "                    float sf_a  = sm_blend_factor(sm_u4(sm_fs.rt_src_alpha, " << rts << "), out_c, dst_c).a;\n";
                ss << "                    float df_a  = sm_blend_factor(sm_u4(sm_fs.rt_dst_alpha, " << rts << "), out_c, dst_c).a;\n";
                ss << "                    out_c = vec4(sm_blend_combine3(sm_u4(sm_fs.rt_blend_op, " << rts << "), out_c.rgb, dst_c.rgb, sf_rgb, df_rgb), "
                   << "sm_blend_combine1(sm_u4(sm_fs.rt_alpha_op, " << rts << "), out_c.a, dst_c.a, sf_a, df_a));\n";
                ss << "                }\n";
            }
            ss << "                if (wm != 0xFu) out_c = mix(dst_c, out_c, bvec4((wm&1u)!=0u, (wm&2u)!=0u, (wm&4u)!=0u, (wm&8u)!=0u));\n";
            ss << "            }\n";
            ss << "            if (fc == 4u)       { vec4 s16 = sm_safe_f16(out_c); " << buf << "[rpidx*2u] = packHalf2x16(s16.rg); " << buf << "[rpidx*2u+1u] = packHalf2x16(s16.ba); }\n";
            ss << "            else if (fc == 10u) { " << buf << "[rpidx*2u] = floatBitsToUint(out_c.r); " << buf << "[rpidx*2u+1u] = floatBitsToUint(out_c.g); }\n";
            ss << "            else if (fc == 15u) { " << buf << "[rpidx*2u] = packUnorm2x16(clamp(out_c.rg,0.0,1.0)); " << buf << "[rpidx*2u+1u] = packUnorm2x16(clamp(out_c.ba,0.0,1.0)); }\n";
            ss << "            else if (fc == 16u) { " << buf << "[rpidx*2u] = packSnorm2x16(clamp(out_c.rg,-1.0,1.0)); " << buf << "[rpidx*2u+1u] = packSnorm2x16(clamp(out_c.ba,-1.0,1.0)); }\n";
            ss << "            else if (fc == 8u)  { " << buf << "[rpidx*4u]=floatBitsToUint(out_c.r); " << buf << "[rpidx*4u+1u]=floatBitsToUint(out_c.g); " << buf << "[rpidx*4u+2u]=floatBitsToUint(out_c.b); " << buf << "[rpidx*4u+3u]=floatBitsToUint(out_c.a); }\n";
            ss << "            else if (fc == 17u) { uint sh=(rpidx&3u)<<3u; uint v8=uint(round(clamp(out_c.r,0.0,1.0)*255.0))&0xFFu; atomicAnd(" << buf << "[rpidx>>2u], ~(0xFFu<<sh)); atomicOr(" << buf << "[rpidx>>2u], v8<<sh); }\n";
            ss << "            else if (fc == 18u) { uint sh=(rpidx&1u)<<4u; uint v16=(uint(round(clamp(out_c.r,0.0,1.0)*255.0))&0xFFu)|((uint(round(clamp(out_c.g,0.0,1.0)*255.0))&0xFFu)<<8u); atomicAnd(" << buf << "[rpidx>>1u], ~(0xFFFFu<<sh)); atomicOr(" << buf << "[rpidx>>1u], v16<<sh); }\n";
            ss << "            else if (fc == 19u) { uint sh=(rpidx&1u)<<4u; uint v16=packHalf2x16(vec2(sm_safe_f16(out_c).r,0.0))&0xFFFFu; atomicAnd(" << buf << "[rpidx>>1u], ~(0xFFFFu<<sh)); atomicOr(" << buf << "[rpidx>>1u], v16<<sh); }\n";
            ss << "            else if (fc == 20u) { uint sh=(rpidx&1u)<<4u; uint v16=uint(round(clamp(out_c.r,0.0,1.0)*65535.0))&0xFFFFu; atomicAnd(" << buf << "[rpidx>>1u], ~(0xFFFFu<<sh)); atomicOr(" << buf << "[rpidx>>1u], v16<<sh); }\n";
            ss << "            else { " << buf << "[rpidx] = sm_pack_u32_fmt(out_c, fc); }\n";
            ss << "        }\n";
        }
        ss << "    }\n";
    };

    ss << "bool sm_is_top_left(vec2 d) { return (d.y < 0.0) || (d.y == 0.0 && d.x > 0.0); }\n";
    ss << "float sm_edge_eval(vec2 a, vec2 b, float fx, float fy) {\n"
       << "    if (a.y < b.y || (a.y == b.y && a.x < b.x)) { vec2 d = b - a; return d.x * (fy - a.y) - d.y * (fx - a.x); }\n"
       << "    else { vec2 d = a - b; return -(d.x * (fy - b.y) - d.y * (fx - b.x)); }\n"
       << "}\n";
    ss << "void main() {\n";
    ss << "    uint tid = gl_LocalInvocationIndex;\n";
    ss << "    uint quad_id = tid >> 2u;\n";
    ss << "    sm_quad_lane = tid & 3u;\n";
    ss << "    uint lx = ((quad_id & 7u) << 1u) | (sm_quad_lane & 1u);\n";
    ss << "    uint ly = ((quad_id >> 3u) << 1u) | ((sm_quad_lane >> 1u) & 1u);\n";
    ss << "    uvec2 p;\n";
    ss << "    bool in_bounds;\n";
    ss << "    uint pidx;\n";
    ss << "    bool active_px;\n";
    ss << "    uint my_tri;\n";
    ss << "    if (sm_fs.direct_2d == 3u) {\n";
    ss << "        p = gl_WorkGroupID.xy * 16u + uvec2(lx, ly);\n";
    ss << "        in_bounds = (p.x < sm_fs.width && p.y < sm_fs.height);\n";
    ss << "        pidx = in_bounds ? (p.y * sm_fs.width + p.x) : 0u;\n";
    ss << "        uint n_in = min(sm_fs.quad_meta & 15u, 2u);\n";
    ss << "        if (tid == 0u) { sm_fq_cnt = 0u; sm_fq_clipped = 0u; }\n";
    ss << "        barrier();\n";
    ss << "        if (tid < n_in) {\n";
    ss << "            uint topo = (sm_fs.quad_meta >> 4u) & 7u;\n";
    ss << "            uint qi0, qi1, qi2;\n";
    ss << "            if (topo == 0u) { qi0 = tid * 3u; qi1 = qi0 + 1u; qi2 = qi0 + 2u; }\n";
    ss << "            else if ((tid & 1u) == 0u) { qi0 = tid; qi1 = tid + 1u; qi2 = tid + 2u; }\n";
    ss << "            else { qi0 = tid + 1u; qi1 = tid; qi2 = tid + 2u; }\n";
    ss << "            vec4 c0 = sm_clip_verts[qi0 * sm_fs.vert_stride_vec4];\n";
    ss << "            vec4 c1 = sm_clip_verts[qi1 * sm_fs.vert_stride_vec4];\n";
    ss << "            vec4 c2 = sm_clip_verts[qi2 * sm_fs.vert_stride_vec4];\n";
    ss << "            bool no_dc = (sm_fs.quad_meta & 2048u) != 0u;\n";
    ss << "            bool ok_tri = !any(isnan(c0)) && !any(isnan(c1)) && !any(isnan(c2)) &&\n";
    ss << "                          !any(isinf(c0)) && !any(isinf(c1)) && !any(isinf(c2)) &&\n";
    ss << "                          !(c0.w <= 1e-5 && c1.w <= 1e-5 && c2.w <= 1e-5) &&\n";
    ss << "                          !(c0.x + c0.w < 0.0 && c1.x + c1.w < 0.0 && c2.x + c2.w < 0.0) &&\n";
    ss << "                          !(c0.w - c0.x < 0.0 && c1.w - c1.x < 0.0 && c2.w - c2.x < 0.0) &&\n";
    ss << "                          !(c0.y + c0.w < 0.0 && c1.y + c1.w < 0.0 && c2.y + c2.w < 0.0) &&\n";
    ss << "                          !(c0.w - c0.y < 0.0 && c1.w - c1.y < 0.0 && c2.w - c2.y < 0.0) &&\n";
    ss << "                          (no_dc || (!(c0.z < 0.0 && c1.z < 0.0 && c2.z < 0.0) && !(c0.z > c0.w && c1.z > c1.w && c2.z > c2.w)));\n";
    ss << "            if (ok_tri) {\n";
    ss << "                vec4 poly_pos[10]; vec3 poly_u[10]; uint n_poly = 3u;\n";
    ss << "                poly_pos[0] = c0; poly_pos[1] = c1; poly_pos[2] = c2;\n";
    ss << "                poly_u[0] = vec3(1.0, 0.0, 0.0); poly_u[1] = vec3(0.0, 1.0, 0.0); poly_u[2] = vec3(0.0, 0.0, 1.0);\n";
    ss << "                const float GB = 4.0;\n";
    ss << "                bool in0 = (c0.w > 1e-4) && (abs(c0.x) <= GB * c0.w) && (abs(c0.y) <= GB * c0.w) && (no_dc || (c0.z >= 0.0 && c0.z <= c0.w));\n";
    ss << "                bool in1 = (c1.w > 1e-4) && (abs(c1.x) <= GB * c1.w) && (abs(c1.y) <= GB * c1.w) && (no_dc || (c1.z >= 0.0 && c1.z <= c1.w));\n";
    ss << "                bool in2 = (c2.w > 1e-4) && (abs(c2.x) <= GB * c2.w) && (abs(c2.y) <= GB * c2.w) && (no_dc || (c2.z >= 0.0 && c2.z <= c2.w));\n";
    ss << "                if (!(in0 && in1 && in2)) {\n";
    ss << "                    atomicOr(sm_fq_clipped, 1u);\n";
    ss << "                    for (uint pl = 0u; pl < 7u && n_poly >= 3u; ++pl) {\n";
    ss << "                        if (no_dc && (pl == 1u || pl == 2u)) continue;\n";
    ss << "                        bool all_in = true, any_in = false; float ds[10];\n";
    ss << "                        for (uint e = 0u; e < n_poly; ++e) {\n";
    ss << "                            vec4 vp = poly_pos[e]; float d;\n";
    ss << "                            if (pl == 0u) d = vp.w - 1e-4; else if (pl == 1u) d = vp.z; else if (pl == 2u) d = vp.w - vp.z;\n";
    ss << "                            else if (pl == 3u) d = vp.x + GB * vp.w; else if (pl == 4u) d = GB * vp.w - vp.x;\n";
    ss << "                            else if (pl == 5u) d = vp.y + GB * vp.w; else d = GB * vp.w - vp.y;\n";
    ss << "                            ds[e] = d; if (d < 0.0) all_in = false; else any_in = true;\n";
    ss << "                        }\n";
    ss << "                        if (all_in) continue;\n";
    ss << "                        if (!any_in) { n_poly = 0u; break; }\n";
    ss << "                        vec4 opos[10]; vec3 ou[10]; uint no = 0u;\n";
    ss << "                        for (uint e = 0u; e < n_poly; ++e) {\n";
    ss << "                            uint ne = (e + 1u == n_poly) ? 0u : (e + 1u);\n";
    ss << "                            float da = ds[e], db = ds[ne]; bool ai = (da >= 0.0), bi = (db >= 0.0);\n";
    ss << "                            if (ai && no < 10u) { opos[no] = poly_pos[e]; ou[no] = poly_u[e]; no++; }\n";
    ss << "                            if (ai != bi && no < 10u) {\n";
    ss << "                                float tt = (abs(da - db) > 1e-12) ? clamp(da / (da - db), 0.0, 1.0) : 0.5;\n";
    ss << "                                vec4 np = mix(poly_pos[e], poly_pos[ne], tt);\n";
    ss << "                                if (pl == 0u) np.w = max(np.w, 1e-4); else if (pl == 1u) np.z = max(np.z, 0.0);\n";
    ss << "                                else if (pl == 2u) { np.w = max(np.w, 1e-4); np.z = min(np.z, np.w); }\n";
    ss << "                                opos[no] = np; ou[no] = mix(poly_u[e], poly_u[ne], tt); no++;\n";
    ss << "                            }\n";
    ss << "                        }\n";
    ss << "                        for (uint k = 0u; k < no; ++k) { poly_pos[k] = opos[k]; poly_u[k] = ou[k]; }\n";
    ss << "                        n_poly = no;\n";
    ss << "                    }\n";
    ss << "                }\n";
    ss << "                for (uint f = 2u; f < n_poly; ++f) {\n";
    ss << "                    vec4 p0 = poly_pos[0], p1 = poly_pos[f - 1u], p2 = poly_pos[f];\n";
    ss << "                    vec3 u0 = poly_u[0],   u1 = poly_u[f - 1u],   u2 = poly_u[f];\n";
    ss << "                    if (!(p0.w > 1e-6 && p1.w > 1e-6 && p2.w > 1e-6)) continue;\n";
    ss << "                    float iw0 = 1.0 / p0.w, iw1 = 1.0 / p1.w, iw2 = 1.0 / p2.w;\n";
    ss << "                    vec2 qs0 = round(vec2(sm_fs.quad_vp.x + (p0.x * iw0 * 0.5 + 0.5) * sm_fs.quad_vp.z, sm_fs.quad_vp.y + (p0.y * iw0 * 0.5 + 0.5) * sm_fs.quad_vp.w) * 16.0) * 0.0625;\n";
    ss << "                    vec2 qs1 = round(vec2(sm_fs.quad_vp.x + (p1.x * iw1 * 0.5 + 0.5) * sm_fs.quad_vp.z, sm_fs.quad_vp.y + (p1.y * iw1 * 0.5 + 0.5) * sm_fs.quad_vp.w) * 16.0) * 0.0625;\n";
    ss << "                    vec2 qs2 = round(vec2(sm_fs.quad_vp.x + (p2.x * iw2 * 0.5 + 0.5) * sm_fs.quad_vp.z, sm_fs.quad_vp.y + (p2.y * iw2 * 0.5 + 0.5) * sm_fs.quad_vp.w) * 16.0) * 0.0625;\n";
    ss << "                    float qz0 = sm_fs.quad_z.x + (p0.z * iw0) * (sm_fs.quad_z.y - sm_fs.quad_z.x);\n";
    ss << "                    float qz1 = sm_fs.quad_z.x + (p1.z * iw1) * (sm_fs.quad_z.y - sm_fs.quad_z.x);\n";
    ss << "                    float qz2 = sm_fs.quad_z.x + (p2.z * iw2) * (sm_fs.quad_z.y - sm_fs.quad_z.x);\n";
    ss << "                    float qa = (qs1.x - qs0.x) * (qs2.y - qs0.y) - (qs1.y - qs0.y) * (qs2.x - qs0.x);\n";
    ss << "                    if (!(abs(qa) >= 1e-6)) continue;\n";
    ss << "                    bool fcw = (sm_fs.quad_meta & 1024u) != 0u;\n";
    ss << "                    bool is_f = fcw ? (qa > 0.0) : (qa < 0.0);\n";
    ss << "                    bool cb = (sm_fs.quad_meta & 256u) != 0u, cf = (sm_fs.quad_meta & 512u) != 0u;\n";
    ss << "                    if ((cb && !is_f) || (cf && is_f)) continue;\n";
    ss << "                    if (qa < 0.0) { vec2 t2 = qs1; qs1 = qs2; qs2 = t2; float tf = qz1; qz1 = qz2; qz2 = tf; tf = iw1; iw1 = iw2; iw2 = tf; vec3 tu = u1; u1 = u2; u2 = tu; qa = -qa; }\n";
    ss << "                    int x0 = max(sm_fs.quad_sc.x, int(floor(clamp(min(qs0.x, min(qs1.x, qs2.x)), -1e6, 1e6))));\n";
    ss << "                    int y0 = max(sm_fs.quad_sc.y, int(floor(clamp(min(qs0.y, min(qs1.y, qs2.y)), -1e6, 1e6))));\n";
    ss << "                    int x1 = min(sm_fs.quad_sc.z - 1, int(ceil(clamp(max(qs0.x, max(qs1.x, qs2.x)), -1e6, 1e6))));\n";
    ss << "                    int y1 = min(sm_fs.quad_sc.w - 1, int(ceil(clamp(max(qs0.y, max(qs1.y, qs2.y)), -1e6, 1e6))));\n";
    ss << "                    if (x0 > x1 || y0 > y1) continue;\n";
    ss << "                    uint sl = atomicAdd(sm_fq_cnt, 1u);\n";
    ss << "                    if (sl < 16u) {\n";
    ss << "                        SmTriSetup qts;\n";
    ss << "                        qts.v0 = vec4(qs0, qz0, iw0); qts.v1 = vec4(qs1, qz1, iw1); qts.v2 = vec4(qs2, qz2, iw2);\n";
    ss << "                        qts.area = vec4(1.0 / qa, float(x0), float(y0), float(x1));\n";
    ss << "                        qts.idx = uvec4(qi0, qi1, qi2, (uint(y1) & 0xFFFFu) | ((tid & 0x7Fu) << 24u) | (is_f ? 0x80000000u : 0u));\n";
    ss << "                        sm_fq_tris[sl] = qts; sm_fq_u0[sl] = u0; sm_fq_u1[sl] = u1; sm_fq_u2[sl] = u2;\n";
    ss << "                    }\n";
    ss << "                }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        barrier();\n";
    ss << "        uint ntri = min(sm_fq_cnt, 16u);\n";
    ss << "        if (ntri == 0u) return;\n";
    ss << "        float tx0 = float(gl_WorkGroupID.x * 16u), ty0 = float(gl_WorkGroupID.y * 16u);\n";
    ss << "        float tx1 = tx0 + 16.0, ty1 = ty0 + 16.0;\n";
    ss << "        bool any_ov = false;\n";
    ss << "        for (uint k = 0u; k < ntri; ++k) {\n";
    ss << "            SmTriSetup qt = sm_fq_tris[k];\n";
    ss << "            if (!(tx1 <= qt.area.y || tx0 > qt.area.w + 1.0 || ty1 <= qt.area.z || ty0 > float(qt.idx.w & 0xFFFFu) + 1.0)) { any_ov = true; break; }\n";
    ss << "        }\n";
    ss << "        if (!any_ov) return;\n";
    ss << "        float qfx = float(p.x) + 0.5, qfy = float(p.y) + 0.5;\n";
    ss << "        my_tri = 0xFFFFFFFFu;\n";
    ss << "        if (in_bounds) {\n";
    ss << "            uint best_ord = 0u;\n";
    ss << "            for (uint k = 0u; k < ntri; ++k) {\n";
    ss << "                SmTriSetup qt = sm_fq_tris[k];\n";
    ss << "                if (qfx < qt.area.y || qfx > qt.area.w + 1.0 || qfy < qt.area.z || qfy > float(qt.idx.w & 0xFFFFu) + 1.0) continue;\n";
    ss << "                vec2 a0 = qt.v0.xy, a1 = qt.v1.xy, a2 = qt.v2.xy;\n";
    ss << "                float e0 = sm_edge_eval(a1, a2, qfx, qfy);\n";
    ss << "                float e1 = sm_edge_eval(a2, a0, qfx, qfy);\n";
    ss << "                float e2 = sm_edge_eval(a0, a1, qfx, qfy);\n";
    ss << "                if ((sm_is_top_left(a2 - a1) ? (e0 < 0.0) : (e0 <= 0.0)) ||\n";
    ss << "                    (sm_is_top_left(a0 - a2) ? (e1 < 0.0) : (e1 <= 0.0)) ||\n";
    ss << "                    (sm_is_top_left(a1 - a0) ? (e2 < 0.0) : (e2 <= 0.0))) continue;\n";
    ss << "                uint ord = ((qt.idx.w >> 24u) & 0x7Fu) + 1u;\n";
    ss << "                if (ord >= best_ord) { best_ord = ord; my_tri = k; }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        active_px = (my_tri != 0xFFFFFFFFu);\n";
    ss << "    } else if (sm_fs.direct_2d == 2u) {\n";
    ss << "        p = gl_WorkGroupID.xy * 16u + uvec2(lx, ly);\n";
    ss << "        in_bounds = (p.x < sm_fs.width && p.y < sm_fs.height);\n";
    ss << "        pidx = in_bounds ? (p.y * sm_fs.width + p.x) : 0u;\n";
    ss << "        uint ntri = sm_tris[8u].idx.x;\n";
    ss << "        if (ntri == 0u) return;\n";
    ss << "        float tx0 = float(gl_WorkGroupID.x * 16u), ty0 = float(gl_WorkGroupID.y * 16u);\n";
    ss << "        float tx1 = tx0 + 16.0, ty1 = ty0 + 16.0;\n";
    ss << "        bool any_ov = false;\n";
    ss << "        for (uint k = 0u; k < ntri; ++k) {\n";
    ss << "            SmTriSetup qt = sm_tris[k];\n";
    ss << "            if (!(tx1 <= qt.area.y || tx0 > qt.area.w + 1.0 || ty1 <= qt.area.z || ty0 > float(qt.idx.w & 0xFFFFu) + 1.0)) { any_ov = true; break; }\n";
    ss << "        }\n";
    ss << "        if (!any_ov) return;\n";
    ss << "        float qfx = float(p.x) + 0.5, qfy = float(p.y) + 0.5;\n";
    ss << "        my_tri = 0xFFFFFFFFu;\n";
    ss << "        if (in_bounds) {\n";
    ss << "            uint best_ord = 0u;\n";
    ss << "            for (uint k = 0u; k < ntri; ++k) {\n";
    ss << "                SmTriSetup qt = sm_tris[k];\n";
    ss << "                if (qfx < qt.area.y || qfx > qt.area.w + 1.0 || qfy < qt.area.z || qfy > float(qt.idx.w & 0xFFFFu) + 1.0) continue;\n";
    ss << "                vec2 a0 = qt.v0.xy, a1 = qt.v1.xy, a2 = qt.v2.xy;\n";
    ss << "                float e0 = sm_edge_eval(a1, a2, qfx, qfy);\n";
    ss << "                float e1 = sm_edge_eval(a2, a0, qfx, qfy);\n";
    ss << "                float e2 = sm_edge_eval(a0, a1, qfx, qfy);\n";
    ss << "                if ((sm_is_top_left(a2 - a1) ? (e0 < 0.0) : (e0 <= 0.0)) ||\n";
    ss << "                    (sm_is_top_left(a0 - a2) ? (e1 < 0.0) : (e1 <= 0.0)) ||\n";
    ss << "                    (sm_is_top_left(a1 - a0) ? (e2 < 0.0) : (e2 <= 0.0))) continue;\n";
    ss << "                uint ord = ((qt.idx.w >> 24u) & 0x7Fu) + 1u;\n";
    ss << "                if (ord >= best_ord) { best_ord = ord; my_tri = k; }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        active_px = (my_tri != 0xFFFFFFFFu);\n";
    ss << "    } else {\n";
    ss << "        if (sm_fs.direct_2d != 0u) {\n";
    ss << "            p = gl_WorkGroupID.xy * 16u + uvec2(lx, ly);\n";
    ss << "        } else {\n";
    ss << "            uint sm_tile_id = sm_active_tiles[gl_WorkGroupID.x];\n";
    ss << "            p = uvec2((sm_tile_id % sm_fs.tiles_x) * 16u + lx, (sm_tile_id / sm_fs.tiles_x) * 16u + ly);\n";
    ss << "        }\n";
    ss << "        in_bounds = (p.x < sm_fs.width && p.y < sm_fs.height);\n";
    ss << "        pidx = in_bounds ? (p.y * sm_fs.width + p.x) : 0u;\n";
    ss << "        uint64_t vis = in_bounds ? sm_vis_px[pidx] : 0xFFFFFFFFFFFFFFFFul;\n";
    ss << "        active_px = in_bounds && (vis != 0xFFFFFFFFFFFFFFFFul);\n";
    ss << "        if (active_px) sm_vis_px[pidx] = 0xFFFFFFFFFFFFFFFFul;\n";
    ss << "        my_tri = active_px ? uint(vis & 0xFFFFFFFFul) : 0xFFFFFFFFu;\n";
    ss << "    }\n";
    ss << "    sm_gl_HelperInvocation = !active_px;\n";
    ss << "    uint t0 = subgroupQuadBroadcast(my_tri, 0u);\n";
    ss << "    uint t1 = subgroupQuadBroadcast(my_tri, 1u);\n";
    ss << "    uint t2 = subgroupQuadBroadcast(my_tri, 2u);\n";
    ss << "    uint t3 = subgroupQuadBroadcast(my_tri, 3u);\n";
    ss << "    uint eff_tri = active_px ? my_tri : min(min(t0, t1), min(t2, t3));\n";
    ss << "    if (eff_tri == 0xFFFFFFFFu) return;\n\n";

    ss << "    SmTriSetup ts = (sm_fs.direct_2d == 3u) ? sm_fq_tris[eff_tri] : sm_tris[eff_tri];\n";
    ss << "    uint sm_draw_id = (ts.idx.w >> 16u) & 0xFFu;\n";
    if (batched_push)    ss << push_init_code;
    if (batched_dyn_ubo) ss << dyn_ubo_init_code;
    ss << "    sm_gl_PrimitiveID = int(eff_tri);\n";
    ss << "    sm_gl_FrontFacing = (ts.idx.w & 0x80000000u) != 0u;\n";
    ss << "    float fx = float(p.x) + 0.5, fy = float(p.y) + 0.5;\n";
    ss << "    vec2 s0 = ts.v0.xy, s1 = ts.v1.xy, s2 = ts.v2.xy;\n";
    ss << "    float l0 = ((s2.x - s1.x) * (fy - s1.y) - (s2.y - s1.y) * (fx - s1.x)) * ts.area.x;\n";
    ss << "    float l1 = ((s0.x - s2.x) * (fy - s2.y) - (s0.y - s2.y) * (fx - s2.x)) * ts.area.x;\n";
    ss << "    float l2 = 1.0 - l0 - l1;\n";
    ss << "    float pz = clamp(l0 * ts.v0.z + l1 * ts.v1.z + l2 * ts.v2.z, 0.0, 1.0);\n";
    ss << "    float w0 = l0 * ts.v0.w, w1 = l1 * ts.v1.w, w2 = l2 * ts.v2.w;\n";
    ss << "    float wsum = w0 + w1 + w2;\n";
    ss << "    float inv_wsum = 1.0 / max(1e-20, wsum);\n";
    ss << "    float b0 = w0 * inv_wsum, b1 = w1 * inv_wsum, b2 = w2 * inv_wsum;\n";
    ss << "    if (sm_fs.direct_2d == 3u && sm_fq_clipped != 0u) {\n";
    ss << "        vec3 ob = b0 * sm_fq_u0[eff_tri] + b1 * sm_fq_u1[eff_tri] + b2 * sm_fq_u2[eff_tri];\n";
    ss << "        b0 = ob.x; b1 = ob.y; b2 = ob.z;\n";
    ss << "        vec3 ol = l0 * sm_fq_u0[eff_tri] + l1 * sm_fq_u1[eff_tri] + l2 * sm_fq_u2[eff_tri];\n";
    ss << "        l0 = ol.x; l1 = ol.y; l2 = ol.z;\n";
    ss << "    } else if (sm_fs.direct_2d == 3u) {\n";
    ss << "        vec3 ob = b0 * sm_fq_u0[eff_tri] + b1 * sm_fq_u1[eff_tri] + b2 * sm_fq_u2[eff_tri];\n";
    ss << "        b0 = ob.x; b1 = ob.y; b2 = ob.z;\n";
    ss << "        vec3 ol = l0 * sm_fq_u0[eff_tri] + l1 * sm_fq_u1[eff_tri] + l2 * sm_fq_u2[eff_tri];\n";
    ss << "        l0 = ol.x; l1 = ol.y; l2 = ol.z;\n";
    ss << "    }\n";
    ss << "    sm_gl_FragCoord = vec4(fx, fy, pz, wsum);\n";
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
            } else if (io.is_flat) {
                ss << "        " << dst_var << " = sm_clip_verts[vi0 + " << off << "u]" << s << ";\n";
            } else if (io.is_noperspective) {
                ss << "        vec4 interp = sm_clip_verts[vi0 + " << off << "u] * l0 + "
                   << "sm_clip_verts[vi1 + " << off << "u] * l1 + "
                   << "sm_clip_verts[vi2 + " << off << "u] * l2;\n";
                ss << "        " << dst_var << " = interp" << s << ";\n";
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
    if (hints.stencil_en) {
        ss << "    if ((sm_fs.stencil_enable & 3u) == 3u) {\n";
        ss << "        uint scfg = sm_gl_FrontFacing ? sm_fs.stencil_front_cfg : sm_fs.stencil_back_cfg;\n";
        ss << "        uint sref = sm_gl_FrontFacing ? (sm_fs.stencil_refs & 0xFFu) : ((sm_fs.stencil_refs >> 8u) & 0xFFu);\n";
        ss << "        uint cmask = (scfg >> 16u) & 0xFFu;\n";
        ss << "        uint wmask = (scfg >> 24u) & 0xFFu;\n";
        ss << "        uint old_s = sm_stencil_px[pidx] & 0xFFu;\n";
        ss << "        bool spass = sm_stencil_cmp(scfg & 7u, sref & cmask, old_s & cmask);\n";
        ss << "        uint sop = spass ? ((scfg >> 6u) & 7u) : ((scfg >> 3u) & 7u);\n";
        ss << "        uint new_s = sm_stencil_apply(sop, old_s, sref) & 0xFFu;\n";
        ss << "        if (wmask != 0u) sm_stencil_px[pidx] = (old_s & (~wmask & 0xFFu)) | (new_s & wmask);\n";
        ss << "        if (!spass) return;\n";
        ss << "    }\n";
    }
    ss << "    if (sm_fs.depth_write != 0u) sm_depth_px[pidx] = " << (uses_frag_depth ? "sm_gl_FragDepth" : "pz") << ";\n";

    // Write ALL reflected MRT color outputs (Location 0..7)!
    for (const auto& io : outputs) {
        if (io.array_size == 0 && io.location < SM_MAX_MRT) emit_rt_store((int)io.location, io);
    }
    ss << "}\n";

    return compile_glsl_to_spv(ss.str(), "fs", out_comp_spv, err);
}

bool sm_build_null_fs_stencil_spv(uint32_t sm_set_idx,
                                  std::vector<uint32_t>& out_comp_spv,
                                  std::string& err) {
    SmCancelDisableGuard _cancel_guard;
    std::ostringstream ss;
    ss << "#version 460\n"
       << "#extension GL_ARB_gpu_shader_int64 : require\n"
       << "layout(local_size_x = 16, local_size_y = 16) in;\n"
       << "struct SmTriSetup { vec4 v0; vec4 v1; vec4 v2; vec4 area; uvec4 idx; };\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 1)  readonly buffer SmTriSetupBuf { SmTriSetup sm_tris[]; };\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 2)  buffer SmVisBuf               { uint64_t sm_vis_px[]; };\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 11) writeonly buffer SmDepthBuf   { float sm_depth_px[]; };\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 12) buffer SmStencilBuf           { uint sm_stencil_px[]; };\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 13) readonly buffer SmFsParams {\n"
       << "    uint width; uint height; uint vert_stride_vec4; uint depth_write;\n"
       << "    uint rt_active_mask; uint stencil_enable; uint stencil_front_cfg; uint stencil_back_cfg; uint stencil_refs; uint tiles_x; uint direct_2d;\n"
       << "} sm_fs;\n"
       << "layout(std430, set = " << sm_set_idx << ", binding = 14) readonly buffer SmActiveTilesBuf { uint sm_active_tiles[]; };\n"
       << "bool sm_stencil_cmp(uint op, uint ref_m, uint val_m) {\n"
       << "    if (op == 0u) return false; if (op == 1u) return ref_m < val_m; if (op == 2u) return ref_m == val_m;\n"
       << "    if (op == 3u) return ref_m <= val_m; if (op == 4u) return ref_m > val_m; if (op == 5u) return ref_m != val_m;\n"
       << "    if (op == 6u) return ref_m >= val_m; return true;\n"
       << "}\n"
       << "uint sm_stencil_apply(uint op, uint cur, uint ref) {\n"
       << "    if (op == 1u) return 0u; if (op == 2u) return ref; if (op == 3u) return min(cur + 1u, 255u);\n"
       << "    if (op == 4u) return (cur > 0u) ? (cur - 1u) : 0u; if (op == 5u) return (~cur) & 0xFFu;\n"
       << "    if (op == 6u) return (cur + 1u) & 0xFFu; if (op == 7u) return (cur - 1u) & 0xFFu; return cur;\n"
       << "}\n"
       << "void main() {\n"
       << "    uvec2 p;\n"
       << "    if (sm_fs.direct_2d != 0u) {\n"
       << "        p = gl_GlobalInvocationID.xy;\n"
       << "    } else {\n"
       << "        uint tile_id = sm_active_tiles[gl_WorkGroupID.x];\n"
       << "        p = uvec2((tile_id % sm_fs.tiles_x) * 16u + gl_LocalInvocationID.x, (tile_id / sm_fs.tiles_x) * 16u + gl_LocalInvocationID.y);\n"
       << "    }\n"
       << "    if (p.x >= sm_fs.width || p.y >= sm_fs.height) return;\n"
       << "    uint pidx = p.y * sm_fs.width + p.x;\n"
       << "    uint64_t vis = sm_vis_px[pidx];\n"
       << "    if (vis == 0xFFFFFFFFFFFFFFFFul) return;\n"
       << "    sm_vis_px[pidx] = 0xFFFFFFFFFFFFFFFFul;\n"
       << "    uint tid = uint(vis & 0xFFFFFFFFul);\n"
       << "    SmTriSetup ts = sm_tris[tid];\n"
       << "    bool ff = (ts.idx.w & 0x80000000u) != 0u;\n"
       << "    if ((sm_fs.stencil_enable & 3u) == 3u) {\n"
       << "        uint scfg = ff ? sm_fs.stencil_front_cfg : sm_fs.stencil_back_cfg;\n"
       << "        uint sref = ff ? (sm_fs.stencil_refs & 0xFFu) : ((sm_fs.stencil_refs >> 8u) & 0xFFu);\n"
       << "        uint cmask = (scfg >> 16u) & 0xFFu;\n"
       << "        uint wmask = (scfg >> 24u) & 0xFFu;\n"
       << "        uint old_s = sm_stencil_px[pidx] & 0xFFu;\n"
       << "        bool spass = sm_stencil_cmp(scfg & 7u, sref & cmask, old_s & cmask);\n"
       << "        uint sop = spass ? ((scfg >> 6u) & 7u) : ((scfg >> 3u) & 7u);\n"
       << "        uint new_s = sm_stencil_apply(sop, old_s, sref) & 0xFFu;\n"
       << "        if (wmask != 0u) sm_stencil_px[pidx] = (old_s & (~wmask & 0xFFu)) | (new_s & wmask);\n"
       << "        if (!spass) return;\n"
       << "    }\n"
       << "    if (sm_fs.depth_write != 0u) {\n"
       << "        float fx = float(p.x) + 0.5, fy = float(p.y) + 0.5;\n"
       << "        vec2 s0 = ts.v0.xy, s1 = ts.v1.xy, s2 = ts.v2.xy;\n"
       << "        float l0 = ((s2.x - s1.x)*(fy - s1.y) - (s2.y - s1.y)*(fx - s1.x)) * ts.area.x;\n"
       << "        float l1 = ((s0.x - s2.x)*(fy - s2.y) - (s0.y - s2.y)*(fx - s2.x)) * ts.area.x;\n"
       << "        float l2 = 1.0 - l0 - l1;\n"
       << "        sm_depth_px[pidx] = clamp(l0*ts.v0.z + l1*ts.v1.z + l2*ts.v2.z, 0.0, 1.0);\n"
       << "    }\n"
       << "}\n";
    return compile_glsl_to_spv(ss.str(), "nullfs_stn", out_comp_spv, err);
}
