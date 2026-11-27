// smrast_wsi.cpp — Zero-Copy MIT-SHM XCB & Xlib WSI Presentation Bridge for SmRast-VK v0.7
// Copies H20 HBM3 swapchain framebuffer -> host staging buffer via pure Compute Queue (qf[2])
// and blits to Xvfb / X11 via xcb_shm_put_image OR XShmPutImage (zero-socket-copy for both XCB & Wine Xlib!).

#include "smrast_internal.h"
#include <xcb/xcb.h>
#include <xcb/shm.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <chrono>
#include <cstdio>
#include <cstring>

static uint64_t wsi_now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

static void init_xcb_shm(SmSwapchain* sc, size_t bytes) {
    sc->shm_checked = true;
    SmSurface* surf = sc->surface;
    if (!surf || !surf->is_xcb || !surf->xcb_conn) return;

    sc->xcb_gc = xcb_generate_id(surf->xcb_conn);
    xcb_create_gc(surf->xcb_conn, sc->xcb_gc, surf->xcb_win, 0, nullptr);

    auto q_cookie = xcb_shm_query_version(surf->xcb_conn);
    auto* q_rep = xcb_shm_query_version_reply(surf->xcb_conn, q_cookie, nullptr);
    if (!q_rep) return;
    free(q_rep);

    for (int k = 0; k < 2; ++k) {
        int shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0666);
        if (shmid < 0) goto fail_xcb;
        void* ptr = shmat(shmid, nullptr, 0);
        if (ptr == (void*)-1) { shmctl(shmid, IPC_RMID, nullptr); goto fail_xcb; }
        xcb_shm_seg_t seg = xcb_generate_id(surf->xcb_conn);
        auto att_c = xcb_shm_attach_checked(surf->xcb_conn, seg, (uint32_t)shmid, 0);
        xcb_generic_error_t* err = xcb_request_check(surf->xcb_conn, att_c);
        shmctl(shmid, IPC_RMID, nullptr);
        if (err) { free(err); shmdt(ptr); goto fail_xcb; }
        sc->shm_ids[k]  = shmid;
        sc->shm_segs[k] = seg;
        sc->shm_ptrs[k] = (uint8_t*)ptr;
    }
    sc->use_shm = true;
    sc->shm_bytes = bytes;
    return;
fail_xcb:
    for (int k = 0; k < 2; ++k) {
        if (sc->shm_segs[k]) { xcb_shm_detach(surf->xcb_conn, sc->shm_segs[k]); sc->shm_segs[k] = 0; }
        if (sc->shm_ptrs[k]) { shmdt(sc->shm_ptrs[k]); sc->shm_ptrs[k] = nullptr; }
    }
}

static bool g_xshm_err_flag = false;
static int xshm_err_handler(Display*, XErrorEvent*) { g_xshm_err_flag = true; return 0; }

static void init_xlib_shm(SmSwapchain* sc, uint32_t w, uint32_t h) {
    sc->shm_checked = true;
    SmSurface* surf = sc->surface;
    if (!surf || surf->is_xcb || !surf->xlib_dpy) return;
    Display* dpy = surf->xlib_dpy;
    sc->xlib_gc = (void*)XCreateGC(dpy, surf->xlib_win, 0, nullptr);
    if (!XShmQueryExtension(dpy)) return;

    int screen = DefaultScreen(dpy);
    Visual* vis = DefaultVisual(dpy, screen);
    size_t bytes = 0;
    for (int k = 0; k < 2; ++k) {
        auto* shminfo = new XShmSegmentInfo();
        memset(shminfo, 0, sizeof(*shminfo));
        XImage* ximg = XShmCreateImage(dpy, vis, 24, ZPixmap, nullptr, shminfo, w, h);
        if (!ximg) { delete shminfo; goto fail_xlib; }
        bytes = (size_t)ximg->bytes_per_line * (size_t)ximg->height;
        int shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0666);
        if (shmid < 0) { XDestroyImage(ximg); delete shminfo; goto fail_xlib; }
        void* ptr = shmat(shmid, nullptr, 0);
        if (ptr == (void*)-1) { shmctl(shmid, IPC_RMID, nullptr); XDestroyImage(ximg); delete shminfo; goto fail_xlib; }
        shminfo->shmid = shmid;
        shminfo->shmaddr = (char*)ptr;
        shminfo->readOnly = False;
        ximg->data = (char*)ptr;

        g_xshm_err_flag = false;
        auto prev_err = XSetErrorHandler(xshm_err_handler);
        Status st = XShmAttach(dpy, shminfo);
        XSync(dpy, False);
        XSetErrorHandler(prev_err);
        shmctl(shmid, IPC_RMID, nullptr);
        if (!st || g_xshm_err_flag) {
            shmdt(ptr);
            ximg->data = nullptr;
            XDestroyImage(ximg);
            delete shminfo;
            goto fail_xlib;
        }
        sc->shm_ids[k] = shmid;
        sc->shm_ptrs[k] = (uint8_t*)ptr;
        sc->xlib_shm_imgs[k] = (void*)ximg;
        sc->xlib_shminfos[k] = (void*)shminfo;
    }
    sc->use_shm = true;
    sc->shm_bytes = bytes;
    return;
fail_xlib:
    for (int k = 0; k < 2; ++k) {
        if (sc->xlib_shminfos[k]) {
            auto* si = (XShmSegmentInfo*)sc->xlib_shminfos[k];
            XShmDetach(dpy, si);
            XSync(dpy, False);
            delete si;
            sc->xlib_shminfos[k] = nullptr;
        }
        if (sc->xlib_shm_imgs[k]) {
            XImage* xi = (XImage*)sc->xlib_shm_imgs[k];
            xi->data = nullptr;
            XDestroyImage(xi);
            sc->xlib_shm_imgs[k] = nullptr;
        }
        if (sc->shm_ptrs[k]) { shmdt(sc->shm_ptrs[k]); sc->shm_ptrs[k] = nullptr; }
    }
}

void sm_wsi_cleanup_swapchain(SmSwapchain* sc) {
    if (!sc || !sc->surface) return;
    SmSurface* surf = sc->surface;
    if (surf->is_xcb && surf->xcb_conn) {
        if (sc->use_shm) {
            for (int k = 0; k < 2; ++k) {
                if (sc->shm_segs[k]) { xcb_shm_detach(surf->xcb_conn, sc->shm_segs[k]); sc->shm_segs[k] = 0; }
                if (sc->shm_ptrs[k]) { shmdt(sc->shm_ptrs[k]); sc->shm_ptrs[k] = nullptr; }
            }
            sc->use_shm = false;
        }
        if (sc->xcb_gc) {
            xcb_free_gc(surf->xcb_conn, sc->xcb_gc);
            sc->xcb_gc = 0;
        }
        xcb_flush(surf->xcb_conn);
    } else if (!surf->is_xcb && surf->xlib_dpy) {
        Display* dpy = surf->xlib_dpy;
        if (sc->use_shm) {
            for (int k = 0; k < 2; ++k) {
                if (sc->xlib_shminfos[k]) {
                    auto* shminfo = (XShmSegmentInfo*)sc->xlib_shminfos[k];
                    XShmDetach(dpy, shminfo);
                    XSync(dpy, False);
                    delete shminfo;
                    sc->xlib_shminfos[k] = nullptr;
                }
                if (sc->xlib_shm_imgs[k]) {
                    XImage* xi = (XImage*)sc->xlib_shm_imgs[k];
                    xi->data = nullptr;
                    XDestroyImage(xi);
                    sc->xlib_shm_imgs[k] = nullptr;
                }
                if (sc->shm_ptrs[k]) { shmdt(sc->shm_ptrs[k]); sc->shm_ptrs[k] = nullptr; }
            }
            sc->use_shm = false;
        }
        if (sc->xlib_gc) {
            XFreeGC(dpy, (GC)sc->xlib_gc);
            sc->xlib_gc = nullptr;
        }
        XFlush(dpy);
    }
}

static uint64_t g_prof_last_report_us = 0;
static uint64_t g_prof_last_frames = 0;

static inline void sm_copy_rgba_to_bgra_fast(uint8_t* __restrict dst, const uint8_t* __restrict src, size_t bytes) {
    uint32_t* __restrict d32 = reinterpret_cast<uint32_t*>(dst);
    const uint32_t* __restrict s32 = reinterpret_cast<const uint32_t*>(src);
    const size_t n = bytes >> 2;
    for (size_t i = 0; i < n; ++i) {
        uint32_t p = s32[i];
        d32[i] = (p & 0x0000FF00u) | ((p & 0x000000FFu) << 16) | ((p >> 16) & 0x000000FFu) | 0xFF000000u;
    }
}

void sm_wsi_blit_to_surface(SmSwapchain* sc, SmImage* img) {
    if (!sc || !sc->surface || !img || !img->host_ptr) return;
    SmDevice* dev = sc->dev;
    uint32_t w = sc->extent.width;
    uint32_t h = sc->extent.height;
    size_t bytes = (size_t)w * h * 4;

    uint64_t t0 = g_sm_prof_en ? wsi_now_us() : 0;

    // 1. Pipelined 2-slot Present Ring (for >=3 swapchain images):
    //    Submit Frame N's GPU->Host DMA copy asynchronously, then wait & XShmPutImage Frame N-1!
    //    Completely hides X11 swizzle/blit + CPU command recording behind GPU compute execution!
    bool pipeline_ok = (sc->images.size() >= 3u && sc->present_fences[0] && sc->present_fences[1] && !getenv("SMRAST_XRAY"));
    uint32_t cur  = sc->present_ring_idx & 1u;
    uint32_t prev = 1u - cur;
    sc->present_ring_idx++;

    // Ensure cur slot's previous DMA (from Frame N-2) is complete before resetting its command pool
    if (sc->pending_present_img[cur] && sc->present_fences[cur]) {
        dev->fn.WaitForFences(dev->real_dev, 1, &sc->present_fences[cur], VK_TRUE, UINT64_MAX);
        sc->pending_present_img[cur] = nullptr;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    dev->fn.ResetCommandPool(dev->real_dev, sc->present_pools[cur], 0);
    dev->fn.BeginCommandBuffer(sc->present_cmds[cur], &bi);
    VkMemoryBarrier pmb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT};
    dev->fn.CmdPipelineBarrier(sc->present_cmds[cur],
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pmb, 0, nullptr, 0, nullptr);
    if (img->attach_buf) {
        VkBufferCopy cp{0, 0, bytes};
        dev->fn.CmdCopyBuffer(sc->present_cmds[cur], img->attach_buf, img->host_buf, 1, &cp);
    } else {
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bic.imageExtent = {w, h, 1};
        dev->fn.CmdCopyImageToBuffer(sc->present_cmds[cur], img->real_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img->host_buf, 1, &bic);
    }
    dev->fn.EndCommandBuffer(sc->present_cmds[cur]);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &sc->present_cmds[cur]};
    if (sc->present_fences[cur]) {
        dev->fn.ResetFences(dev->real_dev, 1, &sc->present_fences[cur]);
        {
            std::lock_guard<std::mutex> qlk(dev->main_queue->mtx);
            dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, sc->present_fences[cur]);
        }
    } else {
        std::lock_guard<std::mutex> qlk(dev->main_queue->mtx);
        dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, VK_NULL_HANDLE);
    }
    sc->pending_present_img[cur] = img;

    SmImage* blit_img = nullptr;
    if (pipeline_ok) {
        if (sc->pending_present_img[prev]) {
            dev->fn.WaitForFences(dev->real_dev, 1, &sc->present_fences[prev], VK_TRUE, UINT64_MAX);
            blit_img = sc->pending_present_img[prev];
            sc->pending_present_img[prev] = nullptr;
        }
    } else {
        if (sc->present_fences[cur]) dev->fn.WaitForFences(dev->real_dev, 1, &sc->present_fences[cur], VK_TRUE, UINT64_MAX);
        else                         dev->fn.QueueWaitIdle(dev->main_queue->real_q);
        blit_img = img;
        sc->pending_present_img[cur] = nullptr;
    }

    uint64_t t1 = g_sm_prof_en ? wsi_now_us() : 0;
    if (g_sm_prof_en) g_sm_prof.wsi_gpu_us.fetch_add(t1 - t0, std::memory_order_relaxed);
    if (!blit_img || !blit_img->host_ptr) { sc->present_frames++; return; }

    const uint8_t* src = (const uint8_t*)blit_img->host_ptr;
    bool need_swap = (sc->format == VK_FORMAT_R8G8B8A8_UNORM || sc->format == VK_FORMAT_R8G8B8A8_SRGB);

    SmSurface* surf = sc->surface;
    uint32_t sidx = sc->shm_idx & 1u;
    if (surf->is_xcb && surf->xcb_conn) {
        if (!sc->shm_checked) init_xcb_shm(sc, bytes);
        if (sc->use_shm && sc->shm_ptrs[sidx] && sc->shm_bytes >= bytes) {
            if (sc->shm_in_flight)
                free(xcb_get_input_focus_reply(surf->xcb_conn, xcb_get_input_focus(surf->xcb_conn), nullptr));
            if (!need_swap) {
                memcpy(sc->shm_ptrs[sidx], src, bytes);
            } else {
                sm_copy_rgba_to_bgra_fast(sc->shm_ptrs[sidx], src, bytes);
            }
            xcb_shm_put_image(surf->xcb_conn, surf->xcb_win, sc->xcb_gc,
                              (uint16_t)w, (uint16_t)h, 0, 0, (uint16_t)w, (uint16_t)h,
                              0, 0, 24, XCB_IMAGE_FORMAT_Z_PIXMAP, 0, sc->shm_segs[sidx], 0);
            xcb_flush(surf->xcb_conn);
            sc->shm_idx++;
            sc->shm_in_flight = true;
        } else {
            std::vector<uint8_t> tmp;
            if (need_swap) {
                tmp.resize(bytes);
                for (size_t i = 0; i < bytes; i += 4) {
                    tmp[i+0] = src[i+2]; tmp[i+1] = src[i+1]; tmp[i+2] = src[i+0]; tmp[i+3] = 255;
                }
                src = tmp.data();
            }
            const uint32_t strip_h = 64;
            for (uint32_t y = 0; y < h; y += strip_h) {
                uint32_t ch = (y + strip_h <= h) ? strip_h : (h - y);
                xcb_put_image(surf->xcb_conn, XCB_IMAGE_FORMAT_Z_PIXMAP, surf->xcb_win, sc->xcb_gc,
                              (uint16_t)w, (uint16_t)ch, 0, (int16_t)y, 0, 24,
                              (uint32_t)(ch * w * 4), src + (size_t)y * w * 4);
            }
            xcb_flush(surf->xcb_conn);
        }
    } else if (!surf->is_xcb && surf->xlib_dpy) {
        if (!sc->shm_checked) init_xlib_shm(sc, w, h);
        if (sc->use_shm && sc->shm_ptrs[sidx] && sc->xlib_shm_imgs[sidx] && sc->xlib_gc && sc->shm_bytes >= bytes) {
            // Ping-pong sync BEFORE writing next SHM slot: hides previous frame's Xvfb blit 100% behind GPU compute!
            if (sc->shm_in_flight) XSync(surf->xlib_dpy, False);
            if (!need_swap) {
                memcpy(sc->shm_ptrs[sidx], src, bytes);
            } else {
                sm_copy_rgba_to_bgra_fast(sc->shm_ptrs[sidx], src, bytes);
            }
            XShmPutImage(surf->xlib_dpy, surf->xlib_win, (GC)sc->xlib_gc, (XImage*)sc->xlib_shm_imgs[sidx],
                         0, 0, 0, 0, w, h, False);
            XFlush(surf->xlib_dpy);
            sc->shm_idx++;
            sc->shm_in_flight = true;
        } else {
            std::vector<uint8_t> tmp;
            if (need_swap) {
                tmp.resize(bytes);
                for (size_t i = 0; i < bytes; i += 4) {
                    tmp[i+0] = src[i+2]; tmp[i+1] = src[i+1]; tmp[i+2] = src[i+0]; tmp[i+3] = 255;
                }
                src = tmp.data();
            }
            GC gc = sc->xlib_gc ? (GC)sc->xlib_gc : XCreateGC(surf->xlib_dpy, surf->xlib_win, 0, nullptr);
            int screen = DefaultScreen(surf->xlib_dpy);
            Visual* vis = DefaultVisual(surf->xlib_dpy, screen);
            XImage* xi = XCreateImage(surf->xlib_dpy, vis, 24, ZPixmap, 0, (char*)src, w, h, 32, (int)(w * 4));
            if (xi) {
                XPutImage(surf->xlib_dpy, surf->xlib_win, gc, xi, 0, 0, 0, 0, w, h);
                xi->data = nullptr;
                XDestroyImage(xi);
            }
            if (!sc->xlib_gc) XFreeGC(surf->xlib_dpy, gc);
            XFlush(surf->xlib_dpy);
        }
    }

    uint64_t t2 = g_sm_prof_en ? wsi_now_us() : 0;
    if (g_sm_prof_en) g_sm_prof.wsi_x11_us.fetch_add(t2 - t1, std::memory_order_relaxed);

    sc->present_frames++;
    if (g_sm_prof_en) {
        if (g_prof_last_report_us == 0) {
            g_prof_last_report_us = t2;
            g_prof_last_frames = sc->present_frames;
        } else if (sc->present_frames - g_prof_last_frames >= 30 || (t2 - g_prof_last_report_us) >= 3000000ULL) {
            uint64_t df = sc->present_frames - g_prof_last_frames;
            double dt_ms = (double)(t2 - g_prof_last_report_us) / 1000.0;
            double ms_pf = df ? (dt_ms / df) : 0.0;
            double fps   = dt_ms > 0.0 ? (df * 1000.0 / dt_ms) : 0.0;
            auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
            double invf = df ? (1.0 / (double)df) : 0.0;
            uint64_t rp   = take(g_sm_prof.rp_begins);
            uint64_t rclr = take(g_sm_prof.rp_clears);
            uint64_t rsk  = take(g_sm_prof.rp_skip_clears);
            uint64_t si2a = take(g_sm_prof.sync_i2a);
            uint64_t sa2i = take(g_sm_prof.sync_a2i);
            uint64_t sask = take(g_sm_prof.sync_a2i_skip);
            uint64_t dr   = take(g_sm_prof.draws_total);
            uint64_t dsm  = take(g_sm_prof.draws_small_tri);
            uint64_t dind = take(g_sm_prof.draws_indirect);
            uint64_t ddep = take(g_sm_prof.draws_depth_only);
            uint64_t dbat = take(g_sm_prof.draws_batchable);
            uint64_t acs  = take(g_sm_prof.app_dispatches);
            uint64_t blt  = take(g_sm_prof.blits);
            uint64_t cpy  = take(g_sm_prof.copies);
            uint64_t sub  = take(g_sm_prof.submits);
            uint64_t wfu  = take(g_sm_prof.wait_fences_us);
            uint64_t wsu  = take(g_sm_prof.wait_sems_us);
            uint64_t wgu  = take(g_sm_prof.wsi_gpu_us);
            uint64_t wxu  = take(g_sm_prof.wsi_x11_us);
            fprintf(stderr,
                "[SmRast-PROF] F#%llu: %.2f FPS (%.1f ms/f) | WSI(gpu=%.1fms x11=%.1fms shm=%d) Wait(f=%.1fms s=%.1fms) | "
                "RP=%.1f (clr=%.1f skip=%.1f) Sync(i2a=%.1f a2i=%.1f skip=%.1f) | "
                "Draws=%.1f (quad=%.1f ind=%.1f dep=%.1f batchable=%.1f) AppCS=%.1f Blit=%.1f Cpy=%.1f Sub=%.1f\n",
                (unsigned long long)sc->present_frames, fps, ms_pf,
                wgu * invf / 1000.0, wxu * invf / 1000.0, sc->use_shm ? 1 : 0,
                wfu * invf / 1000.0, wsu * invf / 1000.0,
                rp * invf, rclr * invf, rsk * invf,
                si2a * invf, sa2i * invf, sask * invf,
                dr * invf, dsm * invf, dind * invf, ddep * invf, dbat * invf,
                acs * invf, blt * invf, cpy * invf, sub * invf);
            fflush(stderr);
            g_prof_last_report_us = t2;
            g_prof_last_frames = sc->present_frames;
        }
    }
    if (sc->present_frames <= 10 || sc->present_frames == 30 || (sc->present_frames % 60) == 0) {
        const uint32_t* px32 = reinterpret_cast<const uint32_t*>(src);
        uint32_t mid = px32[(h / 2) * w + (w / 2)];
        uint32_t nz = 0;
        for (size_t k = 0; k < (size_t)w * h; k += 37) if ((px32[k] & 0x00FFFFFFu) != 0) nz++;
        fprintf(stderr, "[SmRast-WSI] Presented frame #%llu (%ux%u, shm=%d, mid=0x%08x, lit_samples=%u)\n",
                (unsigned long long)sc->present_frames, w, h, sc->use_shm ? 1 : 0, mid, nz);
        fflush(stderr);
    }
}
