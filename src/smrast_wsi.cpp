// smrast_wsi.cpp — Zero-Copy MIT-SHM XCB & Xlib WSI Presentation Bridge for SmRast-VK v0.3
// Copies H20 HBM3 swapchain framebuffer -> host staging buffer via pure Compute Queue (qf[2])
// and blits to Xvfb / X11 via xcb_shm_put_image (MIT-SHM zero-copy!) with automatic fallback.

#include "smrast_internal.h"
#include <xcb/xcb.h>
#include <xcb/shm.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdio>
#include <cstring>

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

    int shmid = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
    if (shmid < 0) return;
    void* ptr = shmat(shmid, nullptr, 0);
    if (ptr == (void*)-1) { shmctl(shmid, IPC_RMID, nullptr); return; }

    xcb_shm_seg_t seg = xcb_generate_id(surf->xcb_conn);
    auto att_c = xcb_shm_attach_checked(surf->xcb_conn, seg, (uint32_t)shmid, 0);
    xcb_generic_error_t* err = xcb_request_check(surf->xcb_conn, att_c);
    // Mark for auto-removal as soon as both client & X server detach
    shmctl(shmid, IPC_RMID, nullptr);
    if (err) {
        free(err);
        shmdt(ptr);
        return;
    }
    sc->use_shm = true;
    sc->shm_id = shmid;
    sc->shm_seg = seg;
    sc->shm_ptr = (uint8_t*)ptr;
    sc->shm_bytes = bytes;
}

void sm_wsi_cleanup_swapchain(SmSwapchain* sc) {
    if (!sc || !sc->surface) return;
    SmSurface* surf = sc->surface;
    if (surf->is_xcb && surf->xcb_conn) {
        if (sc->use_shm) {
            xcb_shm_detach(surf->xcb_conn, sc->shm_seg);
            if (sc->shm_ptr) shmdt(sc->shm_ptr);
            sc->shm_ptr = nullptr;
            sc->use_shm = false;
        }
        if (sc->xcb_gc) {
            xcb_free_gc(surf->xcb_conn, sc->xcb_gc);
            sc->xcb_gc = 0;
        }
        xcb_flush(surf->xcb_conn);
    }
}

void sm_wsi_blit_to_surface(SmSwapchain* sc, SmImage* img) {
    if (!sc || !sc->surface || !img || !img->host_ptr) return;
    SmDevice* dev = sc->dev;
    uint32_t w = sc->extent.width;
    uint32_t h = sc->extent.height;
    size_t bytes = (size_t)w * h * 4;

    // 1. Copy GPU attachment buffer -> host-coherent staging buffer via pure Compute Queue (qf[2])
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    dev->fn.ResetCommandPool(dev->real_dev, sc->present_pool, 0);
    dev->fn.BeginCommandBuffer(sc->present_cmd, &bi);
    if (img->attach_buf) {
        VkBufferCopy cp{0, 0, bytes};
        dev->fn.CmdCopyBuffer(sc->present_cmd, img->attach_buf, img->host_buf, 1, &cp);
    } else {
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bic.imageExtent = {w, h, 1};
        dev->fn.CmdCopyImageToBuffer(sc->present_cmd, img->real_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img->host_buf, 1, &bic);
    }
    dev->fn.EndCommandBuffer(sc->present_cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &sc->present_cmd};
    if (sc->present_fence) {
        dev->fn.ResetFences(dev->real_dev, 1, &sc->present_fence);
        {
            std::lock_guard<std::mutex> qlk(dev->main_queue->mtx);
            dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, sc->present_fence);
        }
        // Wait WITHOUT holding main_queue->mtx so vkd3d-proton timeline-semaphore submits never deadlock!
        dev->fn.WaitForFences(dev->real_dev, 1, &sc->present_fence, VK_TRUE, UINT64_MAX);
    } else {
        {
            std::lock_guard<std::mutex> qlk(dev->main_queue->mtx);
            dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, VK_NULL_HANDLE);
        }
        dev->fn.QueueWaitIdle(dev->main_queue->real_q);
    }

    const uint8_t* src = (const uint8_t*)img->host_ptr;
    bool need_swap = (sc->format == VK_FORMAT_R8G8B8A8_UNORM || sc->format == VK_FORMAT_R8G8B8A8_SRGB);

    SmSurface* surf = sc->surface;
    if (surf->is_xcb && surf->xcb_conn) {
        if (!sc->shm_checked) init_xcb_shm(sc, bytes);
        if (sc->use_shm && sc->shm_ptr && sc->shm_bytes >= bytes) {
            // Fast MIT-SHM Zero-Socket-Copy Blit!
            if (!need_swap) {
                memcpy(sc->shm_ptr, src, bytes);
            } else {
                uint8_t* dst = sc->shm_ptr;
                for (size_t i = 0; i < bytes; i += 4) {
                    dst[i+0] = src[i+2]; dst[i+1] = src[i+1]; dst[i+2] = src[i+0]; dst[i+3] = 255;
                }
            }
            xcb_shm_put_image(surf->xcb_conn, surf->xcb_win, sc->xcb_gc,
                              (uint16_t)w, (uint16_t)h, 0, 0, (uint16_t)w, (uint16_t)h,
                              0, 0, 24, XCB_IMAGE_FORMAT_Z_PIXMAP, 0, sc->shm_seg, 0);
            xcb_flush(surf->xcb_conn);
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
        std::vector<uint8_t> tmp;
        if (need_swap) {
            tmp.resize(bytes);
            for (size_t i = 0; i < bytes; i += 4) {
                tmp[i+0] = src[i+2]; tmp[i+1] = src[i+1]; tmp[i+2] = src[i+0]; tmp[i+3] = 255;
            }
            src = tmp.data();
        }
        GC gc = XCreateGC(surf->xlib_dpy, surf->xlib_win, 0, nullptr);
        int screen = DefaultScreen(surf->xlib_dpy);
        Visual* vis = DefaultVisual(surf->xlib_dpy, screen);
        XImage* xi = XCreateImage(surf->xlib_dpy, vis, 24, ZPixmap, 0, (char*)src, w, h, 32, (int)(w * 4));
        if (xi) {
            XPutImage(surf->xlib_dpy, surf->xlib_win, gc, xi, 0, 0, 0, 0, w, h);
            xi->data = nullptr;
            XDestroyImage(xi);
        }
        XFreeGC(surf->xlib_dpy, gc);
        XFlush(surf->xlib_dpy);
    }
    sc->present_frames++;
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
