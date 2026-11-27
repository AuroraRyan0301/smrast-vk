// smrast_wsi.cpp — Headless XCB & Xlib WSI Presentation Bridge for SmRast-VK
// Copies H20 HBM3 swapchain framebuffer -> host staging buffer via pure Compute Queue (qf[2])
// and blits directly into Xvfb / X11 windows via xcb_put_image or XPutImage!

#include "smrast_internal.h"
#include <xcb/xcb.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstdio>
#include <cstring>

void sm_wsi_blit_to_surface(SmSwapchain* sc, SmImage* img) {
    if (!sc || !sc->surface || !img || !img->host_ptr) return;
    SmDevice* dev = sc->dev;
    uint32_t w = sc->extent.width;
    uint32_t h = sc->extent.height;
    size_t bytes = (size_t)w * h * 4;

    // 1. Copy GPU attachment buffer (or VkImage) -> host-coherent staging buffer via pure Compute Queue!
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
    dev->fn.QueueSubmit(dev->main_queue->real_q, 1, &si, VK_NULL_HANDLE);
    dev->fn.QueueWaitIdle(dev->main_queue->real_q);

    // 2. Blit BGRA/RGBA32 pixels to XCB or Xlib window!
    // X11 24-bit ZPixmap on x86_64 little-endian expects BGRX (same byte layout as VK_FORMAT_B8G8R8A8_UNORM!)
    const uint8_t* src = (const uint8_t*)img->host_ptr;
    std::vector<uint8_t> tmp;
    if (sc->format == VK_FORMAT_R8G8B8A8_UNORM || sc->format == VK_FORMAT_R8G8B8A8_SRGB) {
        tmp.resize(bytes);
        for (size_t i = 0; i < bytes; i += 4) {
            tmp[i+0] = src[i+2];
            tmp[i+1] = src[i+1];
            tmp[i+2] = src[i+0];
            tmp[i+3] = 255;
        }
        src = tmp.data();
    }

    SmSurface* surf = sc->surface;
    if (surf->is_xcb && surf->xcb_conn) {
        xcb_gcontext_t gc = xcb_generate_id(surf->xcb_conn);
        xcb_create_gc(surf->xcb_conn, gc, surf->xcb_win, 0, nullptr);
        // Chunk by strips so we never exceed XCB max request size (~16MB)
        const uint32_t strip_h = 64;
        for (uint32_t y = 0; y < h; y += strip_h) {
            uint32_t ch = (y + strip_h <= h) ? strip_h : (h - y);
            xcb_put_image(surf->xcb_conn, XCB_IMAGE_FORMAT_Z_PIXMAP, surf->xcb_win, gc,
                          (uint16_t)w, (uint16_t)ch, 0, (int16_t)y, 0, 24,
                          (uint32_t)(ch * w * 4), src + (size_t)y * w * 4);
        }
        xcb_free_gc(surf->xcb_conn, gc);
        xcb_flush(surf->xcb_conn);
    } else if (!surf->is_xcb && surf->xlib_dpy) {
        GC gc = XCreateGC(surf->xlib_dpy, surf->xlib_win, 0, nullptr);
        int screen = DefaultScreen(surf->xlib_dpy);
        Visual* vis = DefaultVisual(surf->xlib_dpy, screen);
        // XCreateImage does not copy data; clear xi->data before XDestroyImage so it doesn't free our buffer
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
}
