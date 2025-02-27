/*
 * vaapi_driver.c - Minimal VAAPI driver interface for NVENC-based encoding
 *
 * This file implements the essential VAAPI functions required by clients like FFmpeg.
 * It forwards configuration calls to your NVENC integration (e.g. via nvenc_vaCreateConfig)
 * and provides dummy implementations for surface and context management.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <va/va.h>
#include "nvenc.h"
#include "utils.h"

/* Define maximum numbers for dummy surface and context pools */
#define MAX_SURFACES 256
#define MAX_CONTEXTS 16

/* Dummy surface structure */
typedef struct {
    VASurfaceID surface_id;
    unsigned int width;
    unsigned int height;
    int allocated;
} Surface;

/* Dummy context structure */
typedef struct {
    VAContextID context_id;
    VAConfigID config_id;
    int picture_width;
    int picture_height;
    VASurfaceID *render_targets;
    int num_render_targets;
    int allocated;
} Context;

/* Global pools and counters for surfaces and contexts */
static Surface surface_pool[MAX_SURFACES];
static int next_surface_id = 1;

static Context context_pool[MAX_CONTEXTS];
static int next_context_id = 1;

/* vaInitialize: Initialize the VAAPI driver.
 * Returns major/minor version numbers (here we use 1.12 as an example).
 */
VAStatus vaInitialize(VADisplay dpy, int *major_version, int *minor_version)
{
    if (major_version)
        *major_version = 1;
    if (minor_version)
        *minor_version = 12;
    log_info("vaInitialize called");
    return VA_STATUS_SUCCESS;
}

/* vaTerminate: Terminate the VAAPI driver.
 */
VAStatus vaTerminate(VADisplay dpy)
{
    log_info("vaTerminate called");
    return VA_STATUS_SUCCESS;
}

/* vaQueryConfigProfiles: Return a list of supported profiles.
 * Here we support H.264 (Main, High, ConstrainedBaseline), HEVC Main, and AV1 Main.
 */
VAStatus vaQueryConfigProfiles(VADisplay dpy, VAProfile **profile_list, int *num_profiles)
{
    static VAProfile supported_profiles[] = {
        VAProfileH264Main,
        VAProfileH264High,
        VAProfileH264ConstrainedBaseline,
        VAProfileHEVCMain,
        VAProfileAV1Main
    };
    int count = sizeof(supported_profiles) / sizeof(supported_profiles[0]);
    if (!profile_list || !num_profiles)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    *profile_list = supported_profiles;
    *num_profiles = count;
    log_info("vaQueryConfigProfiles called, returning %d profiles", count);
    return VA_STATUS_SUCCESS;
}

/* vaCreateConfig: Create a VAAPI configuration.
 * This function simply forwards the call to your NVENC-based implementation.
 */
VAStatus vaCreateConfig(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                        VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id)
{
    if (!config_id)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    VAStatus status = nvenc_vaCreateConfig(dpy, profile, entrypoint, attrib_list, num_attribs, config_id);
    log_info("vaCreateConfig called, config_id=%u, status=%d", *config_id, status);
    return status;
}

/* vaGetConfigAttributes: Get configuration attributes.
 * Again, this forwards the call to your NVENC-based implementation.
 */
VAStatus vaGetConfigAttributes(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                               VAConfigAttrib *attrib_list, int num_attribs)
{
    VAStatus status = nvenc_vaGetConfigAttributes(dpy, profile, entrypoint, attrib_list, num_attribs);
    log_info("vaGetConfigAttributes called, status=%d", status);
    return status;
}

/* vaDestroyConfig: Destroy a configuration (stub implementation).
 */
VAStatus vaDestroyConfig(VADisplay dpy, VAConfigID config_id)
{
    log_info("vaDestroyConfig called, config_id=%u", config_id);
    return VA_STATUS_SUCCESS;
}

/* vaCreateSurfaces: Create one or more surfaces.
 * This dummy implementation allocates surface IDs from a pool.
 */
VAStatus vaCreateSurfaces(VADisplay dpy, unsigned int format, unsigned int width,
                          unsigned int height, VASurfaceID *surfaces, unsigned int num_surfaces,
                          VASurfaceAttrib *attrib_list, unsigned int num_attribs)
{
    if (!surfaces)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    for (unsigned int i = 0; i < num_surfaces; i++) {
        int found = 0;
        for (int j = 0; j < MAX_SURFACES; j++) {
            if (!surface_pool[j].allocated) {
                surface_pool[j].allocated = 1;
                surface_pool[j].surface_id = next_surface_id++;
                surface_pool[j].width = width;
                surface_pool[j].height = height;
                surfaces[i] = surface_pool[j].surface_id;
                found = 1;
                break;
            }
        }
        if (!found)
            return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }
    log_info("vaCreateSurfaces called, created %u surfaces", num_surfaces);
    return VA_STATUS_SUCCESS;
}

/* vaDestroySurfaces: Destroy the given surfaces.
 */
VAStatus vaDestroySurfaces(VADisplay dpy, VASurfaceID *surfaces, unsigned int num_surfaces)
{
    if (!surfaces)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    for (unsigned int i = 0; i < num_surfaces; i++) {
        for (int j = 0; j < MAX_SURFACES; j++) {
            if (surface_pool[j].allocated && surface_pool[j].surface_id == surfaces[i]) {
                surface_pool[j].allocated = 0;
                break;
            }
        }
    }
    log_info("vaDestroySurfaces called, destroyed %u surfaces", num_surfaces);
    return VA_STATUS_SUCCESS;
}

/* vaCreateContext: Create a context for rendering.
 * This dummy implementation allocates a context from a pool.
 */
VAStatus vaCreateContext(VADisplay dpy, VAConfigID config_id, int picture_width, int picture_height,
                         int flag, VASurfaceID *render_targets, int num_render_targets, VAContextID *context)
{
    if (!context)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (!context_pool[i].allocated) {
            context_pool[i].allocated = 1;
            context_pool[i].context_id = next_context_id++;
            context_pool[i].config_id = config_id;
            context_pool[i].picture_width = picture_width;
            context_pool[i].picture_height = picture_height;
            context_pool[i].render_targets = render_targets;
            context_pool[i].num_render_targets = num_render_targets;
            *context = context_pool[i].context_id;
            log_info("vaCreateContext called, context_id=%u", *context);
            return VA_STATUS_SUCCESS;
        }
    }
    return VA_STATUS_ERROR_ALLOCATION_FAILED;
}

/* vaDestroyContext: Destroy a previously created context.
 */
VAStatus vaDestroyContext(VADisplay dpy, VAContextID context)
{
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (context_pool[i].allocated && context_pool[i].context_id == context) {
            context_pool[i].allocated = 0;
            log_info("vaDestroyContext called, context_id=%u", context);
            return VA_STATUS_SUCCESS;
        }
    }
    return VA_STATUS_ERROR_INVALID_CONTEXT;
}

/* vaBeginPicture: Begin rendering to a surface.
 */
VAStatus vaBeginPicture(VADisplay dpy, VAContextID context, VASurfaceID surface)
{
    log_info("vaBeginPicture called, context_id=%u, surface=%u", context, surface);
    return VA_STATUS_SUCCESS;
}

/* vaRenderPicture: Submit buffers for rendering.
 */
VAStatus vaRenderPicture(VADisplay dpy, VAContextID context, VABufferID *buffers, int num_buffers)
{
    log_info("vaRenderPicture called, context_id=%u, num_buffers=%d", context, num_buffers);
    return VA_STATUS_SUCCESS;
}

/* vaEndPicture: End rendering to the current surface.
 */
VAStatus vaEndPicture(VADisplay dpy, VAContextID context)
{
    log_info("vaEndPicture called, context_id=%u", context);
    return VA_STATUS_SUCCESS;
}

/* vaSyncSurface: Wait for the given surface to become idle.
 */
VAStatus vaSyncSurface(VADisplay dpy, VASurfaceID surface)
{
    log_info("vaSyncSurface called, surface=%u", surface);
    return VA_STATUS_SUCCESS;
}
