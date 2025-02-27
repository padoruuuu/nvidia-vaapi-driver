#define _GNU_SOURCE

/*
 * vabackend.c - NVIDIA VA-API driver backend for NVDEC, with integrated
 * VAAPI-driver functionality.
 *
 * This file implements the VAAPI interface entry points (vaInitialize, vaTerminate,
 * vaQueryConfigProfiles, vaCreateConfig, vaGetConfigAttributes, vaCreateSurfaces,
 * vaCreateContext, vaBeginPicture, vaRenderPicture, vaEndPicture, vaSyncSurface, etc.)
 * that are used by clients (e.g. FFmpeg) and also shared with the NVENC portion.
 * In future you might refactor common routines into a separate module for both NVDEC and NVENC.
 */

#include "vabackend.h"
#include "backend-common.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <fcntl.h>
#include <sys/param.h>
#include <va/va_backend.h>
#include <va/va_drmcommon.h>
#include <drm_fourcc.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdarg.h>
#include <time.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_include
#define __has_include(x) 0
#endif

#if __has_include(<pthread_np.h>)
#include <pthread_np.h>
#define gettid pthread_getthreadid_np
#define HAVE_GETTID 1
#endif

#ifndef HAVE_GETTID
#include <sys/syscall.h>
#ifdef __BIONIC__
#define HAVE_GETTID 1
#elif !defined(__GLIBC_PREREQ)
#define HAVE_GETTID 0
#elif !__GLIBC_PREREQ(2,30)
#define HAVE_GETTID 0
#else
#define HAVE_GETTID 1
#endif
#endif

static pid_t nv_gettid(void)
{
#if HAVE_GETTID
    return gettid();
#else
    return syscall(__NR_gettid);
#endif
}

static pthread_mutex_t concurrency_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t instances;
static uint32_t max_instances;

static CudaFunctions *cu;
static CuvidFunctions *cv;

extern const NVCodec __start_nvd_codecs[];
extern const NVCodec __stop_nvd_codecs[];

static FILE *LOG_OUTPUT;

static int gpu = -1;
static enum {
    EGL, DIRECT
} backend = DIRECT;

const NVFormatInfo formatsInfo[] =
{
    [NV_FORMAT_NONE] = {0},
    [NV_FORMAT_NV12] = {1, 2, DRM_FORMAT_NV12,     false, false, {{1, DRM_FORMAT_R8,       {0,0}}, {2, DRM_FORMAT_RG88,   {1,1}}},                            {VA_FOURCC_NV12, VA_LSB_FIRST,   12, 0,0,0,0,0}},
    [NV_FORMAT_P010] = {2, 2, DRM_FORMAT_P010,     true,  false, {{1, DRM_FORMAT_R16,      {0,0}}, {2, DRM_FORMAT_RG1616, {1,1}}},                            {VA_FOURCC_P010, VA_LSB_FIRST,   24, 0,0,0,0,0}},
    [NV_FORMAT_P012] = {2, 2, DRM_FORMAT_P012,     true,  false, {{1, DRM_FORMAT_R16,      {0,0}}, {2, DRM_FORMAT_RG1616, {1,1}}},                            {VA_FOURCC_P012, VA_LSB_FIRST,   24, 0,0,0,0,0}},
    [NV_FORMAT_P016] = {2, 2, DRM_FORMAT_P016,     true,  false, {{1, DRM_FORMAT_R16,      {0,0}}, {2, DRM_FORMAT_RG1616, {1,1}}},                            {VA_FOURCC_P016, VA_LSB_FIRST,   24, 0,0,0,0,0}},
    [NV_FORMAT_444P] = {1, 3, DRM_FORMAT_YUV444,   false, true,  {{1, DRM_FORMAT_R8,       {0,0}}, {1, DRM_FORMAT_R8,     {0,0}}, {1, DRM_FORMAT_R8, {0,0}}}, {VA_FOURCC_444P, VA_LSB_FIRST,   24, 0,0,0,0,0}},
#if VA_CHECK_VERSION(1, 20, 0)
    [NV_FORMAT_Q416] = {2, 3, DRM_FORMAT_INVALID,  true,  true,  {{1, DRM_FORMAT_R16,      {0,0}}, {1, DRM_FORMAT_R16,    {0,0}}, {1, DRM_FORMAT_R16,{0,0}}}, {VA_FOURCC_Q416, VA_LSB_FIRST,   48, 0,0,0,0,0}},
#endif
};

static NVFormat nvFormatFromVaFormat(uint32_t fourcc) {
    for (uint32_t i = NV_FORMAT_NONE + 1; i < ARRAY_SIZE(formatsInfo); i++) {
        if (formatsInfo[i].vaFormat.fourcc == fourcc) {
            return i;
        }
    }
    return NV_FORMAT_NONE;
}

/* ======================================================================
   Shared VAAPI-Driver Functionality (integrated from vaapi-driver.c)
   These functions implement the core VAAPI entry points that are used by
   clients (e.g. FFmpeg) and also serve as the common interface between
   the NVDEC and (potentially) the NVENC portions.
   ====================================================================== */

/* vaInitialize - initialize the driver.
   (Note: VAAPI clients call __vaDriverInit_1_0 to load the driver.)
*/
VAStatus vaInitialize(VADisplay dpy, int *major_version, int *minor_version)
{
    if (major_version)
        *major_version = 1;
    if (minor_version)
        *minor_version = 12;
    LOG("vaInitialize called");
    return VA_STATUS_SUCCESS;
}

VAStatus vaTerminate(VADisplay dpy)
{
    LOG("vaTerminate called");
    return VA_STATUS_SUCCESS;
}

/* ======================================================================
   End of shared VAAPI-driver functionality
   ====================================================================== */

/* ======================================================================
   NVDEC-specific implementations follow.
   (The remaining functions implement the VAAPI backend interface.)
   ====================================================================== */

__attribute__ ((constructor))
static void init() {
    char *nvdLog = getenv("NVD_LOG");
    if (nvdLog != NULL) {
        if (strcmp(nvdLog, "1") == 0) {
            LOG_OUTPUT = stdout;
        } else {
            LOG_OUTPUT = fopen(nvdLog, "a");
            if (LOG_OUTPUT == NULL) {
                LOG_OUTPUT = stdout;
            }
        }
    }

    char *nvdGpu = getenv("NVD_GPU");
    if (nvdGpu != NULL) {
        gpu = atoi(nvdGpu);
    }

    char *nvdMaxInstances = getenv("NVD_MAX_INSTANCES");
    if (nvdMaxInstances != NULL) {
        max_instances = atoi(nvdMaxInstances);
    }

    char *nvdBackend = getenv("NVD_BACKEND");
    if (nvdBackend != NULL) {
        if (strncmp(nvdBackend, "direct", 6) == 0) {
            backend = DIRECT;
        } else if (strncmp(nvdBackend, "egl", 6) == 0) {
            backend = EGL;
        }
    }

    // Try to detect the Firefox sandbox and skip loading CUDA if detected.
    int fd = open("/proc/version", O_RDONLY);
    if (fd < 0) {
        LOG("ERROR: Potential Firefox sandbox detected, failing to init!");
        LOG("If running in Firefox, set env var MOZ_DISABLE_RDD_SANDBOX=1 to disable sandbox.");
        if (getenv("NVD_FORCE_INIT") == NULL) {
            return;
        }
    } else {
        close(fd);
    }

    // Initialize the CUDA and NVDEC functions.
    int ret = cuda_load_functions(&cu, NULL);
    if (ret != 0) {
        cu = NULL;
        LOG("Failed to load CUDA functions");
        return;
    }
    ret = cuvid_load_functions(&cv, NULL);
    if (ret != 0) {
        cv = NULL;
        LOG("Failed to load NVDEC functions");
        return;
    }

    CHECK_CUDA_RESULT(cu->cuInit(0));
}

__attribute__ ((destructor))
static void cleanup() {
    if (cv != NULL) {
        cuvid_free_functions(&cv);
    }
    if (cu != NULL) {
        cuda_free_functions(&cu);
    }
}

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(gnu_printf) || (defined(__GNUC__) && !defined(__clang__))
__attribute((format(gnu_printf, 4, 5)))
#endif
void logger(const char *filename, const char *function, int line, const char *msg, ...) {
    if (LOG_OUTPUT == 0) {
        return;
    }

    va_list argList;
    char formattedMessage[1024];

    va_start(argList, msg);
    vsnprintf(formattedMessage, 1024, msg, argList);
    va_end(argList);

    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);

    fprintf(LOG_OUTPUT, "%10ld.%09ld [%d-%d] %s:%4d %24s %s\n", (long)tp.tv_sec, tp.tv_nsec, getpid(), nv_gettid(), filename, line, function, formattedMessage);
    fflush(LOG_OUTPUT);
}

bool checkCudaErrors(CUresult err, const char *file, const char *function, const int line) {
    if (CUDA_SUCCESS != err) {
        const char *errStr = NULL;
        cu->cuGetErrorString(err, &errStr);
        logger(file, function, line, "CUDA ERROR '%s' (%d)\n", errStr, err);
        return true;
    }
    return false;
}

void appendBuffer(AppendableBuffer *ab, const void *buf, uint64_t size) {
    if (ab->buf == NULL) {
        ab->allocated = size * 2;
        ab->buf = memalign(16, ab->allocated);
        ab->size = 0;
    } else if (ab->size + size > ab->allocated) {
        while (ab->size + size > ab->allocated) {
            ab->allocated += ab->allocated >> 1;
        }
        void *nb = memalign(16, ab->allocated);
        memcpy(nb, ab->buf, ab->size);
        free(ab->buf);
        ab->buf = nb;
    }
    memcpy((char*)ab->buf + ab->size, buf, size);
    ab->size += size;
}

static void freeBuffer(AppendableBuffer *ab) {
    if (ab->buf != NULL) {
        free(ab->buf);
        ab->buf = NULL;
        ab->size = 0;
        ab->allocated = 0;
    }
}

static Object allocateObject(NVDriver *drv, ObjectType type, int allocatePtrSize) {
    Object newObj = (Object) calloc(1, sizeof(struct Object_t));
    newObj->type = type;
    newObj->id = (++drv->nextObjId);
    if (allocatePtrSize > 0) {
        newObj->obj = calloc(1, allocatePtrSize);
    }
    pthread_mutex_lock(&drv->objectCreationMutex);
    add_element(&drv->objects, newObj);
    pthread_mutex_unlock(&drv->objectCreationMutex);
    return newObj;
}

static Object getObject(NVDriver *drv, VAGenericID id) {
    Object ret = NULL;
    if (id != VA_INVALID_ID) {
        pthread_mutex_lock(&drv->objectCreationMutex);
        ARRAY_FOR_EACH(Object, o, &drv->objects)
            if (o->id == id) {
                ret = o;
                break;
            }
        END_FOR_EACH
        pthread_mutex_unlock(&drv->objectCreationMutex);
    }
    return ret;
}

static void* getObjectPtr(NVDriver *drv, VAGenericID id) {
    if (id != VA_INVALID_ID) {
        Object o = getObject(drv, id);
        if (o != NULL) {
            return o->obj;
        }
    }
    return NULL;
}

static Object getObjectByPtr(NVDriver *drv, void *ptr) {
    Object ret = NULL;
    if (ptr != NULL) {
        pthread_mutex_lock(&drv->objectCreationMutex);
        ARRAY_FOR_EACH(Object, o, &drv->objects)
            if (o->obj == ptr) {
                ret = o;
                break;
            }
        END_FOR_EACH
        pthread_mutex_unlock(&drv->objectCreationMutex);
    }
    return ret;
}

static void deleteObject(NVDriver *drv, VAGenericID id) {
    if (id == VA_INVALID_ID) {
        return;
    }
    pthread_mutex_lock(&drv->objectCreationMutex);
    ARRAY_FOR_EACH(Object, o, &drv->objects)
        if (o->id == id) {
            remove_element_at(&drv->objects, o_idx);
            free(o->obj);
            free(o);
            break;
        }
    END_FOR_EACH
    pthread_mutex_unlock(&drv->objectCreationMutex);
}

static bool destroyContext(NVDriver *drv, NVContext *nvCtx) {
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), false);
    LOG("Signaling resolve thread to exit");
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 5;
    nvCtx->exiting = true;
    pthread_cond_signal(&nvCtx->resolveCondition);
    LOG("Waiting for resolve thread to exit");
    int ret = pthread_timedjoin_np(nvCtx->resolveThread, NULL, &timeout);
    LOG("pthread_timedjoin_np finished with %d", ret);
    freeBuffer(&nvCtx->sliceOffsets);
    freeBuffer(&nvCtx->bitstreamBuffer);
    bool successful = true;
    if (nvCtx->decoder != NULL) {
        CUresult result = cv->cuvidDestroyDecoder(nvCtx->decoder);
        if (result != CUDA_SUCCESS) {
            LOG("cuvidDestroyDecoder failed: %d", result);
            successful = false;
        }
    }
    nvCtx->decoder = NULL;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), false);
    return successful;
}

static void deleteAllObjects(NVDriver *drv) {
    pthread_mutex_lock(&drv->objectCreationMutex);
    ARRAY_FOR_EACH(Object, o, &drv->objects)
        LOG("Found object %d or type %d", o->id, o->type);
        if (o->type == OBJECT_TYPE_CONTEXT) {
            destroyContext(drv, (NVContext*) o->obj);
            deleteObject(drv, o->id);
        }
    END_FOR_EACH
    pthread_mutex_unlock(&drv->objectCreationMutex);
}

NVSurface* nvSurfaceFromSurfaceId(NVDriver *drv, VASurfaceID surf) {
    Object obj = getObject(drv, surf);
    if (obj != NULL && obj->type == OBJECT_TYPE_SURFACE) {
        NVSurface *suf = (NVSurface*) obj->obj;
        return suf;
    }
    return NULL;
}

int pictureIdxFromSurfaceId(NVDriver *drv, VASurfaceID surfId) {
    NVSurface *surf = nvSurfaceFromSurfaceId(drv, surfId);
    if (surf != NULL) {
        return surf->pictureIdx;
    }
    return -1;
}

static cudaVideoCodec vaToCuCodec(VAProfile profile) {
    for (const NVCodec *c = __start_nvd_codecs; c < __stop_nvd_codecs; c++) {
        cudaVideoCodec cvc = c->computeCudaCodec(profile);
        if (cvc != cudaVideoCodec_NONE) {
            return cvc;
        }
    }
    return cudaVideoCodec_NONE;
}

static bool doesGPUSupportCodec(cudaVideoCodec codec, int bitDepth, cudaVideoChromaFormat chromaFormat, uint32_t *width, uint32_t *height)
{
    CUVIDDECODECAPS videoDecodeCaps = {
        .eCodecType      = codec,
        .eChromaFormat   = chromaFormat,
        .nBitDepthMinus8 = bitDepth - 8
    };
    CHECK_CUDA_RESULT_RETURN(cv->cuvidGetDecoderCaps(&videoDecodeCaps), false);
    if (width != NULL) {
        *width = videoDecodeCaps.nMaxWidth;
    }
    if (height != NULL) {
        *height = videoDecodeCaps.nMaxHeight;
    }
    return (videoDecodeCaps.bIsSupported == 1);
}

static void* resolveSurfaces(void *param) {
    NVContext *ctx = (NVContext*) param;
    NVDriver *drv = ctx->drv;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), NULL);
    LOG("[RT] Resolve thread for %p started", ctx);
    while (!ctx->exiting) {
        pthread_mutex_lock(&ctx->resolveMutex);
        while (ctx->surfaceQueueReadIdx == ctx->surfaceQueueWriteIdx) {
            pthread_cond_wait(&ctx->resolveCondition, &ctx->resolveMutex);
            if (ctx->exiting) {
                pthread_mutex_unlock(&ctx->resolveMutex);
                goto out;
            }
        }
        pthread_mutex_unlock(&ctx->resolveMutex);
        NVSurface *surface = ctx->surfaceQueue[ctx->surfaceQueueReadIdx++];
        if (ctx->surfaceQueueReadIdx >= SURFACE_QUEUE_SIZE) {
            ctx->surfaceQueueReadIdx = 0;
        }
        if (surface->decodeFailed) {
            pthread_mutex_lock(&surface->mutex);
            surface->resolving = 0;
            pthread_cond_signal(&surface->cond);
            pthread_mutex_unlock(&surface->mutex);
            continue;
        }
        CUdeviceptr deviceMemory = (CUdeviceptr) NULL;
        unsigned int pitch = 0;
        CUVIDPROCPARAMS procParams = {
            .progressive_frame = surface->progressiveFrame,
            .top_field_first = surface->topFieldFirst,
            .second_field = surface->secondField
        };
        if (CHECK_CUDA_RESULT(cv->cuvidMapVideoFrame(ctx->decoder, surface->pictureIdx, &deviceMemory, &pitch, &procParams))) {
            pthread_mutex_lock(&surface->mutex);
            surface->resolving = 0;
            pthread_cond_signal(&surface->cond);
            pthread_mutex_unlock(&surface->mutex);
            continue;
        }
        drv->backend->exportCudaPtr(drv, deviceMemory, surface, pitch);
        CHECK_CUDA_RESULT(cv->cuvidUnmapVideoFrame(ctx->decoder, deviceMemory));
    }
out:
    LOG("[RT] Resolve thread for %p exiting", ctx);
    return NULL;
}

#define MAX_PROFILES 32
static VAStatus nvQueryConfigProfiles(
        VADriverContextP ctx,
        VAProfile *profile_list,	/* out */
        int *num_profiles			/* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    int profiles = 0;
    if (doesGPUSupportCodec(cudaVideoCodec_MPEG2, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileMPEG2Simple;
        profile_list[profiles++] = VAProfileMPEG2Main;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_MPEG4, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileMPEG4Simple;
        profile_list[profiles++] = VAProfileMPEG4AdvancedSimple;
        profile_list[profiles++] = VAProfileMPEG4Main;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_VC1, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileVC1Simple;
        profile_list[profiles++] = VAProfileVC1Main;
        profile_list[profiles++] = VAProfileVC1Advanced;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_H264, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileH264Main;
        profile_list[profiles++] = VAProfileH264High;
        profile_list[profiles++] = VAProfileH264ConstrainedBaseline;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_JPEG, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileJPEGBaseline;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_H264_SVC, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileH264StereoHigh;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_H264_MVC, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileH264MultiviewHigh;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileHEVCMain;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_VP8, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileVP8Version0_3;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_VP9, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileVP9Profile0;
    }
    if (doesGPUSupportCodec(cudaVideoCodec_AV1, 8, cudaVideoChromaFormat_420, NULL, NULL)) {
        profile_list[profiles++] = VAProfileAV1Profile0;
    }
    if (drv->supports16BitSurface) {
        if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 10, cudaVideoChromaFormat_420, NULL, NULL)) {
            profile_list[profiles++] = VAProfileHEVCMain10;
        }
        if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 12, cudaVideoChromaFormat_420, NULL, NULL)) {
            profile_list[profiles++] = VAProfileHEVCMain12;
        }
        if (doesGPUSupportCodec(cudaVideoCodec_VP9, 10, cudaVideoChromaFormat_420, NULL, NULL)) {
            profile_list[profiles++] = VAProfileVP9Profile2;
        }
    }
    if (drv->supports444Surface) {
        if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 8, cudaVideoChromaFormat_444, NULL, NULL)) {
            profile_list[profiles++] = VAProfileHEVCMain444;
        }
        if (doesGPUSupportCodec(cudaVideoCodec_VP9, 8, cudaVideoChromaFormat_444, NULL, NULL)) {
            profile_list[profiles++] = VAProfileVP9Profile1;
        }
        if (doesGPUSupportCodec(cudaVideoCodec_AV1, 8, cudaVideoChromaFormat_444, NULL, NULL)) {
            profile_list[profiles++] = VAProfileAV1Profile1;
        }
#if VA_CHECK_VERSION(1, 20, 0)
        if (drv->supports16BitSurface) {
            if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 10, cudaVideoChromaFormat_444, NULL, NULL)) {
                profile_list[profiles++] = VAProfileHEVCMain444_10;
            }
            if (doesGPUSupportCodec(cudaVideoCodec_HEVC, 12, cudaVideoChromaFormat_444, NULL, NULL)) {
                profile_list[profiles++] = VAProfileHEVCMain444_12;
            }
            if (doesGPUSupportCodec(cudaVideoCodec_VP9, 10, cudaVideoChromaFormat_444, NULL, NULL)) {
                profile_list[profiles++] = VAProfileVP9Profile3;
            }
        }
#endif
    }
    for (int i = 0; i < profiles; i++) {
        if (vaToCuCodec(profile_list[i]) == cudaVideoCodec_NONE) {
            for (int x = i; x < profiles-1; x++) {
                profile_list[x] = profile_list[x+1];
            }
            profiles--;
            i--;
        }
    }
    *num_profiles = profiles;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvQueryConfigEntrypoints(
        VADriverContextP ctx,
        VAProfile profile,
        VAEntrypoint  *entrypoint_list,	/* out */
        int *num_entrypoints			/* out */
    )
{
    entrypoint_list[0] = VAEntrypointVLD;
    *num_entrypoints = 1;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvGetConfigAttributes(
        VADriverContextP ctx,
        VAProfile profile,
        VAEntrypoint entrypoint,
        VAConfigAttrib *attrib_list,	/* in/out */
        int num_attribs
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    if (vaToCuCodec(profile) == cudaVideoCodec_NONE) {
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    LOG("Got here with profile: %d == %d", profile, vaToCuCodec(profile));
    for (int i = 0; i < num_attribs; i++) {
        if (attrib_list[i].type == VAConfigAttribRTFormat) {
            attrib_list[i].value = VA_RT_FORMAT_YUV420;
            switch (profile) {
            case VAProfileHEVCMain12:
            case VAProfileVP9Profile2:
                attrib_list[i].value |= VA_RT_FORMAT_YUV420_12;
            case VAProfileHEVCMain10:
            case VAProfileAV1Profile0:
                attrib_list[i].value |= VA_RT_FORMAT_YUV420_10;
                break;
            case VAProfileHEVCMain444_12:
            case VAProfileVP9Profile3:
                attrib_list[i].value |= VA_RT_FORMAT_YUV444_12 | VA_RT_FORMAT_YUV420_12;
            case VAProfileHEVCMain444_10:
            case VAProfileAV1Profile1:
                attrib_list[i].value |= VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV420_10;
            case VAProfileHEVCMain444:
            case VAProfileVP9Profile1:
                attrib_list[i].value |= VA_RT_FORMAT_YUV444;
                break;
            default:
                break;
            }
            if (!drv->supports16BitSurface) {
                attrib_list[i].value &= ~(VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV420_12 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12);
            }
            if (!drv->supports444Surface) {
                attrib_list[i].value &= ~(VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12);
            }
        } else if (attrib_list[i].type == VAConfigAttribMaxPictureWidth) {
            doesGPUSupportCodec(vaToCuCodec(profile), 8, cudaVideoChromaFormat_420, &attrib_list[i].value, NULL);
        } else if (attrib_list[i].type == VAConfigAttribMaxPictureHeight) {
            doesGPUSupportCodec(vaToCuCodec(profile), 8, cudaVideoChromaFormat_420, NULL, &attrib_list[i].value);
        } else {
            LOG("unhandled config attribute: %d", attrib_list[i].type);
        }
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateConfig(
        VADriverContextP ctx,
        VAProfile profile,
        VAEntrypoint entrypoint,
        VAConfigAttrib *attrib_list,
        int num_attribs,
        VAConfigID *config_id		/* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    LOG("got profile: %d with %d attributes", profile, num_attribs);
    cudaVideoCodec cudaCodec = vaToCuCodec(profile);
    if (cudaCodec == cudaVideoCodec_NONE) {
        LOG("Profile not supported: %d", profile);
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    if (entrypoint != VAEntrypointVLD) {
        LOG("Entrypoint not supported: %d", entrypoint);
        return VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT;
    }
    Object obj = allocateObject(drv, OBJECT_TYPE_CONFIG, sizeof(NVConfig));
    NVConfig *cfg = (NVConfig*) obj->obj;
    cfg->profile = profile;
    cfg->entrypoint = entrypoint;
    for (int i = 0; i < num_attribs; i++) {
      LOG("got config attrib: %d %d %d", i, attrib_list[i].type, attrib_list[i].value);
    }
    cfg->cudaCodec = cudaCodec;
    cfg->chromaFormat = cudaVideoChromaFormat_420;
    cfg->surfaceFormat = cudaVideoSurfaceFormat_NV12;
    cfg->bitDepth = 8;
    if (drv->supports16BitSurface) {
        switch(cfg->profile) {
        case VAProfileHEVCMain10:
            cfg->surfaceFormat = cudaVideoSurfaceFormat_P016;
            cfg->bitDepth = 10;
            break;
        case VAProfileHEVCMain12:
            cfg->surfaceFormat = cudaVideoSurfaceFormat_P016;
            cfg->bitDepth = 12;
            break;
        case VAProfileVP9Profile2:
        case VAProfileAV1Profile0:
            if (num_attribs > 0 && attrib_list[0].type == VAConfigAttribRTFormat) {
                switch(attrib_list[0].value) {
                case VA_RT_FORMAT_YUV420_12:
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_P016;
                    cfg->bitDepth = 12;
                    break;
                case VA_RT_FORMAT_YUV420_10:
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_P016;
                    cfg->bitDepth = 10;
                    break;
                default:
                    break;
                }
            } else {
                if (cfg->profile == VAProfileVP9Profile2) {
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_P016;
                    cfg->bitDepth = 10;
                } else {
                    LOG("Unable to determine surface type for VP9/AV1 codec due to no RTFormat specified.");
                }
            }
        default:
            break;
        }
    }
    if (drv->supports444Surface) {
        switch(cfg->profile) {
        case VAProfileHEVCMain444:
        case VAProfileVP9Profile1:
        case VAProfileAV1Profile1:
            cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444;
            cfg->chromaFormat = cudaVideoChromaFormat_444;
            cfg->bitDepth = 8;
            break;
        default:
            break;
        }
    }
    if (drv->supports444Surface && drv->supports16BitSurface) {
        switch(cfg->profile) {
        case VAProfileHEVCMain444_10:
            cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
            cfg->chromaFormat = cudaVideoChromaFormat_444;
            cfg->bitDepth = 10;
            break;
        case VAProfileHEVCMain444_12:
            cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
            cfg->chromaFormat = cudaVideoChromaFormat_444;
            cfg->bitDepth = 12;
            break;
        case VAProfileVP9Profile3:
        case VAProfileAV1Profile1:
            if (num_attribs > 0 && attrib_list[0].type == VAConfigAttribRTFormat) {
                switch(attrib_list[0].value) {
                case VA_RT_FORMAT_YUV444_12:
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
                    cfg->chromaFormat = cudaVideoChromaFormat_444;
                    cfg->bitDepth = 12;
                    break;
                case VA_RT_FORMAT_YUV444_10:
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
                    cfg->chromaFormat = cudaVideoChromaFormat_444;
                    cfg->bitDepth = 10;
                    break;
                case VA_RT_FORMAT_YUV444:
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444;
                    cfg->chromaFormat = cudaVideoChromaFormat_444;
                    cfg->bitDepth = 8;
                    break;
                default:
                    break;
                }
            } else {
                if (cfg->profile == VAProfileVP9Profile3) {
                    cfg->surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
                    cfg->chromaFormat = cudaVideoChromaFormat_444;
                    cfg->bitDepth = 10;
                }
            }
        default:
            break;
        }
    }
    *config_id = obj->id;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvDestroyConfig(
        VADriverContextP ctx,
        VAConfigID config_id
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    deleteObject(drv, config_id);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvQueryConfigAttributes(
        VADriverContextP ctx,
        VAConfigID config_id,
        VAProfile *profile,		/* out */
        VAEntrypoint *entrypoint, 	/* out */
        VAConfigAttrib *attrib_list,	/* out */
        int *num_attribs		/* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVConfig *cfg = (NVConfig*) getObjectPtr(drv, config_id);
    if (cfg == NULL) {
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    *profile = cfg->profile;
    *entrypoint = cfg->entrypoint;
    int i = 0;
    attrib_list[i].value = VA_RT_FORMAT_YUV420;
    attrib_list[i].type = VAConfigAttribRTFormat;
    switch (cfg->profile) {
    case VAProfileHEVCMain12:
    case VAProfileVP9Profile2:
        attrib_list[i].value |= VA_RT_FORMAT_YUV420_12;
    case VAProfileHEVCMain10:
    case VAProfileAV1Profile0:
        attrib_list[i].value |= VA_RT_FORMAT_YUV420_10;
        break;
    case VAProfileHEVCMain444_12:
    case VAProfileVP9Profile3:
        attrib_list[i].value |= VA_RT_FORMAT_YUV444_12 | VA_RT_FORMAT_YUV420_12;
    case VAProfileHEVCMain444_10:
    case VAProfileAV1Profile1:
        attrib_list[i].value |= VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV420_10;
    case VAProfileHEVCMain444:
    case VAProfileVP9Profile1:
        attrib_list[i].value |= VA_RT_FORMAT_YUV444;
        break;
    default:
        break;
    }
    if (!drv->supports16BitSurface) {
        attrib_list[i].value &= ~(VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV420_12 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12);
    }
    if (!drv->supports444Surface) {
        attrib_list[i].value &= ~(VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12);
    }
    i++;
    *num_attribs = i;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateSurfaces2(
            VADriverContextP    ctx,
            unsigned int        format,
            unsigned int        width,
            unsigned int        height,
            VASurfaceID        *surfaces,
            unsigned int        num_surfaces,
            VASurfaceAttrib    *attrib_list,
            unsigned int        num_attribs
        )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    cudaVideoSurfaceFormat nvFormat;
    cudaVideoChromaFormat chromaFormat;
    int bitdepth;
    switch (format)
    {
    case VA_RT_FORMAT_YUV420:
        nvFormat = cudaVideoSurfaceFormat_NV12;
        chromaFormat = cudaVideoChromaFormat_420;
        bitdepth = 8;
        break;
    case VA_RT_FORMAT_YUV420_10:
        nvFormat = cudaVideoSurfaceFormat_P016;
        chromaFormat = cudaVideoChromaFormat_420;
        bitdepth = 10;
        break;
    case VA_RT_FORMAT_YUV420_12:
        nvFormat = cudaVideoSurfaceFormat_P016;
        chromaFormat = cudaVideoChromaFormat_420;
        bitdepth = 12;
        break;
    case VA_RT_FORMAT_YUV444:
        nvFormat = cudaVideoSurfaceFormat_YUV444;
        chromaFormat = cudaVideoChromaFormat_444;
        bitdepth = 8;
        break;
    case VA_RT_FORMAT_YUV444_10:
        nvFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
        chromaFormat = cudaVideoChromaFormat_444;
        bitdepth = 10;
        break;
    case VA_RT_FORMAT_YUV444_12:
        nvFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
        chromaFormat = cudaVideoChromaFormat_444;
        bitdepth = 12;
        break;
    default:
        LOG("Unknown format: %X", format);
        return VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT;
    }
    switch(chromaFormat) {
        case cudaVideoChromaFormat_422:
            width = ROUND_UP(width, 2);
            break;
        case cudaVideoChromaFormat_420:
            width = ROUND_UP(width, 2);
            height = ROUND_UP(height, 2);
            break;
        default:
            break;
    }
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    for (uint32_t i = 0; i < num_surfaces; i++) {
        Object surfaceObject = allocateObject(drv, OBJECT_TYPE_SURFACE, sizeof(NVSurface));
        surfaces[i] = surfaceObject->id;
        NVSurface *suf = (NVSurface*) surfaceObject->obj;
        suf->width = width;
        suf->height = height;
        suf->format = nvFormat;
        suf->pictureIdx = -1;
        suf->bitDepth = bitdepth;
        suf->context = NULL;
        suf->chromaFormat = chromaFormat;
        pthread_mutex_init(&suf->mutex, NULL);
        pthread_cond_init(&suf->cond, NULL);
        LOG("Creating surface %dx%d, format %X (%p)", width, height, format, suf);
    }
    drv->surfaceCount += num_surfaces;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateSurfaces(
        VADriverContextP ctx,
        int width,
        int height,
        int format,
        int num_surfaces,
        VASurfaceID *surfaces		/* out */
    )
{
    return nvCreateSurfaces2(ctx, format, width, height, surfaces, num_surfaces, NULL, 0);
}

static VAStatus nvDestroySurfaces(
        VADriverContextP ctx,
        VASurfaceID *surface_list,
        int num_surfaces
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    for (int i = 0; i < num_surfaces; i++) {
        NVSurface *surface = (NVSurface*) getObjectPtr(drv, surface_list[i]);
        LOG("Destroying surface %d (%p)", surface->pictureIdx, surface);
        drv->backend->detachBackingImageFromSurface(drv, surface);
        deleteObject(drv, surface_list[i]);
    }
    drv->surfaceCount = MAX(drv->surfaceCount - num_surfaces, 0);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateContext(
        VADriverContextP ctx,
        VAConfigID config_id,
        int picture_width,
        int picture_height,
        int flag,
        VASurfaceID *render_targets,
        int num_render_targets,
        VAContextID *context		/* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVConfig *cfg = (NVConfig*) getObjectPtr(drv, config_id);
    if (cfg == NULL) {
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    LOG("creating context with %d render targets, %d surfaces, at %dx%d", num_render_targets, drv->surfaceCount, picture_width, picture_height);
    const NVCodec *selectedCodec = NULL;
    for (const NVCodec *c = __start_nvd_codecs; c < __stop_nvd_codecs; c++) {
        for (int i = 0; i < c->supportedProfileCount; i++) {
            if (c->supportedProfiles[i] == cfg->profile) {
                selectedCodec = c;
                break;
            }
        }
    }
    if (selectedCodec == NULL) {
        LOG("Unable to find codec for profile: %d", cfg->profile);
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    if (num_render_targets) {
        NVSurface *surface = (NVSurface *) getObjectPtr(drv, render_targets[0]);
        if (!surface) {
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        }
        cfg->surfaceFormat = surface->format;
        cfg->chromaFormat = surface->chromaFormat;
        cfg->bitDepth = surface->bitDepth;
    }
    int surfaceCount = num_render_targets > 0 ? num_render_targets : 32;
    if (surfaceCount > 32) {
        LOG("Application requested %d surface(s), limiting to 32. This may cause issues.", surfaceCount);
        surfaceCount = 32;
    }
    int display_area_width = picture_width;
    int display_area_height = picture_height;
    switch(cfg->chromaFormat) {
        case cudaVideoChromaFormat_422:
            display_area_width = ROUND_UP(display_area_width, 2);
            break;
        case cudaVideoChromaFormat_420:
            display_area_width = ROUND_UP(display_area_width, 2);
            display_area_height = ROUND_UP(display_area_height, 2);
            break;
        default:
            break;
    }
    CUVIDDECODECREATEINFO vdci = {
        .ulWidth             = vdci.ulMaxWidth  = vdci.ulTargetWidth  = picture_width,
        .ulHeight            = vdci.ulMaxHeight = vdci.ulTargetHeight = picture_height,
        .CodecType           = cfg->cudaCodec,
        .ulCreationFlags     = cudaVideoCreate_PreferCUVID,
        .ulIntraDecodeOnly   = 0,
        .display_area.right  = display_area_width,
        .display_area.bottom = display_area_height,
        .ChromaFormat        = cfg->chromaFormat,
        .OutputFormat        = cfg->surfaceFormat,
        .bitDepthMinus8      = cfg->bitDepth - 8,
        .DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave,
        .ulNumOutputSurfaces = 1,
        .ulNumDecodeSurfaces = surfaceCount,
    };
    drv->surfaceCount = 0;
    CHECK_CUDA_RESULT_RETURN(cv->cuvidCtxLockCreate(&vdci.vidLock, drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    CUvideodecoder decoder;
    CHECK_CUDA_RESULT_RETURN(cv->cuvidCreateDecoder(&decoder, &vdci), VA_STATUS_ERROR_ALLOCATION_FAILED);
    Object contextObj = allocateObject(drv, OBJECT_TYPE_CONTEXT, sizeof(NVContext));
    NVContext *nvCtx = (NVContext*) contextObj->obj;
    nvCtx->drv = drv;
    nvCtx->decoder = decoder;
    nvCtx->profile = cfg->profile;
    nvCtx->entrypoint = cfg->entrypoint;
    nvCtx->width = picture_width;
    nvCtx->height = picture_height;
    nvCtx->codec = selectedCodec;
    nvCtx->surfaceCount = surfaceCount;
    pthread_mutexattr_t attrib;
    pthread_mutexattr_init(&attrib);
    pthread_mutexattr_settype(&attrib, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&nvCtx->surfaceCreationMutex, &attrib);
    pthread_mutex_init(&nvCtx->resolveMutex, NULL);
    pthread_cond_init(&nvCtx->resolveCondition, NULL);
    int err = pthread_create(&nvCtx->resolveThread, NULL, &resolveSurfaces, nvCtx);
    if (err != 0) {
        LOG("Unable to create resolve thread: %d", err);
        deleteObject(drv, contextObj->id);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    *context = contextObj->id;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvDestroyContext(
        VADriverContextP ctx,
        VAContextID context)
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    LOG("Destroying context: %d", context);
    NVContext *nvCtx = (NVContext*) getObjectPtr(drv, context);
    if (nvCtx == NULL) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    VAStatus ret = VA_STATUS_SUCCESS;
    if (!destroyContext(drv, nvCtx)) {
        ret = VA_STATUS_ERROR_OPERATION_FAILED;
    }
    deleteObject(drv, context);
    return ret;
}

static VAStatus nvCreateBuffer(
        VADriverContextP ctx,
        VAContextID context,		/* in */
        VABufferType type,		/* in */
        unsigned int size,		/* in */
        unsigned int num_elements,	/* in */
        void *data,			/* in */
        VABufferID *buf_id
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVContext *nvCtx = (NVContext*) getObjectPtr(drv, context);
    if (nvCtx == NULL) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    int offset = 0;
    if (nvCtx->profile == VAProfileVP8Version0_3 && type == VASliceDataBufferType) {
        offset = (int)(((uintptr_t)data) & 0xf);
        data = ((char *)data) - offset;
        size += offset;
    }
    Object bufferObject = allocateObject(drv, OBJECT_TYPE_BUFFER, sizeof(NVBuffer));
    *buf_id = bufferObject->id;
    NVBuffer *buf = (NVBuffer*) bufferObject->obj;
    buf->bufferType = type;
    buf->elements = num_elements;
    buf->size = num_elements * size;
    buf->ptr = memalign(16, buf->size);
    buf->offset = offset;
    if (buf->ptr == NULL) {
        LOG("Unable to allocate buffer of %d bytes", buf->size);
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }
    if (data != NULL) {
        memcpy(buf->ptr, data, buf->size);
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus nvBufferSetNumElements(
        VADriverContextP ctx,
        VABufferID buf_id,	/* in */
        unsigned int num_elements	/* in */
    )
{
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvMapBuffer(
        VADriverContextP ctx,
        VABufferID buf_id,	/* in */
        void **pbuf         /* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVBuffer *buf = getObjectPtr(drv, buf_id);
    if (buf == NULL) {
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }
    *pbuf = buf->ptr;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvUnmapBuffer(
        VADriverContextP ctx,
        VABufferID buf_id	/* in */
    )
{
    return VA_STATUS_SUCCESS;
}

static VAStatus nvDestroyBuffer(
        VADriverContextP ctx,
        VABufferID buffer_id
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVBuffer *buf = getObjectPtr(drv, buffer_id);
    if (buf == NULL) {
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }
    if (buf->ptr != NULL) {
        free(buf->ptr);
    }
    deleteObject(drv, buffer_id);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvBeginPicture(
        VADriverContextP ctx,
        VAContextID context,
        VASurfaceID render_target
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVContext *nvCtx = (NVContext*) getObjectPtr(drv, context);
    NVSurface *surface = (NVSurface*) getObjectPtr(drv, render_target);
    if (surface == NULL) {
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    if (surface->context != NULL && surface->context != nvCtx) {
        if (surface->backingImage != NULL) {
            drv->backend->detachBackingImageFromSurface(drv, surface);
        }
        surface->pictureIdx = -1;
    }
    if (surface->pictureIdx == -1) {
        if (nvCtx->currentPictureId == nvCtx->surfaceCount) {
            return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
        }
        surface->pictureIdx = nvCtx->currentPictureId++;
    }
    pthread_mutex_lock(&surface->mutex);
    surface->resolving = 1;
    pthread_mutex_unlock(&surface->mutex);
    memset(&nvCtx->pPicParams, 0, sizeof(CUVIDPICPARAMS));
    nvCtx->renderTarget = surface;
    nvCtx->renderTarget->progressiveFrame = true;
    nvCtx->pPicParams.CurrPicIdx = nvCtx->renderTarget->pictureIdx;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvRenderPicture(
        VADriverContextP ctx,
        VAContextID context,
        VABufferID *buffers,
        int num_buffers
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVContext *nvCtx = (NVContext*) getObjectPtr(drv, context);
    if (nvCtx == NULL) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    CUVIDPICPARAMS *picParams = &nvCtx->pPicParams;
    for (int i = 0; i < num_buffers; i++) {
        NVBuffer *buf = (NVBuffer*) getObject(drv, buffers[i])->obj;
        if (buf == NULL || buf->ptr == NULL) {
            LOG("Invalid buffer detected, skipping: %d", buffers[i]);
            continue;
        }
        HandlerFunc func = nvCtx->codec->handlers[buf->bufferType];
        if (func != NULL) {
            func(nvCtx, buf, picParams);
        } else {
            LOG("Unhandled buffer type: %d", buf->bufferType);
        }
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus nvEndPicture(
        VADriverContextP ctx,
        VAContextID context
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVContext *nvCtx = (NVContext*) getObject(drv, context)->obj;
    if (nvCtx == NULL) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    CUVIDPICPARAMS *picParams = &nvCtx->pPicParams;
    picParams->pBitstreamData = nvCtx->bitstreamBuffer.buf;
    picParams->pSliceDataOffsets = nvCtx->sliceOffsets.buf;
    nvCtx->bitstreamBuffer.size = 0;
    nvCtx->sliceOffsets.size = 0;
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    CUresult result = cv->cuvidDecodePicture(nvCtx->decoder, picParams);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    VAStatus status = VA_STATUS_SUCCESS;
    if (result != CUDA_SUCCESS) {
        LOG("cuvidDecodePicture failed: %d", result);
        status = VA_STATUS_ERROR_DECODING_ERROR;
    }
    NVSurface *surface = nvCtx->renderTarget;
    surface->context = nvCtx;
    surface->topFieldFirst = !picParams->bottom_field_flag;
    surface->secondField = picParams->second_field;
    surface->decodeFailed = status != VA_STATUS_SUCCESS;
    pthread_mutex_lock(&nvCtx->resolveMutex);
    nvCtx->surfaceQueue[nvCtx->surfaceQueueWriteIdx++] = nvCtx->renderTarget;
    if (nvCtx->surfaceQueueWriteIdx >= SURFACE_QUEUE_SIZE) {
        nvCtx->surfaceQueueWriteIdx = 0;
    }
    pthread_mutex_unlock(&nvCtx->resolveMutex);
    pthread_cond_signal(&nvCtx->resolveCondition);
    return status;
}

static VAStatus nvSyncSurface(
        VADriverContextP ctx,
        VASurfaceID render_target
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVSurface *surface = getObjectPtr(drv, render_target);
    if (surface == NULL) {
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    pthread_mutex_lock(&surface->mutex);
    if (surface->resolving) {
        pthread_cond_wait(&surface->cond, &surface->mutex);
    }
    pthread_mutex_unlock(&surface->mutex);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvQuerySurfaceStatus(
        VADriverContextP ctx,
        VASurfaceID render_target,
        VASurfaceStatus *status	/* out */
    )
{
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvQuerySurfaceError(
        VADriverContextP ctx,
        VASurfaceID render_target,
        VAStatus error_status,
        void **error_info /*out*/
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvPutSurface(
        VADriverContextP ctx,
        VASurfaceID surface,
        void* draw,
        short srcx,
        short srcy,
        unsigned short srcw,
        unsigned short srch,
        short destx,
        short desty,
        unsigned short destw,
        unsigned short desth,
        VARectangle *cliprects,
        unsigned int number_cliprects,
        unsigned int flags
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvQueryImageFormats(
        VADriverContextP ctx,
        VAImageFormat *format_list,
        int *num_formats
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    LOG("In %s", __func__);
    *num_formats = 0;
    for (unsigned int i = NV_FORMAT_NONE + 1; i < ARRAY_SIZE(formatsInfo); i++) {
        if (formatsInfo[i].is16bits && !drv->supports16BitSurface) {
            continue;
        }
        if (formatsInfo[i].isYuv444 && !drv->supports444Surface) {
            continue;
        }
        format_list[(*num_formats)++] = formatsInfo[i].vaFormat;
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateImage(
        VADriverContextP ctx,
        VAImageFormat *format,
        int width,
        int height,
        VAImage *image     /* out */
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVFormat nvFormat = nvFormatFromVaFormat(format->fourcc);
    const NVFormatInfo *fmtInfo = &formatsInfo[nvFormat];
    const NVFormatPlane *p = fmtInfo->plane;
    if (nvFormat == NV_FORMAT_NONE) {
        return VA_STATUS_ERROR_INVALID_IMAGE_FORMAT;
    }
    Object imageObj = allocateObject(drv, OBJECT_TYPE_IMAGE, sizeof(NVImage));
    image->image_id = imageObj->id;
    LOG("created image id: %d", imageObj->id);
    NVImage *img = (NVImage*) imageObj->obj;
    img->width = width;
    img->height = height;
    img->format = nvFormat;
    Object imageBufferObject = allocateObject(drv, OBJECT_TYPE_BUFFER, sizeof(NVBuffer));
    NVBuffer *imageBuffer = (NVBuffer*) imageBufferObject->obj;
    imageBuffer->bufferType = VAImageBufferType;
    imageBuffer->size = 0;
    for (uint32_t i = 0; i < fmtInfo->numPlanes; i++) {
        imageBuffer->size += ((width * height) >> (p[i].ss.x + p[i].ss.y)) * fmtInfo->bppc * p[i].channelCount;
    }
    imageBuffer->elements = 1;
    imageBuffer->ptr = memalign(16, imageBuffer->size);
    img->imageBuffer = imageBuffer;
    memcpy(&image->format, format, sizeof(VAImageFormat));
    image->buf = imageBufferObject->id;
    image->width = width;
    image->height = height;
    image->data_size = imageBuffer->size;
    image->num_planes = fmtInfo->numPlanes;
    image->pitches[0] = width * fmtInfo->bppc;
    image->pitches[1] = width * fmtInfo->bppc;
    image->pitches[2] = width * fmtInfo->bppc;
    image->offsets[0] = 0;
    image->offsets[1] = image->offsets[0] + ((width * height) >> (p[0].ss.x + p[0].ss.y)) * fmtInfo->bppc * p[0].channelCount;
    image->offsets[2] = image->offsets[1] + ((width * height) >> (p[1].ss.x + p[1].ss.y)) * fmtInfo->bppc * p[1].channelCount;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvDeriveImage(
        VADriverContextP ctx,
        VASurfaceID surface,
        VAImage *image     /* out */
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_OPERATION_FAILED;
}

static VAStatus nvDestroyImage(
        VADriverContextP ctx,
        VAImageID image
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVImage *img = (NVImage*) getObjectPtr(drv, image);
    if (img == NULL) {
        return VA_STATUS_ERROR_INVALID_IMAGE;
    }
    Object imageBufferObj = getObjectByPtr(drv, img->imageBuffer);
    if (imageBufferObj != NULL) {
        if (img->imageBuffer->ptr != NULL) {
            free(img->imageBuffer->ptr);
        }
        deleteObject(drv, imageBufferObj->id);
    }
    deleteObject(drv, image);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvSetImagePalette(
            VADriverContextP ctx,
            VAImageID image,
            unsigned char *palette
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvGetImage(
        VADriverContextP ctx,
        VASurfaceID surface,
        int x,
        int y,
        unsigned int width,
        unsigned int height,
        VAImageID image
    )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVSurface *surfaceObj = (NVSurface*) getObject(drv, surface)->obj;
    NVImage *imageObj = (NVImage*) getObject(drv, image)->obj;
    NVContext *context = (NVContext*) surfaceObj->context;
    const NVFormatInfo *fmtInfo = &formatsInfo[imageObj->format];
    uint32_t offset = 0;
    if (context == NULL) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    nvSyncSurface(ctx, surface);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    for (uint32_t i = 0; i < fmtInfo->numPlanes; i++) {
        const NVFormatPlane *p = &fmtInfo->plane[i];
        CUDA_MEMCPY2D memcpy2d = {
            .srcXInBytes = 0, .srcY = 0,
            .srcMemoryType = CU_MEMORYTYPE_ARRAY,
            .srcArray = surfaceObj->backingImage->arrays[i],
            .dstXInBytes = 0, .dstY = 0,
            .dstMemoryType = CU_MEMORYTYPE_HOST,
            .dstHost = (char *)imageObj->imageBuffer->ptr + offset,
            .dstPitch = width * fmtInfo->bppc,
            .WidthInBytes = (width >> p->ss.x) * fmtInfo->bppc * p->channelCount,
            .Height = height >> p->ss.y
        };
        CUresult result = cu->cuMemcpy2D(&memcpy2d);
        if (result != CUDA_SUCCESS) {
                LOG("cuMemcpy2D failed: %d", result);
                return VA_STATUS_ERROR_DECODING_ERROR;
        }
        offset += ((width * height) >> (p->ss.x + p->ss.y)) * fmtInfo->bppc * p->channelCount;
    }
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvPutImage(
        VADriverContextP ctx,
        VASurfaceID surface,
        VAImageID image,
        int src_x,
        int src_y,
        unsigned int src_width,
        unsigned int src_height,
        int dest_x,
        int dest_y,
        unsigned int dest_width,
        unsigned int dest_height
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvQuerySubpictureFormats(
        VADriverContextP ctx,
        VAImageFormat *format_list,
        unsigned int *flags,
        unsigned int *num_formats
    )
{
    LOG("In %s", __func__);
    *num_formats = 0;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvCreateSubpicture(
        VADriverContextP ctx,
        VAImageID image,
        VASubpictureID *subpicture
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvDestroySubpicture(
        VADriverContextP ctx,
        VASubpictureID subpicture
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvSetSubpictureImage(
                VADriverContextP ctx,
                VASubpictureID subpicture,
                VAImageID image
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvSetSubpictureChromakey(
        VADriverContextP ctx,
        VASubpictureID subpicture,
        unsigned int chromakey_min,
        unsigned int chromakey_max,
        unsigned int chromakey_mask
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvSetSubpictureGlobalAlpha(
        VADriverContextP ctx,
        VASubpictureID subpicture,
        float global_alpha
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvAssociateSubpicture(
        VADriverContextP ctx,
        VASubpictureID subpicture,
        VASurfaceID *target_surfaces,
        int num_surfaces,
        short src_x,
        short src_y,
        unsigned short src_width,
        unsigned short src_height,
        short dest_x,
        short dest_y,
        unsigned short dest_width,
        unsigned short dest_height,
        unsigned int flags
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvDeassociateSubpicture(
        VADriverContextP ctx,
        VASubpictureID subpicture,
        VASurfaceID *target_surfaces,
        int num_surfaces
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvQueryDisplayAttributes(
        VADriverContextP ctx,
        VADisplayAttribute *attr_list,
        int *num_attributes
        )
{
    LOG("In %s", __func__);
    *num_attributes = 0;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvGetDisplayAttributes(
        VADriverContextP ctx,
        VADisplayAttribute *attr_list,
        int num_attributes
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvSetDisplayAttributes(
        VADriverContextP ctx,
        VADisplayAttribute *attr_list,
        int num_attributes
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvQuerySurfaceAttributes(
        VADriverContextP    ctx,
	    VAConfigID          config,
	    VASurfaceAttrib    *attrib_list,
	    unsigned int       *num_attribs
	)
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    NVConfig *cfg = (NVConfig*) getObjectPtr(drv, config);
    if (cfg == NULL) {
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    LOG("with %d (%d) %p %d", cfg->cudaCodec, cfg->bitDepth, attrib_list, *num_attribs);
    if (cfg->chromaFormat != cudaVideoChromaFormat_420 && cfg->chromaFormat != cudaVideoChromaFormat_444) {
        LOG("Unknown chrome format: %d", cfg->chromaFormat);
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    if ((cfg->chromaFormat == cudaVideoChromaFormat_444 || cfg->surfaceFormat == cudaVideoSurfaceFormat_YUV444_16Bit) && !drv->supports444Surface) {
        LOG("YUV444 surfaces not supported: %d", cfg->chromaFormat);
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    if (cfg->surfaceFormat == cudaVideoSurfaceFormat_P016 && !drv->supports16BitSurface) {
        LOG("16 bits surfaces not supported: %d", cfg->chromaFormat);
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    if (num_attribs != NULL) {
        int cnt = 4;
        if (cfg->chromaFormat == cudaVideoChromaFormat_444) {
            cnt += 1;
#if VA_CHECK_VERSION(1, 20, 0)
            cnt += 1;
#endif
        } else {
            cnt += 1;
            if (drv->supports16BitSurface) {
                cnt += 3;
            }
        }
        *num_attribs = cnt;
    }
    if (attrib_list != NULL) {
        CUVIDDECODECAPS videoDecodeCaps = {
            .eCodecType      = cfg->cudaCodec,
            .eChromaFormat   = cfg->chromaFormat,
            .nBitDepthMinus8 = cfg->bitDepth - 8
        };
        CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
        CHECK_CUDA_RESULT_RETURN(cv->cuvidGetDecoderCaps(&videoDecodeCaps), VA_STATUS_ERROR_OPERATION_FAILED);
        CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
        attrib_list[0].type = VASurfaceAttribMinWidth;
        attrib_list[0].flags = 0;
        attrib_list[0].value.type = VAGenericValueTypeInteger;
        attrib_list[0].value.value.i = videoDecodeCaps.nMinWidth;
        attrib_list[1].type = VASurfaceAttribMinHeight;
        attrib_list[1].flags = 0;
        attrib_list[1].value.type = VAGenericValueTypeInteger;
        attrib_list[1].value.value.i = videoDecodeCaps.nMinHeight;
        attrib_list[2].type = VASurfaceAttribMaxWidth;
        attrib_list[2].flags = 0;
        attrib_list[2].value.type = VAGenericValueTypeInteger;
        attrib_list[2].value.value.i = videoDecodeCaps.nMaxWidth;
        attrib_list[3].type = VASurfaceAttribMaxHeight;
        attrib_list[3].flags = 0;
        attrib_list[3].value.type = VAGenericValueTypeInteger;
        attrib_list[3].value.value.i = videoDecodeCaps.nMaxHeight;
        LOG("Returning constraints: width: %d - %d, height: %d - %d", attrib_list[0].value.value.i, attrib_list[2].value.value.i, attrib_list[1].value.value.i, attrib_list[3].value.value.i);
        int attrib_idx = 4;
        if (cfg->chromaFormat == cudaVideoChromaFormat_444) {
            attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
            attrib_list[attrib_idx].flags = 0;
            attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
            attrib_list[attrib_idx].value.value.i = VA_FOURCC_444P;
            attrib_idx += 1;
#if VA_CHECK_VERSION(1, 20, 0)
            attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
            attrib_list[attrib_idx].flags = 0;
            attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
            attrib_list[attrib_idx].value.value.i = VA_FOURCC_Q416;
            attrib_idx += 1;
#endif
        } else {
            attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
            attrib_list[attrib_idx].flags = 0;
            attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
            attrib_list[attrib_idx].value.value.i = VA_FOURCC_NV12;
            attrib_idx += 1;
            if (drv->supports16BitSurface) {
                attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
                attrib_list[attrib_idx].flags = 0;
                attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
                attrib_list[attrib_idx].value.value.i = VA_FOURCC_P010;
                attrib_idx += 1;
                attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
                attrib_list[attrib_idx].flags = 0;
                attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
                attrib_list[attrib_idx].value.value.i = VA_FOURCC_P012;
                attrib_idx += 1;
                attrib_list[attrib_idx].type = VASurfaceAttribPixelFormat;
                attrib_list[attrib_idx].flags = 0;
                attrib_list[attrib_idx].value.type = VAGenericValueTypeInteger;
                attrib_list[attrib_idx].value.value.i = VA_FOURCC_P016;
                attrib_idx += 1;
            }
        }
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus nvBufferInfo(
           VADriverContextP ctx,
           VABufferID buf_id,
           VABufferType *type,
           unsigned int *size,
           unsigned int *num_elements
)
{
    LOG("In %s", __func__);
    *size = 0;
    *num_elements = 0;
    return VA_STATUS_SUCCESS;
}

static VAStatus nvAcquireBufferHandle(
            VADriverContextP    ctx,
            VABufferID          buf_id,
            VABufferInfo *      buf_info
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvReleaseBufferHandle(
            VADriverContextP    ctx,
            VABufferID          buf_id
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvLockSurface(
        VADriverContextP ctx,
        VASurfaceID surface,
        unsigned int *fourcc,
        unsigned int *luma_stride,
        unsigned int *chroma_u_stride,
        unsigned int *chroma_v_stride,
        unsigned int *luma_offset,
        unsigned int *chroma_u_offset,
        unsigned int *chroma_v_offset,
        unsigned int *buffer_name,
        void **buffer
)
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvUnlockSurface(
        VADriverContextP ctx,
        VASurfaceID surface
)
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvCreateMFContext(
            VADriverContextP ctx,
            VAMFContextID *mfe_context
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvMFAddContext(
            VADriverContextP ctx,
            VAMFContextID mf_context,
            VAContextID context
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvMFReleaseContext(
            VADriverContextP ctx,
            VAMFContextID mf_context,
            VAContextID context
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvMFSubmit(
            VADriverContextP ctx,
            VAMFContextID mf_context,
            VAContextID *contexts,
            int num_contexts
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvCreateBuffer2(
            VADriverContextP ctx,
            VAContextID context,
            VABufferType type,
            unsigned int width,
            unsigned int height,
            unsigned int *unit_size,
            unsigned int *pitch,
            VABufferID *buf_id
    )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvQueryProcessingRate(
            VADriverContextP ctx,
            VAConfigID config_id,
            VAProcessingRateParameter *proc_buf,
            unsigned int *processing_rate
        )
{
    LOG("In %s", __func__);
    return VA_STATUS_ERROR_UNIMPLEMENTED;
}

static VAStatus nvExportSurfaceHandle(
            VADriverContextP    ctx,
            VASurfaceID         surface_id,
            uint32_t            mem_type,
            uint32_t            flags,
            void               *descriptor
)
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    if ((mem_type & VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2) == 0) {
        return VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE;
    }
    if ((flags & VA_EXPORT_SURFACE_SEPARATE_LAYERS) == 0) {
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    NVSurface *surface = (NVSurface*) getObjectPtr(drv, surface_id);
    if (surface == NULL) {
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    if (!drv->backend->realiseSurface(drv, surface)) {
        LOG("Unable to export surface");
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }
    VADRMPRIMESurfaceDescriptor *ptr = (VADRMPRIMESurfaceDescriptor*) descriptor;
    drv->backend->fillExportDescriptor(drv, surface, ptr);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    return VA_STATUS_SUCCESS;
}

static VAStatus nvTerminate( VADriverContextP ctx )
{
    NVDriver *drv = (NVDriver*) ctx->pDriverData;
    LOG("Terminating %p", ctx);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPushCurrent(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    drv->backend->destroyAllBackingImage(drv);
    deleteAllObjects(drv);
    drv->backend->releaseExporter(drv);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxPopCurrent(NULL), VA_STATUS_ERROR_OPERATION_FAILED);
    pthread_mutex_lock(&concurrency_mutex);
    instances--;
    LOG("Now have %d (%d max) instances", instances, max_instances);
    pthread_mutex_unlock(&concurrency_mutex);
    CHECK_CUDA_RESULT_RETURN(cu->cuCtxDestroy(drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    drv->cudaContext = NULL;
    free(drv);
    return VA_STATUS_SUCCESS;
}

extern const NVBackend DIRECT_BACKEND;
extern const NVBackend EGL_BACKEND;

#define VTABLE(func) .va ## func = &nv ## func
static const struct VADriverVTable vtable = {
    VTABLE(Terminate),
    VTABLE(QueryConfigProfiles),
    VTABLE(QueryConfigEntrypoints),
    VTABLE(QueryConfigAttributes),
    VTABLE(CreateConfig),
    VTABLE(DestroyConfig),
    VTABLE(GetConfigAttributes),
    VTABLE(CreateSurfaces),
    VTABLE(CreateSurfaces2),
    VTABLE(DestroySurfaces),
    VTABLE(CreateContext),
    VTABLE(DestroyContext),
    VTABLE(CreateBuffer),
    VTABLE(BufferSetNumElements),
    VTABLE(MapBuffer),
    VTABLE(UnmapBuffer),
    VTABLE(DestroyBuffer),
    VTABLE(BeginPicture),
    VTABLE(RenderPicture),
    VTABLE(EndPicture),
    VTABLE(SyncSurface),
    VTABLE(QuerySurfaceStatus),
    VTABLE(QuerySurfaceError),
    VTABLE(PutSurface),
    VTABLE(QueryImageFormats),
    VTABLE(CreateImage),
    VTABLE(DeriveImage),
    VTABLE(DestroyImage),
    VTABLE(SetImagePalette),
    VTABLE(GetImage),
    VTABLE(PutImage),
    VTABLE(QuerySubpictureFormats),
    VTABLE(CreateSubpicture),
    VTABLE(DestroySubpicture),
    VTABLE(SetSubpictureImage),
    VTABLE(SetSubpictureChromakey),
    VTABLE(SetSubpictureGlobalAlpha),
    VTABLE(AssociateSubpicture),
    VTABLE(DeassociateSubpicture),
    VTABLE(QueryDisplayAttributes),
    VTABLE(GetDisplayAttributes),
    VTABLE(SetDisplayAttributes),
    VTABLE(QuerySurfaceAttributes),
    VTABLE(BufferInfo),
    VTABLE(AcquireBufferHandle),
    VTABLE(ReleaseBufferHandle),
    VTABLE(LockSurface),
    VTABLE(UnlockSurface),
    VTABLE(CreateMFContext),
    VTABLE(MFAddContext),
    VTABLE(MFReleaseContext),
    VTABLE(MFSubmit),
    VTABLE(CreateBuffer2),
    VTABLE(QueryProcessingRate),
    VTABLE(ExportSurfaceHandle),
};

__attribute__((visibility("default")))
VAStatus __vaDriverInit_1_0(VADriverContextP ctx);

__attribute__((visibility("default")))
VAStatus __vaDriverInit_1_0(VADriverContextP ctx) {
    LOG("Initialising NVIDIA VA-API Driver");
    bool isDrm = ctx->drm_state != NULL && ((struct drm_state*) ctx->drm_state)->fd > 0;
    int drmFd = (gpu == -1 && isDrm) ? ((struct drm_state*) ctx->drm_state)->fd : -1;
    LOG("Got DRM FD: %d %d", isDrm, drmFd);
    if (drmFd != -1) {
        if (!isNvidiaDrmFd(drmFd, true)) {
            LOG("Passed in DRM FD does not belong to the NVIDIA driver, ignoring");
            drmFd = -1;
        } else if (!checkModesetParameterFromFd(drmFd)) {
            return VA_STATUS_ERROR_OPERATION_FAILED;
        }
    }
    pthread_mutex_lock(&concurrency_mutex);
    LOG("Now have %d (%d max) instances", instances, max_instances);
    if (max_instances > 0 && instances >= max_instances) {
        pthread_mutex_unlock(&concurrency_mutex);
        return VA_STATUS_ERROR_HW_BUSY;
    }
    instances++;
    pthread_mutex_unlock(&concurrency_mutex);
    if (cu == NULL || cv == NULL) {
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    NVDriver *drv = (NVDriver*) calloc(1, sizeof(NVDriver));
    ctx->pDriverData = drv;
    drv->cu = cu;
    drv->cv = cv;
    drv->useCorrectNV12Format = true;
    drv->cudaGpuId = gpu;
    drv->drmFd = drmFd;
    if (backend == EGL) {
        LOG("Selecting EGL backend");
        drv->backend = &EGL_BACKEND;
    } else if (backend == DIRECT) {
        LOG("Selecting Direct backend");
        drv->backend = &DIRECT_BACKEND;
    }
    ctx->max_profiles = MAX_PROFILES;
    ctx->max_entrypoints = 1;
    ctx->max_attributes = 1;
    ctx->max_display_attributes = 1;
    ctx->max_image_formats = ARRAY_SIZE(formatsInfo) - 1;
    ctx->max_subpic_formats = 1;
    if (backend == DIRECT) {
        ctx->str_vendor = "VA-API NVDEC driver [direct backend]";
    } else if (backend == EGL) {
        ctx->str_vendor = "VA-API NVDEC driver [egl backend]";
    }
    pthread_mutexattr_t attrib;
    pthread_mutexattr_init(&attrib);
    pthread_mutexattr_settype(&attrib, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&drv->objectCreationMutex, &attrib);
    pthread_mutex_init(&drv->imagesMutex, &attrib);
    pthread_mutex_init(&drv->exportMutex, NULL);
    if (!drv->backend->initExporter(drv)) {
        LOG("Exporter failed");
        free(drv);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    if (CHECK_CUDA_RESULT(cu->cuCtxCreate(&drv->cudaContext, CU_CTX_SCHED_BLOCKING_SYNC, drv->cudaGpuId))) {
        drv->backend->releaseExporter(drv);
        free(drv);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    *ctx->vtable = vtable;
    return VA_STATUS_SUCCESS;
}
