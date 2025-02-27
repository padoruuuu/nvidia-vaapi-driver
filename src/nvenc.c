#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <va/va.h>
#include <va/va_drmcommon.h>
#include <cuda.h>

#include "nvenc.h"
#include "utils.h"

/* Logging macros if not already defined */
#ifndef log_info
#define log_info(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#endif
#ifndef log_error
#define log_error(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

#define MAX_ENCODE_PROFILES 10
#define MAX_SURFACES 64
#define CHECK_CUDA(x) \
    do { \
        CUresult result = (x); \
        if (result != CUDA_SUCCESS) { \
            const char *error_string; \
            cuGetErrorString(result, &error_string); \
            ERROR("CUDA error at %s:%d code=%d \"%s\"\n", __FILE__, __LINE__, result, error_string); \
            return VA_STATUS_ERROR_OPERATION_FAILED; \
        } \
    } while (0)

#define NVENC_CALL(x) \
    do { \
        NVENCSTATUS ret = (x); \
        if (ret != NV_ENC_SUCCESS) { \
            ERROR("NVENC call failed at %s:%d with error %d\n", __FILE__, __LINE__, ret); \
            return VA_STATUS_ERROR_OPERATION_FAILED; \
        } \
    } while (0)

/*
 * NVENC-specific VAAPI helper functions.
 * These functions are used to provide a minimal VAAPI interface for encoding.
 */

__attribute__((visibility("default")))
VAStatus nvenc_vaCreateConfig(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                              VAConfigAttrib *attrib_list, int num_attribs, VAConfigID *config_id)
{
    if (!config_id)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    /* For encoding, we simply return a dummy nonzero configuration ID */
    *config_id = 1;
    log_info("nvenc_vaCreateConfig: created dummy config id %u for profile %d, entrypoint %d", *config_id, profile, entrypoint);
    return VA_STATUS_SUCCESS;
}

__attribute__((visibility("default")))
VAStatus nvenc_vaGetConfigAttributes(VADisplay dpy, VAProfile profile, VAEntrypoint entrypoint,
                                     VAConfigAttrib *attrib_list, int num_attribs)
{
    for (int i = 0; i < num_attribs; i++) {
        if (attrib_list[i].type == VAConfigAttribRTFormat)
            attrib_list[i].value = VA_RT_FORMAT_YUV420;
        else
            attrib_list[i].value = 0;
    }
    return VA_STATUS_SUCCESS;
}

__attribute__((visibility("default")))
VAStatus nvenc_init_driver(VADisplay va_display)
{
    VAProfile profile = VAProfileH264Main;
    VAEntrypoint entrypoint = VAEntrypointEncSlice;
    VAConfigID dummy;
    VAStatus status = nvenc_vaCreateConfig(va_display, profile, entrypoint, NULL, 0, &dummy);
    if (status != VA_STATUS_SUCCESS) {
         log_error("nvenc_init_driver: failed to create dummy config for profile %d, entrypoint %d", profile, entrypoint);
         return status;
    }
    log_info("nvenc_init_driver: NVENC VAAPI driver (encoding) initialized successfully");
    return VA_STATUS_SUCCESS;
}

/*
 * NVENC encoder implementation.
 */

typedef struct _NVEncoder {
    void *nvenc_lib;
    NV_ENCODE_API_FUNCTION_LIST nvenc_funcs;
    CUcontext cuda_ctx;
    NV_ENC_DEVICE_TYPE device_type;
    void *encoder;
    GUID codec_guid;
    GUID profile_guid;
    GUID preset_guid;
    
    // Encoder parameters
    unsigned int width;
    unsigned int height;
    unsigned int bitrate;
    unsigned int max_bitrate;
    unsigned int vbv_buffer_size;
    unsigned int frame_rate_num;
    unsigned int frame_rate_den;
    unsigned int gop_length;
    unsigned int b_frames;
    int rc_mode; // 0=CQP, 1=VBR, 2=CBR

    // Resources
    struct nv_frame frames[MAX_SURFACES];
    NV_ENC_OUTPUT_PTR bitstream_buffers[MAX_SURFACES];
    int num_frames;
    int initialized;
} NVEncoder;

static NVEncoder *g_encoder = NULL;

/* Function pointer for NVENC API entry point */
typedef NVENCSTATUS (NVENCAPI *PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST *);

/* Check if NVENC is available and supported */
int nvenc_is_available(void)
{
    void *nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!nvenc_lib) {
        fprintf(stderr, "Failed to load NVENC library: %s\n", dlerror());
        return 0;
    }
    PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance = dlsym(nvenc_lib, "NvEncodeAPICreateInstance");
    if (!nvEncodeAPICreateInstance) {
        fprintf(stderr, "Failed to get NvEncodeAPICreateInstance: %s\n", dlerror());
        dlclose(nvenc_lib);
        return 0;
    }
    NV_ENCODE_API_FUNCTION_LIST nvenc_funcs = {0};
    nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS status = nvEncodeAPICreateInstance(&nvenc_funcs);
    if (status != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to create NVENC instance: %d\n", status);
        dlclose(nvenc_lib);
        return 0;
    }
    CUresult cuda_status = cuInit(0);
    if (cuda_status != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA: %d\n", cuda_status);
        dlclose(nvenc_lib);
        return 0;
    }
    int device_count = 0;
    cuda_status = cuDeviceGetCount(&device_count);
    if (cuda_status != CUDA_SUCCESS || device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        dlclose(nvenc_lib);
        return 0;
    }
    dlclose(nvenc_lib);
    return 1;
}

/* Get supported encoding profiles */
int nvenc_get_profiles(VAProfile *profiles, int *num_profiles)
{
    if (!profiles || !num_profiles || *num_profiles <= 0)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    if (!nvenc_is_available()) {
        *num_profiles = 0;
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    int profile_count = 0;
    if (profile_count < *num_profiles) {
        profiles[profile_count++] = VAProfileH264Main;
    }
    if (profile_count < *num_profiles) {
        profiles[profile_count++] = VAProfileH264High;
    }
    if (profile_count < *num_profiles) {
        profiles[profile_count++] = VAProfileH264ConstrainedBaseline;
    }
    if (profile_count < *num_profiles) {
        profiles[profile_count++] = VAProfileHEVCMain;
    }
    /* Check for AV1 support on newer NVIDIA cards */
    void *nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (nvenc_lib) {
        PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance = dlsym(nvenc_lib, "NvEncodeAPICreateInstance");
        if (nvEncodeAPICreateInstance) {
            NV_ENCODE_API_FUNCTION_LIST nvenc_funcs = {0};
            nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
            if (nvEncodeAPICreateInstance(&nvenc_funcs) == NV_ENC_SUCCESS) {
                CUcontext cuda_ctx;
                if (cuInit(0) == CUDA_SUCCESS) {
                    CUdevice cuda_device;
                    if (cuDeviceGet(&cuda_device, 0) == CUDA_SUCCESS &&
                        cuCtxCreate(&cuda_ctx, 0, cuda_device) == CUDA_SUCCESS) {
                        
                        void *encoder = NULL;
                        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_params = {0};
                        open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
                        open_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
                        open_params.device = cuda_ctx;
                        open_params.apiVersion = NVENCAPI_VERSION;
                        
                        if (nvenc_funcs.nvEncOpenEncodeSessionEx(&open_params, &encoder) == NV_ENC_SUCCESS) {
                            uint32_t count = 0;
                            if (nvenc_funcs.nvEncGetEncodeGUIDCount(encoder, &count) == NV_ENC_SUCCESS && count > 0) {
                                GUID *guids = malloc(count * sizeof(GUID));
                                if (guids && nvenc_funcs.nvEncGetEncodeGUIDs(encoder, guids, count, &count) == NV_ENC_SUCCESS) {
                                    for (uint32_t i = 0; i < count; i++) {
                                        if (memcmp(&guids[i], &NV_ENC_CODEC_AV1_GUID, sizeof(GUID)) == 0 && profile_count < *num_profiles) {
                                            profiles[profile_count++] = VAProfileAV1Main;
                                            break;
                                        }
                                    }
                                }
                                free(guids);
                            }
                            nvenc_funcs.nvEncDestroyEncoder(encoder);
                        }
                        cuCtxDestroy(cuda_ctx);
                    }
                }
            }
        }
        dlclose(nvenc_lib);
    }
    *num_profiles = profile_count;
    return VA_STATUS_SUCCESS;
}

/* Initialize the NVENC encoder with specific parameters */
int nvenc_init(unsigned int width, unsigned int height, unsigned int bitrate)
{
    if (!nvenc_is_available())
        return VA_STATUS_ERROR_OPERATION_FAILED;
    if (g_encoder) {
        nvenc_terminate(); // Clean up existing encoder
    }
    g_encoder = calloc(1, sizeof(NVEncoder));
    if (!g_encoder)
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    g_encoder->nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!g_encoder->nvenc_lib) {
        ERROR("Failed to load NVENC library: %s\n", dlerror());
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance = dlsym(g_encoder->nvenc_lib, "NvEncodeAPICreateInstance");
    if (!nvEncodeAPICreateInstance) {
        ERROR("Failed to get NvEncodeAPICreateInstance: %s\n", dlerror());
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    memset(&g_encoder->nvenc_funcs, 0, sizeof(g_encoder->nvenc_funcs));
    g_encoder->nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS status = nvEncodeAPICreateInstance(&g_encoder->nvenc_funcs);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to create NVENC instance: %d\n", status);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    /* Initialize CUDA */
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        ERROR("Failed to initialize CUDA: %d\n", cu_result);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    CUdevice cu_device;
    cu_result = cuDeviceGet(&cu_device, 0);
    if (cu_result != CUDA_SUCCESS) {
        ERROR("Failed to get CUDA device: %d\n", cu_result);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    cu_result = cuCtxCreate(&g_encoder->cuda_ctx, 0, cu_device);
    if (cu_result != CUDA_SUCCESS) {
        ERROR("Failed to create CUDA context: %d\n", cu_result);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    /* Initialize encoder parameters */
    g_encoder->width = width;
    g_encoder->height = height;
    g_encoder->bitrate = bitrate;
    g_encoder->frame_rate_num = 30;
    g_encoder->frame_rate_den = 1;
    g_encoder->gop_length = 30;
    g_encoder->device_type = NV_ENC_DEVICE_TYPE_CUDA;
    g_encoder->codec_guid = NV_ENC_CODEC_H264_GUID;
    g_encoder->profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
    g_encoder->preset_guid = NV_ENC_PRESET_P4_GUID;
    g_encoder->rc_mode = 2; // CBR by default
    /* Initialize encoder session */
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_params = {0};
    open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    open_params.deviceType = g_encoder->device_type;
    open_params.device = g_encoder->cuda_ctx;
    open_params.apiVersion = NVENCAPI_VERSION;
    status = g_encoder->nvenc_funcs.nvEncOpenEncodeSessionEx(&open_params, &g_encoder->encoder);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to open encode session: %d\n", status);
        cuCtxDestroy(g_encoder->cuda_ctx);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    /* Create encoder configuration */
    NV_ENC_INITIALIZE_PARAMS init_params = {0};
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID = g_encoder->codec_guid;
    init_params.presetGUID = g_encoder->preset_guid;
    init_params.encodeWidth = g_encoder->width;
    init_params.encodeHeight = g_encoder->height;
    init_params.darWidth = g_encoder->width;
    init_params.darHeight = g_encoder->height;
    init_params.frameRateNum = g_encoder->frame_rate_num;
    init_params.frameRateDen = g_encoder->frame_rate_den;
    init_params.enablePTD = 1;
    
    NV_ENC_CONFIG enc_config = {0};
    enc_config.version = NV_ENC_CONFIG_VER;
    init_params.encodeConfig = &enc_config;
    
    status = g_encoder->nvenc_funcs.nvEncGetEncodePresetConfig(g_encoder->encoder, 
                                                    g_encoder->codec_guid, 
                                                    g_encoder->preset_guid, 
                                                    &enc_config);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to get preset config: %d\n", status);
        g_encoder->nvenc_funcs.nvEncDestroyEncoder(g_encoder->encoder);
        cuCtxDestroy(g_encoder->cuda_ctx);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    /* Set rate control parameters */
    if (g_encoder->rc_mode == 1) // VBR
        enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    else if (g_encoder->rc_mode == 2) // CBR
        enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    else // CQP
        enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    
    enc_config.rcParams.averageBitRate = g_encoder->bitrate;
    enc_config.rcParams.maxBitRate = g_encoder->max_bitrate > 0 ? g_encoder->max_bitrate : g_encoder->bitrate;
    enc_config.rcParams.vbvBufferSize = g_encoder->vbv_buffer_size > 0 ? g_encoder->vbv_buffer_size : g_encoder->bitrate / 1000;
    
    /* Set codec-specific parameters */
    if (g_encoder->codec_guid == NV_ENC_CODEC_H264_GUID) {
        enc_config.profileGUID = g_encoder->profile_guid;
        enc_config.encodeCodecConfig.h264Config.idrPeriod = g_encoder->gop_length;
        enc_config.encodeCodecConfig.h264Config.maxNumRefFrames = 3;
        enc_config.encodeCodecConfig.h264Config.sliceMode = 0;
        enc_config.encodeCodecConfig.h264Config.sliceModeData = 0;
    } else if (g_encoder->codec_guid == NV_ENC_CODEC_HEVC_GUID) {
        enc_config.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
        enc_config.encodeCodecConfig.hevcConfig.idrPeriod = g_encoder->gop_length;
        enc_config.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 3;
        enc_config.encodeCodecConfig.hevcConfig.sliceMode = 0;
        enc_config.encodeCodecConfig.hevcConfig.sliceModeData = 0;
    } else if (g_encoder->codec_guid == NV_ENC_CODEC_AV1_GUID) {
        enc_config.profileGUID = NV_ENC_AV1_PROFILE_MAIN_GUID;
        enc_config.encodeCodecConfig.av1Config.idrPeriod = g_encoder->gop_length;
    }
    
    /* Create bitstream buffers */
    for (int i = 0; i < MAX_SURFACES; i++) {
        NV_ENC_CREATE_BITSTREAM_BUFFER create_params = {0};
        create_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        status = g_encoder->nvenc_funcs.nvEncCreateBitstreamBuffer(g_encoder->encoder, &create_params);
        if (status != NV_ENC_SUCCESS) {
            ERROR("Failed to create bitstream buffer: %d\n", status);
            break;
        }
        g_encoder->bitstream_buffers[i] = create_params.bitstreamBuffer;
    }
    
    /* Initialize encoder */
    status = g_encoder->nvenc_funcs.nvEncInitializeEncoder(g_encoder->encoder, &init_params);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to initialize encoder: %d\n", status);
        for (int i = 0; i < MAX_SURFACES; i++) {
            if (g_encoder->bitstream_buffers[i]) {
                g_encoder->nvenc_funcs.nvEncDestroyBitstreamBuffer(g_encoder->encoder, g_encoder->bitstream_buffers[i]);
                g_encoder->bitstream_buffers[i] = NULL;
            }
        }
        g_encoder->nvenc_funcs.nvEncDestroyEncoder(g_encoder->encoder);
        cuCtxDestroy(g_encoder->cuda_ctx);
        dlclose(g_encoder->nvenc_lib);
        free(g_encoder);
        g_encoder = NULL;
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    g_encoder->initialized = 1;
    return VA_STATUS_SUCCESS;
}

/* Helper function to map VA surface to CUDA.
   This stub should ideally query the common backend for the actual mapping.
*/
static int map_va_surface_to_cuda(VASurfaceID surface, CUdeviceptr *cuda_ptr, unsigned int *pitch)
{
    if (!g_encoder || !cuda_ptr || !pitch)
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    
    VASurfaceAttrib attribs[1];
    attribs[0].type = VASurfaceAttribExternalBufferDescriptor;
    attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attribs[0].value.type = VAGenericValueTypePointer;
    
    VASurfaceAttribExternalBuffers external_buffer;
    memset(&external_buffer, 0, sizeof(external_buffer));
    attribs[0].value.value.p = &external_buffer;
    
    *cuda_ptr = 0;
    *pitch = g_encoder->width;
    
    return VA_STATUS_SUCCESS;
}

/* Prepare frame buffers for encoding */
int nvenc_prepare_frame(struct nv_frame *frame, VASurfaceID surface)
{
    if (!g_encoder || !g_encoder->initialized || !frame)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    
    CUdeviceptr cuda_ptr = 0;
    unsigned int pitch = 0;
    
    int ret = map_va_surface_to_cuda(surface, &cuda_ptr, &pitch);
    if (ret != VA_STATUS_SUCCESS)
        return ret;
    
    /* Register the CUDA resource with NVENC */
    NV_ENC_REGISTER_RESOURCE reg_resource = {0};
    reg_resource.version = NV_ENC_REGISTER_RESOURCE_VER;
    reg_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    reg_resource.width = g_encoder->width;
    reg_resource.height = g_encoder->height;
    reg_resource.pitch = pitch;
    reg_resource.resourceToRegister = (void*)cuda_ptr;
    reg_resource.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    
    NVENCSTATUS status = g_encoder->nvenc_funcs.nvEncRegisterResource(g_encoder->encoder, &reg_resource);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to register resource: %d\n", status);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    /* Map the registered resource */
    NV_ENC_MAP_INPUT_RESOURCE map_resource = {0};
    map_resource.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map_resource.registeredResource = reg_resource.registeredResource;
    
    status = g_encoder->nvenc_funcs.nvEncMapInputResource(g_encoder->encoder, &map_resource);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to map input resource: %d\n", status);
        g_encoder->nvenc_funcs.nvEncUnregisterResource(g_encoder->encoder, reg_resource.registeredResource);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    frame->input_buffer = (void*)cuda_ptr;
    frame->input_surface = map_resource.mappedResource;
    frame->output_buffer = g_encoder->bitstream_buffers[g_encoder->num_frames % MAX_SURFACES];
    frame->width = g_encoder->width;
    frame->height = g_encoder->height;
    frame->pitch = pitch;
    frame->va_surface = surface;
    
    g_encoder->num_frames++;
    return VA_STATUS_SUCCESS;
}

/* Encode a frame using NVENC */
int nvenc_encode_frame(struct nv_frame *frame)
{
    if (!g_encoder || !g_encoder->initialized || !frame || !frame->input_surface)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    
    NV_ENC_PIC_PARAMS pic_params = {0};
    pic_params.version = NV_ENC_PIC_PARAMS_VER;
    pic_params.inputBuffer = frame->input_surface;
    pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pic_params.inputWidth = frame->width;
    pic_params.inputHeight = frame->height;
    pic_params.outputBitstream = frame->output_buffer;
    pic_params.completionEvent = NULL;
    
    NVENCSTATUS status = g_encoder->nvenc_funcs.nvEncEncodePicture(g_encoder->encoder, &pic_params);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to encode picture: %d\n", status);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    NV_ENC_LOCK_BITSTREAM lock_params = {0};
    lock_params.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock_params.outputBitstream = frame->output_buffer;
    lock_params.doNotWait = 0;
    
    status = g_encoder->nvenc_funcs.nvEncLockBitstream(g_encoder->encoder, &lock_params);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to lock bitstream: %d\n", status);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    status = g_encoder->nvenc_funcs.nvEncUnlockBitstream(g_encoder->encoder, frame->output_buffer);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to unlock bitstream: %d\n", status);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    status = g_encoder->nvenc_funcs.nvEncUnmapInputResource(g_encoder->encoder, frame->input_surface);
    if (status != NV_ENC_SUCCESS) {
        ERROR("Failed to unmap input resource: %d\n", status);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }
    
    return VA_STATUS_SUCCESS;
}

/* Terminate NVENC encoder and free resources */
void nvenc_terminate(void)
{
    if (!g_encoder)
        return;
    
    if (g_encoder->encoder) {
        for (int i = 0; i < MAX_SURFACES; i++) {
            if (g_encoder->bitstream_buffers[i]) {
                g_encoder->nvenc_funcs.nvEncDestroyBitstreamBuffer(g_encoder->encoder, g_encoder->bitstream_buffers[i]);
                g_encoder->bitstream_buffers[i] = NULL;
            }
        }
        g_encoder->nvenc_funcs.nvEncDestroyEncoder(g_encoder->encoder);
        g_encoder->encoder = NULL;
    }
    
    if (g_encoder->cuda_ctx) {
        cuCtxDestroy(g_encoder->cuda_ctx);
        g_encoder->cuda_ctx = NULL;
    }
    
    if (g_encoder->nvenc_lib) {
        dlclose(g_encoder->nvenc_lib);
        g_encoder->nvenc_lib = NULL;
    }
    
    free(g_encoder);
    g_encoder = NULL;
}
