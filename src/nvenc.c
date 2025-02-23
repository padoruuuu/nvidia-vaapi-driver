#include "nvenc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>

// Global NVENC state
static NV_ENCODE_API_FUNCTION_LIST nvEncFuncList = { NV_ENCODE_API_FUNCTION_LIST_VER };
static void *nvencLib = NULL;
static void *session = NULL;
static CUcontext nvencCudaCtx = NULL;
static NV_ENC_INITIALIZE_PARAMS initParams = { 0 };
static NV_ENC_CONFIG encodeConfig = { 0 };

int nvenc_is_available(void) {
    void *lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!lib) {
        return 0;
    }
    dlclose(lib);

    // Check CUDA availability
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    int deviceCount = 0;
    result = cuDeviceGetCount(&deviceCount);
    if (result != CUDA_SUCCESS || deviceCount == 0) {
        return 0;
    }

    // Check if NVENC is supported on the first device
    CUdevice cuDevice;
    result = cuDeviceGet(&cuDevice, 0);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    // Try to create a CUDA context
    CUcontext tempContext;
    result = cuCtxCreate(&tempContext, CU_CTX_SCHED_AUTO, cuDevice);
    if (result != CUDA_SUCCESS) {
        return 0;
    }
    
    cuCtxDestroy(tempContext);
    return 1;
}

int nvenc_init(unsigned int width, unsigned int height, unsigned int bitrate) {
    NVENCSTATUS nvStatus;
    CUresult cuResult;

    // Load NVENC library
    nvencLib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
    if (!nvencLib) {
        fprintf(stderr, "Failed to load NVENC library: %s\n", dlerror());
        return -1;
    }

    // Get API entry point
    typedef NVENCSTATUS (NVENCAPI *NvEncodeAPICreateInstance_t)(NV_ENCODE_API_FUNCTION_LIST*);
    NvEncodeAPICreateInstance_t nvEncodeAPICreateInstance = dlsym(nvencLib, "NvEncodeAPICreateInstance");
    if (!nvEncodeAPICreateInstance) {
        fprintf(stderr, "Failed to get NVENC API entry point\n");
        goto fail;
    }

    nvStatus = nvEncodeAPICreateInstance(&nvEncFuncList);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to create NVENC API instance\n");
        goto fail;
    }

    // Initialize CUDA
    cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        goto fail;
    }

    CUdevice cuDevice;
    cuResult = cuDeviceGet(&cuDevice, 0);
    if (cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get CUDA device\n");
        goto fail;
    }

    cuResult = cuCtxCreate(&nvencCudaCtx, CU_CTX_SCHED_AUTO, cuDevice);
    if (cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create CUDA context\n");
        goto fail;
    }

    // Open NVENC session
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { 0 };
    sessionParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sessionParams.device = nvencCudaCtx;
    sessionParams.apiVersion = NVENCAPI_VERSION;

    nvStatus = nvEncFuncList.nvEncOpenEncodeSessionEx(&sessionParams, &session);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to open NVENC encode session\n");
        goto fail;
    }

    // Query encoder capabilities
    NV_ENC_CAPS_PARAM capsParam = { 0 };
    capsParam.version = NV_ENC_CAPS_PARAM_VER;
    capsParam.capsToQuery = NV_ENC_CAPS_SUPPORT_YUV444_ENCODE;
    
    int yuv444Support = 0;
    nvStatus = nvEncFuncList.nvEncGetEncodeCaps(session, NV_ENC_CODEC_H264_GUID, &capsParam, &yuv444Support);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to query encoder capabilities\n");
        goto fail;
    }

    // Initialize encoder
    initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
    initParams.encodeWidth = width;
    initParams.encodeHeight = height;
    initParams.darWidth = width;
    initParams.darHeight = height;
    initParams.frameRateNum = 30;
    initParams.frameRateDen = 1;
    initParams.enablePTD = 1;
    initParams.reportSliceOffsets = 0;
    initParams.enableSubFrameWrite = 0;
    initParams.maxEncodeWidth = width;
    initParams.maxEncodeHeight = height;
    initParams.presetGUID = NV_ENC_PRESET_P4_GUID;

    encodeConfig.version = NV_ENC_CONFIG_VER;
    initParams.encodeConfig = &encodeConfig;

    // Set rate control parameters
    encodeConfig.rcParams.version = NV_ENC_RC_PARAMS_VER;
    encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    encodeConfig.rcParams.averageBitRate = bitrate;
    encodeConfig.rcParams.maxBitRate = bitrate * 1.2;
    encodeConfig.rcParams.vbvBufferSize = bitrate;
    encodeConfig.rcParams.vbvInitialDelay = bitrate;

    nvStatus = nvEncFuncList.nvEncInitializeEncoder(session, &initParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVENC encoder\n");
        goto fail;
    }

    printf("NVENC: Successfully initialized encoder\n");
    return 0;

fail:
    nvenc_terminate();
    return -1;
}

int nvenc_prepare_frame(struct nv_frame *frame, VASurfaceID surface) {
    // Create input buffer
    NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams = { 0 };
    createInputBufferParams.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    createInputBufferParams.width = frame->width;
    createInputBufferParams.height = frame->height;
    createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    createInputBufferParams.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;

    NVENCSTATUS nvStatus = nvEncFuncList.nvEncCreateInputBuffer(session, &createInputBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to create NVENC input buffer\n");
        return -1;
    }

    frame->input_surface = createInputBufferParams.inputBuffer;

    // Create output buffer
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams = { 0 };
    createBitstreamBufferParams.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    createBitstreamBufferParams.size = frame->width * frame->height * 2; // Conservative estimate

    nvStatus = nvEncFuncList.nvEncCreateBitstreamBuffer(session, &createBitstreamBufferParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to create NVENC output buffer\n");
        return -1;
    }

    frame->output_buffer = createBitstreamBufferParams.bitstreamBuffer;
    frame->va_surface = surface;

    return 0;
}

int nvenc_encode_frame(struct nv_frame *frame) {
    NV_ENC_PIC_PARAMS picParams = { 0 };
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.inputBuffer = frame->input_surface;
    picParams.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    picParams.inputWidth = frame->width;
    picParams.inputHeight = frame->height;
    picParams.outputBitstream = frame->output_buffer;
    picParams.completionEvent = NULL;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

    NVENCSTATUS nvStatus = nvEncFuncList.nvEncEncodePicture(session, &picParams);
    if (nvStatus != NV_ENC_SUCCESS) {
        fprintf(stderr, "Failed to encode frame\n");
        return -1;
    }

    return 0;
}

int nvenc_get_profiles(VAProfile *profiles, int *num_profiles) {
    if (!profiles || !num_profiles || *num_profiles <= 0) {
        return -1;
    }

    int count = 0;
    
    // Add H.264 profiles
    if (count < *num_profiles) profiles[count++] = VAProfileH264Main;
    if (count < *num_profiles) profiles[count++] = VAProfileH264High;
    if (count < *num_profiles) profiles[count++] = VAProfileH264ConstrainedBaseline;
    
    // Add HEVC profiles if supported
    if (count < *num_profiles) profiles[count++] = VAProfileHEVCMain;
    
    *num_profiles = count;
    return 0;
}

void nvenc_terminate(void) {
    if (session) {
        nvEncFuncList.nvEncDestroyEncoder(session);
        session = NULL;
    }
    if (nvencCudaCtx) {
        cuCtxDestroy(nvencCudaCtx);
        nvencCudaCtx = NULL;
    }
    if (nvencLib) {
        dlclose(nvencLib);
        nvencLib = NULL;
    }
}
