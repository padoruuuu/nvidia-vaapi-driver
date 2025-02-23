#ifndef NVENC_H
#define NVENC_H

#include <nvEncodeAPI.h>
#include <va/va.h>

/* Structure to hold NVENC frame data */
struct nv_frame {
    void* input_buffer;
    NV_ENC_INPUT_PTR input_surface;
    NV_ENC_OUTPUT_PTR output_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    VASurfaceID va_surface;
};

/* Check if NVENC is available and supported */
int nvenc_is_available(void);

/* Get supported encoding profiles */
int nvenc_get_profiles(VAProfile *profiles, int *num_profiles);

/* Initialize the NVENC encoder with specific parameters */
int nvenc_init(unsigned int width, unsigned int height, unsigned int bitrate);

/* Prepare frame buffers for encoding */
int nvenc_prepare_frame(struct nv_frame *frame, VASurfaceID surface);

/* Encode a frame using NVENC */
int nvenc_encode_frame(struct nv_frame *frame);

/* Terminate NVENC encoder and free resources */
void nvenc_terminate(void);

#endif // NVENC_H
