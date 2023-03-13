#pragma once

// c
#include <stdint.h>

// flv
#include "flv_client.h"


#if !defined(WIN32)
#define FLV_HWACCEL_DECODER_EXPORT extern "C"
#else
#define FLV_HWACCEL_DECODER_EXPORT extern "C" __declspec(dllexport)
#endif



FLV_HWACCEL_DECODER_EXPORT int start_flv_client(void **flv, const char *ip, const int port, const char *gbcode, flv_client::recv_host_rgb24_to_np_cb recv_host_buf_to_np, flv_client::recv_host_rgb24_cb recv_host_buf_cb, flv_client::recv_device_rgb24_cb recv_device_buf_cb);
FLV_HWACCEL_DECODER_EXPORT void free_flv_client_buf(void *p, int64_t buf);
FLV_HWACCEL_DECODER_EXPORT void stop_flv_client(void *p);
