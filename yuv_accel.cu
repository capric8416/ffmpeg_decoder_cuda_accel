// self
#include "yuv_accel.h"

// log
#include "stdout_log.h"

// c
#include <stdint.h>

// c++
#include <chrono>

// cuda
#include <device_launch_parameters.h>



//#define BT601_RESTRICTED
//#define BT601_FULL
//#define BT709_RESTRICTED
#define BT709_FULL
//#define BT2020_RESTRICTED
//#define BT2020_FULL
#define SKIP_FETCH_RGB24
#define SKIP_SAVE_RGB_FILE



/*
* 在device端执行的函数，将float颜色值范围缩到[0, 255]
* 
* 参数列表
*   value: float, 颜色值
* 
* 返回值
*   uint8_t
*/
__device__
uint8_t clamp_float_to_byte(float value)
{
    value += 0.5f;
    if (value >= 255.0f) return 255;
    if (value <= 0.0f) return 0;
    return (uint8_t)value;
}


/*
* 在device端执行的函数，将int颜色值范围缩到[0, 255]
*
* 参数列表
*   value: float, 颜色值
*
* 返回值
*   uint8_t
*/
__device__
uint8_t clamp_int_to_byte(int value)
{
    if (value >= 255) return 255;
    if (value <= 0) return 0;
    return (uint8_t)value;
}


/*
* 在device端执行的核(线程)函数，执行单个像素由NV12向RGB24转换
* 
* 参数列表
*   buf_y: uint8_t *, y分量数据
*   buf_uv: uint8_t *, uv分量数据 (交错存放uvuvuv...)
*   linesize_y: int, y数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   linesize_uv: int, uv数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   buf_rgb24: uint8_t *, 输出的RGB24数据
*   width: int, 图像宽度
*   height: int, 图像高度
*   rgb_channels: int, RGB通道数
* 
* 返回值
*   无
*
* 
* 参考资料
* 
* https://en.wikipedia.org/wiki/YUV
* https://www.cnblogs.com/betterwgo/p/16712604.html
* https://www.cnblogs.com/riddick/p/7724877.html
* https://blog.csdn.net/liyuanbhu/article/details/68951683
*
*
* from mpv readme
*
* Available color spaces are :
*     ...
*     : bt.601 : ITU - R BT.601 (SD)
*     : bt.709 : ITU - R BT.709 (HD)
*     : bt.2020 - ncl : ITU - R BT.2020 non - constant luminance system
*     : bt.2020 - cl : ITU - R BT.2020 constant luminance system
*     ...
*
*
* from ffmpeg source
*
* enum AVColorSpace {
*     ...
*     AVCOL_SPC_BT709       = 1,  ///< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / derived in SMPTE RP 177 Annex B
*     ...
*     AVCOL_SPC_BT470BG     = 5,  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
*     AVCOL_SPC_SMPTE170M   = 6,  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC / functionally identical to above
*     ...
*     AVCOL_SPC_BT2020_NCL  = 9,  ///< ITU-R BT2020 non-constant luminance system
*     AVCOL_SPC_BT2020_CL   = 10, ///< ITU-R BT2020 constant luminance system
*     ...
*     AVCOL_SPC_ICTCP       = 14, ///< ITU-R BT.2100-0, ICtCp
*     ...
* };
*
*
* from libuv source
*
* // Conversion matrix for YVU to BGR
* LIBYUV_API extern const struct YuvConstants kYvuI601Constants;   // BT.601
* LIBYUV_API extern const struct YuvConstants kYvuJPEGConstants;   // BT.601 full
* LIBYUV_API extern const struct YuvConstants kYvuH709Constants;   // BT.709
* LIBYUV_API extern const struct YuvConstants kYvuF709Constants;   // BT.709 full
* LIBYUV_API extern const struct YuvConstants kYvu2020Constants;   // BT.2020
* LIBYUV_API extern const struct YuvConstants kYvuV2020Constants;  // BT.2020 full
*/
__global__
void nv12_to_rgb24_kernel(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, uint8_t *buf_rgb24, int width, int height, int rgb_channels)
{
    const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_id_x >= width && thread_id_y >= height) {
        return;
    }

    int index_y, index_u, index_v;
    index_y = thread_id_y * linesize_y + thread_id_x;

    uint8_t y, u, v;
    y = buf_y[index_y];

    if (thread_id_x % 2 == 0) {
        index_u = thread_id_y / 2 * linesize_uv + thread_id_x;
        index_v = thread_id_y / 2 * linesize_uv + thread_id_x + 1;
        u = buf_uv[index_u];
        v = buf_uv[index_v];
    } else if (thread_id_x % 2 == 1) {
        index_v = thread_id_y / 2 * linesize_uv + thread_id_x;
        index_u = thread_id_y / 2 * linesize_uv + thread_id_x - 1;
        u = buf_uv[index_u];
        v = buf_uv[index_v];
    }

#ifdef BT601_RESTRICTED
    // bt.601 局部色域
    // R
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 0] = clamp_float_to_byte(1.164 * (y - 16) + 1.596 * (v - 128));
    // G
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 1] = clamp_float_to_byte(1.164 * (y - 16) - 0.391 * (u - 128) - 0.813 * (v - 128));
    // B
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 2] = clamp_float_to_byte(1.164 * (y - 16) + 2.018 * (u - 128));
#endif // BT601_RESTRICTED

#ifdef BT601_FULL
    // bt.601 全色域
    // R
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 0] = clamp_float_to_byte(y + 1.403 * (v - 128));
    // G
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 1] = clamp_float_to_byte(y - 0.343 * (u - 128) - 0.714 * (v - 128));
    // B
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 2] = clamp_float_to_byte(y + 1.770 * (u - 128));
#endif // BT601_LIMIT

#ifdef BT709_RESTRICTED
    // bt.709 局部色域
    // R
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 0] = clamp_float_to_byte(1.164 * (y - 16) + 1.793 * (v - 128));
    // G
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 1] = clamp_float_to_byte(1.164 * (y - 16) - 0.213 * (u - 128) - 0.533 * (v - 128));
    // B
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 2] = clamp_float_to_byte(1.164 * (y - 16) + 2.112 * (u - 128));
#endif // BT709_RESTRICTED

#ifdef BT709_FULL
    // bt.709 全色域
    // R
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 0] = clamp_float_to_byte(y + 1.280 * (v - 128));
    // G
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 1] = clamp_float_to_byte(y - 0.215 * (u - 128) - 0.381 * (v - 128));
    // B
    buf_rgb24[(thread_id_y * width + thread_id_x) * rgb_channels + 2] = clamp_float_to_byte(y + 2.128 * (u - 128));
#endif // BT709_FULL

#ifdef BT2020_RESTRICTED
    // bt.2020 局部色域
#endif // BT2020_RESTRICTED

#ifdef BT2020_FULL
    // bt.2020 全色域
#endif // BT2020_FULL
}


/*
* cuda资源准备，然后等待转换完成
*
* 参数列表
*   buf_y: uint8_t *, y分量数据
*   buf_uv: uint8_t *, uv分量数据 (交错存放uvuvuv...)
*   linesize_y: int, y数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   linesize_uv: int, uv数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   buf_rgb24: uint8_t *, 输出的RGB24数据
*   width: int, 图像宽度
*   height: int, 图像高度
*
* 返回值
*   cudaError_t
*/
cudaError_t launch_nv12_to_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, uint8_t *buf_rgb24, int width, int height)
{
    // 检查指针
    if (!buf_y || !buf_uv || !buf_rgb24) {
        return cudaErrorInvalidDevicePointer;
    }

    // 检查参数
    if (width <= 0 || height <= 0 || linesize_y < width || linesize_uv < width) {
        return cudaErrorInvalidValue;
    }

    // 网格和块配置 (目标是一个线程处理一个像素，注意冲突，参照nvidia gpu架构和编程指南)
    dim3 block(32, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    // 并行执行
    nv12_to_rgb24_kernel <<<grid, block>>> (buf_y, buf_uv, linesize_y, linesize_uv, buf_rgb24, width, height, 3);

    // 等待完成
    return cudaDeviceSynchronize();
}


/*
* host和device资源交换
*
* 参数列表
*   buf_y: uint8_t *, y分量数据
*   buf_uv: uint8_t *, uv分量数据 (交错存放uvuvuv...)
*   linesize_y: int, y数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   linesize_uv: int, uv数据一行的宽度 (注意，在CPU上通常等于width，而在GPU上通常会大于width)
*   width: int, 图像宽度
*   height: int, 图像高度
*   path: char *, 要保存的rgb文件路径，如果为空就不保存
*
* 返回值
*   cudaError_t
*/
cudaError_t save_nv12_to_rgb24_file(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, const char *path)
{
    cudaError_t error = cudaSuccess;

    // rgb24大小
    const int nbytes_rgb24 = 3 * width * height;

    // device端内存
    uint8_t *d_buf_rgb24 = nullptr;
    // host端内存
    uint8_t *h_buf_rgb24 = nullptr;
    // 本地文件
    FILE *file_rgb24 = nullptr;

    do {
        // 分配device端内存
        if (cudaSuccess != (error = cudaMalloc(&d_buf_rgb24, nbytes_rgb24))) {
            LOG_ERROR("cudaMalloc rgb24 buffer fail, size: %d, code: %d\n", nbytes_rgb24, error);
            break;
        }

        // 执行转换
        if (cudaSuccess != (error = launch_nv12_to_rgb24(buf_y, buf_uv, linesize_y, linesize_uv, d_buf_rgb24, width, height))) {
            LOG_ERROR(
                "launch_nv12_to_rgb24 fail, pInputYData: %p, pInputUVData: %p, linesizeY: %d, linesizeUV: %d, pDeviceOutputRGB: %p, width: %d, height: %d, code: %d\n",
                buf_y, buf_uv, linesize_y, linesize_uv, d_buf_rgb24, width, height, error
            );
            break;
        }

        // 分配host端内存
        h_buf_rgb24 = new uint8_t[nbytes_rgb24];
        if (!h_buf_rgb24) {
            LOG_ERROR("new host rgb24 buffer fail, size: %d, code: %d\n", nbytes_rgb24, error);
            break;
        }

#ifndef SKIP_FETCH_RGB24
        // 拷贝device端内存到host端内存
        if (cudaSuccess != (error = cudaMemcpy(h_buf_rgb24, d_buf_rgb24, nbytes_rgb24, cudaMemcpyDeviceToHost))) {
            LOG_ERROR("cudaMemcpy device rgb24 buffer to host fail, code: %d\n", error);
            break;
        }

        if (!path) {
            // 无需保存到文件
            break;
        }

#ifndef SKIP_SAVE_RGB_FILE
        // 保存rgb到文件
        static int i = 0;
        char filepath[256];
        sprintf(filepath, "%s/%d.rgb", path, i++);
        file_rgb24 = fopen(filepath, "wb");
        if (!file_rgb24) {
            LOG_ERROR("fopen fail, path: %s\n", path);
            break;
        }

        size_t written = fwrite(h_buf_rgb24, 1, nbytes_rgb24, file_rgb24);
        if (written != nbytes_rgb24) {
            LOG_ERROR("fwrite fail, path: %s\n", path);
            break;
        }
#endif // SKIP_SAVE_RGB_FILE
#endif // SKIP_FETCH_RGB24
    } while (false);

    // 释放device端内存
    if (d_buf_rgb24) {
        if (cudaSuccess != (error = cudaFree(d_buf_rgb24))) {
            LOG_ERROR("cudaFree device rgb24 buffer fail, error: %d\n", error);
        }
        d_buf_rgb24 = nullptr;
    }

    // 释放host端内存
    if (h_buf_rgb24) {
        delete[] h_buf_rgb24;
        h_buf_rgb24 = nullptr;
    }

    // 关闭文件
    if (file_rgb24) {
        fclose(file_rgb24);
        file_rgb24 = nullptr;
    }

    return error;
}


cudaError receive_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, int64_t &buf_host, int64_t &buf_device, bool copy)
{
    cudaError_t error = cudaSuccess;

    // rgb24大小
    const int nbytes_rgb24 = 3 * width * height;

    // device端内存
    uint8_t *d_buf_rgb24 = nullptr;
    // host端内存
    uint8_t *h_buf_rgb24 = nullptr;

    do {
        // 分配device端内存
        if (cudaSuccess != (error = cudaMalloc(&d_buf_rgb24, nbytes_rgb24))) {
            LOG_ERROR("cudaMalloc rgb24 buffer fail, size: %d, code: %d\n", nbytes_rgb24, error);
            break;
        }

        // 执行转换
        if (cudaSuccess != (error = launch_nv12_to_rgb24(buf_y, buf_uv, linesize_y, linesize_uv, d_buf_rgb24, width, height))) {
            LOG_ERROR(
                "launch_nv12_to_rgb24 fail, pInputYData: %p, pInputUVData: %p, linesizeY: %d, linesizeUV: %d, pDeviceOutputRGB: %p, width: %d, height: %d, code: %d\n",
                buf_y, buf_uv, linesize_y, linesize_uv, d_buf_rgb24, width, height, error
            );
            break;
        }

        buf_device = int64_t(d_buf_rgb24);

        if (copy) {
            // 分配host端内存
            h_buf_rgb24 = new uint8_t[nbytes_rgb24];
            if (!h_buf_rgb24) {
                LOG_ERROR("new host rgb24 buffer fail, size: %d, code: %d\n", nbytes_rgb24, error);
                break;
            }

            // 拷贝device端内存到host端内存
            if (cudaSuccess != (error = cudaMemcpy(h_buf_rgb24, d_buf_rgb24, nbytes_rgb24, cudaMemcpyDeviceToHost))) {
                LOG_ERROR("cudaMemcpy device rgb24 buffer to host fail, code: %d\n", error);
                break;
            }

            buf_host = int64_t(h_buf_rgb24);
        }
    } while (false);

    // 释放device端内存
    if (copy && d_buf_rgb24) {
        if (cudaSuccess != (error = cudaFree(d_buf_rgb24))) {
            LOG_ERROR("cudaFree device rgb24 buffer fail, error: %d\n", error);
        }
        d_buf_rgb24 = nullptr;
        buf_device = 0;
    }

    // 释放host端内存
    if (!copy && h_buf_rgb24) {
        delete[] h_buf_rgb24;
        h_buf_rgb24 = nullptr;
        buf_host = 0;
    }

    return error;
}


// nvcc -Ioutput/include -Loutput/lib -lavcodec -lavutil -lavformat cuda_nv12_to_rgb24.cu -o cuda_nv12_to_rgb24.out
//
// 000001B8F1746290.1920x1080.hevc.25fps.9595.00060680.363kbps.265
// Windows 10, Intel Core i5-9400 2.8GHz, NVIDIA GeForce RTX 2070 8G
//   拷贝RGB回内存 38255ms -> 250 fps -> 22% CPU / 93% 3D / 14% Copy / 42% Video Decode
// 不拷贝RGB回内存 22616ms -> 424 fps -> 14% CPU / 96% 3D / 07% Copy / 25% Video Decode

