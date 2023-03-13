#pragma once


// c
#include <stdint.h>

// cuda
#include <cuda_runtime.h>



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
cudaError_t launch_nv12_to_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, uint8_t *buf_rgb24, int width, int height);


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
cudaError_t save_nv12_to_rgb24_file(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, const char *path = nullptr);


cudaError receive_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, int64_t &buf_host, int64_t &buf_device, bool copy);
