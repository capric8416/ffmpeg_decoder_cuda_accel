#pragma once


// c
#include <stdint.h>

// cuda
#include <cuda_runtime.h>



/*
* cuda��Դ׼����Ȼ��ȴ�ת�����
*
* �����б�
*   buf_y: uint8_t *, y��������
*   buf_uv: uint8_t *, uv�������� (������uvuvuv...)
*   linesize_y: int, y����һ�еĿ�� (ע�⣬��CPU��ͨ������width������GPU��ͨ�������width)
*   linesize_uv: int, uv����һ�еĿ�� (ע�⣬��CPU��ͨ������width������GPU��ͨ�������width)
*   buf_rgb24: uint8_t *, �����RGB24����
*   width: int, ͼ����
*   height: int, ͼ��߶�
*
* ����ֵ
*   cudaError_t
*/
cudaError_t launch_nv12_to_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, uint8_t *buf_rgb24, int width, int height);


/*
* host��device��Դ����
*
* �����б�
*   buf_y: uint8_t *, y��������
*   buf_uv: uint8_t *, uv�������� (������uvuvuv...)
*   linesize_y: int, y����һ�еĿ�� (ע�⣬��CPU��ͨ������width������GPU��ͨ�������width)
*   linesize_uv: int, uv����һ�еĿ�� (ע�⣬��CPU��ͨ������width������GPU��ͨ�������width)
*   width: int, ͼ����
*   height: int, ͼ��߶�
*   path: char *, Ҫ�����rgb�ļ�·�������Ϊ�վͲ�����
*
* ����ֵ
*   cudaError_t
*/
cudaError_t save_nv12_to_rgb24_file(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, const char *path = nullptr);


cudaError receive_rgb24(uint8_t *buf_y, uint8_t *buf_uv, int linesize_y, int linesize_uv, int width, int height, int64_t &buf_host, int64_t &buf_device, bool copy);
