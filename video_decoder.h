#pragma once

// c/c++
#include <ctime>
#include <list>
#include <mutex>
#include <thread>

// ffmpeg
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}


// link ffmpeg
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avutil.lib")



class ffmpeg_hw_decoder
{
public:
    ffmpeg_hw_decoder();
    ~ffmpeg_hw_decoder();

    // 初始化硬解码器
    bool init(enum AVCodecID codec_id = AV_CODEC_ID_HEVC, enum AVPixelFormat pixel_format = AV_PIX_FMT_CUDA, enum AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_CUDA);
    // 销毁硬解码器资源
    void uninit();
    // 判断硬解码器是否初始化失败
    bool is_failed();

    // 清空已解码的帧
    void flush();

    // 送包到解码器
    int send_packet(uint8_t *buffer, int size, int64_t dts, int64_t pts);
    // 获取已解码的帧
    AVFrame *receive_frame(int &ret, int &key_frame, uint64_t &dts, uint64_t &pts, int &width, int &height);
    // 销毁已解码的帧
    void free_frame(AVFrame *frame);

    // 获取最新帧的宽度
    int get_width();
    // 获取最新帧的高度
    int get_height();
    // 获取帧率
    double get_fps();

    // 获取解码器ID
    enum AVCodecID get_codec_id();
    // 获取像素格式
    enum AVPixelFormat get_pixel_format();
    // 获取硬件加速器类型
    enum AVHWDeviceType get_hw_device_type();

private:
    // 硬件加速器查找回调
    static enum AVPixelFormat get_pixel_format_callback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

    // 解码器
    AVCodec *m_decoder;
    // 硬件加速器上下文
    AVBufferRef *m_hw_device_context;
    // 解码器上下文
    AVCodecContext *m_decoder_context;

    // 解码器ID
    enum AVCodecID m_codec_id;
    // 像素格式
    enum AVPixelFormat m_pixel_format;
    // 硬件加速器类型
    enum AVHWDeviceType m_hw_device_type;

    int m_width;
    int m_height;
    double m_fps;

    bool m_failed;

    std::mutex m_mutex;
};


