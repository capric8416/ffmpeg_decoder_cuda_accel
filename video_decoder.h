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

    // ��ʼ��Ӳ������
    bool init(enum AVCodecID codec_id = AV_CODEC_ID_HEVC, enum AVPixelFormat pixel_format = AV_PIX_FMT_CUDA, enum AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_CUDA);
    // ����Ӳ��������Դ
    void uninit();
    // �ж�Ӳ�������Ƿ��ʼ��ʧ��
    bool is_failed();

    // ����ѽ����֡
    void flush();

    // �Ͱ���������
    int send_packet(uint8_t *buffer, int size, int64_t dts, int64_t pts);
    // ��ȡ�ѽ����֡
    AVFrame *receive_frame(int &ret, int &key_frame, uint64_t &dts, uint64_t &pts, int &width, int &height);
    // �����ѽ����֡
    void free_frame(AVFrame *frame);

    // ��ȡ����֡�Ŀ��
    int get_width();
    // ��ȡ����֡�ĸ߶�
    int get_height();
    // ��ȡ֡��
    double get_fps();

    // ��ȡ������ID
    enum AVCodecID get_codec_id();
    // ��ȡ���ظ�ʽ
    enum AVPixelFormat get_pixel_format();
    // ��ȡӲ������������
    enum AVHWDeviceType get_hw_device_type();

private:
    // Ӳ�����������һص�
    static enum AVPixelFormat get_pixel_format_callback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

    // ������
    AVCodec *m_decoder;
    // Ӳ��������������
    AVBufferRef *m_hw_device_context;
    // ������������
    AVCodecContext *m_decoder_context;

    // ������ID
    enum AVCodecID m_codec_id;
    // ���ظ�ʽ
    enum AVPixelFormat m_pixel_format;
    // Ӳ������������
    enum AVHWDeviceType m_hw_device_type;

    int m_width;
    int m_height;
    double m_fps;

    bool m_failed;

    std::mutex m_mutex;
};


