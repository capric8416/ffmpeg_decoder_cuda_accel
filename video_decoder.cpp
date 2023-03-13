// self
#include "video_decoder.h"

// log
#include "stdout_log.h"



ffmpeg_hw_decoder::ffmpeg_hw_decoder()
    : m_decoder(nullptr)
    , m_decoder_context(nullptr)
    , m_hw_device_context(nullptr)

    , m_codec_id(AV_CODEC_ID_NONE)
    , m_pixel_format(AV_PIX_FMT_NONE)
    , m_hw_device_type(AV_HWDEVICE_TYPE_NONE)

    , m_width(0)
    , m_height(0)
    , m_fps(0)

    , m_failed(false)
{
    av_log_set_level(AV_LOG_INFO);
    av_log_set_callback(
        [](void *ptr, int level, const char *fmt, va_list vl) {
            if (level > av_log_get_level()) {
                return;
            }

            vprintf(fmt, vl);
        }
    );
}


ffmpeg_hw_decoder::~ffmpeg_hw_decoder()
{
    uninit();
}


bool ffmpeg_hw_decoder::init(enum AVCodecID codec_id, enum AVPixelFormat pixel_format, enum AVHWDeviceType hw_device_type)
{
    if (m_decoder != nullptr) {
        return false;
    }

    LOG_INFO("<begin> codec: %s\n", codec_id == AV_CODEC_ID_H264 ? "h264" : "h265");

    std::lock_guard<std::mutex> locker(m_mutex);

    m_failed = true;

    m_codec_id = codec_id;
    m_pixel_format = pixel_format;
    m_hw_device_type = hw_device_type;

    int ret = 0;
    do {
        ret = av_hwdevice_ctx_create(&m_hw_device_context, m_hw_device_type, "auto", NULL, 0);
        if (ret != 0) {
            LOG_ERROR("av_hwdevice_ctx_create fail, code: %d\n", ret);
            break;
        }

        m_decoder = (AVCodec *)avcodec_find_decoder(m_codec_id);
        if (m_decoder == NULL) {
            LOG_ERROR("avcodec_find_decoder fail\n");
            break;
        }

        for (int i = 0;; i++) {
            const AVCodecHWConfig *config = avcodec_get_hw_config(m_decoder, i);
            if (config == nullptr) {
                LOG_ERROR("avcodec_get_hw_config, decoder does not support device type(%s)\n", av_hwdevice_get_type_name(m_hw_device_type));
                break;
            }

            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == m_hw_device_type) {
                if (config->pix_fmt != m_pixel_format) {
                    LOG_INFO("reset m_HWDeviceType(%s) with config->pix_fmt(%s)\n", av_get_pix_fmt_name(m_pixel_format), av_get_pix_fmt_name(config->pix_fmt));
                    m_pixel_format = config->pix_fmt;
                }
                break;
            }
        }

        m_decoder_context = avcodec_alloc_context3(m_decoder);
        if (m_decoder_context == NULL) {
            LOG_ERROR("avcodec_alloc_context3 fail\n");
            break;
        }

        m_decoder_context->opaque = this;
        m_decoder_context->hw_device_ctx = av_buffer_ref(m_hw_device_context);
        m_decoder_context->get_format = get_pixel_format_callback;

        ret = avcodec_open2(m_decoder_context, m_decoder, NULL);
        if (m_decoder_context == NULL) {
            LOG_ERROR("avcodec_open2 fail, code: %d\n", ret);
            break;
        }

        m_width = m_decoder_context->width;
        m_height = m_decoder_context->height;

        m_failed = false;
    } while (false);

    LOG_INFO("</end>\n");

    return true;
}


void ffmpeg_hw_decoder::uninit()
{
    LOG_INFO("<begin>\n");

    std::lock_guard<std::mutex> locker(m_mutex);

    flush();

    if (m_decoder_context != nullptr) {
        av_buffer_unref(&m_hw_device_context);

        avcodec_free_context(&m_decoder_context);
        m_decoder_context = nullptr;
    }

    m_decoder = nullptr;

    m_codec_id = AV_CODEC_ID_NONE;

    LOG_INFO("</end>\n");
}


bool ffmpeg_hw_decoder::is_failed()
{
    return m_failed;
}


void ffmpeg_hw_decoder::flush()
{
    if (m_decoder_context == nullptr) {
        return;
    }

    int ret = 0;
    int keyFrame = 0;
    uint64_t dts = 0;
    uint64_t pts = 0;
    while (true) {
        AVFrame *pFrame = receive_frame(ret, keyFrame, dts, pts, m_width, m_height);
        if (pFrame != nullptr) {
            av_frame_unref(pFrame);
            continue;  // flush next decoded frame
        }

        break;  // all decoded frames have been flushed
    }
}


int ffmpeg_hw_decoder::send_packet(uint8_t *buffer, int size, int64_t dts, int64_t pts)
{
    AVPacket *packet = av_packet_alloc();
    if (!packet) {
        LOG_WARNING("av_packet_alloc fail\n");
        return -1;
    }

    packet->dts = dts;
    packet->pts = pts;
    packet->size = size;
    packet->data = buffer;
    int ret = avcodec_send_packet(m_decoder_context, packet);
    if (ret < 0) {
        LOG_WARNING("error during decoding with code %d\n", ret);
    }

    av_packet_free(&packet);

    return ret;
}


AVFrame *ffmpeg_hw_decoder::receive_frame(int &ret, int &key_frame, uint64_t &dts, uint64_t &pts, int &width, int &height)
{
    ret = 0;

    while (ret >= 0) {
        AVFrame *frame = NULL;
        if (!(frame = av_frame_alloc())) {
            LOG_WARNING("can not alloc frame\n");
            ret = AVERROR(ENOMEM);
            break;
        }

        ret = avcodec_receive_frame(m_decoder_context, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&frame);
            break;
        }
        else if (ret < 0) {
            LOG_WARNING("error while decoding with code 0x%x\n", ret);
            av_frame_free(&frame);
            break;
        }

        key_frame = frame->key_frame;
        dts = (uint64_t)frame->pkt_dts;
        pts = (uint64_t)frame->pts;
        width = frame->width;
        height = frame->height;
        m_width = width;
        m_height = height;

        return frame;
    }

    ret = -1;
    return nullptr;
}


void ffmpeg_hw_decoder::free_frame(AVFrame *frame)
{
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
}


int ffmpeg_hw_decoder::get_width()
{
    return m_width;
}


int ffmpeg_hw_decoder::get_height()
{
    return m_height;
}


double ffmpeg_hw_decoder::get_fps()
{
    if (m_fps < 1e-15) {
        m_fps = (double)m_decoder_context->framerate.den / m_decoder_context->framerate.num;
    }
    return m_fps;
}


AVCodecID ffmpeg_hw_decoder::get_codec_id()
{
    return m_codec_id;
}


AVPixelFormat ffmpeg_hw_decoder::get_pixel_format()
{
    return m_pixel_format;
}


AVHWDeviceType ffmpeg_hw_decoder::get_hw_device_type()
{
    return m_hw_device_type;
}


enum AVPixelFormat ffmpeg_hw_decoder::get_pixel_format_callback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    auto thiz = (ffmpeg_hw_decoder *)ctx->opaque;

    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == thiz->m_pixel_format) {
            LOG_INFO("get hw surface format(%s) succeed\n", av_get_pix_fmt_name(*p));
            thiz->m_failed = false;
            return *p;
        }
    }

    LOG_ERROR("get hw surface format(%s) failed\n", av_get_pix_fmt_name(thiz->m_pixel_format));

    thiz->m_failed = true;

    return AV_PIX_FMT_NONE;
}


#if defined(TEST_FFMPEG_DECODER)
bool ReadNALUIndex(std::list<size_t> &results, const char *path)
{
    FILE *fp_index = fopen(path, "rb");
    if (fp_index == NULL) {
        return false;
    }

    results.clear();

    uint8_t buf[sizeof(int)];
    while (true) {
        size_t len = fread(buf, sizeof(uint8_t), sizeof(int32_t), fp_index);
        if (len == 0) {
            break;
        }

        int32_t nbytes = 0;
        std::memcpy(&nbytes, buf, sizeof(int32_t));

        results.push_back(nbytes);
    }

    fclose(fp_index);

    return true;
}


int main(int argc, char **argv)
{
    FFmpegDecoder decoder;
    decoder.Init(AV_CODEC_ID_H265);
    bool failed = decoder.IsFailed();

    while (true) {
        std::list<size_t> index;
        ReadNALUIndex(index, "E:/ElementaryStream/000001B89AB13DD0.1920x1080.hevc.25fps.9665.00060660.336kbps.265.index");
        FILE *fp = fopen("E:/ElementaryStream/000001B89AB13DD0.1920x1080.hevc.25fps.9665.00060660.336kbps.265", "rb");
        for (auto iter = index.begin(); iter != index.end(); iter++) {
            uint8_t *buf = new uint8_t[*iter];
            size_t count = fread(buf, 1, *iter, fp);

            decoder.SendPacket(buf, count, 0, 0);

            delete[] buf;

            int ret = 0;
            int key = 0;
            uint64_t dts = 0;
            uint64_t pts = 0;
            int width = 0;
            int height = 0;
            AVFrame *pFrame = decoder.ReceiveFrame(ret, key, dts, pts, width, height);

            if (pFrame != nullptr) {
                decoder.FreeFrame(pFrame);
            }
        }
    }

    return 0;
}
#endif

