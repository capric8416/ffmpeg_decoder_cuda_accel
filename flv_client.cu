// self
#include "flv_client.h"

// cuda yuv accel
#include "yuv_accel.h"

// log
#include "stdout_log.h"

// qt
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>



#define FLV_HEADER_LEN      9
#define TAG_HEADER_LEN	    11
#define TAG_TIME_POS        4
#define STREAM_ALIVE_S      195
#define GET_TIMESTAMP_S     std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count



flv_client::flv_client(uv_loop_t *loop)
    : m_loop(loop)
    , m_gateway_ip("")
    , m_gateway_port(0)
    , m_gbcode("")
    , m_stream_ip("")
    , m_stream_port(0)
    , m_stream_id(0)
    , m_stream_path("")
    , m_stream_query("")
    , m_flv_ring_buffer(nullptr)
    , m_flv_ring_buffer_length(4 * 1024 * 1024)
    , m_flv_ring_buffer_write_offset(0)
    , m_frame_buffer_length(1024 * 1024)
    , m_frame_buffer(nullptr)
    , m_sps_len(0)
    , m_pps_len(0)
    , m_need_search_i_frame(true)
    , m_found_flv_header(false)
    , m_need_search_magic(false)
    , m_stream_keepaliving(false)
    , m_stream_addr_not_found(false)
    , m_stream_recv_eof(false)
    , m_stream_recv_fail(false)
    , m_stream_keepalive_fail(false)
    , m_stream_close_fail(false)
    , m_last_keepalive_ts(0)
    , m_recv_host_rgb24_cb(nullptr)
    , m_recv_device_rgb24_cb(nullptr)
{
    m_http_client.set_loop(loop);

    memset(m_pps, 0, 64);
    memset(m_sps, 0, 64);

    // 创建4M内部缓冲区用于网络数据拼帧解析
    m_flv_ring_buffer = new uint8_t[m_flv_ring_buffer_length];
    m_flv_ring_buffer_write_offset = 0;

    // 保存拼接出的1帧数据
    m_frame_buffer = new uint8_t[m_frame_buffer_length];

    // 初始化定时器
    uv_timer_init(loop, &m_stream_keepalive_timer);
    uv_timer_init(loop, &m_stream_status_check_timer);

    // 开始状态检测
    start_stream_status_checker();
}


flv_client::~flv_client()
{
    stop_stream_keepalive();
    uv_close((uv_handle_t *)&m_stream_keepalive_timer, NULL);

    stop_stream_status_checker();
    uv_close((uv_handle_t *)&m_stream_status_check_timer, NULL);

    request_stream_close();

    if (m_flv_ring_buffer) {
        delete[] m_flv_ring_buffer;
        m_flv_ring_buffer = nullptr;
    }

    if (m_frame_buffer) {
        delete[] m_frame_buffer;
        m_frame_buffer = nullptr;
    }
}


void flv_client::stop()
{
    uv_loop_close(m_loop);
}


void flv_client::set_gateway_addr(const char *ip, const int port)
{
    m_gateway_ip = ip;
    m_gateway_port = port;
}


void flv_client::set_output_callbacks(recv_host_rgb24_to_np_cb recv_host_to_np, recv_host_rgb24_cb recv_host, recv_device_rgb24_cb recv_device)
{
    m_recv_host_rgb24_to_np_cb = recv_host_to_np;
    m_recv_host_rgb24_cb = recv_host;
    m_recv_device_rgb24_cb = recv_device;
}


void flv_client::free_output_buf(uint8_t *buf)
{
    if (m_recv_device_rgb24_cb && buf) {
        cudaFree(buf);
    }

    if (m_recv_host_rgb24_cb && buf) {
        delete[] buf;
    }
}


void flv_client::request_stream_info()
{
    m_stream_addr_not_found = false;

    if (m_last_keepalive_ts == 0) {
        m_last_keepalive_ts = GET_TIMESTAMP_S() + 15;
    }

    auto params = new async_http_client::req_params_t;
    params->ctx = this;
    params->ip = m_gateway_ip;
    params->port = m_gateway_port;
    params->path = "/vtdu/http/play";
    params->query = "gb_code=" + m_gbcode;
    params->recv_eof_cb = receive_stream_info_success;
    params->recv_fail_cb = receive_stream_info_fail;
    params->conn_fail_cb = [](void *ctx, int code) {
        auto params = (async_http_client::req_params_t *)ctx;
        if (!params) {
            return;
        }

        auto thiz = (flv_client *)params->ctx;
        thiz->receive_stream_info_fail(params, code);
    };
    params->write_fail_cb = params->conn_fail_cb;
    m_http_client.request(params);
}


void flv_client::receive_stream_info_success(void *ctx)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    QJsonParseError parseError;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QByteArray((const char *)params->resp_body.data(), (int)params->resp_body.size()), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        // 解析失败
        LOG_ERROR("parse json fail\n");
    }
    else {
        QJsonObject rootObj = jsonDoc.object();
        int error_code = rootObj.value("error_code").toInt();
        if (error_code != 0) {
            // 非200
            if (params->resp_body.back() != '\0') {
                params->resp_body.push_back('\0');
            }
            LOG_ERROR("%s\n", params->resp_body.data());

            thiz->set_stream_info(0, "", 0, "", "");

            thiz->m_stream_addr_not_found = true;
        }
        else {
            std::string url = std::string(rootObj.value("http_uri").toString().toUtf8());

            // 移除左侧的schema
            std::string ip = url.substr(url.find("://") + 3);
            // 移除右侧的端口
            ip = ip.substr(0, ip.find(":"));

            // 端口
            int port = rootObj.value("http_port").toInt();

            // 路径和参数
            std::string path = "/live";
            std::string query = url.substr(url.find("?") + 1);

            // 流标识
            std::string stream = url.substr(url.find("stream=") + 7);
            int stream_id = stream.length() > 0 ? std::atoi(stream.c_str()) : 0;

            thiz->set_stream_info(stream_id, ip, port, path, query);

            // 开始请求流
            thiz->connect_stream();

            thiz->m_stream_addr_not_found = false;
            thiz->m_last_keepalive_ts = GET_TIMESTAMP_S() + 15;
        }
    }

    params->http_client->remove_request_history(params);
}


void flv_client::receive_stream_info_fail(void *ctx, ssize_t code)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    params->http_client->remove_request_history(params);

    thiz->m_stream_addr_not_found = true;
}


void flv_client::request_stream_keep_alive()
{
    m_stream_keepalive_fail = false;

    auto params = new async_http_client::req_params_t;
    params->ctx = this;
    params->ip = m_gateway_ip;
    params->port = m_gateway_port;
    params->path = "/vtdu/http/kpalive";
    params->query = "session_id=" + std::to_string(m_stream_id);
    params->recv_eof_cb = receive_stream_keep_alive_success;
    params->recv_fail_cb = receive_stream_keep_alive_fail;
    params->conn_fail_cb = [](void *ctx, int code) {
        auto params = (async_http_client::req_params_t *)ctx;
        if (!params) {
            return;
        }

        auto thiz = (flv_client *)params->ctx;
        thiz->receive_stream_keep_alive_fail(params, code);
    };
    params->write_fail_cb = params->conn_fail_cb;
    m_http_client.request(params);
}


void flv_client::receive_stream_keep_alive_success(void *ctx)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    QJsonParseError parseError;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QByteArray((const char *)params->resp_body.data(), (int)params->resp_body.size()), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        // 解析失败
        LOG_ERROR("parse json fail\n");
    }
    else {
        QJsonObject rootObj = jsonDoc.object();
        int error_code = rootObj.value("error_code").toInt();
        if (error_code != 0) {
            // 非200
            if (params->resp_body.back() != '\0') {
                params->resp_body.push_back('\0');
            }
            LOG_ERROR("%s\n", params->resp_body.data());
        }
        else {
            thiz->m_stream_keepalive_fail = false;
            thiz->m_last_keepalive_ts = GET_TIMESTAMP_S();
        }
    }

    thiz->m_http_client.remove_request_history(params);
}


void flv_client::receive_stream_keep_alive_fail(void *ctx, ssize_t code)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    params->http_client->remove_request_history(params);

    thiz->m_stream_keepalive_fail = true;
}


void flv_client::request_stream_close()
{
    m_stream_close_fail = false;

    // 流媒体默认值：如果3分钟没有保活就把流关闭
    // 做时间检查，避免把别人的流关了
    int64_t delta_s = GET_TIMESTAMP_S() - m_last_keepalive_ts;
    if (delta_s > STREAM_ALIVE_S) {
        //LOG_INFO("skip stream closing because %lld seconds have passed since the last keepalive\n", delta_s);
        m_stream_id = 0;
        return;
    }

    if (m_stream_id <= 0) {
        return;
    }

    auto params = new async_http_client::req_params_t;
    params->ctx = this;
    params->ip = m_gateway_ip;
    params->port = m_gateway_port;
    params->path = "/sdk/dlg/close";
    params->query = "session_id=" + std::to_string(m_stream_id);
    params->recv_eof_cb = receive_stream_close_success;
    params->recv_fail_cb = receive_stream_close_fail;
    params->conn_fail_cb = [](void *ctx, int code) {
        auto params = (async_http_client::req_params_t *)ctx;
        if (!params) {
            return;
        }

        auto thiz = (flv_client *)params->ctx;
        thiz->receive_stream_close_fail(params, code);
    };
    params->write_fail_cb = params->conn_fail_cb;
    m_http_client.request(params);
}


void flv_client::receive_stream_close_success(void *ctx)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    QJsonParseError parseError;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QByteArray((const char *)params->resp_body.data(), (int)params->resp_body.size()), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        // 解析失败
        LOG_ERROR("parse json fail\n");
    }
    else {
        QJsonObject rootObj = jsonDoc.object();
        int error_code = rootObj.value("error_code").toInt();
        if (error_code != 0) {
            // 非200
            if (params->resp_body.back() != '\0') {
                params->resp_body.push_back('\0');
            }
            LOG_ERROR("%s\n", params->resp_body.data());
            // 流不存在
            if (error_code == 70000008) {
                thiz->m_stream_addr_not_found = true;
            }
        }

        thiz->m_stream_id = 0;
        thiz->m_stream_close_fail = false;
    }

    thiz->m_http_client.remove_request_history(params);
}


void flv_client::receive_stream_close_fail(void *ctx, ssize_t code)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    params->http_client->remove_request_history(params);

    thiz->m_stream_close_fail = true;
}


void flv_client::set_stream_info(const int id, const std::string ip, const int port, const std::string path, const std::string query)
{
    m_stream_id = id;
    m_stream_ip = ip;
    m_stream_port = port;
    m_stream_path = path;
    m_stream_query = query;
}


void flv_client::connect_stream()
{
    request_stream_content();
}


void flv_client::connect_stream(std::string gbcode)
{
    if (m_gbcode != gbcode) {
        m_gbcode = gbcode;
    }

    request_stream_info();
}


void flv_client::disconnect_stream(async_http_client::req_params_t *params, bool close)
{
    if (params) {
        params->http_client->remove_request_history(params);
    }

    if (close) {
        request_stream_close();
    }
}


void flv_client::reconnect_stream(async_http_client::req_params_t *params, bool close)
{

    if (!close) {
        // 使用现有的流id发起拉流请求
        disconnect_stream(params, false);
        connect_stream();
    }
    else {
        // 使用现有的国标码先请求流信息再发起拉流请求
        disconnect_stream(params, true);
        connect_stream(m_gbcode);
    }
}


void flv_client::start_stream_keepalive()
{
    if (m_stream_keepaliving) {
        return;
    }

    m_stream_keepaliving = true;

    m_stream_keepalive_timer.data = this;

    uv_timer_start(
        &m_stream_keepalive_timer,
        [](uv_timer_t *handle) {
            auto thiz = (flv_client *)handle->data;
            thiz->request_stream_keep_alive();
        },
        14000,
        14000
    );
}


void flv_client::stop_stream_keepalive()
{
    m_stream_keepaliving = false;
    uv_timer_stop(&m_stream_keepalive_timer);
}


void flv_client::start_stream_status_checker()
{
    m_stream_status_check_timer.data = this;

    uv_timer_start(
        &m_stream_status_check_timer,
        [](uv_timer_t *handle) {
            auto thiz = (flv_client *)handle->data;

            // 流媒体默认值：如果3分钟没有保活就把流关闭
            // 做时间检查，直接重新取流
            int64_t delta_s = GET_TIMESTAMP_S() - thiz->m_last_keepalive_ts;
            if (!thiz->m_stream_addr_not_found && delta_s > STREAM_ALIVE_S) {
                LOG_INFO("request_stream_info because %lld seconds have passed since the last keepalive\n", delta_s);
                thiz->m_stream_addr_not_found = true;
            }

            bool closing = false;
            // 重新请求流信息(无限重试)
            if (thiz->m_stream_addr_not_found) {
                closing = true;
                thiz->request_stream_close();
                thiz->request_stream_info();
            }
            else {
                // 重新拉流(无限重试)
                if (thiz->m_stream_recv_eof || thiz->m_stream_recv_fail) {
                    thiz->stop_stream_keepalive();
                    // 如果当前流id保活过期，则用国标码发起请求
                    closing = delta_s > STREAM_ALIVE_S;
                    thiz->reconnect_stream(thiz->m_latest_stream_params, closing);
                }
                else {
                    // 保活失败重试(最多重试STREAM_ALIVE_S次)
                    if (thiz->m_stream_keepalive_fail) {
                        static int keepalive_times = 0;
                        if (++keepalive_times >= STREAM_ALIVE_S) {
                            thiz->m_stream_keepalive_fail = false;
                            keepalive_times = 0;
                        }
                        thiz->request_stream_keep_alive();
                    }
                }
            }

            // 关流失败重试(最多重试STREAM_ALIVE_S次)
            if (!closing && thiz->m_stream_close_fail && !thiz->m_stream_recv_eof && !thiz->m_stream_recv_fail) {
                static int close_times = 0;
                if (++close_times >= STREAM_ALIVE_S) {
                    thiz->m_stream_close_fail = false;
                    close_times = 0;
                }
                thiz->request_stream_close();
            }
        },
        1000,
        1000
    );
}

void flv_client::stop_stream_status_checker()
{
    uv_timer_stop(&m_stream_status_check_timer);
}


void flv_client::request_stream_content()
{
    m_flv_ring_buffer_write_offset = 0;
    m_need_search_i_frame = true;
    m_found_flv_header = false;
    m_need_search_magic = false;
    m_sps_len = 0;
    m_pps_len = 0;

    m_stream_addr_not_found = false;
    m_stream_recv_eof = false;
    m_stream_recv_fail = false;
    m_stream_keepalive_fail = false;
    m_stream_close_fail = false;

    m_latest_stream_params = new async_http_client::req_params_t;
    m_latest_stream_params->ctx = this;
    m_latest_stream_params->user_agent = "VLC/2.2.0 LibVLC/2.2.0";
    m_latest_stream_params->ip = m_stream_ip;
    m_latest_stream_params->port = m_stream_port;
    m_latest_stream_params->path = m_stream_path;
    m_latest_stream_params->query = m_stream_query;
    m_latest_stream_params->text_json = false;
    m_latest_stream_params->recv_body_cb = receive_stream_content_success;
    m_latest_stream_params->recv_fail_cb = receive_stream_content_fail;
    m_latest_stream_params->recv_eof_cb = receive_stream_content_eof;
    m_latest_stream_params->conn_fail_cb = [](void *ctx, int code) {
        auto params = (async_http_client::req_params_t *)ctx;
        if (!params) {
            return;
        }

        auto thiz = (flv_client *)params->ctx;
        thiz->receive_stream_content_fail(params, code);
    };
    m_latest_stream_params->write_fail_cb = m_latest_stream_params->conn_fail_cb;
    m_http_client.request(m_latest_stream_params);
}


void flv_client::receive_stream_content_success(void *ctx, uint8_t *ptr, ssize_t count)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    thiz->start_stream_keepalive();

    std::lock_guard<std::mutex> locker(params->mtx);

    if (!thiz->m_flv_ring_buffer) {
        LOG_WARNING("buf_input_circle was nullptr\n");
        return;
    }

    if (!ptr || count <= 0) {
        LOG_WARNING("no data\n");
        return;
    }

    int left_space = thiz->m_flv_ring_buffer_length - thiz->m_flv_ring_buffer_write_offset;
    if (left_space < count) {  // linhh 未考虑如何再重新找首部结构，可考虑发送的流中添加MAGIC信息
        thiz->m_need_search_i_frame = true;
        if (thiz->m_found_flv_header) {
            thiz->m_need_search_magic = true;  // 需要查找MAGIC信息
        }
        thiz->m_flv_ring_buffer_write_offset = 0;

        LOG_WARNING("no space\n");

        return;
    }

    char *cur_point = (char *)thiz->m_flv_ring_buffer + thiz->m_flv_ring_buffer_write_offset;
    memmove(cur_point, ptr, count);
    thiz->m_flv_ring_buffer_write_offset += count;  // 数据先保存到缓冲区

    left_space = thiz->m_flv_ring_buffer_write_offset;
    while (left_space > 0) {
        int current_offset = 0;
        if (!thiz->m_found_flv_header) {
            if (left_space >= FLV_HEADER_LEN) {
                bool found_flv_header = false;
                int i = 0;
                for (i = 0; i < left_space - FLV_HEADER_LEN; i++) {  // 查找FLV首部 (9个字节)
                    //    F    L    V
                    // 0x46 0x4C 0x56 ---- ---- ---- ---- ---- 0x9
                    if ((thiz->m_flv_ring_buffer[i] == 0x46) && (thiz->m_flv_ring_buffer[i + 1] == 0x4C) &&
                        (thiz->m_flv_ring_buffer[i + 2] == 0x56) && (thiz->m_flv_ring_buffer[i + 8] == 0x9)) {
                        found_flv_header = true;
                        break;
                    }
                }

                if (!found_flv_header) {  // 没有找到FLV首部
                    memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[left_space - FLV_HEADER_LEN], FLV_HEADER_LEN);
                    thiz->m_flv_ring_buffer_write_offset = FLV_HEADER_LEN;

                    LOG_WARNING("no flv header\n");
                    return;
                }

                thiz->m_found_flv_header = true;
                current_offset = i + FLV_HEADER_LEN;  // FLV的HEADER是9个字节
                thiz->m_need_search_magic = false;
            }
            else {  // 缓存数据太少了，不需要处理
                thiz->m_flv_ring_buffer_write_offset = left_space;

                LOG_WARNING("no flv header\n");
                return;
            }
        }

        if (thiz->m_found_flv_header && thiz->m_need_search_magic) {  // 网络传输中出现丢包了，需要查找MAGIC信息
            int video_data_len = left_space - current_offset;
            if (video_data_len < (TAG_HEADER_LEN + 5 + 10)) {
                if (0 == current_offset) {
                    LOG_WARNING("no flv header\n");
                    return;
                }
                else {
                    memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[current_offset], video_data_len);
                    thiz->m_flv_ring_buffer_write_offset = video_data_len;

                    LOG_WARNING("no flv header\n");
                    return;
                }
            }
            else
            {
                bool found_i_frame = false;
                int i = 0;
                for (i = 0; i < (video_data_len - 6); i++) {
                    if ((thiz->m_flv_ring_buffer[current_offset + i] == 0x13) &&
                        (thiz->m_flv_ring_buffer[current_offset + i + 1] == 0x14) &&
                        (thiz->m_flv_ring_buffer[current_offset + i + 2] == 0x05) &&
                        (thiz->m_flv_ring_buffer[current_offset + i + 3] == 0x20) &&
                        (thiz->m_flv_ring_buffer[current_offset + i + 4] == 0x26) &&
                        (thiz->m_flv_ring_buffer[current_offset + i + 5] == 0x2a)) {
                        found_i_frame = true;
                        break;
                    }
                }

                if (found_i_frame) {
                    current_offset = current_offset + i - 5 - 11;
                    if (current_offset < 0)
                    {
                        current_offset = 0;
                    }
                    thiz->m_need_search_magic = false;
                }
                else {
                    memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[left_space - 6], 6);
                    thiz->m_flv_ring_buffer_write_offset = 6;

                    LOG_WARNING("no flv header\n");
                    return;
                }
            }
        }

        if (thiz->m_found_flv_header) {  // 网络传输中出现丢包了，需要查找MAGIC信息 */
            while (current_offset < (left_space - TAG_HEADER_LEN)) { // FLV的每个TAG首部是11字节 : 后期可以扩展TAG = MAGIC + 帧号 + 帧类型
                int i = current_offset;
                int j = 0;
                for (j = 0; j < (TAG_HEADER_LEN - 2); j++) {
                    // \r\n
                    if ((((unsigned char)thiz->m_flv_ring_buffer[i + j] & 0xFF) == 0x0D) &&
                        (((unsigned char)thiz->m_flv_ring_buffer[i + j + 1] & 0xFF) == 0x0A)) {
                        break;
                    }
                }

                if (j < (TAG_HEADER_LEN - 2)) {  // 说明存在0x0D 0x0A ，此处循环去掉0x0D 与 0x0A控制
                    current_offset += j + 2;
                    int value = thiz->m_flv_ring_buffer[current_offset] & 0xFF;
                    if ((value != 0x12) && (value != 0x08) && (value != 0x09)) {
                        continue;
                    }

                    i = current_offset;
                }

                if ((thiz->m_flv_ring_buffer[i] == 0x12) || (thiz->m_flv_ring_buffer[i] == 0x8)) {             // 视频描述信息处理
                    unsigned int len1 = (unsigned char)thiz->m_flv_ring_buffer[i + 1] & 0xFF;
                    unsigned int len2 = (unsigned char)thiz->m_flv_ring_buffer[i + 2] & 0xFF;
                    unsigned int len3 = (unsigned char)thiz->m_flv_ring_buffer[i + 3] & 0xFF;
                    unsigned int len = (len1 << 16) + (len2 << 8) + (len3);
                    if ((int)(i + TAG_HEADER_LEN + len + 4) <= left_space) {
                        unsigned int len4 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + len] & 0xFF;
                        unsigned int len5 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 1 + len] & 0xFF;
                        unsigned int len6 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 2 + len] & 0xFF;
                        unsigned int len7 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 3 + len] & 0xFF;
                        unsigned int len_check = (len4 << 24) + (len5 << 16) + (len6 << 8) + (len7)-TAG_HEADER_LEN;
                        if (len_check != len) {                    		// linhh 出现了数据异常，暂时没有处理
                            thiz->m_need_search_i_frame = true;
                            thiz->m_need_search_magic = true;
                            thiz->m_flv_ring_buffer_write_offset = 0;

                            LOG_WARNING("no description\n");
                            return;
                        }

                        i += TAG_HEADER_LEN + len + 4;                // 4字节表示当前TAG长度信息 ，添加2字节0D 0A
                        current_offset = i;
                    }
                    else {
                        current_offset = i;                                   // 数据不足，直接返回，等待后期数据
                        memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[current_offset], (left_space - current_offset));
                        thiz->m_flv_ring_buffer_write_offset = left_space - current_offset;

                        LOG_WARNING("wait more data\n");
                        return;
                    }
                }
                else if (thiz->m_flv_ring_buffer[i] == 0x09) {                 // 视频帧处理
                    unsigned int len1 = (unsigned char)thiz->m_flv_ring_buffer[i + 1] & 0xFF;
                    unsigned int len2 = (unsigned char)thiz->m_flv_ring_buffer[i + 2] & 0xFF;
                    unsigned int len3 = (unsigned char)thiz->m_flv_ring_buffer[i + 3] & 0xFF;
                    unsigned int len = (len1 << 16) + (len2 << 8) + (len3);
                    if ((int)(i + TAG_HEADER_LEN + len + 4) <= left_space) {
                        unsigned int len4 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + len] & 0xFF;
                        unsigned int len5 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 1 + len] & 0xFF;
                        unsigned int len6 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 2 + len] & 0xFF;
                        unsigned int len7 = (unsigned char)thiz->m_flv_ring_buffer[i + TAG_HEADER_LEN + 3 + len] & 0xFF;
                        unsigned int len_check = (len4 << 24) + (len5 << 16) + (len6 << 8) + (len7)-TAG_HEADER_LEN;
                        if (len_check != len) {                            // linhh 出现了数据异常，暂时没有处理
                            thiz->m_need_search_i_frame = true;
                            thiz->m_need_search_magic = true;
                            thiz->m_flv_ring_buffer_write_offset = 0;

                            LOG_WARNING("video size error\n");
                            return;
                        }

                        int ret = thiz->receive_elementary_stream(params, (char *)&thiz->m_flv_ring_buffer[i], (unsigned int)(TAG_HEADER_LEN + len));
                        i += TAG_HEADER_LEN + len + 4;
                        current_offset = i;                           // 这边不处理执行下一帧检测
                    }
                    else {
                        current_offset = i;
                        memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[current_offset], (left_space - current_offset));
                        thiz->m_flv_ring_buffer_write_offset = left_space - current_offset;
                        return;
                    }
                }
                else {                                  // linhh 出现了数据异常，暂时没有处理
                    thiz->m_need_search_i_frame = true;
                    thiz->m_need_search_magic = true;
                    thiz->m_flv_ring_buffer_write_offset = 0;

                    LOG_WARNING("unexpected error\n");
                    return;
                }
            }

            memmove(&thiz->m_flv_ring_buffer[0], &thiz->m_flv_ring_buffer[current_offset], (left_space - current_offset));
            thiz->m_flv_ring_buffer_write_offset = left_space - current_offset;

            return;
        }  // 未识别到FLV 首部，在上面代码已经做了处理控制
    }

    if (!thiz->m_found_flv_header) {
        LOG_WARNING("no flv header\n");
    }
}


void flv_client::receive_stream_content_fail(void *ctx, ssize_t code)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    params->http_client->remove_request_history(params);

    thiz->disconnect_stream(params, false);

    if (!params || params == thiz->m_latest_stream_params) {
        thiz->m_latest_stream_params = nullptr;
    }

    thiz->m_stream_recv_fail = true;
}


void flv_client::receive_stream_content_eof(void *ctx)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return;
    }

    auto thiz = (flv_client *)params->ctx;

    params->http_client->remove_request_history(params);

    thiz->disconnect_stream(params, false);

    if (!params || params == thiz->m_latest_stream_params) {
        thiz->m_latest_stream_params = nullptr;
    }

    thiz->m_stream_recv_eof = true;
}


int flv_client::receive_elementary_stream(void *ctx, char *input_buf, unsigned int len)
{
    auto params = (async_http_client::req_params_t *)ctx;
    if (!params) {
        return -1;
    }

    auto thiz = (flv_client *)params->ctx;

    if (!thiz->m_flv_ring_buffer) {
        LOG_WARNING("buf_input_circle was nullptr\n");
        return -2;
    }

    if (!input_buf || len <= 0) {
        LOG_WARNING("no data\n");
        return -3;
    }

    char frame_type = input_buf[TAG_HEADER_LEN] & 0xF0;
    if (0x10 == frame_type || 0x20 == frame_type) {                      // 0x10 为I Frame, 0x20 为P Frame
        char codec_type = input_buf[TAG_HEADER_LEN] & 0x0F;              // 7表示H264, 8表示H265
        char nalu_type = input_buf[TAG_HEADER_LEN + 1] & 0x0F;           // 0表示SPS等信息, 1为NALU信息
        if (0 == nalu_type) {                                            // SPS与PPS信息
            char nalu_head[] = { 0x00, 0x00, 0x00, 0x01 };
            char *current_addr = input_buf + TAG_HEADER_LEN + 5 + 6;     // +6开始到SPS长度信息
            int len1 = current_addr[0] & 0xFF;
            int len2 = current_addr[1] & 0xFF;
            int len = (len1 << 8) + len2;

            thiz->m_sps_len = len + 4;
            memcpy(thiz->m_sps, nalu_head, 4);
            memcpy(thiz->m_sps + 4, current_addr + 2, len);

            current_addr += 2 + len + 1;
            len1 = current_addr[0] & 0xFF;
            len2 = current_addr[1] & 0xFF;
            len = (len1 << 8) + len2;
            thiz->m_pps_len = len + 4;
            memcpy(thiz->m_pps, nalu_head, 4);
            memcpy(thiz->m_pps + 4, current_addr + 2, len);
            return 0;
        }
        else if (1 == nalu_type) {
            if (thiz->m_need_search_i_frame)
            {
                if (0x10 == frame_type) {
                    thiz->m_need_search_i_frame = false;
                }
                else {
                    LOG_WARNING("no node\n");
                }
            }

            // 1字节FrameType|CodecID, 1字节AVC NALU, 3字节时间信息 , 10字节自定义
            char nalu_head[] = { 0x00, 0x00, 0x00, 0x01 };
            unsigned int total_len = len - TAG_HEADER_LEN - 5;
            char *current_addr = input_buf + TAG_HEADER_LEN + 5 + 10;

            // 计算NALU的长度信息
            unsigned int len1 = (unsigned char)current_addr[0] & 0xFF;
            unsigned int len2 = (unsigned char)current_addr[1] & 0xFF;
            unsigned int len3 = (unsigned char)current_addr[2] & 0xFF;
            unsigned int len4 = (unsigned char)current_addr[3] & 0xFF;
            unsigned int current_len = (len1 << 24) + (len2 << 16) + (len3 << 8) + (len4);

            // 计算帧时间信息
            unsigned int frame_time1 = (unsigned char)input_buf[TAG_TIME_POS] & 0xFF;
            unsigned int frame_time2 = (unsigned char)input_buf[TAG_TIME_POS + 1] & 0xFF;
            unsigned int frame_time3 = (unsigned char)input_buf[TAG_TIME_POS + 2] & 0xFF;
            unsigned int frame_time4 = (unsigned char)input_buf[TAG_TIME_POS + 3] & 0xFF;
            unsigned int real_time = (frame_time1 << 16) | (frame_time2 << 8) | frame_time3 | (frame_time4 << 24);
            unsigned int frame_time = (frame_time1 << 16) | (frame_time2 << 8) | frame_time3;

            // 一帧数据读会包括多个NALU单元
            int len = 0;
            while (current_len <= total_len) {
                memmove(thiz->m_frame_buffer + len, nalu_head, 4);
                len += 4;
                memmove(thiz->m_frame_buffer + len, current_addr + 4, current_len);
                len += current_len;

                current_addr += (4 + current_len);
                total_len -= (4 + current_len);

                if (0 == total_len) {
                    break;
                }

                len1 = (unsigned char)current_addr[0] & 0xFF;
                len2 = (unsigned char)current_addr[1] & 0xFF;
                len3 = (unsigned char)current_addr[2] & 0xFF;
                len4 = (unsigned char)current_addr[3] & 0xFF;
                current_len = (len1 << 24) + (len2 << 16) + (len3 << 8) + (len4);
            }

            // 只解码I帧
            if (0x10 != frame_type) {
                return 0;
            }

            // 编码类型不同就重新初始化
            enum AVCodecID codec = (0x07 == codec_type) ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
            if (thiz->m_decoder.get_codec_id() != codec) {
                thiz->m_decoder.uninit();
                thiz->m_decoder.init(codec);
            }

            // 送解码器、送转换器
            if (thiz->m_decoder.send_packet(thiz->m_frame_buffer, len, 0, 0) >= 0) {
                int ret = 0;
                int key_frame = 0;
                uint64_t dts = 0;
                uint64_t pts = 0;
                int width = 0;
                int height = 0;
                AVFrame *frame = thiz->m_decoder.receive_frame(ret, key_frame, dts, pts, width, height);
                if (frame) {
                    static int nv12_frames = 0;
                    LOG_INFO("NV12 FRAMES: %d\n", ++nv12_frames);

                    int64_t host = 0;
                    int64_t device = 0;
                    bool copy = m_recv_host_rgb24_cb ? true : false;
                    cudaError_t error = receive_rgb24(frame->data[0], frame->data[1], frame->linesize[0], frame->linesize[1], width, height, host, device, copy);
                    if (error == cudaSuccess) {
                        static int rgb24_frames = 0;
                        LOG_INFO("RGB24 FRAMES: %d\n", ++rgb24_frames);
                    }

                    if (copy) {
                        if (thiz->m_recv_host_rgb24_to_np_cb) {
                            // 数据复制到numpy数组
                            

                            // 释放host内存
                            free_output_buf((uint8_t *)host);
                        }
                        else {
                            thiz->m_recv_host_rgb24_cb(thiz, error == cudaSuccess ? host : 0, error == cudaSuccess ? width * height * 3 : 0, int(error));
                        }
                    }
                    else {
                        thiz->m_recv_device_rgb24_cb(thiz, error == cudaSuccess ? device : 0, error == cudaSuccess ? width * height * 3 : 0, int(error));
                    }

                    thiz->m_decoder.free_frame(frame);
                }
            }

            return 0;
        }
        else {
            return 0;
        }
    }
    return 0;
}
