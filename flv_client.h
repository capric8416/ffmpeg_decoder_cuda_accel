#pragma once

// http
#include "http_client.h"

// decoder
#include "video_decoder.h"



class flv_client
{
public:
    flv_client(uv_loop_t *loop);
    ~flv_client();

    void stop();

    // 设置流媒体网关地址
    void set_gateway_addr(const char *ip, const int port);

    // 设置接收rgb24回调
    typedef void (*recv_host_rgb24_to_np_cb)(flv_client *ctx, int error);
    typedef void (*recv_host_rgb24_cb)(flv_client *ctx, int64_t buf, int size, int error);
    typedef void (*recv_device_rgb24_cb)(flv_client *ctx, int64_t buf, int size, int error);
    // 设置接收RGB24函数指针
    void set_output_callbacks(recv_host_rgb24_to_np_cb recv_host_to_np, recv_host_rgb24_cb recv_host, recv_device_rgb24_cb recv_device);
    // 释放device/host指针
    void free_output_buf(uint8_t *buf);

    // 请求流信息
    void request_stream_info();
    // 接收流信息
    static void receive_stream_info_success(void *ctx);
    static void receive_stream_info_fail(void *ctx, ssize_t code);

    // 请求流保活
    void request_stream_keep_alive();
    // 接收流保活
    static void receive_stream_keep_alive_success(void *ctx);
    static void receive_stream_keep_alive_fail(void *ctx, ssize_t code);

    // 请求流关闭
    void request_stream_close();
    // 接收流关闭
    static void receive_stream_close_success(void *ctx);
    static void receive_stream_close_fail(void *ctx, ssize_t code);

    // 请求流内容
    void request_stream_content();
    // 接收flv流并解析和缓冲
    static void receive_stream_content_success(void *ctx, uint8_t *ptr, ssize_t count);
    static void receive_stream_content_fail(void *ctx, ssize_t code);
    static void receive_stream_content_eof(void *ctx);

    // 设置流信息
    void set_stream_info(const int id, const std::string ip, const int port, const std::string path, const std::string query);

    // 发起流请求
    void connect_stream();
    void connect_stream(std::string gbcode);

    // 断开流请求
    void disconnect_stream(async_http_client::req_params_t *params, bool close);

    // 重新发起流请求
    void reconnect_stream(async_http_client::req_params_t *params, bool close = false);

    // 流保活
    void start_stream_keepalive();
    void stop_stream_keepalive();

    // 流状态检测
    void start_stream_status_checker();
    void stop_stream_status_checker();

    // 解析es流并送解码
    int receive_elementary_stream(void *ctx, char *input_buf, unsigned int len);


protected:
    // http client
    uv_loop_t *m_loop;
    async_http_client m_http_client;
    async_http_client::req_params_t *m_latest_stream_params;

    // video decoder
    ffmpeg_hw_decoder m_decoder;

    // 流媒体网关
    std::string m_gateway_ip;
    int m_gateway_port;

    // 点位国标码
    std::string m_gbcode;

    // 拉流配置
    std::string m_stream_ip;
    int m_stream_port;
    int m_stream_id;
    std::string m_stream_path;
    std::string m_stream_query;

    // 环形缓冲
    uint8_t *m_flv_ring_buffer;
    uint32_t m_flv_ring_buffer_length;
    uint32_t m_flv_ring_buffer_write_offset;

    // 帧缓冲
    uint32_t m_frame_buffer_length;
    uint8_t *m_frame_buffer;

    // 标志
    bool m_need_search_i_frame;
    bool m_found_flv_header;
    bool m_need_search_magic;

    // sps
    char m_sps[64];
    int m_sps_len;

    // pps
    char m_pps[64];
    int m_pps_len;

    // status
    bool m_stream_keepaliving;
    bool m_stream_addr_not_found;
    bool m_stream_recv_eof;
    bool m_stream_recv_fail;
    bool m_stream_keepalive_fail;
    bool m_stream_close_fail;

    int64_t m_last_keepalive_ts;

    // timer
    uv_timer_t m_stream_keepalive_timer;
    uv_timer_t m_stream_status_check_timer;

    // 输出回调
    recv_host_rgb24_to_np_cb m_recv_host_rgb24_to_np_cb;
    recv_host_rgb24_cb m_recv_host_rgb24_cb;
    recv_device_rgb24_cb m_recv_device_rgb24_cb;
};
