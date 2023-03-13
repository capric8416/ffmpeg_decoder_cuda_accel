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

    // ������ý�����ص�ַ
    void set_gateway_addr(const char *ip, const int port);

    // ���ý���yuv�ӵ�
    typedef void (*recv_host_rgb24_to_np_cb)(flv_client *ctx, int error);
    typedef void (*recv_host_rgb24_cb)(flv_client *ctx, int64_t buf, int size, int error);
    typedef void (*recv_device_rgb24_cb)(flv_client *ctx, int64_t buf, int size, int error);
    // ���ý���RGB24����ָ��
    void set_output_callbacks(recv_host_rgb24_to_np_cb recv_host_to_np, recv_host_rgb24_cb recv_host, recv_device_rgb24_cb recv_device);
    // �ͷ�device/hostָ��
    void free_output_buf(uint8_t *buf);

    // ��������Ϣ
    void request_stream_info();
    // ��������Ϣ
    static void receive_stream_info_success(void *ctx);
    static void receive_stream_info_fail(void *ctx, ssize_t code);

    // ����������
    void request_stream_keep_alive();
    // ����������
    static void receive_stream_keep_alive_success(void *ctx);
    static void receive_stream_keep_alive_fail(void *ctx, ssize_t code);

    // �������ر�
    void request_stream_close();
    // �������ر�
    static void receive_stream_close_success(void *ctx);
    static void receive_stream_close_fail(void *ctx, ssize_t code);

    // ����������
    void request_stream_content();
    // ����flv���������ͻ���
    static void receive_stream_content_success(void *ctx, uint8_t *ptr, ssize_t count);
    static void receive_stream_content_fail(void *ctx, ssize_t code);
    static void receive_stream_content_eof(void *ctx);

    // ��������Ϣ
    void set_stream_info(const int id, const std::string ip, const int port, const std::string path, const std::string query);

    // ����������
    void connect_stream();
    void connect_stream(std::string gbcode);

    // �Ͽ�������
    void disconnect_stream(async_http_client::req_params_t *params, bool close);

    // ���·���������
    void reconnect_stream(async_http_client::req_params_t *params, bool close = false);

    // ������
    void start_stream_keepalive();
    void stop_stream_keepalive();

    // ��״̬���
    void start_stream_status_checker();
    void stop_stream_status_checker();

    // ����es�����ͽ���
    int receive_elementary_stream(void *ctx, char *input_buf, unsigned int len);


protected:
    // http client
    uv_loop_t *m_loop;
    async_http_client m_http_client;
    async_http_client::req_params_t *m_latest_stream_params;

    // video decoder
    ffmpeg_hw_decoder m_decoder;

    // ��ý������
    std::string m_gateway_ip;
    int m_gateway_port;

    // ��λ������
    std::string m_gbcode;

    // ��������
    std::string m_stream_ip;
    int m_stream_port;
    int m_stream_id;
    std::string m_stream_path;
    std::string m_stream_query;

    // ���λ���
    uint8_t *m_flv_ring_buffer;
    uint32_t m_flv_ring_buffer_length;
    uint32_t m_flv_ring_buffer_write_offset;

    // ֡����
    uint32_t m_frame_buffer_length;
    uint8_t *m_frame_buffer;

    // ��־
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

    // ����ص�
    recv_host_rgb24_to_np_cb m_recv_host_rgb24_to_np_cb;
    recv_host_rgb24_cb m_recv_host_rgb24_cb;
    recv_device_rgb24_cb m_recv_device_rgb24_cb;
};