#pragma once

// c++
#include <condition_variable>
#include <mutex>
#include <string>

// libuv
#include <uv.h>


class async_tcp_client
{
public:
    async_tcp_client();
    async_tcp_client(uv_loop_t *loop);
    virtual ~async_tcp_client();

    void set_loop(uv_loop_t *loop);

    // TCP�����뷵��
    typedef struct conn_params_s {
        //////// ���
        void *ctx = nullptr;  // �û��Զ���ָ��
        async_tcp_client *tcp_client = nullptr;  // tcp clientָ��
        std::string ip = "";  // ���ӵķ�����IP
        int port = 0;  //  ���ӵķ������˿�
        int timeout_ms = INT_MAX;  // ���ӳ�ʱʱ��
        uv_connect_cb connect_cb = nullptr;  // ���ӻص�
        //////// ����
        int status = -1;  // ����״̬
        uv_tcp_t *client = nullptr;  // tcp clientָ��
        uv_connect_t *connection = nullptr;  // tcp connectָ��
        //////// ͬ��
        std::mutex mtx;  // ������
        std::condition_variable cv;  // ��������
    } conn_params_t;

    // ����tcp����
    int connect(conn_params_t *conn_params);

    // �ڴ����ص�
    static void on_alloc_callback(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    // �ر���Դ�ص�
    static void on_close_callback(uv_handle_t *handle);
    // �ر���Դ�ص�
    static void on_shutdown_callback(uv_shutdown_t *sd, int status);


protected:
    // uvѭ��
    uv_loop_t *m_loop;
};
