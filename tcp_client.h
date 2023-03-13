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

    // TCP连接与返回
    typedef struct conn_params_s {
        //////// 入参
        void *ctx = nullptr;  // 用户自定义指针
        async_tcp_client *tcp_client = nullptr;  // tcp client指针
        std::string ip = "";  // 连接的服务器IP
        int port = 0;  //  连接的服务器端口
        int timeout_ms = INT_MAX;  // 连接超时时间
        uv_connect_cb connect_cb = nullptr;  // 连接回调
        //////// 出参
        int status = -1;  // 返回状态
        uv_tcp_t *client = nullptr;  // tcp client指针
        uv_connect_t *connection = nullptr;  // tcp connect指针
        //////// 同步
        std::mutex mtx;  // 互斥量
        std::condition_variable cv;  // 条件变量
    } conn_params_t;

    // 发起tcp连接
    int connect(conn_params_t *conn_params);

    // 内存分配回调
    static void on_alloc_callback(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    // 关闭资源回调
    static void on_close_callback(uv_handle_t *handle);
    // 关闭资源回调
    static void on_shutdown_callback(uv_shutdown_t *sd, int status);


protected:
    // uv循环
    uv_loop_t *m_loop;
};
