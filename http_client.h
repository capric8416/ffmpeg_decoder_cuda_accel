#pragma once

// tcp
#include "tcp_client.h"

// c++
#include <map>
#include <vector>



class async_http_client
{
public:
    async_http_client();
    async_http_client(uv_loop_t *loop);
    virtual ~async_http_client();

    void set_loop(uv_loop_t *loop);

    // 返回头回调
    typedef void (*uv_recv_header_cb)(void *ctx, std::vector<std::string> *content);
    // 返回正文回调
    typedef void (*uv_recv_body_cb)(void *ctx, uint8_t *content, ssize_t nbytes);
    // 返回结束回调
    typedef void (*uv_recv_eof_cb)(void *ctx);
    // 返回错误回调
    typedef void (*uv_recv_fail_cb)(void *ctx, ssize_t code);
    // 连接错误回调
    typedef void (*uv_connect_fail_cb)(void *ctx, int code);
    // 请求错误回调
    typedef void (*uv_write_fail_cb)(void *ctx, int code);

    // HTTP请求与返回
    typedef struct req_params_s {
        //////// 入参
        int64_t id = 0;  // 标识
        void *ctx = nullptr;  // 用户自定义指针
        async_http_client *http_client = nullptr;  // http client指针
        FILE *fp = nullptr;  // 文件句柄
        std::string ip = "";  // 请求的服务器IP
        int port = 0;  // 请求的服务器端口
        std::string user_agent = "aysnc http client powered by libuv/0.1";  // 用户代理标识
        std::string method = "GET";  // 请求的方法
        std::string path = "/";  //  请求的资源路径
        std::string query = "";  // 请求的资源查询参数
        std::string fragment = "";  // 请求的资源锚标识
        bool text_json = true;  // 默认返回json文本，会自动移除头部的非法字符
        int conn_timeout_ms = INT_MAX;  // 连接超时时间
        int req_timeout_ms = INT_MAX;  // 请求超时时间
        uv_write_cb write_cb = nullptr;  // 通常情况下无需填充
        uv_read_cb read_cb = nullptr;  // 通常情况下无需填充
        uv_recv_header_cb recv_header_cb = nullptr;  // 通常情况下无需填充，仅当你需要读取返回头的时候设置
        uv_recv_body_cb recv_body_cb = nullptr;  // 如果接收大的文件或者流媒体，请设置此回调，否则reponse_body可能会耗光内存
        uv_recv_eof_cb recv_eof_cb = nullptr;  // 请求返回结束回调
        uv_recv_fail_cb recv_fail_cb = nullptr;  // 请求返回错误回调
        uv_connect_fail_cb conn_fail_cb = nullptr;  // 连接错误回调
        uv_write_fail_cb write_fail_cb = nullptr;  // 请求错误回调
        //////// 出参
        int status = -1;  // 返回状态
        std::vector<char> resp_temp;  // 临时存放返回数据
        std::vector<std::string> resp_headers;  // 返回头
        std::vector<uint8_t> resp_body;  // 返回正文
        // 从TCP连接复制
        uv_tcp_t *client = nullptr;  // tcp client指针
        uv_connect_t *connection = nullptr;  // tcp connect指针
        //////// 同步
        std::mutex mtx;  // 互斥量
        std::condition_variable cv;  // 条件变量
    } req_params_t;

    // 发起http请求(请求参数必须new分配然后在recv_finished_cb和recv_unexpected_cb中delete释放)
    int request(req_params_t *req_params);

    // 记录请求
    void record_request_history(req_params_t *params);
    // 移除所有请求记录
    void remove_all_request_histories();
    // 移除请求记录
    void remove_request_history(req_params_t *&params);
    // 查找请求记录
    req_params_t *find_request_history(int64_t id);


protected:
    // uv循环
    uv_loop_t *m_loop;
    // tcp client
    async_tcp_client m_tcp_client;

    // 互斥量
    std::mutex m_mtx;
    // 请求记录
    int64_t m_requiest_id;
    std::map<int64_t, req_params_t *> m_request_ids;
    std::map<req_params_t *, async_http_client *> m_request_clients;
};
