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

    // ����ͷ�ص�
    typedef void (*uv_recv_header_cb)(void *ctx, std::vector<std::string> *content);
    // �������Ļص�
    typedef void (*uv_recv_body_cb)(void *ctx, uint8_t *content, ssize_t nbytes);
    // ���ؽ����ص�
    typedef void (*uv_recv_eof_cb)(void *ctx);
    // ���ش���ص�
    typedef void (*uv_recv_fail_cb)(void *ctx, ssize_t code);
    // ���Ӵ���ص�
    typedef void (*uv_connect_fail_cb)(void *ctx, int code);
    // �������ص�
    typedef void (*uv_write_fail_cb)(void *ctx, int code);

    // HTTP�����뷵��
    typedef struct req_params_s {
        //////// ���
        int64_t id = 0;  // ��ʶ
        void *ctx = nullptr;  // �û��Զ���ָ��
        async_http_client *http_client = nullptr;  // http clientָ��
        FILE *fp = nullptr;  // �ļ����
        std::string ip = "";  // ����ķ�����IP
        int port = 0;  // ����ķ������˿�
        std::string user_agent = "aysnc http client powered by libuv/0.1";  // �û������ʶ
        std::string method = "GET";  // ����ķ���
        std::string path = "/";  //  �������Դ·��
        std::string query = "";  // �������Դ��ѯ����
        std::string fragment = "";  // �������Դê��ʶ
        bool text_json = true;  // Ĭ�Ϸ���json�ı������Զ��Ƴ�ͷ���ķǷ��ַ�
        int conn_timeout_ms = INT_MAX;  // ���ӳ�ʱʱ��
        int req_timeout_ms = INT_MAX;  // ����ʱʱ��
        uv_write_cb write_cb = nullptr;  // ͨ��������������
        uv_read_cb read_cb = nullptr;  // ͨ��������������
        uv_recv_header_cb recv_header_cb = nullptr;  // ͨ�������������䣬��������Ҫ��ȡ����ͷ��ʱ������
        uv_recv_body_cb recv_body_cb = nullptr;  // ������մ���ļ�������ý�壬�����ô˻ص�������reponse_body���ܻ�Ĺ��ڴ�
        uv_recv_eof_cb recv_eof_cb = nullptr;  // ���󷵻ؽ����ص�
        uv_recv_fail_cb recv_fail_cb = nullptr;  // ���󷵻ش���ص�
        uv_connect_fail_cb conn_fail_cb = nullptr;  // ���Ӵ���ص�
        uv_write_fail_cb write_fail_cb = nullptr;  // �������ص�
        //////// ����
        int status = -1;  // ����״̬
        std::vector<char> resp_temp;  // ��ʱ��ŷ�������
        std::vector<std::string> resp_headers;  // ����ͷ
        std::vector<uint8_t> resp_body;  // ��������
        // ��TCP���Ӹ���
        uv_tcp_t *client = nullptr;  // tcp clientָ��
        uv_connect_t *connection = nullptr;  // tcp connectָ��
        //////// ͬ��
        std::mutex mtx;  // ������
        std::condition_variable cv;  // ��������
    } req_params_t;

    // ����http����(�����������new����Ȼ����recv_finished_cb��recv_unexpected_cb��delete�ͷ�)
    int request(req_params_t *req_params);

    // ��¼����
    void record_request_history(req_params_t *params);
    // �Ƴ����������¼
    void remove_all_request_histories();
    // �Ƴ������¼
    void remove_request_history(req_params_t *&params);
    // ���������¼
    req_params_t *find_request_history(int64_t id);


protected:
    // uvѭ��
    uv_loop_t *m_loop;
    // tcp client
    async_tcp_client m_tcp_client;

    // ������
    std::mutex m_mtx;
    // �����¼
    int64_t m_requiest_id;
    std::map<int64_t, req_params_t *> m_request_ids;
    std::map<req_params_t *, async_http_client *> m_request_clients;
};
