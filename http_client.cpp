// self
#include "http_client.h"

// log
#include "stdout_log.h"

// c
#include <string.h>



async_http_client::async_http_client()
    : m_loop(nullptr)
    , m_requiest_id(0)
{
}


async_http_client::async_http_client(uv_loop_t *loop)
    : m_loop(loop)
    , m_requiest_id(0)
{
    m_tcp_client.set_loop(loop);
}


async_http_client::~async_http_client()
{
    remove_all_request_histories();
}


void async_http_client::set_loop(uv_loop_t *loop)
{
    m_loop = loop;
    m_tcp_client.set_loop(loop);
}


int async_http_client::request(req_params_t *req_params)
{
    record_request_history(req_params);

    req_params->resp_temp.clear();
    req_params->resp_body.clear();
    req_params->resp_headers.clear();
    req_params->http_client = this;

    auto conn_params = new async_tcp_client::conn_params_t;
    conn_params->ctx = req_params;
    conn_params->ip = req_params->ip;
    conn_params->port = req_params->port;
    conn_params->timeout_ms = req_params->conn_timeout_ms;
    conn_params->connect_cb = [](uv_connect_t *req, int status) {
        auto my_conn_params = (async_tcp_client::conn_params_t *)req->data;
        if (!my_conn_params) {
            LOG_ERROR("conn_params_t was nullptr\n");
            return;
        }

        auto my_req_params = (req_params_t *)my_conn_params->ctx;

        if (!my_req_params) {
            LOG_ERROR("request_params_t was nullptr\n");
            return;
        }

        if (status < 0) {
            LOG_ERROR(
                "%s %s:%d%s%s%s%s%s fail, error: %s\n",
                my_req_params->method.c_str(), my_req_params->ip.c_str(), my_req_params->port,
                my_req_params->path.c_str(),
                my_req_params->query.size() > 0 ? "?" : "", my_req_params->query.c_str(),
                my_req_params->fragment.size() > 0 ? "#" : "", my_req_params->fragment.c_str(),
                uv_strerror(status)
            );
            if (my_req_params->conn_fail_cb) {
                my_req_params->conn_fail_cb(my_req_params, status);
            }

            delete my_conn_params; // 销毁conn_params
        }
        else {
            char req_buf[1024];
            size_t req_buf_len = 0;

            snprintf(
                req_buf, sizeof(req_buf),
                "%s %s%s%s%s%s HTTP/1.1\r\n"
                "Host: %s:%d\r\n"
                "User-Agent: %s\r\n"
                "Range: bytes=0-\r\n"
                "Connection: close\r\n"
                "Icy-MetaData: 1\r\n\r\n",
                my_req_params->method.c_str(), my_req_params->path.c_str(),
                my_req_params->query.size() > 0 ? "?" : "", my_req_params->query.c_str(),
                my_req_params->fragment.size() > 0 ? "#" : "", my_req_params->fragment.c_str(),
                my_req_params->ip.c_str(), my_req_params->port,
                my_req_params->user_agent.c_str()
            );

            req_buf_len = strlen(req_buf);
            req_buf[req_buf_len] = '\0';

            LOG_INFO("\n%s", req_buf);

            uv_buf_t bufs[] = {
#if defined(WIN32)
                {(unsigned long)req_buf_len, req_buf}
#else
                {req_buf, (unsigned long)req_buf_len}
#endif
            };

            uv_write_t write;
            uv_stream_t *handle = my_conn_params->connection->handle;
            my_req_params->status = uv_write(&write, handle, bufs, 1, my_req_params->write_cb);
            if (my_req_params->status != 0) {
                LOG_ERROR("uv_write fail, error: %s\n", uv_strerror(my_req_params->status));
                delete my_conn_params; // 销毁conn_params
                return;
            }

            write.data = my_req_params;
            handle->data = my_req_params;
            my_req_params->status = uv_read_start(handle, async_tcp_client::on_alloc_callback, my_req_params->read_cb);
            if (my_req_params->status != 0) {
                LOG_ERROR("uv_read_start fail, error: %s\n", uv_strerror(my_req_params->status));
                delete my_conn_params; // 销毁conn_params
                return;
            }

            delete my_conn_params; // 销毁conn_params
        }
    };

    req_params->write_cb = [](uv_write_t *req, int status) {
        if (status == 0) {
            // success
            return;
        }
        else {
            LOG_WARNING("uv_write fail, error: %s\n", uv_strerror(status));

            auto params = (req_params_t *)req->data;
            if (params->conn_fail_cb) {
                params->conn_fail_cb(params, status);
            }

            if (status == UV_ECANCELED) {
                LOG_WARNING("uv_write_cb status == UV_ECANCELED\n");
            }

            if (status != UV_EPIPE) {
                LOG_WARNING("uv_write_cb status != UV_EPIPE\n");
            }
        }

        uv_close((uv_handle_t *)req->handle, async_tcp_client::on_close_callback);
        free(req);
    };

    req_params->read_cb = [](uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf) {
        auto params = (req_params_t *)stream->data;
        if (nread < 0) {
            // close stream
            uv_close((uv_handle_t *)stream, async_tcp_client::on_close_callback);

            // free connection
            free(params->connection);

            if (nread != UV_EOF) {
                // error
                LOG_ERROR("read_cb fail, error: %s\n", uv_strerror((int)nread));
                if (params->recv_fail_cb) {
                    params->recv_fail_cb(params, nread);
                }
            }
            else {
                // eof

                // notify finished
                if (params->recv_eof_cb) {
                    params->recv_eof_cb(params);
                }
            }
        }
        else if (nread > 0) {
            if (params->resp_temp.size() == 1 && params->resp_temp[0] == 127) {
                // 保存正文
                if (params->recv_body_cb) {
                    params->recv_body_cb(params, (uint8_t *)buf->base, nread);
                }
                else {
                    params->resp_body.insert(params->resp_body.end(), buf->base, buf->base + nread);
                }
            }
            else {
                // 解析返回头
                params->resp_temp.insert(params->resp_temp.end(), buf->base, buf->base + nread);
                char *begin = params->resp_temp.data();
                char *end = strstr(begin, "\r\n\r\n");
                if (end) {
                    // store headers line by line
                    std::string headers;
                    headers.insert(headers.end(), begin, end);
                    std::string::size_type prev_pos = 0, pos = 0;
                    while ((pos = headers.find("\r\n", pos)) != std::string::npos) {
                        params->resp_headers.push_back(headers.substr(prev_pos, pos - prev_pos));
                        pos += 2;
                        prev_pos = pos;
                    }
                    params->resp_headers.push_back(headers.substr(prev_pos, pos - prev_pos));
                    if (params->recv_header_cb) {
                        params->recv_header_cb(params, &params->resp_headers);
                    }

                    // skip unprintable chars after headers
                    char *addr_begin = end + 4;
                    ssize_t count = nread - headers.size() - 4;
                    while (params->text_json && count > 0 && (addr_begin[0] != '[' && addr_begin[0] != '{')) {
                        addr_begin++;
                        count--;
                    }
                    char *addr_end = addr_begin + count;
                    // skip unprintable chars after content
                    while (params->text_json && count > 0 && (addr_end[0] != ']' && addr_end[0] != '}')) {
                        addr_end--;
                        count--;
                    }
                    if (count > 0) {
                        // store left chars to body
                        if (params->recv_body_cb) {
                            params->recv_body_cb(params, (uint8_t *)addr_begin, count);
                        }
                        else {
                            params->resp_body.insert(params->resp_body.end(), addr_begin, addr_end + 1);
                        }
                    }

                    // clear temp and set flag
                    params->resp_temp.clear();
                    params->resp_temp.push_back(127);
                }
            }
        }

        free(buf->base);
    };

    conn_params->status = m_tcp_client.connect(conn_params);

    req_params->client = conn_params->client;
    req_params->connection = conn_params->connection;
    req_params->connection->data = conn_params;

    return conn_params->status;
}


void async_http_client::record_request_history(req_params_t *params)
{
    std::lock_guard<std::mutex> locker(m_mtx);
    params->id = ++m_requiest_id;
    m_request_ids[params->id] = params;
    m_request_clients[params] = this;
}


void async_http_client::remove_all_request_histories()
{
    std::lock_guard<std::mutex> locker(m_mtx);

    for (auto iter = m_request_ids.begin(); iter != m_request_ids.end(); iter++) {
        delete iter->second;
    }
    m_request_ids.clear();

    m_request_clients.clear();
}


void async_http_client::remove_request_history(req_params_t *&params)
{
    std::lock_guard<std::mutex> locker(m_mtx);
    
    auto iter = m_request_clients.find(params);
    if (iter != m_request_clients.end()) {
        m_request_clients.erase(iter);
    }

    auto jter = m_request_ids.find(params->id);
    if (jter != m_request_ids.end()) {
        m_request_ids.erase(jter);
    }

    if (params != nullptr) {
        delete params;
        params = nullptr;
    }
}


async_http_client::req_params_t *async_http_client::find_request_history(int64_t id)
{
    auto iter = m_request_ids.find(id);
    return iter != m_request_ids.end() ? iter->second : nullptr;
}
