// self
#include "tcp_client.h"

// log
#include "stdout_log.h"



async_tcp_client::async_tcp_client()
    : m_loop(nullptr)
{
}


async_tcp_client::async_tcp_client(uv_loop_t *loop)
    : m_loop(loop)
{
}


async_tcp_client::~async_tcp_client()
{
}


void async_tcp_client::set_loop(uv_loop_t *loop)
{
    m_loop = loop;
}


int async_tcp_client::connect(conn_params_t *conn_params)
{
    conn_params->tcp_client = this;

    uv_tcp_t *client = (uv_tcp_t *)malloc(sizeof(uv_tcp_t));
    conn_params->client = client;
    if (!client) {
        LOG_ERROR("malloc uv_tcp_t fail\n");
        return -1;
    }

    int result;
    result = uv_tcp_init(m_loop, client);
    if (result < 0) {
        LOG_ERROR("uv_tcp_init fail, error: %s\n", uv_strerror(result));
    }

    struct sockaddr_in addr;
    if ((result = uv_ip4_addr(conn_params->ip.c_str(), conn_params->port, &addr)) != 0) {
        LOG_ERROR("uv_ip4_addr fail, ip: %s, port: %d, error: %s\n", conn_params->ip.c_str(), conn_params->port, uv_strerror(result));
        return result;
    }

    uv_connect_t *connection = (uv_connect_t *)malloc(sizeof(uv_connect_t));
    conn_params->connection = connection;
    if (!connection) {
        LOG_ERROR("malloc uv_connect_t fail\n");
        return -1;
    }

    connection->data = conn_params;

    if ((result = uv_tcp_connect(connection, client, (const struct sockaddr *)&addr, conn_params->connect_cb)) != 0) {
        LOG_ERROR("uv_tcp_connect fail, error: %s\n", uv_strerror(result));
        return result;
    }

    return 0;
}


void async_tcp_client::on_alloc_callback(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf)
{
    buf->base = (char *)malloc(suggested_size);
    if (!buf->base) {
        buf->len = 0;
        LOG_ERROR("malloc uv_buf_t->base fail\n");
    }
    else {
        buf->len = (unsigned long)suggested_size;
    }
}


void async_tcp_client::on_close_callback(uv_handle_t *handle)
{
    free(handle);
}


void async_tcp_client::on_shutdown_callback(uv_shutdown_t *sd, int status)
{
    if (status < 0) {
        LOG_ERROR("shutdown fail, error: %s\n", uv_strerror(status));
        return;
    }
    else {
        uv_close((uv_handle_t *)sd->handle, on_close_callback);
    }

    free(sd);
}
