// self
#include "dynamic_lib.h"


int start_flv_client(void **flv, const char *ip, const int port, const char *gbcode, flv_client::recv_host_rgb24_to_np_cb recv_host_buf_to_np, flv_client::recv_host_rgb24_cb recv_host_buf_cb, flv_client::recv_device_rgb24_cb recv_device_buf_cb)
{
    uv_loop_t *loop = uv_default_loop();

    auto client = new flv_client(loop);
    *flv = client;

    client->set_gateway_addr(ip, port);
    client->set_output_callbacks(recv_host_buf_to_np, recv_host_buf_cb, recv_device_buf_cb);
    client->connect_stream(gbcode);
    
    return uv_run(loop, UV_RUN_DEFAULT);
}


void free_flv_client_buf(void *p, int64_t buf)
{
    auto client = (flv_client *)p;
    client->free_output_buf((uint8_t *)buf);
}


void stop_flv_client(void *p)
{
    auto client = (flv_client *)p;
    client->stop();
    delete client;
}
