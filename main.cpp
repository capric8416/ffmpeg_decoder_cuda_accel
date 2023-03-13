// http
#include "flv_client.h"

// log
#include "stdout_log.h"



int http_get_file(const char *ip, int port, const char *path, const char *filename) {
    uv_loop_t *loop = uv_default_loop();

    async_http_client http_client(loop);

    auto req_params = new async_http_client::req_params_t;
    req_params->fp = fopen(filename, "wb");
    req_params->ip = ip;
    req_params->port = port;
    req_params->method = "GET";
    req_params->path = path;
    req_params->conn_timeout_ms = 3000;
    req_params->req_timeout_ms = 10000;
    req_params->text_json = false;
    req_params->recv_body_cb = [](void *ctx, uint8_t *content, ssize_t nbytes) {
        auto params = (async_http_client::req_params_t *)ctx;
        fwrite(content, 1, nbytes, params->fp);
    };

    http_client.request(req_params);

    int code = uv_run(loop, UV_RUN_DEFAULT);

    fclose(req_params->fp);

    return code;
}


#define TEST_FLV_STREAM
int main(int argc, char **argv) {

#if defined(TEST_HTTP_FILE)
    http_get_file("127.0.0.1", 80, "/uv_a.lib", "uv_a.lib");
    return http_get_file("127.0.0.1", 80, "/cn_windows_7_home_basic_with_sp1_x86_dvd_u_676500.iso", "cn_windows_7_home_basic_with_sp1_x86_dvd_u_676500.iso");
#endif

#if defined(TEST_FLV_STREAM)
    if (argc < 4) {
        LOG_ERROR("Usage: %s <stream_gateway_ip> <stream_gateway_port> <camera_gbcode>\n", argv[0]);
        return -1;
    }

    uv_loop_t *loop = uv_default_loop();

    flv_client flv(loop);
    flv.set_gateway_addr(argv[1], std::atoi(argv[2]));
    flv.set_output_callbacks(
        nullptr,
        [](flv_client *ctx, int64_t buf, int size, int error) {
            static int i = 0;
            char path[256];
            sprintf(path, "images/%d.rgb", i++);
            FILE *fp = fopen(path, "wb");
            fwrite((void *)buf, 1, size, fp);
            fclose(fp);

            ctx->free_output_buf((uint8_t *)buf);
        },
        nullptr
    );
    flv.connect_stream(argv[3]);

    return uv_run(loop, UV_RUN_DEFAULT);
#endif

}
