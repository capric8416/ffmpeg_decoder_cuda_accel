import numpy

# 导入C/C++动态库支持
from ctypes import *

from dataclasses import dataclass


@CFUNCTYPE(None, c_void_p, c_int64, c_int, c_int)
def recv_host_rgb24_buf(ctx, buf, size, error):
    """
    接收host端rgb24内存
    """
    # 释放host端rgb24内存
    globals()['sampler'].free_flv_client_buf(c_void_p(ctx), c_int64(buf))


@CFUNCTYPE(None, c_void_p, c_int64, c_int, c_int)
def recv_device_rgb24_buf(ctx, buf, size, error):
    """
    接收device端rgb24内存
    """
    # 释放device端rgb24内存
    globals()['sampler'].free_flv_client_buf(c_void_p(ctx), c_int64(buf))


class VideoSample4AI:
    def __init__(self, path_dynamic_library):
        """
        加载动态库
        """
        self.library = CDLL(path_dynamic_library)
        self.flv_client = POINTER(c_void_p)()
        self.start_flv_client = self.library.start_flv_client
        self.free_flv_client_buf = self.library.free_flv_client_buf
        self.stop_flv_client = self.library.stop_flv_client

        globals()['sampler'] = self

    def start(self, ip, port, gbcode):
        """
        开始拉流、解码、转换、输出
        """
        self.start_flv_client(byref(self.flv_client), ip.encode(), port, gbcode.encode(), recv_host_rgb24_buf, None)

    def stop(self):
        """
        停止拉流、解码、转换、输出
        """
        self.stop_flv_client(self.flv_client)

