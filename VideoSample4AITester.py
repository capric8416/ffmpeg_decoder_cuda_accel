import platform
import signal

from VideoSample4AI import *



def quit(signum, frame):
    globals()['sampler'].stop()


if __name__ == '__main__':
    globals()['sampler'] = VideoSample4AI(f'./Sample4AI.{"dll" if "Windows" in platform.system() else "so"}')
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    globals()['sampler'].start("192.168.9.6", 8082, "19216811401000001102")
