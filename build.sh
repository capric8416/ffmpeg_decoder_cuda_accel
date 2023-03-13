# apt install libuv1-dev

nvcc                                                        \
	-shared                                                 \
	                                                        \
	-o Sample4AI.so                                         \
	                                                        \
	--std=c++17                                             \
	                                                        \
	-I/home/capric/source/output/include                    \
	-I/home/capric/Qt5.12.12.1/5.12.12/gcc_64/include/      \
	-L/home/capric/source/output/lib                        \
	-L/home/capric/Qt5.12.12.1/5.12.12/gcc_64/lib/          \
	                                                        \
	-luv                                                    \
	-lQt5Core                                               \
	-lavcodec -lavformat -lavutil                           \
	                                                        \
	--compiler-options '-fPIC'                              \
	                                                        \
	dynamic_lib.cpp                                         \
	http_client.cpp                                         \
	tcp_client.cpp                                          \
	video_decoder.cpp                                       \
	flv_client.cu                                           \
	yuv_accel.cu


mv Sample4AI.so ../x64/Release/ 
