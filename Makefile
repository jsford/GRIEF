all:
	g++ -std=c++17 grief.cpp -o grief -lstdc++fs -Wall -I./ -I/usr/include/eigen3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lfmt -O3
