CC = g++
SRCS = gelsight/tracking_class.cpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CFLAGS = -g -Wall -std=c++11 -O3 -Wall -fPIC -shared -std=c++11 `python3 -m pybind11 --includes` -I ./
    PROG = gelsight/util/find_marker.so
endif

ifeq ($(UNAME_S),Darwin)
    CFLAGS = -g -Wall -std=c++11 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` -I ./
    PROG = gelsight/util/find_marker`python3-config --extension-suffix`
endif

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

clean:
	rm $(PROG)
