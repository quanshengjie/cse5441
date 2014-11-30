KERNELS=$(wildcard *.cu)

BINS=$(KERNELS:.cu=.bin)

all: mmdriver $(BINS)

mmdriver: mmdriver.c
	nvcc -O -arch sm_20 -o mmdriver -lcuda mmdriver.c

%.bin: %.cu
	nvcc -O -cubin -arch sm_20 -o $@ $<

clean:
	rm mmdriver $(BINS)
