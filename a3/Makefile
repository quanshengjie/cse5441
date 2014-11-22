all: mmdriver k1.bin k2.bin k3.bin k4.bin

mmdriver: mmdriver.c
	nvcc -O -o mmdriver -lcuda mmdriver.c

k1.bin: k1.cu
	nvcc -O -cubin -arch sm_20 -o k1.bin k1.cu
 
k2.bin: k2.cu
	nvcc -O -cubin -arch sm_20 -o k2.bin k2.cu
 
k3.bin: k3.cu
	nvcc -O -cubin -arch sm_20 -o k3.bin k3.cu
 
k4.bin: k4.cu
	nvcc -O -cubin -arch sm_20 -o k4.bin k4.cu 

clean:
	rm mmdriver *.bin
