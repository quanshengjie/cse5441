all: mmdriver k6.bin k7.bin k8.bin k9.bin k10.bin

mmdriver: mmdriver.c
	nvcc -O -arch sm_20 -o mmdriver -lcuda mmdriver.c

k6.bin: k6.cu
	nvcc -O -cubin -arch sm_20 -o k6.bin k6.cu
 
k7.bin: k7.cu
	nvcc -O -cubin -arch sm_20 -o k7.bin k7.cu
 
k8.bin: k8.cu
	nvcc -O -cubin -arch sm_20 -o k8.bin k8.cu
 
k9.bin: k9.cu
	nvcc -O -cubin -arch sm_20 -o k9.bin k9.cu 

k10.bin: k10.cu
	nvcc -O -cubin -arch sm_20 -o k10.bin k10.cu 

clean:
	rm mmdriver *.bin
