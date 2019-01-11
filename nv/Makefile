GCC ?= gcc
CUDA ?= /usr/local/cuda

TARGETS := nv_gb
NVCC_FLAGS := -O2 --disable-warnings

all: ${TARGETS}

%.o: %.cu
	${CUDA}/bin/nvcc ${NVCC_FLAGS} -o $@ -c $<
%: %.o
	${CUDA}/bin/nvcc ${NVCC_FLAGS} $< -o $@
clean:
	rm -rf *.o ${TARGETS}
