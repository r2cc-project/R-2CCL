# Builds the NCCL topology dumper against this repository's NCCL build (used by cloudlab_dump_topo.sh).
NVCC      := nvcc
TARGET    := dump_nccl_topo
SRC       := dump_nccl_topo.cc

NCCL_HOME ?= $(abspath ../../../build)
CUDA_HOME ?= /usr/local/cuda

NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu
INCLUDES   := -I$(NCCL_HOME)/include
LIBS       := -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64 -lnccl -lcudart -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
