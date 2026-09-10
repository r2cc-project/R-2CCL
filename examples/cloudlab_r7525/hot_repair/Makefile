# Builds the hot-repair test program against this repository's NCCL build.
NVCC      := nvcc
TARGET    := test_hot_repair
SRC       := test_hot_repair.cc

NCCL_HOME ?= $(abspath ../../../build)
MPI_HOME  ?= /mydata/openMpi
CUDA_HOME ?= /usr/local/cuda

NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu
INCLUDES   := -I$(NCCL_HOME)/include -I$(MPI_HOME)/include
LIBS       := -L$(NCCL_HOME)/lib -L$(MPI_HOME)/lib -L$(CUDA_HOME)/lib64 -lnccl -lmpi -lcudart -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
