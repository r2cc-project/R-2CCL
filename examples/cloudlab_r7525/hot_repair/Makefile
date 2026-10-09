# Builds the hot-repair test programs against this repository's NCCL build:
# test_hot_repair (01/02) and graph_timing (08).
NVCC      := nvcc
TARGETS   := test_hot_repair graph_timing

NCCL_HOME ?= $(abspath ../../../build)
MPI_HOME  ?= /mydata/openMpi
CUDA_HOME ?= /usr/local/cuda

NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu
INCLUDES   := -I$(NCCL_HOME)/include -I$(MPI_HOME)/include
LIBS       := -L$(NCCL_HOME)/lib -L$(MPI_HOME)/lib -L$(CUDA_HOME)/lib64 -lnccl -lmpi -lcudart -lpthread

all: $(TARGETS)

test_hot_repair: test_hot_repair.cc
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

graph_timing: graph_timing.cc
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)
