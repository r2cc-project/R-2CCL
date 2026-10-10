# Builds the hot-repair test programs against this repository's NCCL build:
# test_hot_repair (01/02) and test_hot_repair_graph (08), both with the checks of test_common.h.
NVCC      := nvcc
TARGETS   := test_hot_repair test_hot_repair_graph

NCCL_HOME ?= $(abspath ../../../build)
MPI_HOME  ?= /mydata/openMpi
CUDA_HOME ?= /usr/local/cuda

NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu
INCLUDES   := -I$(NCCL_HOME)/include -I$(MPI_HOME)/include
LIBS       := -L$(NCCL_HOME)/lib -L$(MPI_HOME)/lib -L$(CUDA_HOME)/lib64 -lnccl -lmpi -lcudart -lpthread

all: $(TARGETS)

test_hot_repair: test_hot_repair.cc test_common.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

test_hot_repair_graph: test_hot_repair_graph.cc test_common.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)
