NVCC := nvcc
TARGET := dump_nccl_topo
SRC := dump_nccl_topo.cc

# Flags
NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu

# Libraries
LIBS := -L../../install/lib -lnccl -lcudart -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
