
NVCC := nvcc
TARGET := test_hot_repair
SRC := test_hot_repair.cc

# Flags
NVCC_FLAGS := -O3 -g -std=c++11 -Xcompiler -pthread -x cu

# Libraries
# Assumes generic paths or that environment variables (LD_LIBRARY_PATH etc)# Libraries
LIBS := -L../../../install/lib -lnccl -lmpi -lcudart -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
