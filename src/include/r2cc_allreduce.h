#ifndef NCCL_R2CC_ALLREDUCE_H_
#define NCCL_R2CC_ALLREDUCE_H_
#include "info.h"

ncclResult_t r2ccPrepareBalance(struct ncclComm* comm, int mode);
ncclResult_t r2ccAllReduce(struct ncclInfo* info);
ncclResult_t r2ccAllReduceDestroy(struct ncclComm* comm, bool abort);
#endif
