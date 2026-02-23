#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <dirent.h>
#include <sys/wait.h>
#include <sys/stat.h>

#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

static int get_env_int(const char* name, int def_val) {
  const char* v = getenv(name);
  if (!v || v[0] == '\0') return def_val;
  return atoi(v);
}

static bool path_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::vector<std::string> split_list(const char* list_str) {
  std::vector<std::string> items;
  if (!list_str || list_str[0] == '\0') return items;
  const char* p = list_str;
  while (*p) {
    while (*p == ' ' || *p == ',') p++;
    if (*p == '\0') break;
    const char* start = p;
    while (*p && *p != ',') p++;
    items.emplace_back(start, p - start);
  }
  return items;
}

static void sanitize_env_list(const char* env_name, const char* sysfs_root) {
  const char* v = getenv(env_name);
  if (!v || v[0] == '\0') return;
  auto items = split_list(v);
  std::string kept;
  for (const auto& it : items) {
    if (it.empty()) continue;
    std::string path = std::string(sysfs_root) + "/" + it;
    if (path_exists(path)) {
      if (!kept.empty()) kept += ",";
      kept += it;
    }
  }
  if (kept.empty()) {
    unsetenv(env_name);
  } else {
    setenv(env_name, kept.c_str(), 1);
  }
}

static std::vector<std::string> discover_devices() {
  // Priority:
  // 1) R2CC_XML_GPU_LIST (comma-separated tokens)
  // 2) CUDA_VISIBLE_DEVICES (comma-separated tokens, may be UUIDs)
  // 3) R2CC_XML_NUM_GPUS (N -> 0..N-1)
  // 4) /proc/driver/nvidia/gpus (count -> 0..N-1)
  std::vector<std::string> devs;
  const char* list_env = getenv("R2CC_XML_GPU_LIST");
  if (list_env && list_env[0] != '\0') {
    return split_list(list_env);
  }
  const char* cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
  if (cuda_visible && cuda_visible[0] != '\0') {
    return split_list(cuda_visible);
  }
  int num = get_env_int("R2CC_XML_NUM_GPUS", -1);
  if (num > 0) {
    for (int i = 0; i < num; i++) devs.push_back(std::to_string(i));
    return devs;
  }
  DIR* dir = opendir("/proc/driver/nvidia/gpus");
  if (dir) {
    int count = 0;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
      if (ent->d_name[0] == '.') continue;
      count++;
    }
    closedir(dir);
    for (int i = 0; i < count; i++) devs.push_back(std::to_string(i));
  }
  return devs;
}

static int run_single_rank(int rank, int nranks, const ncclUniqueId& id) {
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaFree(0));
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  ncclCommDestroy(comm);
  return 0;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  // Clear or sanitize stale network env unless user asks to preserve.
  if (getenv("R2CC_XML_PRESERVE_NET_ENV") == nullptr) {
    sanitize_env_list("NCCL_SOCKET_IFNAME", "/sys/class/net");
    sanitize_env_list("NCCL_IB_HCA", "/sys/class/infiniband");
  }

  // Decide output file name (if not already set)
  if (getenv("NCCL_TOPO_DUMP_FILE") == nullptr) {
    setenv("NCCL_TOPO_DUMP_FILE", "topo.xml", 1);
  }
  if (getenv("NCCL_TOPO_DUMP_FILE_RANK") == nullptr) {
    setenv("NCCL_TOPO_DUMP_FILE_RANK", "0", 1);
  }

  // Build device list without CUDA (safe before fork).
  std::vector<std::string> devs = discover_devices();
  if (devs.empty()) {
    printf("No CUDA devices found. Set R2CC_XML_NUM_GPUS or R2CC_XML_GPU_LIST.\n");
    return 1;
  }

  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));

  if (devs.size() == 1) {
    // Single GPU: no fork needed.
    setenv("CUDA_VISIBLE_DEVICES", devs[0].c_str(), 1);
    int rc = run_single_rank(0, 1, id);
    if (rc != 0) return rc;
  } else {
    // Multi-GPU: fork one process per GPU to avoid NCCL thread-safety issues.
    std::vector<pid_t> pids;
    pids.reserve(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
      pid_t pid = fork();
      if (pid < 0) {
        perror("fork");
        return 1;
      }
      if (pid == 0) {
        setenv("CUDA_VISIBLE_DEVICES", devs[i].c_str(), 1);
        int rc = run_single_rank((int)i, (int)devs.size(), id);
        _exit(rc);
      }
      pids.push_back(pid);
    }
    int status = 0;
    for (pid_t pid : pids) {
      int st = 0;
      if (waitpid(pid, &st, 0) < 0) {
        perror("waitpid");
        return 1;
      }
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        status = 1;
      }
    }
    if (status != 0) return 1;
  }

  const char* out = getenv("NCCL_TOPO_DUMP_FILE");
  printf("NCCL topo dumped to %s (devices=%zu)\n", out ? out : "<unset>", devs.size());
  return 0;
}
