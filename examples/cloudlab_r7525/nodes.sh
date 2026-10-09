#!/usr/bin/env bash
# Node list, shared by common.sh, hot_repair/run_hot_repair.sh, nic/ and tools/. Source it.
#
# node-1 runs mpirun; REMOTE_HOSTS defaults to every node-K (K >= 2) in /etc/hosts, i.e. node-2 on the
# two-node profile and node-2 node-3 on the three-node one. REMOTE_HOSTS=node-2 limits a run to two nodes.
#
# Addresses set by /mydata/02.setup_network_and_nic.sh on node-K:
#   eno33np0  (mlx5_0)  10.10.1.(3K-2)
#   ens5f0np0 (mlx5_2)  10.10.2.(3K-1)
#   ens5f1np1 (mlx5_3)  10.10.3.(3K)

if [[ -z "${REMOTE_HOSTS:-}" ]]; then
  REMOTE_HOSTS="${REMOTE_HOST:-}"
  if [[ -z "${REMOTE_HOSTS}" ]]; then
    for ((_k = 2; ; _k++)); do
      getent hosts "node-${_k}" >/dev/null 2>&1 || break
      REMOTE_HOSTS+="${REMOTE_HOSTS:+ }node-${_k}"
    done
    unset _k
  fi
fi
read -r -a REMOTE_HOST_LIST <<< "${REMOTE_HOSTS}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NNODES=$((1 + ${#REMOTE_HOST_LIST[@]}))
NRANKS=$((NNODES * GPUS_PER_NODE))
MPI_HOSTS="localhost:${GPUS_PER_NODE}"
for _h in "${REMOTE_HOST_LIST[@]}"; do MPI_HOSTS+=",${_h}:${GPUS_PER_NODE}"; done
unset _h

# node_ips node-K -> "eno33np0 ens5f0np0 ens5f1np1" addresses
node_ips() {
  local k="${1#node-}"
  echo "10.10.1.$((3 * k - 2)) 10.10.2.$((3 * k - 1)) 10.10.3.$((3 * k))"
}
