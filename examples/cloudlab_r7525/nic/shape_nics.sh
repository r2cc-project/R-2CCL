#!/usr/bin/env bash
# Egress-shape every NIC port of every node to the same rate, so that the three NICs really are equal and the
# NICs, not the GPU side of the servers, bound the collectives (README.md, section 1). Run from node-1 after
# Phase 2. The host-side limit is lost on a host reboot, the SmartNIC-side limit on a BlueField reboot.
#
# Usage: nic/shape_nics.sh [rate_gbps|off|status]      (default: 10)
#   mlx5_0 (eno33np0, host ConnectX-5): ETS limit on all eight traffic classes (RoCE does not run on TC0 there)
#   mlx5_2 / mlx5_3 (BlueField ports p0 / p1): ETS limit on TC0, set on the SmartNIC
# Only transmitted traffic is limited; with every port shaped, each link is bounded by its sender.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nodes.sh"

arg="${1:-10}"
case "${arg}" in
  off)    host_rate="0,0,0,0,0,0,0,0"; bf_rate="0,0,0,0,0,0,0,0" ;;
  status) ;;
  *)      [[ "${arg}" =~ ^[0-9]+$ ]] || { echo "usage: $0 [rate_gbps|off|status]" >&2; exit 1; }
          host_rate="${arg},${arg},${arg},${arg},${arg},${arg},${arg},${arg}"; bf_rate="${arg},0,0,0,0,0,0,0" ;;
esac

on_node() {  # on_node <host> <command>
  if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi
}

NIC_SSH="ssh -o BatchMode=yes -o LogLevel=ERROR nic"
LIMIT="grep -m1 -oE 'ratelimit: [^,]+' || true"
for h in localhost "${REMOTE_HOST_LIST[@]}"; do
  if [[ "${arg}" != status ]]; then
    on_node "${h}" "sudo mlnx_qos -i eno33np0 -r ${host_rate} > /dev/null
                    ${NIC_SSH} 'for p in p0 p1; do sudo mlnx_qos -i \$p -r ${bf_rate} > /dev/null; done'"
  fi
  on_node "${h}" "echo \"\$(hostname -s): mlx5_0 \$(sudo mlnx_qos -i eno33np0 | ${LIMIT}) | mlx5_2 \$(${NIC_SSH} sudo mlnx_qos -i p0 | ${LIMIT}) | mlx5_3 \$(${NIC_SSH} sudo mlnx_qos -i p1 | ${LIMIT})\""
done
