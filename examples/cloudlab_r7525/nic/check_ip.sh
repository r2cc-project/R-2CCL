#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nodes.sh"

for h in "${REMOTE_HOST_LIST[@]}"; do
  for ip in $(node_ips "${h}"); do
    if ping -c 1 -W 1 "${ip}" >/dev/null 2>&1; then
      echo "${h} ${ip} reachable"
    else
      echo "${h} ${ip} unreachable"
    fi
  done
done
