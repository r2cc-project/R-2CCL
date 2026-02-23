#!/usr/bin/env bash
set -euo pipefail

ips=(
  "10.10.1.4"
  "10.10.2.5"
  "10.10.3.6"
)

for ip in "${ips[@]}"; do
  if ping -c 1 -W 1 "${ip}" >/dev/null 2>&1; then
    echo "${ip} reachable"
  else
    echo "${ip} unreachable"
  fi
done
