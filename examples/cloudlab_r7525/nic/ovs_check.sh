#!/usr/bin/env bash
set -euo pipefail

NIC_TARGET="${NIC_TARGET:-nic}"

echo "Checking OVS status on ${NIC_TARGET} ..."
hw_offload="$(
  ssh "${NIC_TARGET}" \
    "sudo ovs-vsctl --no-wait get Open_vSwitch . other_config:hw-offload 2>/dev/null || echo unknown"
)"

offloaded_rules="$(
  ssh "${NIC_TARGET}" \
    "sudo ovs-appctl dpctl/dump-flows type=offloaded 2>/dev/null | grep -c . || true"
)"

bridge_status="$(
  ssh "${NIC_TARGET}" \
    "sudo ovs-vsctl show | awk '/^    Bridge / {print; next} /^        Port / {print; next} /^            Interface / {print; next} /^                type: / {print; next}'"
)"

echo "hw-offload: ${hw_offload}"
echo "hw-offload rule count: ${offloaded_rules}"
echo
echo "bridge status:"
echo "${bridge_status}"
