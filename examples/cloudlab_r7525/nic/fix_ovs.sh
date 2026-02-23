#!/usr/bin/env bash
set -euo pipefail

NIC_TARGET="${NIC_TARGET:-nic}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="${SCRIPT_DIR}/ovs_check.sh"

if [[ ! -x "${CHECK_SCRIPT}" ]]; then
  echo "Missing executable check script: ${CHECK_SCRIPT}"
  exit 1
fi

echo "[fix_ovs] current status on ${NIC_TARGET}:"
CHECK_OUT="$(NIC_TARGET="${NIC_TARGET}" "${CHECK_SCRIPT}" || true)"
echo "${CHECK_OUT}"
echo

if echo "${CHECK_OUT}" | grep -Eq '^hw-offload:\s*"?true"?\s*$' \
  && echo "${CHECK_OUT}" | grep -q '^    Bridge ovsbr1$' \
  && echo "${CHECK_OUT}" | grep -q '^    Bridge ovsbr2$'; then
  echo "[fix_ovs] OVS looks healthy (hw-offload=true, both ovsbr1/ovsbr2 present). No fix needed."
  exit 0
fi

echo "[fix_ovs] applying fix on ${NIC_TARGET} ..."
ssh "${NIC_TARGET}" 'bash -s' <<'EOF'
set -euo pipefail

# Ensure kernel-offload mode is enabled.
sudo ovs-vsctl --no-wait set Open_vSwitch . other_config:hw-offload=true
sudo ovs-vsctl --no-wait set Open_vSwitch . other_config:dpdk-init=false || true

ensure_port_on_bridge() {
  local br="$1"
  local port="$2"

  if ! ip link show "${port}" >/dev/null 2>&1; then
    echo "[WARN] ${port} not found on NIC, skip adding to ${br}"
    return 0
  fi

  local cur_br=""
  cur_br="$(sudo ovs-vsctl port-to-br "${port}" 2>/dev/null || true)"
  if [[ -n "${cur_br}" && "${cur_br}" != "${br}" ]]; then
    sudo ovs-vsctl --if-exists del-port "${cur_br}" "${port}"
  fi

  sudo ovs-vsctl --may-exist add-port "${br}" "${port}"
}

# Ensure both bridges exist.
sudo ovs-vsctl --may-exist add-br ovsbr1
sudo ovs-vsctl --may-exist add-br ovsbr2

# Ensure expected ports are attached.
ensure_port_on_bridge ovsbr1 p0
ensure_port_on_bridge ovsbr1 pf0hpf
ensure_port_on_bridge ovsbr1 en3f0pf0sf0

ensure_port_on_bridge ovsbr2 p1
ensure_port_on_bridge ovsbr2 pf1hpf
ensure_port_on_bridge ovsbr2 en3f1pf1sf0
EOF

echo
echo "[fix_ovs] status after fix:"
NIC_TARGET="${NIC_TARGET}" "${CHECK_SCRIPT}"
