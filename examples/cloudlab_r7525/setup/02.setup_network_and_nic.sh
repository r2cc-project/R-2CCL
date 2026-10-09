#!/bin/bash
# Phase 2: Configure network interfaces and SmartNIC (run as normal user)
# Usage: /mydata/02.setup_network_and_nic.sh
set -euo pipefail

echo "============================================"
echo "  Phase 2: Network & SmartNIC Setup"
echo "============================================"
echo ""

NODE_ID=$(geni-get client_id)

# node-K: eno33np0 10.10.1.(3K-2), ens5f0np0 10.10.2.(3K-1), ens5f1np1 10.10.3.(3K)
# (node-1: .1/.2/.3, node-2: .4/.5/.6, node-3: .7/.8/.9)
node_index() {
    local k="${1#node-}"
    [[ "$k" =~ ^[0-9]+$ ]] && echo "$k"
}

# [1/8] Disable ACS
echo "[1/8] Disabling ACS..."
sudo bash /mydata/host/acs.sh
echo "[1/8] Done."

# [2/8] Configure network interfaces
echo "[2/8] Configuring network interfaces for $NODE_ID..."
K=$(node_index "$NODE_ID" || true)
if [ -n "$K" ]; then
    sudo ifconfig eno33np0 10.10.1.$((3 * K - 2))/24
    sudo ifconfig ens5f0np0 10.10.2.$((3 * K - 1))/24
    sudo ifconfig ens5f1np1 10.10.3.$((3 * K))/24
else
    echo "WARNING: Unknown node '$NODE_ID', skipping IP configuration."
fi
# All NICs share one VLAN: answer ARP only on the interface that owns the address, otherwise a peer can
# cache another NIC's MAC and RoCE traffic to that address is dropped.
sudo sysctl -q -w net.ipv4.conf.all.arp_ignore=1 net.ipv4.conf.all.arp_announce=2
sudo ip neigh flush dev eno33np0
sudo ip neigh flush dev ens5f0np0
sudo ip neigh flush dev ens5f1np1
echo "[2/8] Done."

# [3/8] Configure inter-node SSH (every node-J of the experiment)
# /etc/hosts lists all three CloudLab addresses of a node and two of them are gone after step 2, so pin each
# peer to its eno33np0 address.
echo "[3/8] Configuring inter-node SSH..."
cp /mydata/key/cloudlab_key "${HOME}/.ssh/cloudlab_key"
chmod 600 "${HOME}/.ssh/cloudlab_key"

CLOUDLAB_PUB=$(cat /mydata/key/cloudlab_key.pub)
if ! grep -qF "$CLOUDLAB_PUB" "${HOME}/.ssh/authorized_keys" 2>/dev/null; then
    echo "$CLOUDLAB_PUB" >> "${HOME}/.ssh/authorized_keys"
fi

SSH_CONFIG="${HOME}/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"
TMP_CONFIG="$(mktemp)"
awk '/^# BEGIN r2cc nodes$/ {skip=1; next} /^# END r2cc nodes$/ {skip=0; next} !skip' "$SSH_CONFIG" > "$TMP_CONFIG"
{
    cat "$TMP_CONFIG"
    echo "# BEGIN r2cc nodes"
    for ((J = 1; ; J++)); do
        getent hosts "node-$J" >/dev/null || break
        [ "node-$J" = "$NODE_ID" ] && continue
        cat <<EOF
Host node-$J
    HostName 10.10.1.$((3 * J - 2))
    IdentityFile ~/.ssh/cloudlab_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF
    done
    echo "# END r2cc nodes"
} > "$SSH_CONFIG"
rm -f "$TMP_CONFIG"
chmod 600 "$SSH_CONFIG"
echo "[3/8] Done."

# [4/8] Wait for BlueField NIC
echo "[4/8] Waiting for BlueField NIC (192.168.100.2)..."
while ! ping -c 1 -W 2 192.168.100.2 &>/dev/null; do
    sleep 5
done
echo "[4/8] Done. BlueField NIC is up."

# [5/8] Configure SSH config for nic
echo "[5/8] Adding 'nic' to ~/.ssh/config..."
SSH_CONFIG="${HOME}/.ssh/config"
if ! grep -q "Host nic" "$SSH_CONFIG" 2>/dev/null; then
    cat >> "$SSH_CONFIG" <<EOF

Host nic
    HostName 192.168.100.2
    User ubuntu
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    chmod 600 "$SSH_CONFIG"
fi
ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "192.168.100.2" 2>/dev/null || true
echo "[5/8] Done."

# [6/8] Generate SSH key if needed
echo "[6/8] Checking SSH key..."
if [ ! -f "${HOME}/.ssh/id_ed25519.pub" ]; then
    echo "       No SSH key found. Generating one..."
    ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519" -N ""
fi
echo "[6/8] Done."

# [7/8] Copy SSH key to NIC
echo "[7/8] Setting up passwordless SSH to NIC..."
NIC_PASSWORD="${NIC_PASSWORD:-cloudlab}"
PUBKEY="${HOME}/.ssh/id_ed25519.pub"
if [ ! -f "$PUBKEY" ]; then
    PUBKEY="${HOME}/.ssh/id_rsa.pub"
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if command -v sshpass >/dev/null 2>&1; then
    sshpass -p "${NIC_PASSWORD}" ssh-copy-id "${SSH_OPTS[@]}" -i "${PUBKEY}" nic
else
    ASKPASS_SCRIPT="$(mktemp)"
    trap 'rm -f "${ASKPASS_SCRIPT}"' EXIT
    cat > "${ASKPASS_SCRIPT}" <<EOF
#!/usr/bin/env bash
echo '${NIC_PASSWORD}'
EOF
    chmod 700 "${ASKPASS_SCRIPT}"
    DISPLAY=:0 SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force \
        setsid -w ssh-copy-id "${SSH_OPTS[@]}" -i "${PUBKEY}" nic < /dev/null
fi
echo "[7/8] Done. Testing connection..."
ssh nic whoami
echo ""

# [8/8] Fix OVS on NIC
echo "[8/8] Fixing OVS configuration on NIC..."
/mydata/nic/fix_ovs.sh
echo "[8/8] Done."

echo ""
echo "============================================"
echo "  Phase 2 complete for $NODE_ID."
echo "  You can now use: ssh nic"
echo "============================================"
