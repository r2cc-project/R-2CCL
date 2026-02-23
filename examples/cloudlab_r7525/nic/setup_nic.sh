#!/usr/bin/env bash
set -euo pipefail

# Configure passwordless SSH to SmartNIC by appending local public key
# to remote ~/.ssh/authorized_keys.
#
# Defaults are aligned with CloudLab environment:
# - target host alias: nic
# - target password: cloudlab
#
# Optional env overrides:
#   NIC_TARGET=nic
#   NIC_PASSWORD=cloudlab
#   NIC_KEY_DIR=~/.ssh
#   NIC_KEY_PREFIX=nic
#   NIC_KEY_PRIV=~/.ssh/nic_dedicated
#   NIC_KEY_PUB=~/.ssh/nic_dedicated.pub
#   NIC_SETUP_SSH_CONFIG=1

NIC_TARGET="${NIC_TARGET:-nic}"
NIC_PASSWORD="${NIC_PASSWORD:-cloudlab}"
NIC_KEY_DIR="${NIC_KEY_DIR:-${HOME}/.ssh}"
NIC_KEY_PREFIX="${NIC_KEY_PREFIX:-nic}"
NIC_KEY_PRIV="${NIC_KEY_PRIV:-}"
NIC_KEY_PUB="${NIC_KEY_PUB:-}"
NIC_SETUP_SSH_CONFIG="${NIC_SETUP_SSH_CONFIG:-1}"

mkdir -p "${NIC_KEY_DIR}"
chmod 700 "${NIC_KEY_DIR}"

is_private_key_file() {
  local f="$1"
  [[ -f "${f}" ]] && head -n 1 "${f}" | grep -Eq '^-+BEGIN .*PRIVATE KEY-+$'
}

discover_nic_key_pair() {
  local pub priv candidate

  while IFS= read -r pub; do
    priv="${pub%.pub}"
    if is_private_key_file "${priv}"; then
      NIC_KEY_PRIV="${priv}"
      NIC_KEY_PUB="${pub}"
      return 0
    fi
  done < <(find "${NIC_KEY_DIR}" -maxdepth 1 -type f -name "${NIC_KEY_PREFIX}*.pub" | sort)

  while IFS= read -r candidate; do
    [[ "${candidate}" == *.pub ]] && continue
    if ! is_private_key_file "${candidate}"; then
      continue
    fi
    pub="${candidate}.pub"
    NIC_KEY_PRIV="${candidate}"
    NIC_KEY_PUB="${pub}"
    return 0
  done < <(find "${NIC_KEY_DIR}" -maxdepth 1 -type f -name "${NIC_KEY_PREFIX}*" | sort)

  return 1
}

generate_dedicated_key_pair() {
  mkdir -p "$(dirname "${NIC_KEY_PRIV}")"
  ssh-keygen -t rsa -b 4096 -f "${NIC_KEY_PRIV}" -N "" -C "nic-dedicated-key"
  NIC_KEY_PUB="${NIC_KEY_PRIV}.pub"
}

if [[ -n "${NIC_KEY_PRIV}" || -n "${NIC_KEY_PUB}" ]]; then
  if [[ -n "${NIC_KEY_PRIV}" && -z "${NIC_KEY_PUB}" ]]; then
    NIC_KEY_PUB="${NIC_KEY_PRIV}.pub"
  elif [[ -z "${NIC_KEY_PRIV}" && -n "${NIC_KEY_PUB}" ]]; then
    NIC_KEY_PRIV="${NIC_KEY_PUB%.pub}"
  fi
elif discover_nic_key_pair; then
  :
else
  NIC_KEY_PRIV="${NIC_KEY_DIR}/${NIC_KEY_PREFIX}_dedicated"
  NIC_KEY_PUB="${NIC_KEY_PRIV}.pub"
fi

if [[ -f "${NIC_KEY_PRIV}" && -f "${NIC_KEY_PUB}" ]]; then
  echo "Using existing NIC dedicated SSH key pair."
elif [[ -f "${NIC_KEY_PRIV}" && ! -f "${NIC_KEY_PUB}" ]]; then
  echo "NIC public key missing; deriving it from existing private key ..."
  ssh-keygen -y -f "${NIC_KEY_PRIV}" > "${NIC_KEY_PUB}"
elif [[ ! -f "${NIC_KEY_PRIV}" && -f "${NIC_KEY_PUB}" ]]; then
  echo "NIC private key missing; regenerating dedicated NIC SSH key pair ..."
  generate_dedicated_key_pair
else
  echo "Generating dedicated NIC SSH key pair ..."
  generate_dedicated_key_pair
fi

PUBKEY="${NIC_KEY_PUB}"
if [[ ! -f "${PUBKEY}" ]]; then
  echo "Failed to prepare dedicated NIC public key."
  exit 1
fi
chmod 644 "${PUBKEY}"

if [[ "${NIC_SETUP_SSH_CONFIG}" == "1" ]]; then
  SSH_CONFIG_FILE="${HOME}/.ssh/config"
  touch "${SSH_CONFIG_FILE}"
  chmod 600 "${SSH_CONFIG_FILE}"
  TMP_CONFIG="$(mktemp)"
  awk '
    BEGIN {skip=0}
    /^# BEGIN setup_nic managed block$/ {skip=1; next}
    /^# END setup_nic managed block$/ {skip=0; next}
    skip==0 {print}
  ' "${SSH_CONFIG_FILE}" > "${TMP_CONFIG}"
  mv "${TMP_CONFIG}" "${SSH_CONFIG_FILE}"
  cat >> "${SSH_CONFIG_FILE}" <<EOF
# BEGIN setup_nic managed block
Host ${NIC_TARGET}
  IdentityFile ${NIC_KEY_PRIV}
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
# END setup_nic managed block
EOF
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

if command -v ssh-copy-id >/dev/null 2>&1; then
  if command -v sshpass >/dev/null 2>&1; then
    sshpass -p "${NIC_PASSWORD}" ssh-copy-id -f "${SSH_OPTS[@]}" -i "${PUBKEY}" "${NIC_TARGET}"
  else
    echo "sshpass not found; using SSH_ASKPASS fallback."
    ASKPASS_SCRIPT="$(mktemp)"
    trap 'rm -f "${ASKPASS_SCRIPT}"' EXIT
    cat > "${ASKPASS_SCRIPT}" <<EOF
#!/usr/bin/env bash
echo '${NIC_PASSWORD}'
EOF
    chmod 700 "${ASKPASS_SCRIPT}"
    DISPLAY=:0 SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force \
      setsid -w ssh-copy-id -f "${SSH_OPTS[@]}" -i "${PUBKEY}" "${NIC_TARGET}" < /dev/null
  fi
else
  echo "ssh-copy-id not found; using ssh fallback."
  if command -v sshpass >/dev/null 2>&1; then
    cat "${PUBKEY}" | sshpass -p "${NIC_PASSWORD}" ssh "${SSH_OPTS[@]}" "${NIC_TARGET}" \
      'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; while IFS= read -r k; do grep -qxF "$k" ~/.ssh/authorized_keys || echo "$k" >> ~/.ssh/authorized_keys; done'
  else
    echo "sshpass not found; falling back to interactive password prompt."
    cat "${PUBKEY}" | ssh "${SSH_OPTS[@]}" "${NIC_TARGET}" \
      'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; while IFS= read -r k; do grep -qxF "$k" ~/.ssh/authorized_keys || echo "$k" >> ~/.ssh/authorized_keys; done'
  fi
fi

echo "Done. Test with: ssh ${NIC_TARGET}"
echo "Dedicated key is configured for host alias '${NIC_TARGET}'."
