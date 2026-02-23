#!/usr/bin/env bash
set -euo pipefail

# Remove the drop rule added by disconnect_nic1.sh (match by cookie).
ssh nic "sudo -n ovs-ofctl -O OpenFlow13 del-flows ovsbr1 'cookie=0xabc1/0xffffffffffffffff'"
