#!/usr/bin/env bash
set -euo pipefail

# Drop all traffic on ovsbr1 (host PF0 <-> uplink p0) via a high-priority rule.
# Make it idempotent by removing any existing cookie first.
ssh nic "sudo -n ovs-ofctl -O OpenFlow13 del-flows ovsbr1 'cookie=0xabc1/0xffffffffffffffff'"
ssh nic "sudo -n ovs-ofctl -O OpenFlow13 add-flow ovsbr1 'cookie=0xabc1,priority=65000,actions=drop'"
