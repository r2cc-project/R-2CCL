#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

make clean
make
./dump_nccl_topo
sed -i -E 's/speed="(25000|100000)"/speed="10000"/g' topo.xml
cp topo.xml ~/
