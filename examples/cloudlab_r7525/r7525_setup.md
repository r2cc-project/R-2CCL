# R2CC CloudLab (r7525) setup guide

How to bring up the two-node r7525 testbed used by the experiments in [README.md](README.md).

## Prerequisites

- CloudLab account with access to the `r7525` node type (Clemson cluster, may need to reserve nodes).
- Instantiate the R2CC profile: <https://www.cloudlab.us/p/SoftMeasure/R2CC-r7525>
  - It provides 2x r7525 nodes with BlueField SmartNICs and the R2CC image-backed dataset mounted at `/mydata`.
  - After the experiment is ready and you can SSH into the nodes, follow the steps below.

## Setup workflow

Run the following scripts **on each node**, in order. Some take 10-20+ minutes; use SSH keepalive or
`tmux`/`screen` to avoid disconnection.

```bash
# On each node:
/mydata/01.setup_flash_firmware.sh        # then wait for reboot
/mydata/02.setup_network_and_nic.sh       # after reboot
/mydata/03.setup_topo.sh                  # dump topology

# Then run the experiments (from node-1), see README.md:
cd /mydata/R2CC/examples/cloudlab_r7525 && ./01.hot_repair_to_balance.sh
```

### Phase 1: flash BlueField firmware

```bash
/mydata/01.setup_flash_firmware.sh
```

- Injects the R2CC environment variables (CUDA, OpenMPI, NCCL paths) into your `.bashrc`.
- Flashes the BlueField SmartNIC firmware and waits for the NIC to come up.
- Reboots the node (may take 10-20 minutes).

### Phase 2: network and SmartNIC configuration

After the reboot, SSH back in and run:

```bash
/mydata/02.setup_network_and_nic.sh
```

- Disables ACS.
- Configures the host network interfaces (IP addresses based on node-1 or node-2).
- Waits for the BlueField NIC to be reachable.
- Sets up an SSH config so you can use `ssh nic` to access the SmartNIC.
- Generates an SSH key if you don't have one and copies it to the SmartNIC for passwordless access.
- Fixes the OVS (Open vSwitch) bridge configuration on the SmartNIC.

### Phase 3: dump the NCCL topology

```bash
/mydata/03.setup_topo.sh
```

- Builds and runs the NCCL topology dumper (`xml/dump_nccl_topo`).
- Generates `topo.xml`, **rewrites every NIC speed to 10000** (see README.md for why) and copies it to your
  home directory; the tests pass it to NCCL through `NCCL_TOPO_FILE=~/topo.xml`.

## Directory structure

```
/mydata/
  01.setup_flash_firmware.sh   # Phase 1: firmware flash + reboot
  02.setup_network_and_nic.sh  # Phase 2: network + SmartNIC setup
  03.setup_topo.sh             # Phase 3: NCCL topology dump
  bluefield/                   # BlueField firmware and flash scripts
  host/                        # Host environment config (.bashrc, acs.sh)
  nic/                         # SmartNIC scripts (setup_nic, fix_ovs, etc.)
  R2CC/                        # R2CC source and pre-built binaries (this repository)
  nccl-tests/                  # created by R2CC/examples/cloudlab_r7525/tools/build_nccl_tests.sh
  cuda-12.2/                   # CUDA toolkit
  openMpi/                     # OpenMPI installation
```

## Notes

- Both nodes get the same image-backed dataset, but as **separate copies**: after rebuilding anything under
  `/mydata/R2CC` run `examples/cloudlab_r7525/tools/sync.sh` to rsync it to node-2.
- After Phase 2 you can access the SmartNIC with `ssh nic`. `nic/disconnect_nic1.sh` installs an OVS drop rule
  for the whole `mlx5_2` port of node-1 (the "NIC failure" used by the experiments); `nic/connect_nic1.sh`
  removes it. Every test script reconnects the NIC before it starts.
- Environment variables (CUDA, OpenMPI, NCCL paths) are added to `.bashrc` during Phase 1.
- **The Phase 2 host settings do not survive a host reboot.** After a reboot CloudLab re-assigns the two
  BlueField ports addresses in the `10.10.1.x` subnet and PCIe ACS is enabled again, so the test scripts fail in
  their preflight (`ping 10.10.2.5 failed`). Re-run `/mydata/02.setup_network_and_nic.sh` on the rebooted node
  (it is idempotent and does not reboot), or just its first two steps:
  ```bash
  sudo bash /mydata/host/acs.sh                 # disable ACS again (needed for GPUDirect RDMA)
  sudo ifconfig ens5f0np0 10.10.2.2/24           # node-1 (node-2: 10.10.2.5)
  sudo ifconfig ens5f1np1 10.10.3.3/24           # node-1 (node-2: 10.10.3.6)
  ```
  The SmartNIC itself keeps its OVS configuration across host reboots. `~/topo.xml` lives on the persistent
  home directory and does not need to be regenerated.
