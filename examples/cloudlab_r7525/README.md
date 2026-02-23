# R2CC CloudLab Setup Guide

## Prerequisites

- CloudLab account with access to the `r7525` node type (Clemson cluster, may need to reserve nodes)
- Instantiate the R2CC profile: <https://www.cloudlab.us/p/SoftMeasure/R2CC-r7525>
  - This profile provides 2x r7525 nodes with BlueField SmartNICs and the R2CC image-backed dataset mounted at `/mydata`
  - After instantiating the profile and the experiment is ready, simply follow the steps below to reproduce the demo video experiment

## Setup Workflow

After the experiment is ready and you can SSH into the nodes, run the following scripts **on each node** in order.

**Note:** Some scripts take 10-20+ minutes. Use SSH keepalive or `tmux`/`screen` to avoid disconnection.

```bash
# On each node:
/mydata/01.setup_flash_firmware.sh        # then wait for reboot
/mydata/02.setup_network_and_nic.sh       # after reboot
/mydata/03.setup_topo.sh                  # dump topology

# Run the demo (from node1):
cd /mydata/R2CC/examples/cloudlab_r7525 && ./run_hot_repair.sh
```

### Details

#### Phase 1: Flash BlueField Firmware

```bash
/mydata/01.setup_flash_firmware.sh
```

This script:
- Injects R2CC environment variables (CUDA, OpenMPI, NCCL paths) into your `.bashrc`
- Flashes BlueField SmartNIC firmware and waits for the NIC to come up
- Reboots the node (may take 10-20 minutes)

#### Phase 2: Network & SmartNIC Configuration

After reboot, SSH back in and run:

```bash
/mydata/02.setup_network_and_nic.sh
```

This script:
- Disables ACS
- Configures host network interfaces (IP addresses based on node-1 or node-2)
- Waits for BlueField NIC to be reachable
- Sets up SSH config so you can use `ssh nic` to access the SmartNIC
- Generates an SSH key if you don't have one
- Copies your SSH key to the SmartNIC for passwordless access
- Fixes OVS (Open vSwitch) bridge configuration on the SmartNIC

#### Phase 3: Dump NCCL Topology

```bash
/mydata/03.setup_topo.sh
```

This script:
- Builds and runs the NCCL topology dumper
- Generates `topo.xml` and copies it to your home directory

## Directory Structure

```
/mydata/
  01.setup_flash_firmware.sh   # Phase 1: firmware flash + reboot
  02.setup_network_and_nic.sh  # Phase 2: network + SmartNIC setup
  03.setup_topo.sh             # Phase 3: NCCL topology dump
  bluefield/                   # BlueField firmware and flash scripts
  host/                        # Host environment config (.bashrc, acs.sh)
  nic/                         # SmartNIC scripts (setup_nic, fix_ovs, etc.)
  R2CC/                        # R2CC source and pre-built binaries
  cuda-12.2/                   # CUDA toolkit
  openMpi/                     # OpenMPI installation
```

## Notes

- Both nodes share the same image-backed dataset, so `/mydata` contents are identical.
- After Phase 2, you can access the SmartNIC with `ssh nic`.
- Environment variables (CUDA, OpenMPI, NCCL paths) are automatically added to your `.bashrc` during Phase 1.
