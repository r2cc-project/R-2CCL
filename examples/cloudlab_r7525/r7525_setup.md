# R2CC CloudLab (r7525) setup guide

How to bring up the r7525 testbed used by the experiments in [README.md](README.md): two or three servers
(`node-1`, `node-2`[, `node-3`]). The reference logs were produced on three servers; R2CC-AllReduce can only be
faster than R2CC-Balance with three or more (README.md, section 1.3).

## Prerequisites

- CloudLab account with access to the `r7525` node type (Clemson cluster, may need to reserve nodes). Only the
  `r7525` nodes that still carry a BlueField-2 SmartNIC can be used.
- Instantiate the R2CC profile: <https://www.cloudlab.us/p/SoftMeasure/R2CC-r7525>
  - It provides r7525 nodes with BlueField SmartNICs and the R2CC image-backed dataset mounted at `/mydata`.
    For three servers use three such nodes, named `node-1`, `node-2`, `node-3`, each with its copy of the dataset
    and with all their experiment interfaces in one LAN.
  - After the experiment is ready and you can SSH into the nodes, follow the steps below.

## Setup workflow

Run Phases 1-3 **on each node**, in order, then Phase 4 once from node-1. Some steps take 10-20+ minutes; use
SSH keepalive or `tmux`/`screen` to avoid disconnection.

```bash
# On each node:
/mydata/01.setup_flash_firmware.sh        # then wait for reboot
/mydata/02.setup_network_and_nic.sh       # after reboot
/mydata/03.setup_topo.sh                  # dump topology

# Once, on node-1:
cd /mydata/R2CC/examples/cloudlab_r7525
./nic/check_ip.sh                         # every address of the other nodes must be reachable
./nic/shape_nics.sh 10                    # all NIC ports to 10 Gb/s (README.md, section 1.2)

# Then run the experiments (from node-1), see README.md:
./01.hot_repair_to_balance.sh
```

### Phase 1: flash BlueField firmware

```bash
/mydata/01.setup_flash_firmware.sh
```

- Injects the R2CC environment variables (CUDA, OpenMPI, NCCL paths) into your `.bashrc`.
- Flashes the BlueField SmartNIC image (DOCA 2.0.2) and waits for the NIC to come up.
- Reboots the node (may take 10-20 minutes).

### Phase 2: network and SmartNIC configuration

After the reboot, SSH back in and run:

```bash
/mydata/02.setup_network_and_nic.sh
```

- Disables ACS.
- Configures the host network interfaces. All nine experiment ports share one LAN, so every port gets its own
  subnet, by node number K:

  | | `eno33np0` (mlx5_0, 25G) | `ens5f0np0` (mlx5_2, BlueField port 0) | `ens5f1np1` (mlx5_3, BlueField port 1) |
  |---|---|---|---|
  | node-K | 10.10.1.(3K-2) | 10.10.2.(3K-1) | 10.10.3.(3K) |
  | node-1 / node-2 / node-3 | .1 / .4 / .7 | .2 / .5 / .8 | .3 / .6 / .9 |

- Sets `arp_ignore=1` and `arp_announce=2`: otherwise a node may answer an ARP request for one port's address
  with the MAC of another port, the peer caches it, and RoCE traffic to that address is dropped while ping still
  works.
- Sets up SSH between all nodes (every `node-J` is pinned to its `eno33np0` address, because the other addresses
  CloudLab lists in `/etc/hosts` are gone after the step above).
- Waits for the BlueField NIC, sets up `ssh nic` with passwordless access, and fixes the OVS bridge configuration
  on the SmartNIC.

[`setup/02.setup_network_and_nic.sh`](setup/02.setup_network_and_nic.sh) is a copy of this script. If the copy in
your dataset predates three-node support (it prints `Unknown node 'node-3'`), run the copy from this directory.

### Phase 3: dump the NCCL topology

```bash
/mydata/03.setup_topo.sh
```

- Builds and runs the NCCL topology dumper (`xml/dump_nccl_topo`).
- Generates `topo.xml`, **rewrites every NIC speed to 10000** (10 Gb/s, see README.md) and copies it to your
  home directory; the tests pass it to NCCL through `NCCL_TOPO_FILE=~/topo.xml`.

### Phase 4: shape the NICs (node-1)

```bash
cd /mydata/R2CC/examples/cloudlab_r7525
./nic/shape_nics.sh 10        # rate limit every NIC port of every node to 10 Gb/s
./nic/shape_nics.sh status    # prints the limit of mlx5_0, mlx5_2 and mlx5_3 on every node
./nic/shape_nics.sh off       # removes the limits
```

The limits make the declared speed of `topo.xml` the real one (README.md, section 1.2). Every test prints the
limit of node-1's `mlx5_0` in a `[testbed]` line.

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

- Every node gets the same image-backed dataset, but as **separate copies**: after rebuilding anything under
  `/mydata/R2CC` on node-1, run `examples/cloudlab_r7525/tools/sync.sh` to rsync it to all other nodes.
- After Phase 2 you can access the SmartNIC with `ssh nic`. `nic/disconnect_nic1.sh` installs an OVS drop rule
  for the whole `mlx5_2` port of node-1 (the "NIC failure" used by the experiments); `nic/connect_nic1.sh`
  removes it. Every test script reconnects the NIC before it starts.
- Environment variables (CUDA, OpenMPI, NCCL paths) are added to `.bashrc` during Phase 1.
- **The Phase 2 host settings and the host-side NIC limit do not survive a host reboot.** After a reboot CloudLab
  re-assigns the experiment ports addresses in the `10.10.1.x` subnet and PCIe ACS is enabled again, so the test
  scripts fail in their preflight (`ping 10.10.2.5 failed`). Re-run `/mydata/02.setup_network_and_nic.sh` on the
  rebooted node (it is idempotent and does not reboot), then `nic/shape_nics.sh 10` on node-1 and check the
  result with `nic/shape_nics.sh status`. The SmartNIC keeps its OVS configuration across host reboots.
  `~/topo.xml` lives on the persistent home directory and does not need to be regenerated.

## Troubleshooting

- **`/mydata` is empty after instantiation** (`could not mount on /mydata` in the boot log, `error loading
  journal` in `dmesg`): the dataset snapshot was taken while it was mounted. Repair the node's copy and mount it:
  ```bash
  sudo e2fsck -fy /dev/emulab/bs_node1     # bs_node2 / bs_node3 on the other nodes
  sudo mount /mydata
  ```
- **The SmartNIC has no ports**: on the SmartNIC `ip -br link` shows no `p0`/`pf0hpf`, `dmesg` shows
  `mlx5_core ... Firmware over 120000 MS in pre-initializing state`, and `sudo ovs-vsctl show` reports
  `could not open network device p0`. Then the card is in NIC mode (`sudo mlxconfig -d 81:00.0 q` on the host shows
  `INTERNAL_CPU_OFFLOAD_ENGINE DISABLED(1)`) and the OVS drop rule of the hot-repair tests cannot take effect,
  because the traffic does not pass the SmartNIC's Arm cores. Restore the default (DPU) mode on the host,
  power-cycle the node (`sudo apt-get install -y ipmitool` if needed), and run Phase 2 again:
  ```bash
  sudo mlxconfig -d 81:00.0 -y reset
  sudo ipmitool chassis power cycle
  ```
- **A connection between two nodes stalls although ping works**: check RDMA on every port pair with `ib_write_bw`
  (`-d mlx5_2 -x 3`), and make sure Phase 2 set `arp_ignore`/`arp_announce` on every node (`ip neigh` must show
  the MAC of the matching port for every `10.10.x.y` address).
