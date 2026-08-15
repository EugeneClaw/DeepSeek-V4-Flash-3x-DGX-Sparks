# Fabric: pairwise QSFP triangle + NCCL mesh

Three DGX Sparks with two ConnectX-7 ports each almost never form **one**
RoCE `/24`. The usual wiring is a **triangle of pairwise links**:

```
node0 ←— 192.168.100.0/24 —→ node1
node0 ←— 192.168.101.0/24 —→ node2
node1 ←— 192.168.102.0/24 —→ node2
```

There is no single HCA that can see all three peers. Stock NCCL IB
assumes rail *i* is one L2 domain. On this triangle the W1–W2 hop is
often **cross-ported** (f0 ↔ f1), so NCCL pairs the wrong GID and
`ibv_modify_qp` INIT→RTR times out.

## What this recipe uses

| Plane | Interface | Role |
|---|---|---|
| Control | Shared LAN (example `enP7s7`) | Gloo, TCPStore, `--master-addr`, SSH |
| Data | Both CX7 ports via **NCCL mesh plugin** | TP allreduce + EP allgather/reducescatter |

`NCCL_NET=Mesh`, `NCCL_NET_PLUGIN=mesh`, `NCCL_ALGO=Ring`,
`NCCL_SOCKET_IFNAME==enP7s7` (leading `=` = exact match, NCCL syntax).

The plugin picks an IPv4-mapped RoCEv2 GID **per port**. Do **not** pin
`NCCL_IB_GID_INDEX` — the IPv4 RoCEv2 slot is not the same index on
every HCA.

## What does not work (we measured it)

- Stock `NCCL_NET=IB` even with `MERGE_NICS=0` and
  `SUBNET_AWARE_ROUTING=1`
- TCP on the QSFP Ethernet names (`cxsock`)
- Two-hop IP forward across the triangle (adjacent ICMP works; the
  opposite `/24` does not)

Socket on the shared LAN (`enP7s7`) **does** boot TP/EP. It is much
slower (~9–35 tok/s depending on MTP). Use it only to debug Gloo.

## Build the plugin

On node0:

```bash
./scripts/build-mesh-plugin.sh
```

`start.sh` rsyncs `nccl-mesh/libnccl-net-mesh.so` to the workers.
Requires `libibverbs-dev` on the build host.

Upstream: [autoscriptlabs/nccl-mesh-plugin](https://github.com/autoscriptlabs/nccl-mesh-plugin).
