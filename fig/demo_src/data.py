# Builds data.json for the demo video.
#   python3 data.py <raw dir>
# <raw dir> holds the files of the run behind examples/cloudlab_r7525/logs/rx_per_node.log (2026-10-10): the
# timestamped test output t01_stdout.log / t02_stdout.log and the 5 ms port_rcv_data samples
# t0{1,2}_node-{1,2,3}.tsv (time, mlx5_0, mlx5_2, mlx5_3). They are kept on node-1 in /mydata/r2cc_checks/rx_per_node.
# The single-backup time is read from the reference log of test 08.
import json, os, re, sys, bisect, statistics

RAW = sys.argv[1] if len(sys.argv) > 1 else "/mydata/r2cc_checks/rx_per_node"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG08 = os.path.join(HERE, "..", "..", "examples", "cloudlab_r7525", "logs", "08.hot_repair_cuda_graph.log")

def events(tag):
    it, other = {}, {}
    for l in open(f"{RAW}/{tag}_stdout.log"):
        ts, _, rest = l.partition(" ")
        m = re.search(r"Iter (\d+)/10 (START|END)(?:: OK \(elapsed (\d+) ms\))?", rest)
        if m:
            it.setdefault(int(m.group(1)), {})[m.group(2)] = float(ts)
            if m.group(3): it[int(m.group(1))]["ms"] = int(m.group(3))
        if "disconnect command completed" in rest: other["disconnect_done"] = float(ts)
        if "Verification:" in rest: other["verification"] = rest.split("] ", 1)[1].strip()
    return it, other

def load(tag, n):
    T, V = [], []
    for l in open(f"{RAW}/{tag}_node-{n}.tsv"):
        a = l.split(); T.append(float(a[0])); V.append([int(x) for x in a[1:]])
    return T, V

def at(T, V, t):
    i = min(max(bisect.bisect_left(T, t), 1), len(T) - 1)
    a, b = T[i-1], T[i]; w = (t - a) / (b - a)
    return [V[i-1][k] + w * (V[i][k] - V[i-1][k]) for k in range(3)]

out = {}
for tag in ("t02", "t01"):
    it, other = events(tag)
    t0 = it[1]["START"]
    run = {"iters": [{"n": n, "start": round(it[n]["START"] - t0, 3), "end": round(it[n]["END"] - t0, 3),
                      "ms": it[n]["ms"]} for n in sorted(it)],
           "disconnect_done": round(other["disconnect_done"] - t0, 3), "verification": other["verification"]}
    T, V = load(tag, 1)
    last = max(i for i in range(1, len(T)) if V[i][1] > V[i-1][1] and T[i] < it[4]["START"])
    run["cut"] = round(T[last] - t0, 3)          # last increase of node-1's mlx5_2 counter before AllReduce #4
    if tag == "t02":                              # cumulative MiB per port on a 10 ms grid, for the animation
        step, lo, hi = 0.01, -0.3, it[10]["END"] - t0 + 0.5
        grid = [round(lo + k * step, 3) for k in range(int((hi - lo) / step) + 1)]
        series = {}
        for n in (1, 2, 3):
            T, V = load(tag, n); base = at(T, V, t0 + lo)
            series[f"node-{n}"] = [[round((at(T, V, t0 + g)[k] - base[k]) * 4 / 2**20, 1) for g in grid] for k in range(3)]
        run["grid"] = {"t0": lo, "step": step, "n": len(grid)}
        run["mib"] = series
    run["steady_ms"] = round(statistics.mean(x["ms"] for x in run["iters"] if x["n"] >= 5))
    out[tag] = run
out["rx_table"] = {"healthy": [7032, 7032, 7032], "balance": [7078, 7078, 7078], "allreduce": [6150, 8230, 8218]}
m = re.search(r"Pre-failure graph, HotRepair\s+([\d.]+) s", open(LOG08).read())
out["single_backup_s"] = float(m.group(1))       # the failed port's traffic all on one backup port
json.dump(out, open(os.path.join(HERE, "data.json"), "w"), separators=(",", ":"))
print("data.json written; cut", out["t02"]["cut"], "s, steady", out["t02"]["steady_ms"], "/", out["t01"]["steady_ms"],
      "ms, single backup", out["single_backup_s"], "s")
