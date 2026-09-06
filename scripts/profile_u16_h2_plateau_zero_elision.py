#!/usr/bin/env python3
"""Profile U16 H2 plateau-zero persistence elision on/off."""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")
LEAF_RE = re.compile(
    r"PROFILE_U16_PERSIST_LEAF dimension=h2 key_bytes=4 plateau_zero_elision=(on|off) "
    r"zero_persistence_pairs_elided=(\d+) union_attempts=(\d+) successful_unions=(\d+) "
    r"scalar_order_seconds=([0-9.]+) local_sweep_seconds=([0-9.]+)"
)


def command(binary: Path, stack: Path, depth: int, conn: int, out: Path) -> list[str]:
    return [
        str(binary), str(stack), str(depth), str(conn), "h2-scalar-hierarchical-stream", str(out),
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", "direct",
        "--h2-hier-outside-structural-pruning", "off",
    ]


def run_once(cmd: list[str], timing_path: Path, enabled: bool) -> dict[str, float | int | str]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1"
    env["BETTI_PERSIST_H2_PLATEAU_ZERO_ELISION"] = "1" if enabled else "0"
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    timing = timing_path.read_text()
    match = LEAF_RE.search(proc.stdout)
    if not match:
        raise RuntimeError("missing PROFILE_U16_PERSIST_LEAF H2 line")
    return {
        "wall_seconds": wall,
        "max_rss_mib": float(RSS_RE.search(timing).group(1)) / 1024.0,
        "user_seconds": float(USER_RE.search(timing).group(1)),
        "system_seconds": float(SYS_RE.search(timing).group(1)),
        "plateau_zero_elision": match.group(1),
        "zero_persistence_pairs_elided": int(match.group(2)),
        "union_attempts": int(match.group(3)),
        "successful_unions": int(match.group(4)),
        "scalar_order_seconds": float(match.group(5)),
        "local_sweep_seconds": float(match.group(6)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--foreground-connectivity", type=int, default=26)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=Path("profiles/v22_u16_h2_plateau_zero.csv"))
    args = ap.parse_args()
    stack = args.stack.resolve(); binary = args.binary.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="u16_h2_plateau_zero_prof_") as td_raw:
        td = Path(td_raw); seq = 0
        for backend, enabled in (("off", False), ("on", True)):
            for depth in args.slab_depths:
                for _ in range(args.warmup):
                    seq += 1
                    print(f"warmup plateau={backend} h2 d={depth}")
                    run_once(command(binary, stack, depth, args.foreground_connectivity, td/f"{seq}.csv"), td/f"{seq}.time", enabled)
                for rep in range(1, args.repeats + 1):
                    seq += 1
                    result = run_once(command(binary, stack, depth, args.foreground_connectivity, td/f"{seq}.csv"), td/f"{seq}.time", enabled)
                    row = {"backend": backend, "slab_depth": depth, "repeat": rep, **result}
                    rows.append(row)
                    print(f"plateau={backend} d={depth} r={rep}: {result['wall_seconds']:.3f}s {result['max_rss_mib']:.1f} MiB")
    fields = list(rows[0])
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = args.output.with_name(args.output.stem + "_summary.csv")
    groups: dict[tuple[str,int], list[dict[str,object]]] = {}
    for row in rows: groups.setdefault((str(row["backend"]), int(row["slab_depth"])), []).append(row)
    fields2 = [
        "backend","slab_depth","repeats","median_wall_seconds","median_max_rss_mib",
        "median_user_seconds","median_system_seconds","median_zero_persistence_pairs_elided",
        "median_union_attempts","median_successful_unions","median_scalar_order_seconds","median_local_sweep_seconds",
    ]
    with summary.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields2); w.writeheader()
        for (backend, depth), rs in sorted(groups.items()):
            med=lambda name: float(statistics.median(float(r[name]) for r in rs))
            w.writerow({
                "backend":backend,"slab_depth":depth,"repeats":len(rs),
                "median_wall_seconds":med("wall_seconds"),"median_max_rss_mib":med("max_rss_mib"),
                "median_user_seconds":med("user_seconds"),"median_system_seconds":med("system_seconds"),
                "median_zero_persistence_pairs_elided":med("zero_persistence_pairs_elided"),
                "median_union_attempts":med("union_attempts"),"median_successful_unions":med("successful_unions"),
                "median_scalar_order_seconds":med("scalar_order_seconds"),"median_local_sweep_seconds":med("local_sweep_seconds"),
            })
    print(f"wrote {args.output} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
