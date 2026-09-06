#!/usr/bin/env python3
"""Profile compact U16 persistence keys against the v19 wide64 U16 fallback."""
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
    r"PROFILE_U16_PERSIST_LEAF dimension=(h[02]) key_bytes=(\d+) "
    r"zero_persistence_pairs_elided=(\d+) union_attempts=(\d+) "
    r"successful_unions=(\d+) scalar_order_seconds=([0-9.]+) "
    r"local_sweep_seconds=([0-9.]+)"
)


def command(binary: Path, stack: Path, depth: int, conn: int, dim: str, out: Path) -> list[str]:
    mode = f"{dim}-scalar-hierarchical-stream"
    cmd = [str(binary), str(stack), str(depth), str(conn), mode, str(out)]
    if dim == "h0":
        cmd += [
            "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct",
            "--global-h0-uf-layout", "packed",
            "--h0-hier-attach-pruning", "elder-dominated",
        ]
    else:
        cmd += [
            "--local-h2-birth-state", "compact",
            "--global-h2-birth-state", "compact",
            "--global-h2-uf-layout", "packed",
            "--h2-hier-cross-storage", "direct",
            "--h2-hier-outside-structural-pruning", "off",
        ]
    return cmd


def run_once(cmd: list[str], timing_path: Path, native: bool) -> dict[str, float | int | str]:
    env = os.environ.copy()
    env["BETTI_PERSIST_U16_NATIVE_KEYS"] = "1" if native else "0"
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(timing_path), *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wall = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    timing = timing_path.read_text()
    result: dict[str, float | int | str] = {
        "wall_seconds": wall,
        "max_rss_mib": float(RSS_RE.search(timing).group(1)) / 1024.0,
        "user_seconds": float(USER_RE.search(timing).group(1)),
        "system_seconds": float(SYS_RE.search(timing).group(1)),
        "key_bytes": 8 if not native else 4,
        "zero_persistence_pairs_elided": 0,
        "union_attempts": 0,
        "successful_unions": 0,
        "scalar_order_seconds": 0.0,
        "local_sweep_seconds": 0.0,
    }
    match = LEAF_RE.search(proc.stdout)
    if native and match:
        result.update(
            key_bytes=int(match.group(2)),
            zero_persistence_pairs_elided=int(match.group(3)),
            union_attempts=int(match.group(4)),
            successful_unions=int(match.group(5)),
            scalar_order_seconds=float(match.group(6)),
            local_sweep_seconds=float(match.group(7)),
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    ap.add_argument("--slab-depths", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--foreground-connectivity", type=int, default=26)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=Path("profiles/v20_u16_native_keys.csv"))
    args = ap.parse_args()

    stack = args.stack.resolve()
    binary = args.binary.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="u16_native_key_prof_") as td_raw:
        td = Path(td_raw)
        seq = 0
        for dim in ("h0", "h2"):
            for backend, native in (("wide64", False), ("native32", True)):
                for depth in args.slab_depths:
                    for _ in range(args.warmup):
                        seq += 1
                        out = td / f"{seq}_{dim}_{backend}_d{depth}_warm.csv"
                        timing = td / f"{seq}.time"
                        cmd = command(binary, stack, depth, args.foreground_connectivity, dim, out)
                        print(f"warmup {backend}/{dim} d={depth}")
                        run_once(cmd, timing, native)
                    for rep in range(1, args.repeats + 1):
                        seq += 1
                        out = td / f"{seq}_{dim}_{backend}_d{depth}_r{rep}.csv"
                        timing = td / f"{seq}.time"
                        cmd = command(binary, stack, depth, args.foreground_connectivity, dim, out)
                        result = run_once(cmd, timing, native)
                        row = {
                            "dimension": dim,
                            "backend": backend,
                            "slab_depth": depth,
                            "repeat": rep,
                            **result,
                        }
                        rows.append(row)
                        print(
                            f"{backend}/{dim} d={depth} r={rep}: "
                            f"{result['wall_seconds']:.3f}s {result['max_rss_mib']:.1f} MiB"
                        )

    fields = list(rows[0])
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = args.output.with_name(args.output.stem + "_summary.csv")
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dimension"]), str(row["backend"]), int(row["slab_depth"])), []
        ).append(row)

    summary_fields = [
        "dimension", "backend", "slab_depth", "repeats",
        "median_wall_seconds", "median_max_rss_mib", "median_user_seconds", "median_system_seconds",
        "median_key_bytes", "median_zero_persistence_pairs_elided", "median_union_attempts",
        "median_successful_unions", "median_scalar_order_seconds", "median_local_sweep_seconds",
    ]
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for (dim, backend, depth), rs in sorted(groups.items()):
            def med(name: str) -> float:
                return float(statistics.median(float(r[name]) for r in rs))
            writer.writerow({
                "dimension": dim,
                "backend": backend,
                "slab_depth": depth,
                "repeats": len(rs),
                "median_wall_seconds": med("wall_seconds"),
                "median_max_rss_mib": med("max_rss_mib"),
                "median_user_seconds": med("user_seconds"),
                "median_system_seconds": med("system_seconds"),
                "median_key_bytes": med("key_bytes"),
                "median_zero_persistence_pairs_elided": med("zero_persistence_pairs_elided"),
                "median_union_attempts": med("union_attempts"),
                "median_successful_unions": med("successful_unions"),
                "median_scalar_order_seconds": med("scalar_order_seconds"),
                "median_local_sweep_seconds": med("local_sweep_seconds"),
            })

    print(f"wrote {args.output} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
