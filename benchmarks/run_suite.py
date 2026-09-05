#!/usr/bin/env python3
"""Run reproducible Rust benchmark sweeps with wall/RSS/I/O accounting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable

from PIL import Image

TIME_FIELDS = {
    "user_seconds": re.compile(r"^\s*User time \(seconds\):\s*(.+)$"),
    "system_seconds": re.compile(r"^\s*System time \(seconds\):\s*(.+)$"),
    "peak_rss_kib": re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$"),
    "major_faults": re.compile(r"^\s*Major \(requiring I/O\) page faults:\s*(\d+)\s*$"),
    "minor_faults": re.compile(r"^\s*Minor \(reclaiming a frame\) page faults:\s*(\d+)\s*$"),
    "fs_inputs": re.compile(r"^\s*File system inputs:\s*(\d+)\s*$"),
    "fs_outputs": re.compile(r"^\s*File system outputs:\s*(\d+)\s*$"),
}
ELAPSED = re.compile(r"^\s*Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)\s*$")


def wall_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(value)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def stack_shape(root: Path) -> tuple[int, int, int]:
    paths = sorted([*root.glob("*.tif"), *root.glob("*.tiff")], key=natural_key)
    if not paths:
        raise SystemExit(f"no TIFF files in {root}")
    with Image.open(paths[0]) as im:
        width, height = im.size
    return width, height, len(paths)


def preset_args(mode: str) -> list[str]:
    if mode == "h0-scalar-hierarchical-stream":
        return [
            "--f32-key-mode", "native32",
            "--h0-birth-buffer", "reuse-input",
            "--h0-event-storage", "direct",
            "--global-h0-uf-layout", "packed",
            "--h0-hier-attach-pruning", "elder-dominated",
        ]
    if mode == "h2-scalar-hierarchical-stream":
        return [
            "--f32-key-mode", "native32",
            "--local-h2-birth-state", "compact",
            "--global-h2-birth-state", "compact",
            "--global-h2-uf-layout", "packed",
            "--h2-hier-cross-storage", "direct",
            "--h2-hier-outside-structural-pruning", "outside-dominated",
        ]
    return []


def parse_time(log: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for line in log.splitlines():
        m = ELAPSED.match(line)
        if m:
            result["wall_seconds"] = wall_seconds(m.group(1))
        for key, pattern in TIME_FIELDS.items():
            match = pattern.match(line)
            if match:
                result[key] = float(match.group(1)) if "seconds" in key else int(match.group(1))
    if "wall_seconds" not in result:
        raise RuntimeError("could not parse GNU time elapsed line")
    return result


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stack", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    parser.add_argument("--modes", nargs="+", default=["h0-scalar-stream", "h2-scalar-stream"])
    parser.add_argument("--slab-depths", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    if not Path("/usr/bin/time").exists():
        raise SystemExit("benchmark runner currently requires GNU /usr/bin/time")
    if not args.binary.exists():
        raise SystemExit(f"binary not found: {args.binary}")
    width, height, depth = stack_shape(args.stack)
    binary_hash = hash_file(args.binary)
    rows: list[dict] = []

    for repeat in range(1, args.repeats + 1):
        for threads in args.threads:
            for slab in args.slab_depths:
                for mode in args.modes:
                    with tempfile.TemporaryDirectory(prefix="stream_betti_bench_") as tmp:
                        output = Path(tmp) / "output.csv"
                        command = [
                            str(args.binary), str(args.stack), str(slab),
                            str(args.foreground_connectivity), mode, str(output),
                            *preset_args(mode), *args.extra_arg,
                        ]
                        env = os.environ.copy()
                        env["RAYON_NUM_THREADS"] = str(threads)
                        if args.temp_dir:
                            args.temp_dir.mkdir(parents=True, exist_ok=True)
                            env["BETTI_TEMP_DIR"] = str(args.temp_dir)
                        wrapped = ["/usr/bin/time", "-v", *command]
                        cp = subprocess.run(wrapped, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        if cp.returncode != 0:
                            raise RuntimeError(f"benchmark command failed ({cp.returncode}): {' '.join(command)}\n{cp.stdout}")
                        metrics = parse_time(cp.stdout)
                        output_rows = max(0, sum(1 for _ in output.open()) - 1) if output.exists() else 0
                        row = {
                            "width": width, "height": height, "depth": depth,
                            "voxels": width * height * depth,
                            "mode": mode, "slab_depth": slab,
                            "foreground_connectivity": args.foreground_connectivity,
                            "threads": threads, "repeat": repeat,
                            **metrics,
                            "peak_rss_mib": metrics.get("peak_rss_kib", 0) / 1024.0,
                            "output_rows": output_rows,
                            "binary_sha256": binary_hash,
                            "command": " ".join(command),
                        }
                        rows.append(row)
                        append_csv(args.output, rows)
                        print(
                            f"{mode} d={slab} threads={threads} repeat={repeat}: "
                            f"wall={row['wall_seconds']:.3f}s rss={row['peak_rss_mib']:.1f} MiB"
                        )


if __name__ == "__main__":
    main()
