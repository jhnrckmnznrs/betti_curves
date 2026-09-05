#!/usr/bin/env python3
"""Profile optimized hierarchical H2 on real F32 TIFF prefixes.

The input prefix is staged as symlinks; TIFF data are never copied. Each run
uses a dedicated BETTI_TEMP_DIR subtree that is sampled while the process is
running so peak temporary storage can be estimated. The intended production
candidate is native32 + compact births + packed UF + direct cross +
outside-dominated structural pruning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

TIFF_SUFFIXES = {".tif", ".tiff"}
TIME_FIELDS = {
    "User time (seconds)": "user_seconds",
    "System time (seconds)": "system_seconds",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
}


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="directory containing real F32 TIFF slices")
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--temp-dir", type=Path, required=True,
                   help="large scratch filesystem; one private subtree is created per run")
    p.add_argument("--slice-counts", type=int, nargs="+", default=(64, 128))
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cross-storage", choices=("disk", "direct"), default="direct")
    p.add_argument("--outside-structural-pruning", choices=("off", "outside-dominated"),
                   default="outside-dominated")
    p.add_argument("--sample-seconds", type=float, default=2.0)
    p.add_argument("--abort-free-gib", type=float, default=10.0,
                   help="terminate if scratch free space falls below this floor; 0 disables")
    p.add_argument("--output", type=Path, default=Path("t1_s26_h2_hierarchical_prefix.csv"))
    p.add_argument("--keep-logs", action="store_true")
    p.add_argument("--keep-outputs", action="store_true")
    return p.parse_args()


def discover_slices(src: Path):
    files = sorted(
        (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES),
        key=natural_key,
    )
    if not files:
        raise SystemExit(f"no TIFF slices found in {src}")
    return files


def inspect_first_slice(path: Path):
    try:
        import tifffile
    except ImportError as exc:
        raise SystemExit("this profiler requires tifffile (python3 -m pip install tifffile)") from exc
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        shape = tuple(int(x) for x in page.shape)
        dtype = str(page.dtype)
    if len(shape) != 2:
        raise SystemExit(f"expected 2D TIFF slices, got {shape} in {path}")
    if dtype not in {"float32", "<f4", ">f4"}:
        raise SystemExit(f"real-prefix profiler requires F32 TIFF input; got {dtype} in {path}")
    return shape, dtype


def stage_symlink_prefix(files, count: int, dst: Path):
    if count <= 0:
        raise ValueError("slice count must be positive")
    if count > len(files):
        raise SystemExit(f"requested {count} slices but only {len(files)} are available")
    dst.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files[:count]):
        (dst / f"slice_{i:06d}{src.suffix.lower()}").symlink_to(src.resolve())
    return dst


def directory_size(path: Path):
    total = 0
    files = 0
    if not path.exists():
        return 0, 0
    for root, _, names in os.walk(path):
        for name in names:
            p = Path(root) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            total += st.st_size
            files += 1
    return total, files


class ScratchMonitor(threading.Thread):
    def __init__(self, path: Path, interval: float, abort_free_bytes: int):
        super().__init__(daemon=True)
        self.path = path
        self.interval = max(0.25, interval)
        self.abort_free_bytes = abort_free_bytes
        self.stop_event = threading.Event()
        self.peak_bytes = 0
        self.peak_files = 0
        self.min_free_bytes = None
        self.abort_reason = None
        self.process = None

    def run(self):
        while not self.stop_event.is_set():
            size, files = directory_size(self.path)
            self.peak_bytes = max(self.peak_bytes, size)
            self.peak_files = max(self.peak_files, files)
            try:
                free = shutil.disk_usage(self.path).free
                self.min_free_bytes = free if self.min_free_bytes is None else min(self.min_free_bytes, free)
                if self.abort_free_bytes and free < self.abort_free_bytes and self.process is not None:
                    self.abort_reason = f"scratch free space fell below safety floor: {free / 2**30:.2f} GiB"
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    return
            except FileNotFoundError:
                pass
            self.stop_event.wait(self.interval)
        size, files = directory_size(self.path)
        self.peak_bytes = max(self.peak_bytes, size)
        self.peak_files = max(self.peak_files, files)

    def stop(self):
        self.stop_event.set()


def parse_time(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.strip().split(":", 1)
        if k in TIME_FIELDS:
            try:
                out[TIME_FIELDS[k]] = float(v.strip())
            except ValueError:
                pass
    return out


def kv_line(log: str, prefix: str):
    line = next((x.strip() for x in log.splitlines() if x.startswith(prefix)), None)
    if line is None:
        return {}
    return parse_kv_tail(line[len(prefix):])


def kv_lines(log: str, prefix: str):
    return [parse_kv_tail(x.strip()[len(prefix):])
            for x in log.splitlines() if x.startswith(prefix)]


def parse_kv_tail(text: str):
    out = {}
    for token in text.strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def int_or_zero(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def output_fingerprints(path: Path):
    """Return ordered SHA256 plus order-independent diagnostic fingerprints.

    The commutative fingerprints are not a formal replacement for exact Counter
    comparison; they are intended to detect accidental d8/d16 divergence on huge
    prefix outputs without sorting millions of CSV rows.
    """
    ordered = hashlib.sha256()
    sum256 = 0
    xor256 = 0
    rows = 0
    mod = 1 << 256
    with path.open("rb") as h:
        header = h.readline()
        if header.strip() != b"birth,death":
            raise RuntimeError(f"bad persistence header in {path}")
        for raw in h:
            line = raw.strip()
            if not line:
                continue
            rows += 1
            ordered.update(raw)
            v = int.from_bytes(hashlib.sha256(line).digest(), "big")
            sum256 = (sum256 + v) % mod
            xor256 ^= v
    return ordered.hexdigest(), f"{sum256:064x}", f"{xor256:064x}", rows, path.stat().st_size


def require_profile(row, log, requested_cross, requested_outside):
    profile = kv_line(log, "PROFILE_H2_HIER_STREAM ")
    final = kv_line(log, "PROFILE_H2_HIER_STREAM_FINAL ")
    config = kv_line(log, "PROFILE_CONFIG scalar_h2_hierarchical_stream ")
    if not profile:
        raise RuntimeError("missing PROFILE_H2_HIER_STREAM")
    if not final:
        raise RuntimeError("missing PROFILE_H2_HIER_STREAM_FINAL")
    if not config:
        raise RuntimeError("missing PROFILE_CONFIG scalar_h2_hierarchical_stream")
    for k, v in profile.items():
        row[f"hier_{k}"] = v
    for k, v in final.items():
        row[f"final_{k}"] = v
    for k, v in config.items():
        row[f"config_{k}"] = v

    checks = {
        "disk_key_bytes": "4",
        "global_h2_birth_state": "compact",
        "global_h2_uf_layout": "packed",
        "root_materialized": "false",
        "final_interface_nodes": "0",
    }
    for k, expected in checks.items():
        got = profile.get(k)
        if got is None and k in {"global_h2_birth_state", "global_h2_uf_layout"}:
            got = config.get(k)
        if str(got).lower() != expected:
            raise RuntimeError(f"expected {k}={expected}, got {got}")
    for k in ("root_attach_bytes", "root_outside_bytes", "root_interface_bytes"):
        if int_or_zero(profile.get(k)) != 0:
            raise RuntimeError(f"hierarchical H2 unexpectedly wrote {k}={profile.get(k)}")
    if config.get("h2_hier_cross_storage") != requested_cross:
        raise RuntimeError(
            f"expected h2_hier_cross_storage={requested_cross}, got {config.get('h2_hier_cross_storage')}"
        )
    if config.get("h2_hier_outside_structural_pruning") != requested_outside:
        raise RuntimeError(
            "expected h2_hier_outside_structural_pruning="
            f"{requested_outside}, got {config.get('h2_hier_outside_structural_pruning')}"
        )

    leaves = kv_lines(log, "PROFILE_H2_HIER_STREAM_LEAF ")
    combines = kv_lines(log, "PROFILE_H2_HIER_STREAM_COMBINE ")
    row["leaf_profile_records"] = len(leaves)
    row["combine_profile_records"] = len(combines)
    for field in ("attach_bytes", "outside_bytes", "interface_bytes"):
        row[f"leaf_{field}_total"] = sum(int_or_zero(x.get(field)) for x in leaves)
    for field in ("parent_attach_bytes", "parent_outside_bytes", "parent_interface_bytes"):
        row[f"{field}_total"] = sum(int_or_zero(x.get(field)) for x in combines)
    for field in ("cross_run_bytes_avoided", "outside_structural_elided",
                  "attach_finalized_early", "attach_propagated", "outside_propagated"):
        row[f"combine_{field}_total"] = sum(int_or_zero(x.get(field)) for x in combines)
    row["hier_summary_bytes_aggregate"] = (
        row["leaf_attach_bytes_total"] + row["leaf_outside_bytes_total"] + row["leaf_interface_bytes_total"]
        + row["parent_attach_bytes_total"] + row["parent_outside_bytes_total"] + row["parent_interface_bytes_total"]
    )


def run_once(a, fixture: Path, width: int, height: int, slices: int, slab: int, repeat: int, root: Path):
    tag = f"z{slices}_d{slab}_r{repeat}_{a.cross_storage}_{a.outside_structural_pruning}"
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = a.temp_dir.resolve() / f"betti_h2_prefix_{os.getpid()}_{tag}"
    scratch.mkdir(parents=True, exist_ok=False)
    out = run_dir / "intervals.csv"
    timing = run_dir / "time.txt"
    log_path = run_dir / "run.log"

    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing), str(a.binary.resolve()),
        str(fixture.resolve()), str(slab), str(a.foreground_connectivity),
        "h2-scalar-hierarchical-stream", str(out),
        "--f32-key-mode", "native32",
        "--local-h2-birth-state", "compact",
        "--global-h2-birth-state", "compact",
        "--global-h2-uf-layout", "packed",
        "--h2-hier-cross-storage", a.cross_storage,
        "--h2-hier-outside-structural-pruning", a.outside_structural_pruning,
    ]
    env = os.environ.copy()
    env["BETTI_TEMP_DIR"] = str(scratch)
    abort_free_bytes = int(max(0.0, a.abort_free_gib) * 2**30)
    mon = ScratchMonitor(scratch, a.sample_seconds, abort_free_bytes)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logh:
        proc = subprocess.Popen(
            cmd, cwd=run_dir, env=env, text=True,
            stdout=logh, stderr=subprocess.STDOUT, start_new_session=True,
        )
        mon.process = proc
        mon.start()
        rc = proc.wait()
    wall = time.perf_counter() - t0
    mon.stop(); mon.join()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if mon.abort_reason:
        raise RuntimeError(f"{tag}: {mon.abort_reason}")
    if rc:
        raise RuntimeError(f"{tag} failed with exit status {rc}:\n{log[-8000:]}")
    if "source pixel type: F32" not in log:
        raise RuntimeError(f"{tag}: binary did not report F32 input")

    ordered, sum256, xor256, intervals, output_bytes = output_fingerprints(out)
    row = {
        "slices": slices,
        "slab_depth": slab,
        "repeat": repeat,
        "width": width,
        "height": height,
        "voxels": width * height * slices,
        "wall_seconds": wall,
        "interval_count": intervals,
        "output_sha256_ordered": ordered,
        "output_multiset_sum256": sum256,
        "output_multiset_xor256": xor256,
        "output_bytes": output_bytes,
        "peak_scratch_bytes": mon.peak_bytes,
        "peak_scratch_files": mon.peak_files,
        "min_scratch_free_bytes": mon.min_free_bytes or 0,
        "requested_cross_storage": a.cross_storage,
        "requested_outside_structural_pruning": a.outside_structural_pruning,
    }
    row.update(parse_time(timing))
    require_profile(row, log, a.cross_storage, a.outside_structural_pruning)
    if row.get("max_rss_kb"):
        row["max_rss_mib"] = row["max_rss_kb"] / 1024.0
        row["max_rss_gib"] = row["max_rss_kb"] / 1024.0 / 1024.0
    row["peak_scratch_gib"] = mon.peak_bytes / 2**30
    row["throughput_mvox_s"] = (width * height * slices) / wall / 1e6
    row["wall_seconds_per_slice"] = wall / slices
    row["peak_scratch_bytes_per_voxel"] = mon.peak_bytes / (width * height * slices)
    row["hier_summary_mib_aggregate"] = row["hier_summary_bytes_aggregate"] / 2**20

    shutil.rmtree(scratch, ignore_errors=True)
    if not a.keep_outputs:
        out.unlink(missing_ok=True)
    if not a.keep_logs:
        timing.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        try:
            run_dir.rmdir()
        except OSError:
            pass
    return row


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_summary(path: Path, rows):
    groups = {}
    for row in rows:
        key = (row["slices"], row["slab_depth"], row["requested_cross_storage"],
               row["requested_outside_structural_pruning"])
        groups.setdefault(key, []).append(row)
    out = []
    median_fields = [
        "wall_seconds", "user_seconds", "system_seconds", "max_rss_mib",
        "peak_scratch_gib", "throughput_mvox_s", "peak_scratch_bytes_per_voxel",
        "hier_summary_mib_aggregate", "fs_inputs", "fs_outputs",
    ]
    for key, vals in sorted(groups.items()):
        item = {
            "slices": key[0], "slab_depth": key[1], "cross_storage": key[2],
            "outside_structural_pruning": key[3], "repeats": len(vals),
            "interval_count": vals[0]["interval_count"],
            "max_pair_nodes": int_or_zero(vals[0].get("hier_max_pair_nodes")),
            "outside_structural_elided": int_or_zero(vals[0].get("hier_outside_structural_elided")),
            "attach_propagated": int_or_zero(vals[0].get("hier_attach_propagated")),
            "outside_propagated": int_or_zero(vals[0].get("hier_outside_propagated")),
        }
        for field in median_fields:
            nums = [float(v[field]) for v in vals if v.get(field) not in (None, "")]
            if nums:
                item[f"median_{field}"] = statistics.median(nums)
        out.append(item)
    write_csv(path, out)


def main():
    a = parse_args()
    if not a.binary.is_file():
        raise SystemExit(f"binary not found: {a.binary}")
    a.temp_dir.mkdir(parents=True, exist_ok=True)
    files = discover_slices(a.input)
    (height, width), dtype = inspect_first_slice(files[0])
    print(f"input: {len(files)} slices, {width} x {height}, dtype={dtype}")
    free_gib = shutil.disk_usage(a.temp_dir).free / 2**30
    print(f"scratch: {a.temp_dir} free={free_gib:.1f} GiB")
    print(f"H2: cross={a.cross_storage} outside_structural={a.outside_structural_pruning}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="betti_h2_prefix_links_") as td:
        staging_root = Path(td)
        fixtures = {}
        for slices in sorted(set(a.slice_counts)):
            fixtures[slices] = stage_symlink_prefix(files, slices, staging_root / f"z{slices}")
        for slices in a.slice_counts:
            for slab in a.slab_depths:
                if slab > slices:
                    raise SystemExit(f"slab depth {slab} exceeds prefix length {slices}")
                for repeat in range(1, a.repeats + 1):
                    print(f"RUN H2 slices={slices} d={slab} repeat={repeat}", flush=True)
                    row = run_once(a, fixtures[slices], width, height, slices, slab, repeat,
                                   staging_root / "runs")
                    rows.append(row)
                    print(
                        f"  wall={row['wall_seconds']:.1f}s rss={row.get('max_rss_gib', float('nan')):.2f} GiB "
                        f"scratch_peak={row['peak_scratch_gib']:.2f} GiB "
                        f"throughput={row['throughput_mvox_s']:.2f} Mvox/s "
                        f"summary={row['hier_summary_mib_aggregate']:.1f} MiB",
                        flush=True,
                    )
                    write_csv(a.output, rows)
                    write_summary(a.output.with_name(a.output.stem + "_summary.csv"), rows)

    # d8/d16 should agree on count and commutative persistence fingerprints for each prefix.
    by_slices = {}
    for row in rows:
        by_slices.setdefault(row["slices"], []).append(row)
    for slices, vals in by_slices.items():
        signatures = {(v["interval_count"], v["output_multiset_sum256"], v["output_multiset_xor256"])
                      for v in vals}
        if len(signatures) != 1:
            print(
                f"WARNING: H2 persistence diagnostic fingerprints differ across slab depths/repeats "
                f"for {slices} slices; preserve outputs and run an exact external multiset comparison.",
                file=sys.stderr,
            )

    print(f"PASS: completed {len(rows)} real-prefix hierarchical H2 profiles")
    print(f"wrote {a.output}")
    print(f"wrote {a.output.with_name(a.output.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
