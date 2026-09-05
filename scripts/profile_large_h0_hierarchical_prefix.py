#!/usr/bin/env python3
"""Profile v1.22 hierarchical H0 on real F32 prefixes without copying TIFF data.

The input prefix is staged as symlinks, so the profiler does not duplicate the
large image stack.  Each run gets a dedicated BETTI_TEMP_DIR subtree that is
sampled while the process runs to estimate peak temporary-run storage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
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
    p.add_argument("input", type=Path, help="directory containing the real F32 TIFF slices")
    p.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    p.add_argument("--temp-dir", type=Path, required=True,
                   help="large scratch filesystem; a dedicated subtree is created here")
    p.add_argument("--slice-counts", type=int, nargs="+", default=(64, 128))
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8,))
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--attach-pruning", choices=("off", "elder-dominated"), default="elder-dominated",
                   help="hierarchical attach propagation strategy")
    p.add_argument("--sample-seconds", type=float, default=2.0,
                   help="temporary-storage sampling interval")
    p.add_argument("--abort-free-gib", type=float, default=10.0,
                   help="terminate a run if scratch free space falls below this value; 0 disables")
    p.add_argument("--output", type=Path, default=Path("t1_s26_h0_hierarchical_prefix.csv"))
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
        target = dst / f"slice_{i:06d}{src.suffix.lower()}"
        target.symlink_to(src.resolve())
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
                    self.abort_reason = (
                        f"scratch free space fell below safety floor: {free / 2**30:.2f} GiB"
                    )
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
    out = {}
    for token in line[len(prefix):].strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def file_digest_and_rows(path: Path):
    """Order-sensitive checksum plus row count; intended as a run identity, not a canonical multiset hash."""
    dig = hashlib.sha256()
    rows = 0
    with path.open("rb") as h:
        header = h.readline()
        if header.strip() != b"birth,death":
            raise RuntimeError(f"bad persistence header in {path}")
        for line in h:
            if line.strip():
                rows += 1
                dig.update(line)
    return dig.hexdigest(), rows, path.stat().st_size


def require_profile(row, log):
    profile = kv_line(log, "PROFILE_H0_HIER_STREAM ")
    final = kv_line(log, "PROFILE_H0_HIER_STREAM_FINAL ")
    config = kv_line(log, "PROFILE_CONFIG scalar_h0_hierarchical_stream ")
    if not profile:
        raise RuntimeError("missing PROFILE_H0_HIER_STREAM")
    if not final:
        raise RuntimeError("missing PROFILE_H0_HIER_STREAM_FINAL")
    for k, v in profile.items():
        row[f"hier_{k}"] = v
    for k, v in final.items():
        row[f"final_{k}"] = v
    for k, v in config.items():
        row[f"config_{k}"] = v

    if profile.get("disk_key_bytes") != "4":
        raise RuntimeError(f"expected 4-byte disk keys, got {profile.get('disk_key_bytes')}")
    if profile.get("root_materialized") not in {"false", "False"}:
        raise RuntimeError("v1.22 unexpectedly materialized the root summary")
    if int(profile.get("root_attach_bytes", "-1")) != 0:
        raise RuntimeError("v1.22 wrote root attach bytes")
    if int(profile.get("root_interface_bytes", "-1")) != 0:
        raise RuntimeError("v1.22 wrote root interface bytes")
    if int(profile.get("final_interface_nodes", "-1")) != 0:
        raise RuntimeError("v1.22 retained final interface nodes")


def run_once(a, fixture: Path, width: int, height: int, slices: int, slab: int, repeat: int, root: Path):
    tag = f"z{slices}_d{slab}_r{repeat}"
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = a.temp_dir.resolve() / f"betti_prefix_{os.getpid()}_{tag}"
    scratch.mkdir(parents=True, exist_ok=False)
    out = run_dir / "intervals.csv"
    timing = run_dir / "time.txt"
    log_path = run_dir / "run.log"

    cmd = [
        "/usr/bin/time", "-v", "-o", str(timing), str(a.binary.resolve()),
        str(fixture.resolve()), str(slab), str(a.foreground_connectivity),
        "h0-scalar-hierarchical-stream", str(out),
        "--f32-key-mode", "native32",
        "--h0-birth-buffer", "reuse-input",
        "--h0-event-storage", "direct",
        "--event-order", "verify",
        "--global-h0-uf-layout", "packed",
        "--h0-hier-attach-pruning", a.attach_pruning,
    ]
    env = os.environ.copy()
    env["BETTI_TEMP_DIR"] = str(scratch)
    abort_free_bytes = int(max(0.0, a.abort_free_gib) * 2**30)
    mon = ScratchMonitor(scratch, a.sample_seconds, abort_free_bytes)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logh:
        proc = subprocess.Popen(
            cmd, cwd=run_dir, env=env, text=True,
            stdout=logh, stderr=subprocess.STDOUT,
            start_new_session=True,
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
        raise RuntimeError(f"{tag} failed with exit status {rc}:\n{log[-6000:]}")
    if "source pixel type: F32" not in log:
        raise RuntimeError(f"{tag}: binary did not report F32 input")

    digest, intervals, output_bytes = file_digest_and_rows(out)
    row = {
        "slices": slices,
        "slab_depth": slab,
        "repeat": repeat,
        "width": width,
        "height": height,
        "voxels": width * height * slices,
        "wall_seconds": wall,
        "interval_count": intervals,
        "output_sha256_ordered": digest,
        "output_bytes": output_bytes,
        "peak_scratch_bytes": mon.peak_bytes,
        "peak_scratch_files": mon.peak_files,
        "min_scratch_free_bytes": mon.min_free_bytes or 0,
        "requested_attach_pruning": a.attach_pruning,
    }
    row.update(parse_time(timing))
    require_profile(row, log)
    if row.get("config_h0_hier_attach_pruning") != a.attach_pruning:
        raise RuntimeError(
            f"{tag}: expected h0_hier_attach_pruning={a.attach_pruning}, "
            f"got {row.get('config_h0_hier_attach_pruning')}"
        )
    if row.get("max_rss_kb"):
        row["max_rss_mib"] = row["max_rss_kb"] / 1024.0
        row["max_rss_gib"] = row["max_rss_kb"] / 1024.0 / 1024.0
    row["peak_scratch_gib"] = mon.peak_bytes / 2**30
    row["throughput_mvox_s"] = (width * height * slices) / wall / 1e6
    row["wall_seconds_per_slice"] = wall / slices
    row["peak_scratch_bytes_per_voxel"] = mon.peak_bytes / (width * height * slices)

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

    rows = []
    with tempfile.TemporaryDirectory(prefix="betti_prefix_links_") as td:
        staging_root = Path(td)
        fixtures = {}
        for slices in sorted(set(a.slice_counts)):
            fixtures[slices] = stage_symlink_prefix(files, slices, staging_root / f"z{slices}")
        for slices in a.slice_counts:
            for slab in a.slab_depths:
                if slab > slices:
                    raise SystemExit(f"slab depth {slab} exceeds prefix length {slices}")
                for repeat in range(1, a.repeats + 1):
                    print(f"RUN slices={slices} d={slab} repeat={repeat}", flush=True)
                    row = run_once(a, fixtures[slices], width, height, slices, slab, repeat, staging_root / "runs")
                    rows.append(row)
                    print(
                        f"  wall={row['wall_seconds']:.1f}s rss={row.get('max_rss_gib', float('nan')):.2f} GiB "
                        f"scratch_peak={row['peak_scratch_gib']:.2f} GiB "
                        f"throughput={row['throughput_mvox_s']:.2f} Mvox/s",
                        flush=True,
                    )
                    write_csv(a.output, rows)

    print(f"PASS: completed {len(rows)} real-prefix hierarchical H0 profiles")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
