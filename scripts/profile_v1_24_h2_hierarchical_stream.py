#!/usr/bin/env python3
"""Profile flat native32/packed H2 against v1.24 hierarchical outside-aware H2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median

import sys
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "validation")))
from f32_fixture import stage_f32_subset

TIME_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(?P<rss>\d+)")
USER_RE = re.compile(r"User time \(seconds\):\s*(?P<v>[0-9.]+)")
SYS_RE = re.compile(r"System time \(seconds\):\s*(?P<v>[0-9.]+)")
WALL_RE = re.compile(
    r"^\s*Elapsed \(wall clock\) time .*?\):\s*"
    r"(?P<v>\d+(?::\d+){0,2}(?:\.\d+)?)\s*$",
    re.MULTILINE,
)
HIER_RE = re.compile(
    r"PROFILE_H2_HIER_STREAM\s+leaf_slabs=(?P<leaf>\d+)\s+combines=(?P<combines>\d+)\s+"
    r"max_live_summaries=(?P<live>\d+)\s+max_pair_nodes=(?P<pair>\d+)\s+"
    r"max_pair_state_bytes=(?P<state>\d+)\s+final_interface_nodes=(?P<final>\d+)\s+"
    r"finalized_pair_bytes=(?P<pair_bytes>\d+)\s+root_attach_bytes=(?P<root_attach>\d+)\s+"
    r"root_outside_bytes=(?P<root_outside>\d+)\s+root_interface_bytes=(?P<root_interface>\d+)\s+"
    r"root_materialized=(?P<root>\S+)\s+disk_key_bytes=(?P<key>\d+).*?"
    r"leaf_attach_events=(?P<leaf_attach>\d+)\s+leaf_outside_events=(?P<leaf_outside>\d+)\s+"
    r"attach_finalized_early=(?P<finalized>\d+)\s+attach_propagated=(?P<propagated>\d+)\s+"
    r"outside_propagated=(?P<outside_prop>\d+)"
)
FLAT_STORAGE_RE = re.compile(
    r"PROFILE_H2_STORAGE scalar_h2_stream pipeline=(?P<pipeline>\S+) disk_key_bytes=(?P<key>\d+) "
    r"interface_nodes=(?P<nodes>\d+) interface_birth_bytes=(?P<birth_bytes>\d+) "
    r"local_pair_bytes=(?P<pair_bytes>\d+) attach_bytes=(?P<attach>\d+) outside_bytes=(?P<outside>\d+) "
    r"interface_bytes=(?P<interface>\d+) cross_bytes=(?P<cross>\d+) total_run_bytes=(?P<total>\d+) "
    r"global_h2_birth_state=(?P<birth>\S+) global_h2_uf_layout=(?P<layout>\S+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=Path(os.environ.get("BETTI_CURVES_BINARY", "target/release/betti_curves")))
    p.add_argument("--slice-limit", type=int, default=0)
    p.add_argument("--slab-depths", type=int, nargs="+", default=(8, 16, 32))
    p.add_argument("--foreground-connectivity", choices=(6, 26), type=int, default=26)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("cx09t1_v1_24_h2_hierarchical_stream.csv"))
    return p.parse_args()


def wall_seconds(text: str) -> float:
    """Convert GNU time's elapsed value to seconds.

    Accepted forms are plain seconds, m:ss[.ff], and h:mm:ss[.ff].
    """
    text = text.strip()
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"unrecognized elapsed-time value: {text!r}")


def wall_seconds_from_log(log: str) -> float:
    m = WALL_RE.search(log)
    if not m:
        elapsed_lines = [line for line in log.splitlines() if "Elapsed (wall clock) time" in line]
        raise RuntimeError(
            "could not parse GNU time elapsed-time line; found: "
            + repr(elapsed_lines[:3])
        )
    return wall_seconds(m.group("v"))


def persistence_counter(path: Path):
    with path.open(newline="", encoding="utf-8") as h:
        r = csv.reader(h)
        if next(r, None) != ["birth", "death"]:
            raise RuntimeError(f"bad header in {path}")
        return Counter(tuple(row) for row in r if row)


def canonical_digest(counter: Counter) -> str:
    h = hashlib.sha256()
    for (birth, death), count in sorted(counter.items()):
        h.update(birth.encode()); h.update(b","); h.update(death.encode()); h.update(b",")
        h.update(str(count).encode()); h.update(b"\n")
    return h.hexdigest()


def run_one(time_bin: str, binary: Path, inp: Path, work: Path, slab: int, fg: int, variant: str):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "intervals.csv"
    if variant == "flat":
        mode = "h2-scalar-stream"
        extra = ("--f32-key-mode", "native32", "--local-h2-birth-state", "compact",
                 "--global-h2-birth-state", "compact", "--global-h2-uf-layout", "packed")
    elif variant == "hierarchical":
        mode = "h2-scalar-hierarchical-stream"
        extra = ("--f32-key-mode", "native32", "--local-h2-birth-state", "compact",
                 "--global-h2-birth-state", "compact", "--global-h2-uf-layout", "packed")
    else:
        raise ValueError(variant)
    cmd = [time_bin, "-v", str(binary.resolve()), str(inp.resolve()), str(slab), str(fg), mode, str(out), *extra]
    cp = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{cp.stdout}")
    log = cp.stdout
    rss = int(TIME_RE.search(log).group("rss"))
    user = float(USER_RE.search(log).group("v"))
    system = float(SYS_RE.search(log).group("v"))
    wall = wall_seconds_from_log(log)
    counter = persistence_counter(out)
    row = dict(
        variant=variant, slab_depth=slab, wall_seconds=wall, user_seconds=user,
        system_seconds=system, peak_rss_kib=rss, interval_count=sum(counter.values()),
        canonical_sha256=canonical_digest(counter),
    )
    if variant == "hierarchical":
        m = HIER_RE.search(log)
        if not m:
            raise RuntimeError("missing PROFILE_H2_HIER_STREAM")
        for k in ("leaf", "combines", "live", "pair", "state", "final", "pair_bytes",
                  "root_attach", "root_outside", "root_interface", "key", "leaf_attach",
                  "leaf_outside", "finalized", "propagated", "outside_prop"):
            row[k] = int(m.group(k))
        row["root_materialized"] = m.group("root")
    else:
        m = FLAT_STORAGE_RE.search(log)
        if not m:
            raise RuntimeError("missing PROFILE_H2_STORAGE")
        row.update(
            flat_interface_nodes=int(m.group("nodes")), flat_interface_birth_bytes=int(m.group("birth_bytes")),
            flat_local_pair_bytes=int(m.group("pair_bytes")), flat_attach_bytes=int(m.group("attach")),
            flat_outside_bytes=int(m.group("outside")), flat_interface_bytes=int(m.group("interface")),
            flat_cross_bytes=int(m.group("cross")), flat_total_run_bytes=int(m.group("total")),
        )
    return row, counter


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    a = parse_args()
    if not a.binary.is_file(): raise SystemExit(f"binary not found: {a.binary}")
    time_bin = shutil.which("/usr/bin/time") or shutil.which("time")
    if not time_bin: raise SystemExit("/usr/bin/time is required")
    with tempfile.TemporaryDirectory(prefix="betti_v124_h2_profile_") as td:
        root = Path(td)
        f32 = stage_f32_subset(a.input, root / "f32_input", a.slice_limit)
        rows=[]
        for slab in a.slab_depths:
            reference = None
            # Alternate order across repeats to reduce cache-order bias.
            for rep in range(1, a.repeats+1):
                order = ("flat", "hierarchical") if rep % 2 else ("hierarchical", "flat")
                pair_results = {}
                for variant in order:
                    print(f"RUN d{slab} repeat={rep} {variant}", flush=True)
                    row, counter = run_one(time_bin, a.binary, f32, root/f"d{slab}_r{rep}_{variant}", slab,
                                           a.foreground_connectivity, variant)
                    row["repeat"] = rep
                    rows.append(row); pair_results[variant]=counter
                    write_csv(a.output, rows)
                if pair_results["flat"] != pair_results["hierarchical"]:
                    raise SystemExit(
                        f"FAIL d{slab} repeat={rep}: flat/hierarchy persistence differ\n"
                        f"missing={(pair_results['flat']-pair_results['hierarchical']).most_common(10)}\n"
                        f"extra={(pair_results['hierarchical']-pair_results['flat']).most_common(10)}"
                    )
                if reference is None: reference = pair_results["flat"]
                elif reference != pair_results["flat"]:
                    raise SystemExit(f"FAIL d{slab}: repeat persistence changed")

        write_csv(a.output, rows)
        summary=[]
        for slab in a.slab_depths:
            for variant in ("flat", "hierarchical"):
                rr=[r for r in rows if r["slab_depth"]==slab and r["variant"]==variant]
                summary.append(dict(
                    slab_depth=slab, variant=variant, repeats=len(rr),
                    median_wall_seconds=median(r["wall_seconds"] for r in rr),
                    median_peak_rss_mib=median(r["peak_rss_kib"] for r in rr)/1024,
                    median_user_seconds=median(r["user_seconds"] for r in rr),
                    median_system_seconds=median(r["system_seconds"] for r in rr),
                    interval_count=rr[0]["interval_count"], canonical_sha256=rr[0]["canonical_sha256"],
                    max_pair_nodes=rr[0].get("pair", ""),
                    max_pair_state_mib=(rr[0].get("state", 0) / 2**20) if variant=="hierarchical" else "",
                    attach_finalized_early=rr[0].get("finalized", ""),
                    attach_propagated=rr[0].get("propagated", ""),
                    flat_total_run_mib=(rr[0].get("flat_total_run_bytes", 0)/2**20) if variant=="flat" else "",
                ))
        summary_path=a.output.with_name(a.output.stem+"_summary.csv")
        write_csv(summary_path, summary)
        print(f"Wrote {a.output}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__": main()
