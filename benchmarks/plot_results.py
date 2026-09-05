#!/usr/bin/env python3
"""Generate reviewer-facing scaling plots from historical or fresh benchmark CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def series_label(row_name: str) -> str:
    return row_name.replace("-scalar-hierarchical-stream", " hierarchical").replace("-scalar-stream", " flat")


def runtime_vs_voxels(df: pd.DataFrame, out: Path) -> None:
    if "voxels" not in df or "runtime_s" not in df and "wall_seconds" not in df:
        return
    runtime_col = "runtime_s" if "runtime_s" in df else "wall_seconds"
    if "dataset" in df:
        subset = df[df["dataset"].astype(str).str.startswith("T1_S26 prefix")]
        groups = [("H0 hierarchical", subset)] if not subset.empty else []
    else:
        label_col = "mode" if "mode" in df else None
        groups = list(df.groupby(label_col)) if label_col else [("benchmark", df)]
    if not groups or max((g["voxels"].nunique() for _, g in groups), default=0) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, group in groups:
        group = group.groupby("voxels", as_index=False)[runtime_col].median().sort_values("voxels")
        ax.plot(group["voxels"] / 1e9, group[runtime_col] / 60.0, marker="o", label=series_label(str(label)))
    ax.set_xlabel("Voxels (billions)")
    ax.set_ylabel("Runtime (minutes)")
    ax.set_title("Runtime scaling")
    ax.grid(True, alpha=0.25)
    if len(groups) > 1:
        ax.legend()
    save(fig, out)


def memory_vs_voxels(df: pd.DataFrame, out: Path) -> None:
    if "voxels" not in df or "peak_ram_mib" not in df and "peak_rss_mib" not in df:
        return
    mem_col = "peak_ram_mib" if "peak_ram_mib" in df else "peak_rss_mib"
    if "dataset" in df:
        subset = df[df["dataset"].astype(str).str.startswith("T1_S26 prefix")]
        groups = [("H0 hierarchical", subset)] if not subset.empty else []
    else:
        label_col = "mode" if "mode" in df else None
        groups = list(df.groupby(label_col)) if label_col else [("benchmark", df)]
    if not groups or max((g["voxels"].nunique() for _, g in groups), default=0) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, group in groups:
        group = group.groupby("voxels", as_index=False)[mem_col].median().sort_values("voxels")
        ax.plot(group["voxels"] / 1e9, group[mem_col] / 1024.0, marker="o", label=series_label(str(label)))
    ax.set_xlabel("Voxels (billions)")
    ax.set_ylabel("Peak RSS (GiB)")
    ax.set_title("Peak-memory scaling")
    ax.grid(True, alpha=0.25)
    if len(groups) > 1:
        ax.legend()
    save(fig, out)


def runtime_vs_slab_depth(df: pd.DataFrame, out: Path) -> None:
    if "slab_depth" not in df:
        return
    runtime_col = "runtime_s" if "runtime_s" in df else "wall_seconds" if "wall_seconds" in df else None
    if runtime_col is None:
        return
    if "dataset" in df:
        subset = df[df["dataset"] == "CX09T1"]
        label_col = "homology"
    else:
        subset = df
        label_col = "mode" if "mode" in df else None
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = subset.groupby(label_col) if label_col else [("benchmark", subset)]
    group_count = 0
    for label, group in groups:
        group = group.groupby("slab_depth", as_index=False)[runtime_col].median().sort_values("slab_depth")
        ax.plot(group["slab_depth"], group[runtime_col], marker="o", label=series_label(str(label)))
        group_count += 1
    ax.set_xlabel("Slab depth (slices)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime versus slab depth")
    ax.set_xticks(sorted(subset["slab_depth"].unique()))
    ax.grid(True, alpha=0.25)
    if group_count > 1:
        ax.legend()
    save(fig, out)


def speedup_vs_threads(df: pd.DataFrame, out: Path) -> None:
    if "threads" not in df:
        return
    runtime_col = "runtime_s" if "runtime_s" in df else "wall_seconds" if "wall_seconds" in df else None
    if runtime_col is None:
        return
    numeric = df.copy()
    numeric["threads_numeric"] = pd.to_numeric(numeric["threads"], errors="coerce")
    numeric = numeric.dropna(subset=["threads_numeric"])
    if numeric["threads_numeric"].nunique() < 2:
        return
    label_col = "mode" if "mode" in numeric else None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = numeric.groupby(label_col) if label_col else [("benchmark", numeric)]
    group_count = 0
    for label, group in groups:
        grouped = group.groupby("threads_numeric", as_index=False)[runtime_col].median().sort_values("threads_numeric")
        base = grouped.iloc[0][runtime_col]
        grouped["speedup"] = base / grouped[runtime_col]
        ax.plot(grouped["threads_numeric"], grouped["speedup"], marker="o", label=series_label(str(label)))
        group_count += 1
    ax.set_xlabel("Rayon threads")
    ax.set_ylabel("Speedup versus smallest thread count")
    ax.set_title("Thread scaling")
    ax.grid(True, alpha=0.25)
    if group_count > 1:
        ax.legend()
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/plots"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    runtime_vs_voxels(df, args.output_dir / "runtime_vs_voxels.svg")
    memory_vs_voxels(df, args.output_dir / "peak_memory_vs_voxels.svg")
    runtime_vs_slab_depth(df, args.output_dir / "runtime_vs_slab_depth.svg")
    speedup_vs_threads(df, args.output_dir / "speedup_vs_threads.svg")


if __name__ == "__main__":
    main()
