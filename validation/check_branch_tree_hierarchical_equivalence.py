#!/usr/bin/env python3
"""Compare hierarchical branch trees with plateau-canonical flat references."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


def read_nodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["node"]))
    return rows


def canonicalize(rows: list[dict[str, str]], dimension: str) -> list[dict[str, str]]:
    original = [dict(row) for row in rows]
    by_id = {int(row["node"]): row for row in original}
    out = [dict(row) for row in original]

    for row in out:
        parent_text = row["parent"]
        if not parent_text:
            continue
        threshold_key = "death_value" if dimension == "h0" else "birth_value"
        threshold = row[threshold_key]
        parent = int(parent_text)
        steps = 0
        while True:
            parent_row = by_id[parent]
            if parent_row[threshold_key] != threshold or not parent_row["parent"]:
                break
            parent = int(parent_row["parent"])
            steps += 1
            if steps > len(rows):
                raise RuntimeError(f"cycle while canonicalizing {dimension}")
        row["parent"] = str(parent)
    return out


def run_mode(
    binary: Path,
    stack: Path,
    slab_depth: int,
    foreground_connectivity: int,
    mode: str,
    workdir: Path,
) -> Path:
    command = [
        str(binary),
        str(stack),
        str(slab_depth),
        str(foreground_connectivity),
        mode,
    ]
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} failed with exit code {completed.returncode}\n{completed.stdout}"
        )

    names = {
        "branch-tree-h0": "h0_merge_tree_nodes.csv",
        "branch-tree-h0-hierarchical": "h0_branch_tree_hierarchical_nodes.csv",
        "branch-tree-h2": "h2_merge_tree_nodes.csv",
        "branch-tree-h2-hierarchical": "h2_branch_tree_hierarchical_nodes.csv",
    }
    output = workdir / names[mode]
    if not output.exists():
        raise RuntimeError(f"{mode} did not create {output.name}\n{completed.stdout}")
    return output


def compare_dimension(
    dimension: str,
    binary: Path,
    stack: Path,
    slab_depth: int,
    foreground_connectivity: int,
    root: Path,
) -> None:
    flat_mode = f"branch-tree-{dimension}"
    hierarchical_mode = f"branch-tree-{dimension}-hierarchical"
    flat_dir = root / f"{dimension}_flat"
    hierarchical_dir = root / f"{dimension}_hierarchical"
    flat_dir.mkdir(parents=True)
    hierarchical_dir.mkdir(parents=True)

    flat = canonicalize(
        read_nodes(
            run_mode(
                binary,
                stack,
                slab_depth,
                foreground_connectivity,
                flat_mode,
                flat_dir,
            )
        ),
        dimension,
    )
    hierarchical = read_nodes(
        run_mode(
            binary,
            stack,
            slab_depth,
            foreground_connectivity,
            hierarchical_mode,
            hierarchical_dir,
        )
    )

    if flat != hierarchical:
        limit = min(len(flat), len(hierarchical))
        first = next((i for i in range(limit) if flat[i] != hierarchical[i]), None)
        details = [
            f"{dimension.upper()} branch-tree mismatch",
            f"flat canonical nodes={len(flat)} hierarchical nodes={len(hierarchical)}",
        ]
        if first is not None:
            details.extend(
                [
                    f"first differing row={first}",
                    f"flat={flat[first]}",
                    f"hierarchical={hierarchical[first]}",
                ]
            )
        raise AssertionError("\n".join(details))

    print(
        f"PASS {dimension.upper()}: {len(flat)} plateau-canonical nodes; "
        "hierarchical tree matches flat reference"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Validate hierarchical H0/H2 branch trees against the existing flat "
            "reducer after plateau canonicalization."
        )
    )
    result.add_argument("stack", type=Path, help="directory containing TIFF slices")
    result.add_argument("--binary", type=Path, default=Path("target/release/betti_curves"))
    result.add_argument("--slab-depth", type=int, default=8)
    result.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    result.add_argument(
        "--dimensions", nargs="+", choices=("h0", "h2"), default=("h0", "h2")
    )
    result.add_argument("--work-dir", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    binary = args.binary.resolve()
    stack = args.stack.resolve()
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for dimension in args.dimensions:
            compare_dimension(
                dimension,
                binary,
                stack,
                args.slab_depth,
                args.foreground_connectivity,
                root,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="branch_tree_hier_validation_") as directory:
            root = Path(directory)
            for dimension in args.dimensions:
                compare_dimension(
                    dimension,
                    binary,
                    stack,
                    args.slab_depth,
                    args.foreground_connectivity,
                    root,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
