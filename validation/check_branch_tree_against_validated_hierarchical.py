#!/usr/bin/env python3
"""Compare one hierarchical branch-tree binary against an already validated hierarchical binary.

This avoids rerunning the memory-heavy flat/in-memory oracle.  It is intended
for optimization iterations after a checkpoint binary has already passed the
flat-vs-hierarchical validator.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path

NAMES = {
    "h0": "h0_branch_tree_hierarchical_nodes.csv",
    "h2": "h2_branch_tree_hierarchical_nodes.csv",
}


def read_nodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["node"]))
    return rows


def run_hierarchical(
    binary: Path,
    stack: Path,
    slab_depth: int,
    foreground_connectivity: int,
    dimension: str,
    workdir: Path,
    env_overrides: dict[str, str] | None = None,
) -> Path:
    mode = f"branch-tree-{dimension}-hierarchical"
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    completed = subprocess.run(
        [
            str(binary),
            str(stack),
            str(slab_depth),
            str(foreground_connectivity),
            mode,
        ],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{binary} {mode} failed with exit code {completed.returncode}\n"
            f"{completed.stdout}"
        )
    output = workdir / NAMES[dimension]
    if not output.exists():
        raise RuntimeError(f"{mode} did not create {output.name}\n{completed.stdout}")
    return output


def compare_dimension(
    dimension: str,
    candidate_binary: Path,
    reference_binary: Path,
    stack: Path,
    slab_depth: int,
    foreground_connectivity: int,
    root: Path,
    candidate_env: dict[str, str] | None = None,
    reference_env: dict[str, str] | None = None,
) -> None:
    ref_dir = root / f"{dimension}_reference"
    cand_dir = root / f"{dimension}_candidate"
    ref_dir.mkdir(parents=True)
    cand_dir.mkdir(parents=True)

    reference = read_nodes(
        run_hierarchical(
            reference_binary,
            stack,
            slab_depth,
            foreground_connectivity,
            dimension,
            ref_dir,
            reference_env,
        )
    )
    candidate = read_nodes(
        run_hierarchical(
            candidate_binary,
            stack,
            slab_depth,
            foreground_connectivity,
            dimension,
            cand_dir,
            candidate_env,
        )
    )

    if reference != candidate:
        limit = min(len(reference), len(candidate))
        first = next((i for i in range(limit) if reference[i] != candidate[i]), None)
        details = [
            f"{dimension.upper()} hierarchical checkpoint mismatch",
            f"reference nodes={len(reference)} candidate nodes={len(candidate)}",
        ]
        if first is not None:
            details.extend(
                [
                    f"first differing row={first}",
                    f"reference={reference[first]}",
                    f"candidate={candidate[first]}",
                ]
            )
        raise AssertionError("\n".join(details))

    print(
        f"PASS {dimension.upper()}: {len(candidate)} nodes; "
        "candidate hierarchical tree exactly matches validated checkpoint"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare hierarchical branch trees against a validated checkpoint binary."
    )
    p.add_argument("stack", type=Path)
    p.add_argument("--candidate-binary", type=Path, required=True)
    p.add_argument("--reference-binary", type=Path, required=True)
    p.add_argument("--slab-depth", type=int, default=8)
    p.add_argument("--foreground-connectivity", type=int, choices=(6, 26), default=26)
    p.add_argument("--dimensions", nargs="+", choices=("h0", "h2"), default=("h0", "h2"))
    p.add_argument("--work-dir", type=Path)
    p.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment override for the candidate binary; may be repeated",
    )
    p.add_argument(
        "--reference-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment override for the reference binary; may be repeated",
    )
    return p


def parse_env(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid environment override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"invalid environment override {item!r}; key is empty")
        result[key] = value
    return result


def main() -> int:
    args = parser().parse_args()
    candidate = args.candidate_binary.resolve()
    reference = args.reference_binary.resolve()
    stack = args.stack.resolve()
    candidate_env = parse_env(args.candidate_env)
    reference_env = parse_env(args.reference_env)

    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for dimension in args.dimensions:
            compare_dimension(
                dimension,
                candidate,
                reference,
                stack,
                args.slab_depth,
                args.foreground_connectivity,
                root,
                candidate_env,
                reference_env,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="branch_tree_checkpoint_validation_") as directory:
            root = Path(directory)
            for dimension in args.dimensions:
                compare_dimension(
                    dimension,
                    candidate,
                    reference,
                    stack,
                    args.slab_depth,
                    args.foreground_connectivity,
                    root,
                    candidate_env,
                    reference_env,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
