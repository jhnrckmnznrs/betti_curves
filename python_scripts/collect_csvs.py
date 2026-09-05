#!/usr/bin/env python3
"""Collect per-structure H0 persistence CSVs into one directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_INPUT_NAME = "h0_persistence_scalar.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy h0_persistence_scalar.csv from each structure_* subdirectory "
            "and rename it to <subdirectory>.csv."
        )
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=Path("h0_baseline"),
        help="Directory containing structure_* subdirectories (default: h0_baseline)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("h0_baseline_collected"),
        help="Directory for the renamed CSV files (default: h0_baseline_collected)",
    )
    parser.add_argument(
        "--pattern",
        default="structure_*",
        help="Subdirectory glob pattern (default: structure_*)",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"CSV filename to collect (default: {DEFAULT_INPUT_NAME})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination CSV files that already exist",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any matching subdirectory lacks the requested CSV",
    )
    return parser.parse_args()


def collect_csvs(
    source_dir: Path,
    output_dir: Path,
    pattern: str,
    input_name: str,
    overwrite: bool,
    strict: bool,
) -> tuple[int, list[Path]]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    subdirectories = sorted(path for path in source_dir.glob(pattern) if path.is_dir())
    if not subdirectories:
        raise FileNotFoundError(
            f"No subdirectories matching {pattern!r} found in {source_dir}"
        )

    sources: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for subdirectory in subdirectories:
        source_csv = subdirectory / input_name
        if source_csv.is_file():
            sources.append((source_csv, output_dir / f"{subdirectory.name}.csv"))
        else:
            missing.append(source_csv)

    if strict and missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing input CSV files:\n{missing_text}")

    if not sources:
        raise FileNotFoundError(f"No {input_name!r} files were found")

    conflicts = [destination for _, destination in sources if destination.exists()]
    if conflicts and not overwrite:
        conflict_text = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "Destination files already exist; use --overwrite to replace them:\n"
            f"{conflict_text}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for source_csv, destination_csv in sources:
        shutil.copy2(source_csv, destination_csv)

    return len(sources), missing


def main() -> int:
    args = parse_args()
    try:
        copied, missing = collect_csvs(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            input_name=args.input_name,
            overwrite=args.overwrite,
            strict=args.strict,
        )
    except (FileNotFoundError, FileExistsError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Copied {copied} CSV file(s) to {args.output_dir.expanduser().resolve()}")
    if missing:
        print(f"Warning: skipped {len(missing)} subdirector(ies) with no input CSV:")
        for path in missing:
            print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
