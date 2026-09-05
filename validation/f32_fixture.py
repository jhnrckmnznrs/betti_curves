"""Helpers for staging an exact-value F32 TIFF subset from an integer/F32 stack."""
from __future__ import annotations

from pathlib import Path
import re


def stage_f32_subset(src: Path, dst: Path, limit: int) -> Path:
    try:
        import numpy as np
        import tifffile
    except ImportError as exc:  # pragma: no cover - user-facing validation helper
        raise SystemExit("staging an F32 fixture requires numpy and tifffile") from exc

    def natural_key(path: Path):
        return [int(part) if part.isdigit() else part.casefold()
                for part in re.split(r"(\d+)", path.name)]

    slices = sorted(
        (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}),
        key=natural_key,
    )
    if not slices:
        raise SystemExit(f"no TIFF slices found in {src}")
    if limit > 0:
        if len(slices) < limit:
            raise SystemExit(f"requested {limit} slices but found {len(slices)}")
        slices = slices[:limit]

    dst.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(slices):
        arr = tifffile.imread(path)
        if arr.ndim != 2:
            raise SystemExit(f"expected a 2D TIFF slice, got shape {arr.shape} in {path}")
        if arr.dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")):
            raise SystemExit(
                f"exact F32 fixture requires U8, U16, or F32 input; got {arr.dtype} in {path}"
            )
        out = np.asarray(arr, dtype=np.float32)
        # U8/U16 values are exactly representable in IEEE-754 F32; F32 sources are unchanged.
        tifffile.imwrite(dst / f"slice_{i:06d}.tif", out)
    return dst
