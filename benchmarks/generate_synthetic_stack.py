#!/usr/bin/env python3
"""Generate deterministic 3-D grayscale TIFF stacks for public benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_volume(shape: tuple[int, int, int], pattern: str, seed: int) -> np.ndarray:
    width, height, depth = shape
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((depth, height, width), dtype=np.float32)

    if pattern == "noise":
        return rng.random((depth, height, width), dtype=np.float32)
    if pattern == "layers":
        gradient = (x / max(width - 1, 1) + y / max(height - 1, 1) + z / max(depth - 1, 1)) / 3.0
        ripple = 0.12 * np.sin(x / 7.0) * np.cos(y / 9.0)
        return np.clip(gradient + ripple, 0.0, 1.0).astype(np.float32)
    if pattern == "shell":
        cx, cy, cz = (width - 1) / 2, (height - 1) / 2, (depth - 1) / 2
        scale = max(min(width, height, depth) / 2.0, 1.0)
        radius = np.sqrt(((x - cx) / scale) ** 2 + ((y - cy) / scale) ** 2 + ((z - cz) / scale) ** 2)
        return np.clip(np.abs(radius - 0.62) * 2.5, 0.0, 1.0).astype(np.float32)
    if pattern == "blobs":
        volume = np.ones((depth, height, width), dtype=np.float32)
        count = max(8, int(round((width * height * depth) ** (1 / 3) / 4)))
        for _ in range(count):
            cx = rng.uniform(0, width - 1)
            cy = rng.uniform(0, height - 1)
            cz = rng.uniform(0, depth - 1)
            radius = rng.uniform(0.08, 0.22) * min(width, height, depth)
            dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            blob = np.clip(dist2 / max(radius * radius, 1e-6), 0.0, 1.0)
            volume = np.minimum(volume, blob.astype(np.float32))
        noise = rng.random(volume.shape, dtype=np.float32) * 0.03
        return np.clip(volume + noise, 0.0, 1.0)
    raise ValueError(f"unknown pattern: {pattern}")


def encode(volume: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f32":
        return volume.astype(np.float32)
    if dtype == "u8":
        return np.rint(volume * 255.0).astype(np.uint8)
    if dtype == "u16":
        return np.rint(volume * 65535.0).astype(np.uint16)
    raise ValueError(dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("WIDTH", "HEIGHT", "DEPTH"), default=(128, 128, 128))
    parser.add_argument("--pattern", choices=("noise", "blobs", "layers", "shell"), default="blobs")
    parser.add_argument("--dtype", choices=("u8", "u16", "f32"), default="f32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(v <= 0 for v in args.shape):
        parser.error("all shape dimensions must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error(f"{args.output} is not empty; pass --overwrite to replace generated TIFFs")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in args.output.glob("*.tif*"):
            path.unlink()

    volume = encode(make_volume(tuple(args.shape), args.pattern, args.seed), args.dtype)
    for z, plane in enumerate(volume):
        image = Image.fromarray(plane)
        image.save(args.output / f"slice_{z:05d}.tif")

    metadata = {
        "shape_width_height_depth": list(args.shape),
        "pattern": args.pattern,
        "dtype": args.dtype,
        "seed": args.seed,
        "generator": "benchmarks/generate_synthetic_stack.py",
    }
    (args.output / "benchmark_fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
