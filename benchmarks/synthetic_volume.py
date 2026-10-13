"""Reproducible, lazy, correlated 3D synthetic synchrotron-like U16 volume.

This is NOT a physical simulator or surrogate for actual bone tomography.
The samples are deterministic functions of absolute (z,y,x) voxel coordinates,
so arbitrary tile reads agree with full-volume generation, including seams.

Memory used for generation is proportional to requested block size; the volume
can have a shape far larger than available RAM or disk space. Use materialize()
only on a machine with enough disk space for the actual uncompressed output.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np

TARGET = (2048, 3700, 3700)
LARGER = [(4096, 3700, 3700), (4096, 4096, 4096)]

class SyntheticVolume:
    dtype = np.dtype('uint16')

    def __init__(self, shape=TARGET, seed=2026):
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError('shape must be three positive dimensions')
        self.seed = int(seed)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise TypeError('expected three slices (z, y, x)')
        coordinates = []
        collapse = []
        for axis, key in enumerate(index):
            if isinstance(key, slice):
                start, stop, step = key.indices(self.shape[axis])
                if step <= 0:
                    raise ValueError('positive steps only')
                coordinates.append(np.arange(start, stop, step, dtype=np.float32))
                collapse.append(False)
            elif isinstance(key, (int, np.integer)):
                k = int(key)
                if k < 0:
                    k += self.shape[axis]
                if k < 0 or k >= self.shape[axis]:
                    raise IndexError(key)
                coordinates.append(np.array([k], dtype=np.float32))
                collapse.append(True)
            else:
                raise TypeError('only int and slice indices supported')
        z, y, x = coordinates
        # Continuous 3D geometry: coherent ridges, connected sheets and pores.
        # Frequencies avoid repeating a small 2D texture in successive slices.
        z3 = z[:, None, None]
        y3 = y[None, :, None]
        x3 = x[None, None, :]
        p = self.seed * np.float32(0.00017)
        with np.errstate(over='raise', invalid='raise'):
            main = np.sin(np.float32(.042) * x3 + np.float32(.020) * z3 + p)
            main = main * np.cos(np.float32(.053) * y3 - np.float32(.013) * z3)
            main += np.float32(.42) * np.sin(np.float32(.037) * y3 + np.float32(.029) * z3 + p)
            main += np.float32(.29) * np.cos(np.float32(.026) * x3 + np.float32(.031) * y3)
            main += np.float32(.12) * np.sin(np.float32(.21) * x3 - np.float32(.15) * y3 + np.float32(.11) * z3)
            # Quantized scanner intensity with correlated fine-grained texture.
            # Smooth transition from marrow-like (dark) to bone-like (bright).
            fraction = np.clip((main - np.float32(.15)) * np.float32(2.7) + np.float32(.5), 0, 1)
            main = np.float32(9500) + np.float32(44500) * fraction
            main += np.float32(900) * np.sin(np.float32(.43) * x3 + np.float32(.39) * y3 + np.float32(.31) * z3)
            main = np.clip(main, 0, 65535)
            out = main.astype(np.uint16)
        if any(collapse):
            out = out[tuple(0 if c else slice(None) for c in collapse)]
        return out

    def materialize(self, dest, slab_depth=2):
        """Write the entire real U16 .npy, never loading more than a slab in RAM.

        Need at least prod(shape)*2 bytes of free disk, plus results/scratch.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        needed = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize + 256
        free = shutil.disk_usage(dest.parent).free
        if free < needed + 128 * 1024**2:
            raise OSError(
                f'insufficient free disk for full U16 output: need at least '
                f'{(needed + 128*1024**2)/1024**3:.2f} GiB; '
                f'available {free/1024**3:.2f} GiB. '
                'The lazy generator can still sample this shape without a full file.')
        out = np.lib.format.open_memmap(dest, mode='w+', dtype=self.dtype, shape=self.shape)
        try:
            for z in range(0, self.shape[0], slab_depth):
                out[z:z + slab_depth] = self[z:z + slab_depth, :, :]
            out.flush()
        finally:
            del out
        return dest


def shape_stats(shape):
    voxels = int(np.prod(shape, dtype=np.int64))
    return {'shape': list(shape), 'voxels': voxels, 'raw_u16_bytes': 2 * voxels,
            'raw_u16_gib': 2 * voxels / 1024**3,
            'raw_u8_gib': voxels / 1024**3}


def main():
    ap = argparse.ArgumentParser(description='Create a deterministic synthetic 3D U16 image')
    ap.add_argument('--shape', type=int, nargs=3, default=TARGET)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--slab-depth', type=int, default=2)
    args = ap.parse_args()
    volume = SyntheticVolume(args.shape, args.seed)
    print(shape_stats(args.shape), flush=True)
    volume.materialize(args.output, args.slab_depth)
    print('Generated:', args.output)

if __name__ == '__main__':
    main()
