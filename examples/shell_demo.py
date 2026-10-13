"""Create a shell TIFF fixture and compute its H0/H2 barcodes."""
from pathlib import Path

import numpy as np
import tifffile

from betti_curves_py import both_persistence, betti_curve


def main():
    volume = np.full((3, 3, 3), 1, dtype=np.uint16)
    volume[1, 1, 1] = 4
    directory = Path('shell_tiff_example')
    directory.mkdir(parents=True, exist_ok=True)
    for z, image in enumerate(volume):
        tifffile.imwrite(directory / f'slice_{z:04d}.tif', image)
    h0, h2 = both_persistence(volume, foreground_connectivity=6)
    print('H0:', h0.pairs())
    print('H2:', h2.pairs())
    print('H2 curve:', betti_curve(h2))
    print('TIFF files:', directory.resolve())


if __name__ == '__main__':
    main()
