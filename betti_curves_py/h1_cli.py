"""Optional command-line entry point for exact H1 persistence and Betti-1."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .h1_morse import analyze_volume


def main(argv=None):
    parser = argparse.ArgumentParser(description='3D H1 Betti curve and optional exact Morse persistence')
    parser.add_argument('source', help='3D uint8/uint16 .npy array or TIFF slice directory')
    parser.add_argument('--output-dir', default='betti_h1_results')
    parser.add_argument('--backend', choices=('curve', 'morse'), default='curve',
                        help='curve is faster; morse computes individual H1 intervals')
    parser.add_argument('--connectivity', type=int, choices=(6,26), default=26)
    parser.add_argument('--max-cells', type=int, default=3_000_000)
    parser.add_argument('--max-seconds', type=float, default=90.)
    parser.add_argument('--block-voxels', type=int, default=8)
    parser.add_argument('--no-validation', action='store_true')
    args = parser.parse_args(argv)
    path = Path(args.source)
    if path.is_dir():
        from .io import read_tiff_stack
        volume = read_tiff_stack(path)
    elif path.suffix.lower()=='.npy':
        volume = np.load(path,mmap_mode='r')
    else:
        parser.error('Source must be .npy or a directory of TIFF slices')
    from .io import atomic_csv
    output = Path(args.output_dir)
    options = ({'max_cells':args.max_cells,
                'max_seconds':args.max_seconds,'block_voxels':args.block_voxels}
               if args.backend=='morse' else {})
    result=analyze_volume(
        volume, connectivity=args.connectivity, h1_backend=args.backend,
        output_dir=output/'morse_catalog' if args.backend=='morse' else None,
        validate_betti=not args.no_validation, **options,
    )
    with atomic_csv(output/'h1_betti_curve.csv',['threshold','betti1']) as rows:
        rows.writerows(result.h1_betti_curve)
    if result.h1 is not None:
        with atomic_csv(output/'h1_persistence.csv',['pair_id','birth','death']) as rows:
            for pid,(b,d) in zip(result.h1.pair_ids,result.h1.pairs()):
                rows.writerow((pid,b,d))
    print(f'H1 curve: {output / "h1_betti_curve.csv"}')
    if result.h1 is not None:
        print(f'H1 intervals: {len(result.h1)}; verified: {result.validated}; '
              f'catalog: {result.h1.catalog_path or "skipped (beta1=0)"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
