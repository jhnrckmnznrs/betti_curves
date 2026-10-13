"""Command-line entry point for betti-curves-py."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from .streaming import streaming_betti_at
from .io import atomic_csv

from .core import persistence
from .hierarchical import hierarchical_persistence, export_hierarchical_pairs_csv
from .slab import slab_persistence
from .io import read_tiff_stack, write_betti_curve, write_branch_tree, write_persistence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='betti-curves-py',
        description='Exact H0/H2 Betti curves, barcodes, and elder trees for 3-D TIFF stacks.',
    )
    parser.add_argument('stack', type=Path, help='directory of grayscale TIFF slices, or a 3-D .npy volume for hierarchical mode')
    parser.add_argument('--mode', choices=('h0', 'h2', 'both', 'tree-h0', 'tree-h2', 'tree-both',
                                          'threshold-h0', 'threshold-h2', 'threshold-both'),
                        default='both')
    parser.add_argument('--connectivity', type=int, choices=(6, 26), default=6,
                        help='foreground connectivity (background uses dual)')
    parser.add_argument('--backend', choices=('auto', 'numba', 'numba-reference', 'python', 'hierarchical', 'slab'), default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('betti_py_output'))
    parser.add_argument('--thresholds', default=None,
                        help='comma-separated threshold values, required for threshold-* streaming modes')
    parser.add_argument('--slab-depth', type=int, default=8,
                        help='z-slices per slab for slab persistence or threshold-* mode')
    parser.add_argument('--mmap-path', type=Path, default=None,
                        help='raw memmap path for TIFF ingestion; with hierarchical, scratch created automatically if omitted')
    parser.add_argument('--tile-edge', type=int, default=48,
                        help='largest tile side for hierarchical exact persistence (default: 48)')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel subtree processes for hierarchical mode (default: 1)')
    parser.add_argument('--parallel-backend', choices=('threads', 'processes'),
                        default='threads', help='hierarchical worker type (default: threads)')
    parser.add_argument('--scheduling', choices=('adaptive', 'fixed'), default='fixed',
                        help='hierarchical scheduler (default: fixed; adaptive is experimental)')
    parser.add_argument('--task-factor', type=int, default=3,
                        help='maximum adaptive leaf jobs per worker (default: 3)')
    parser.add_argument('--pending-summary-mib', type=float, default=128,
                        help='RAM budget for waiting intermediate boundary summaries (default: 128 MiB)')
    parser.add_argument('--intra-slab', choices=('full', 'two-face', 'boundary-aware', 'boundary-forest'),
                        default='full',
                        help='intra-slab graph contraction; boundary-aware drops completed physical X/Y sides early; boundary-forest also compresses 26-connected internal seams')
    parser.add_argument('--slab-seam-strategy', choices=('forest', 'fused', 'legacy', 'certificate', 'paged-tree', 'stream-parent', 'compact-parent', 'stream26-blocked'),
                        default='forest', help='cross-Z seam strategy; stream26-blocked uses bounded 26-connected candidate events')
    parser.add_argument('--certificate-memory-mib', type=float, default=128.,
                        help='spill large compact cross-Z graph buffers to disk above this estimated size')
    parser.add_argument('--implicit-frontier', action='store_true',
                        help='experimental affine forward-face IDs for paged-tree slab persistence')
    parser.add_argument('--interval-chunk', type=int, default=65536,
                        help='events processed per compact-parent output batch')
    parser.add_argument('--uf-layout', choices=('separate','interleaved'), default='separate',
                        help='stream26-blocked union-find layout (default: separate; interleaved is experimental)')
    parser.add_argument('--seam-batch-activations', type=int, default=2048,
                        help='maximum activated face pixels per streamed seam buffer')
    parser.add_argument('--uf-block-entries', type=int, default=65536,
                        help='union-find initialization block size')
    parser.add_argument('--disk-retained', action='store_true',
                        help='stream compact-parent retained edges to scratch disk')
    parser.add_argument('--event-page-mib', type=float, default=64.,
                        help='per-buffer RAM-to-memmap threshold for paged-tree event arrays')
    parser.add_argument('--merge-strategy', choices=('fused', 'legacy'), default='fused',
                        help='parent seam construction method (default: fused)')
    parser.add_argument('--no-compact-ids', action='store_true',
                        help='disable compact uint32 global IDs in hierarchical summaries')
    parser.add_argument('--no-curve', action='store_true',
                        help='skip sparse curve generation in hierarchical mode')
    parser.add_argument('--pairs-csv', action='store_true',
                        help='also convert binary hierarchical intervals into a larger CSV file')
    args = parser.parse_args(argv)
    if args.mode.startswith('threshold-'):
        if args.backend in ('hierarchical', 'slab'):
            parser.error('hierarchical backend computes full intervals; use --mode h0, h2, or both')
        if not args.thresholds:
            parser.error('--thresholds is required for threshold-* mode')
        try:
            levels = [int(t.strip()) if t.strip().lstrip('+-').isdigit() else float(t.strip())
                      for t in args.thresholds.split(',') if t.strip()]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for dim in (0, 2):
                key = 'h0' if dim == 0 else 'h2'
                if args.mode not in ('threshold-both', f'threshold-{key}'):
                    continue
                values = streaming_betti_at(args.stack, levels, dimension=dim,
                                            foreground_connectivity=args.connectivity,
                                            slab_depth=args.slab_depth,
                                            backend='numba' if args.backend == 'numba-reference' else args.backend)
                dest = args.output_dir / f'{key}_threshold_counts.csv'
                with atomic_csv(dest, ['threshold', 'betti0' if dim == 0 else 'betti2']) as writer:
                    writer.writerows(values)
                print(f'{key.upper()} threshold counts: {len(values)} values; wrote {dest}', flush=True)
        except (ValueError, OSError, TypeError, ImportError) as error:
            parser.exit(2, f'error: {error}\n')
        return 0
    if args.backend in ('hierarchical', 'slab'):
        if args.mode.startswith('tree-'):
            parser.error('hierarchical/slab backend does not export branch-tree ancestry; use --mode h0, h2 or both')
        if args.backend == 'slab' and args.workers != 1:
            parser.error('slab backend is sequential; use --workers 1')
        try:
            # TIFF slices are decompressed one at a time to a disk-backed raw
            # volume. The hierarchy subsequently loads just the current tile.
            with tempfile.TemporaryDirectory(prefix='betti_input_') as scratch:
                if args.stack.is_file() and args.stack.suffix.lower() == '.npy':
                    volume = np.load(args.stack, mmap_mode='r', allow_pickle=False)
                else:
                    raw = args.mmap_path or Path(scratch) / 'volume.raw'
                    volume = read_tiff_stack(args.stack, mmap_path=raw)
                print(f'{args.backend.capitalize()} input: {volume.shape}, dtype={volume.dtype}; '
                      f'tile-edge={args.tile_edge}; workers={args.workers} 'f'({args.parallel_backend})', flush=True)
                for dim in (0, 2):
                    key = 'h0' if dim == 0 else 'h2'
                    if args.mode not in ('both', key):
                        continue
                    result = (slab_persistence if args.backend == 'slab' else hierarchical_persistence)(
                        volume, dim, args.output_dir,
                        foreground_connectivity=args.connectivity,
                        tile_edge=args.tile_edge, write_curve=not args.no_curve,
                        **(dict(slab_depth=args.slab_depth, intra_slab=args.intra_slab, certificate_memory_mib=args.certificate_memory_mib, event_page_mib=args.event_page_mib,
                            interval_chunk=args.interval_chunk, disk_retained=args.disk_retained, uf_layout=args.uf_layout, seam_batch_activations=args.seam_batch_activations, uf_block_entries=args.uf_block_entries, implicit_frontier=args.implicit_frontier) if args.backend == 'slab'
                           else dict(parallel_workers=args.workers,
                                     parallel_backend=args.parallel_backend,
                                     scheduling=args.scheduling, task_factor=args.task_factor,
                                     pending_summary_mib=args.pending_summary_mib)),
                        merge_strategy=(args.slab_seam_strategy if args.backend == 'slab'
                                        else args.merge_strategy),
                        compact_ids=not args.no_compact_ids)
                    if args.pairs_csv:
                        export_hierarchical_pairs_csv(result, args.output_dir / f'{key}_persistence.csv')
                    print(f'{key.upper()}: {result.pair_count} finite intervals; '
                          f'{result.tiles} tiles; peak compressed graph '
                          f'{result.peak_summary_nodes} nodes / {result.peak_summary_edges} edges; '
                          f'wrote {result.pairs_path} and {result.metadata_path}', flush=True)
        except (ValueError, OSError, TypeError, ImportError, MemoryError) as error:
            parser.exit(2, f'error: {error}\n')
        return 0
    if args.stack.is_file() and args.stack.suffix.lower() == '.npy':
        parser.error('the .npy input is supported by --backend hierarchical; use a TIFF directory otherwise')
    try:
        volume = read_tiff_stack(args.stack, mmap_path=args.mmap_path)
        print(f'Volume: {volume.shape} (z,y,x), dtype={volume.dtype}; '
              f'foreground connectivity={args.connectivity}', flush=True)
        for dim in (0, 2):
            key = 'h0' if dim == 0 else 'h2'
            if args.mode not in ('both', 'tree-both', key, f'tree-{key}'):
                continue
            want_tree = args.mode.startswith('tree-')
            result = persistence(volume, dim, args.connectivity,
                                 backend=args.backend, branch_tree=want_tree)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            p = args.output_dir / f'{key}_persistence.csv'
            c = args.output_dir / f'{key}_betti_curve.csv'
            write_persistence(p, result)
            write_betti_curve(c, result)
            print(f'{key.upper()}: {len(result)} positive intervals; wrote {p}, {c}', flush=True)
            if want_tree:
                nodes, edges = write_branch_tree(args.output_dir / f'{key}_branch_tree', result)
                print(f'      branch tree: {nodes}, {edges}', flush=True)
    except (ValueError, OSError, TypeError, ImportError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
