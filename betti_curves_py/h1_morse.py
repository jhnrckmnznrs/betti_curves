"""Optional, exact H1 persistence and original-grid loop reconstruction.

This uses the same closed-voxel sublevel cubical model as the **26-connected**
foreground mode of betti_curves.  It is NOT a 6-connected H1 backend.

The native C++17 backend is compiled lazily, only on explicit use.  It has
configurable cell, reduction-work, trace, heap and replay budgets and is not
an out-of-core general-purpose solver.  Existing Betti-curve commands do not
import it or need its optional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import numpy as np


def _closed_voxel_volume(volume):
    if isinstance(volume, (str, Path)):
        p = Path(volume)
        if p.suffix.lower() != '.npy':
            raise ValueError('H1 Morse path input must be a 3D .npy file; pass TIFF data as an array')
        volume = np.load(p, mmap_mode='r')
    v = np.asarray(volume)
    if v.ndim != 3 or not all(v.shape):
        raise ValueError('H1 Morse requires a nonempty 3D volume')
    if v.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError('H1 Morse currently accepts uint8 and uint16 intensities only')
    return v


def _check_connectivity(connectivity):
    if connectivity != 26:
        raise ValueError(
            'H1 Morse uses closed-voxel cubical sublevels (26-connected foreground); '
            'the 6-connected foreground convention is a different filtration'
        )


@dataclass
class H1MorseResult:
    """Finite positive-length H1 intervals with immutable, ordered pair IDs.

    ``catalog_path`` contains the saved native reduced complex and contraction
    history.  Supplying ``output_dir`` to :func:`h1_persistence` makes it
    durable.  Otherwise the result owns a temporary directory until ``close``.
    """
    births: np.ndarray
    deaths: np.ndarray
    pair_ids: tuple[str, ...]
    catalog_path: Path | None
    statistics: dict[str, Any] = field(default_factory=dict)
    skipped_empty: bool = False
    compact_pair_ids: bool = False
    _scratch: Any = field(default=None, repr=False, compare=False)

    @property
    def dimension(self):
        return 1

    @property
    def finite_count(self):
        return len(self.births)

    def __len__(self):
        return self.finite_count

    def pairs(self):
        return [(int(b), int(d)) for b, d in zip(self.births, self.deaths)]

    def diagram_rows(self):
        return np.stack([self.births, self.deaths], axis=-1) if len(self) else np.empty((0, 2), dtype=np.float64)

    def at(self, thresholds):
        """Exact Betti-1 values at arbitrary thresholds, half-open [birth,death)."""
        t = np.asarray(thresholds)
        births = np.sort(self.births.astype(np.int64, copy=False))
        deaths = np.sort(self.deaths.astype(np.int64, copy=False))
        result = np.searchsorted(births, t, side='right') - np.searchsorted(deaths, t, side='right')
        return int(result) if result.ndim == 0 else result

    def betti_curve(self):
        """Sparse right-continuous change points as (threshold, beta1) pairs."""
        if not len(self):
            return []
        event_levels, indices = np.unique(
            np.concatenate([self.births, self.deaths]), return_inverse=True
        )
        n = len(self)
        deltas = np.bincount(indices[:n], minlength=len(event_levels)).astype(np.int64)
        deltas -= np.bincount(indices[n:], minlength=len(event_levels))
        counts = np.cumsum(deltas)
        change = np.flatnonzero(deltas)
        return [(int(event_levels[i]), int(counts[i])) for i in change]

    def pair_id_at(self, index):
        """Return one exact creator:destroyer ID without expanding all IDs."""
        if not 0 <= index < len(self):
            raise IndexError('H1 pair index out of range')
        if not self.compact_pair_ids:
            return self.pair_ids[index]
        if self.catalog_path is None:
            raise RuntimeError('No H1 catalog available')
        from ._morse.native_backend import CompactMorse
        return CompactMorse.open(self.catalog_path).h1_pair_at(index)[0]

    def iter_pair_ids(self):
        """Stream exact pair IDs without making a tuple of every ID."""
        if not self.compact_pair_ids:
            yield from self.pair_ids
            return
        if self.catalog_path is None:
            raise RuntimeError('No H1 catalog available')
        from ._morse.native_backend import CompactMorse
        morse = CompactMorse.open(self.catalog_path)
        for row in morse.h1_records():
            creator = int(morse.ids[row['creator']])
            destroyer = int(morse.ids[row['destroyer']])
            yield f'{creator}:{destroyer}'

    def reconstruct(self, pair_id, *, max_operations=100_000_000,
                    max_output_entries=2_000_000, max_seconds=120.):
        """Selectively lift and verify a single persistence loop by exact pair ID."""
        if self.catalog_path is None:
            raise ValueError('Pair ID not found in this catalog')
        from ._morse.native_backend import CompactMorse
        try:
            return CompactMorse.open(self.catalog_path).replay_h1_native(
                pair_id, max_operations=max_operations,
                max_output_entries=max_output_entries, max_seconds=max_seconds
            )
        except ValueError as exc:
            raise ValueError('Pair ID not found in this catalog') from exc

    def close(self):
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
            self.catalog_path = None


def h1_persistence(volume, *, connectivity=26, output_dir=None,
                   pre_collapse=True, block_voxels=8, max_cells=3_000_000,
                   max_operations=8_000_000, max_fill=128,
                   max_heap_entries=8_000_000, max_trace_bytes=300_000_000,
                   max_pairs=10_000_000, max_seconds=90., local_first=False,
                   cohomology_max_operations=100_000_000,
                   cohomology_max_seconds=120.,
                   store_pair_ids=True,indexed_scheduler=False,
                   cohomology_max_annotation_entries=50_000_000,
                   cohomology_max_output_pairs=10_000_000):
    """Exact finite H1 persistence with bounded native Morse reduction.

    The full cubical complex currently must fit in memory.  Larger inputs
    require explicitly increasing ``max_cells`` and provisioning enough RAM.
    The on-disk catalog preserves creator/destroyer cell IDs for replay.
    """
    _check_connectivity(connectivity)
    v = _closed_voxel_volume(volume)
    scratch = None
    if output_dir is None:
        scratch = TemporaryDirectory(prefix='betti_h1_morse_')
        output_dir = scratch.name
    try:
        from ._morse.native_backend import CompactMorse
        morse = CompactMorse(
            v, output_dir, pre_collapse=pre_collapse,
            block_voxels=block_voxels, max_cells=max_cells,
            max_operations=max_operations, max_fill=max_fill,
            max_heap_entries=max_heap_entries, max_trace_bytes=max_trace_bytes,
            max_pairs=max_pairs, max_seconds=max_seconds, local_first=local_first,
            indexed_scheduler=indexed_scheduler,
        )
        records = morse.h1_records(
            max_operations=cohomology_max_operations,
            max_seconds=cohomology_max_seconds,
            max_annotation_entries=cohomology_max_annotation_entries,
            max_output_pairs=cohomology_max_output_pairs,
        )
        pair_ids = (() if not store_pair_ids else
                    tuple(f'{int(morse.ids[r["creator"]])}:{int(morse.ids[r["destroyer"]])}'
                          for r in records))
        # Persisted cohomology statistics are part of the native completed
        # catalog. Resource limits apply to sparse entries, not whole-process RSS.
        import json
        cohomology_meta = json.loads((morse.path / 'h1_pairs_metadata.json').read_text())
        statistics = dict(morse.stats)
        statistics['cohomology_peak_annotation_entries'] = int(
            cohomology_meta.get('peak_annotation_entries', 0))
        statistics['cohomology_operations'] = int(cohomology_meta.get('operations', 0))
        result = H1MorseResult(
            births=np.asarray(records['birth'], dtype=v.dtype).copy(),
            deaths=np.asarray(records['death'], dtype=v.dtype).copy(),
            pair_ids=pair_ids,
            catalog_path=Path(output_dir).resolve(),
            statistics=statistics, compact_pair_ids=not store_pair_ids,
            _scratch=scratch,
        )
        if np.any(result.births >= result.deaths):
            raise AssertionError('Native engine emitted a non-positive persistence interval')
        if store_pair_ids and len(set(result.pair_ids)) != len(result):
            raise AssertionError('Native engine emitted duplicate pair identifiers')
        return result
    except BaseException:
        if scratch is not None:
            scratch.cleanup()
        raise


def _independent_beta1(h0, h2, euler, thresholds):
    """Compute beta1 from separately calculated H0, H2 and Euler curves."""
    from .core import betti_at, betti_curve
    b0 = betti_at(betti_curve(h0), thresholds)
    b2 = betti_at(betti_curve(h2), thresholds)
    chi = euler.at(thresholds)
    independent = b0 + b2 - chi
    if np.any(independent < 0):
        i = int(np.flatnonzero(independent < 0)[0])
        raise ValueError(f'Incompatible H0/H2/Euler topology at threshold {int(thresholds[i])}')
    return independent


@dataclass
class VolumeTopologyAnalysis:
    h0: Any
    h2: Any
    euler: Any
    h1_betti_curve: list[tuple[int, int]]
    h1: H1MorseResult | None
    validated: bool
    connectivity: int


def analyze_volume(volume, *, connectivity=26, h1_backend='curve',
                   output_dir=None, validate_betti=True,
                   skip_empty_h1=True, persistence_backend='auto',
                   euler_backend='auto', euler_threads=None, **morse_options):
    """Unified H0/H2/Euler/H1 analysis with optional full H1 barcode.

    ``h1_backend='curve'`` retains the original fast Betti-only behavior.
    ``'morse'`` computes the exact finite H1 barcode, verifies its complete
    Betti curve against independent H0/H2/Euler computations, and supports
    selected original-grid cycles.  The native backend is opt-in only.
    """
    if connectivity not in (6, 26):
        raise ValueError('connectivity must be 6 or 26')
    if h1_backend not in ('curve', 'morse'):
        raise ValueError("h1_backend must be 'curve' or 'morse'")
    if h1_backend == 'morse':
        _check_connectivity(connectivity)
    if h1_backend != 'morse' and morse_options:
        raise ValueError('Morse-specific options require h1_backend="morse"')
    v = _closed_voxel_volume(volume)
    from .core import persistence
    from .euler_stream import euler_curve
    h0 = persistence(v, dimension=0, foreground_connectivity=connectivity,
                     backend=persistence_backend)
    h2 = persistence(v, dimension=2, foreground_connectivity=connectivity,
                     backend=persistence_backend)
    euler = euler_curve(v, connectivity=connectivity,
                        backend=euler_backend, threads=euler_threads)
    thresholds = np.arange(int(v.min()), int(v.max()) + 1, dtype=np.int64)
    expected = _independent_beta1(h0, h2, euler, thresholds)
    if np.any(expected[-1:] != 0):
        raise AssertionError('Closed rectangular filtration has nonzero final beta1')
    changes = np.empty_like(expected, dtype=bool)
    changes[0] = expected[0] != 0
    changes[1:] = expected[1:] != expected[:-1]
    curve = [(int(t), int(b)) for t, b in zip(thresholds[changes], expected[changes])]
    h1 = None
    validated = False
    if h1_backend == 'morse':
        if skip_empty_h1 and not expected.any():
            h1 = H1MorseResult(
                births=np.empty(0, dtype=v.dtype),
                deaths=np.empty(0, dtype=v.dtype), pair_ids=(),
                catalog_path=None, statistics={'skipped_by_betti_identity': True},
                skipped_empty=True,
            )
        else:
            h1 = h1_persistence(v, connectivity=connectivity,
                                output_dir=output_dir, **morse_options)
        if validate_betti:
            found = h1.at(thresholds)
            mismatch = np.flatnonzero(found != expected)
            if len(mismatch):
                i = int(mismatch[0])
                raise AssertionError(
                    f'H1 barcode disagrees with H0+H2-Euler at threshold '
                    f'{int(thresholds[i])}: Morse={int(found[i])}, '
                    f'independent={int(expected[i])}'
                )
            validated = True
    return VolumeTopologyAnalysis(h0, h2, euler, curve, h1, validated, connectivity)
