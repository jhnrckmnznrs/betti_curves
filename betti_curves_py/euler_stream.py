"""Exact streaming 3-D Euler characteristic curves for U8/U16 images.

The input filtration is X_t = {voxel: intensity <= t}, using either
6-connected (vertex-centered cubical complex) or 26-connected
(closed-voxel cubical complex) foreground conventions.

Only two adjacent Z slices are held at once, and all filtration events are
accumulated into an int64 histogram (256 or 65536 bins). This is an Euler
curve, not an H1 persistence barcode.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from numba import njit, prange, get_thread_id, get_num_threads, set_num_threads
except ImportError:
    njit = None
    prange = get_thread_id = get_num_threads = set_num_threads = None


def _jit(f):
    return njit(cache=True)(f) if njit is not None else f


@_jit
def _accumulate6(cur, nxt, has_next, events):
    """Dual cubical cell contributions: +vertex,-edge,+face,-cube."""
    h, w = cur.shape
    for y in range(h):
        for x in range(w):
            v = int(cur[y, x])
            events[v] += 1
            has_x = x + 1 < w
            has_y = y + 1 < h
            if has_x:
                vx = int(cur[y, x + 1])
                events[max(v, vx)] -= 1
            if has_y:
                vy = int(cur[y + 1, x])
                events[max(v, vy)] -= 1
            if has_next:
                vz = int(nxt[y, x])
                events[max(v, vz)] -= 1
            if has_x and has_y:
                vxy = int(cur[y + 1, x + 1])
                events[max(v, vx, vy, vxy)] += 1
            if has_x and has_next:
                vxz = int(nxt[y, x + 1])
                events[max(v, vx, vz, vxz)] += 1
            if has_y and has_next:
                vyz = int(nxt[y + 1, x])
                events[max(v, vy, vz, vyz)] += 1
            if has_x and has_y and has_next:
                vxyz = int(nxt[y + 1, x + 1])
                events[max(v, vx, vy, vz, vxy, vxz, vyz, vxyz)] -= 1


@_jit
def _accumulate26(prev, cur, has_prev, has_cur, events, missing):
    """Closed-cube cell contributions over each 2x2x2 voxel neighborhood.

    Grid vertex owns one positive vertex, three positive edges, three
    positive faces, and one positive cube. Their times are the minima of
    incident voxel intensities. Missing exterior voxels never activate.
    """
    h, w = prev.shape
    for y in range(h + 1):
        for x in range(w + 1):
            yl = y > 0
            yr = y < h
            xl = x > 0
            xr = x < w
            a0 = int(prev[y-1, x-1]) if has_prev and yl and xl else missing
            a1 = int(prev[y-1, x]) if has_prev and yl and xr else missing
            a2 = int(prev[y, x-1]) if has_prev and yr and xl else missing
            a3 = int(prev[y, x]) if has_prev and yr and xr else missing
            a4 = int(cur[y-1, x-1]) if has_cur and yl and xl else missing
            a5 = int(cur[y-1, x]) if has_cur and yl and xr else missing
            a6 = int(cur[y, x-1]) if has_cur and yr and xl else missing
            a7 = int(cur[y, x]) if has_cur and yr and xr else missing
            v = min(a0, a1, a2, a3, a4, a5, a6, a7)
            if v < missing:
                events[v] += 1
            v = min(a1, a3, a5, a7)  # positive X edge
            if v < missing:
                events[v] -= 1
            v = min(a2, a3, a6, a7)  # positive Y edge
            if v < missing:
                events[v] -= 1
            v = min(a4, a5, a6, a7)  # positive Z edge
            if v < missing:
                events[v] -= 1
            v = min(a3, a7)  # XY face
            if v < missing:
                events[v] += 1
            v = min(a5, a7)  # XZ face
            if v < missing:
                events[v] += 1
            v = min(a6, a7)  # YZ face
            if v < missing:
                events[v] += 1
            if a7 < missing:  # 3-cube
                events[a7] -= 1


# The parallel kernels are defined only if Numba is present. Keeping the
# original serial kernels independently available provides a stable oracle.
if njit is not None:
    @njit(cache=True, parallel=True, nogil=True)
    def _accumulate6_parallel(cur, nxt, has_next, thread_events):
        """Process rows concurrently, using one event histogram per Numba thread."""
        h, w = cur.shape
        for y in prange(h):
            events = thread_events[get_thread_id()]
            for x in range(w):
                v = int(cur[y, x])
                events[v] += 1
                has_x = x + 1 < w
                has_y = y + 1 < h
                if has_x:
                    vx = int(cur[y, x + 1])
                    events[max(v, vx)] -= 1
                if has_y:
                    vy = int(cur[y + 1, x])
                    events[max(v, vy)] -= 1
                if has_next:
                    vz = int(nxt[y, x])
                    events[max(v, vz)] -= 1
                if has_x and has_y:
                    vxy = int(cur[y + 1, x + 1])
                    events[max(v, vx, vy, vxy)] += 1
                if has_x and has_next:
                    vxz = int(nxt[y, x + 1])
                    events[max(v, vx, vz, vxz)] += 1
                if has_y and has_next:
                    vyz = int(nxt[y + 1, x])
                    events[max(v, vy, vz, vyz)] += 1
                if has_x and has_y and has_next:
                    vxyz = int(nxt[y + 1, x + 1])
                    events[max(v, vx, vy, vz, vxy, vxz, vyz, vxyz)] -= 1




    @njit(cache=True, parallel=True, nogil=True)
    def _accumulate26_parallel(prev, cur, has_prev, has_cur, thread_events, missing):
        """Process grid rows concurrently with isolated per-thread Euler counters."""
        h, w = prev.shape
        for y in prange(h + 1):
            events = thread_events[get_thread_id()]
            for x in range(w + 1):
                yl = y > 0
                yr = y < h
                xl = x > 0
                xr = x < w
                a0 = int(prev[y-1, x-1]) if has_prev and yl and xl else missing
                a1 = int(prev[y-1, x]) if has_prev and yl and xr else missing
                a2 = int(prev[y, x-1]) if has_prev and yr and xl else missing
                a3 = int(prev[y, x]) if has_prev and yr and xr else missing
                a4 = int(cur[y-1, x-1]) if has_cur and yl and xl else missing
                a5 = int(cur[y-1, x]) if has_cur and yl and xr else missing
                a6 = int(cur[y, x-1]) if has_cur and yr and xl else missing
                a7 = int(cur[y, x]) if has_cur and yr and xr else missing
                v = min(a0, a1, a2, a3, a4, a5, a6, a7)
                if v < missing:
                    events[v] += 1
                v = min(a1, a3, a5, a7)  # positive X edge
                if v < missing:
                    events[v] -= 1
                v = min(a2, a3, a6, a7)  # positive Y edge
                if v < missing:
                    events[v] -= 1
                v = min(a4, a5, a6, a7)  # positive Z edge
                if v < missing:
                    events[v] -= 1
                v = min(a3, a7)  # XY face
                if v < missing:
                    events[v] += 1
                v = min(a5, a7)  # XZ face
                if v < missing:
                    events[v] += 1
                v = min(a6, a7)  # YZ face
                if v < missing:
                    events[v] += 1
                if a7 < missing:  # 3-cube
                    events[a7] -= 1


else:
    _accumulate6_parallel = None
    _accumulate26_parallel = None


@dataclass(frozen=True)
class EulerCurve:
    thresholds: np.ndarray
    values: np.ndarray
    histogram: np.ndarray
    connectivity: int
    shape: tuple[int, int, int]

    def at(self, thresholds):
        query = np.asarray(thresholds)
        out = np.zeros(query.shape, dtype=np.int64)
        if len(self.thresholds):
            i = np.searchsorted(self.thresholds, query, side='right') - 1
            valid = i >= 0
            out[valid] = self.values[i[valid]]
        return int(out) if out.ndim == 0 else out


def _planes(source):
    """Yield slices; validates shape and type, without copying a whole stack."""
    import tifffile

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            from .io import tiff_files
            paths = tiff_files(path)
            raw = (tifffile.imread(p) for p in paths)
        elif path.suffix.lower() == '.npy':
            arr = np.load(path, mmap_mode='r')
            if arr.ndim != 3:
                raise ValueError('.npy must be (Z,Y,X)')
            raw = (arr[i] for i in range(arr.shape[0]))
        else:
            raise ValueError('Source must be a directory of TIFF slices or a 3D .npy')
    else:
        if not hasattr(source, 'shape') or len(source.shape) != 3:
            raise ValueError('Source must have shape (Z,Y,X)')
        raw = (source[i] for i in range(source.shape[0]))
    expected = None
    for i, data in enumerate(raw):
        plane = np.asarray(data)
        if plane.ndim != 2 or not all(plane.shape):
            raise ValueError(f'Slice {i} must be nonempty and 2D')
        if plane.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise TypeError(f'Slice {i}: expected uint8/uint16, got {plane.dtype}')
        if expected is None:
            expected = (plane.shape, plane.dtype)
        elif (plane.shape, plane.dtype) != expected:
            raise ValueError(f'Slice {i} shape/dtype differs from first slice')
        yield np.ascontiguousarray(plane)


def euler_curve(source, *, connectivity=6, backend='auto', threads=None) -> EulerCurve:
    """Stream an exact U8/U16 Euler curve with at most two input planes.

    ``backend='numba'`` is the original serial reference, ``'parallel'``
    uses independent thread-private histograms, and ``'auto'`` selects the
    parallel kernel for wide slices when Numba is installed. ``'python'``
    is intended for small correctness tests. Parallel uses up to four Numba
    threads by default; ``threads`` controls the number explicitly.

    Worker histograms persist throughout the scan and are reduced only once,
    ensuring that no worker ever writes to another worker's histogram.
    """
    if connectivity not in (6, 26):
        raise ValueError('connectivity must be 6 or 26')
    if backend not in ('auto', 'numba', 'parallel', 'python'):
        raise ValueError('backend must be auto, numba, parallel, or python')
    if backend in ('numba', 'parallel') and njit is None:
        raise ImportError('Numba required for --backend numba/parallel')
    if threads is not None:
        if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or threads < 1:
            raise ValueError('threads must be a positive integer')
        if njit is None or backend == 'python':
            raise ValueError('threads requires an available Numba backend')
        # Numba enforces an upper limit determined at initialization.
        from numba import config
        if threads > config.NUMBA_NUM_THREADS:
            raise ValueError(f'threads={threads} exceeds Numba limit {config.NUMBA_NUM_THREADS}')

    it = iter(_planes(source))
    try:
        current = next(it)
    except StopIteration:
        raise ValueError('Empty volume') from None
    height, width = current.shape
    dtype = current.dtype
    levels = 256 if dtype == np.uint8 else 65536

    parallel = (njit is not None and backend == 'parallel') or (
        njit is not None and backend == 'auto' and height * width >= 512 * 512
    )
    if backend == 'numba' or backend == 'python':
        parallel = False
    if threads == 1 and backend == 'auto':
        parallel = False

    use_python = backend == 'python' or njit is None
    fn6 = _accumulate6.py_func if use_python and njit is not None else _accumulate6
    fn26 = _accumulate26.py_func if use_python and njit is not None else _accumulate26

    # Temporarily alter the per-calling-thread Numba thread mask only while
    # running the parallel kernels. Restore it even if an input slice fails.
    original_threads = get_num_threads() if parallel else None
    workers = (int(threads) if threads is not None else min(4, original_threads)) if parallel else 1
    if parallel:
        thread_events = np.zeros((workers, levels), dtype=np.int64)
        set_num_threads(workers)
        events = None
    else:
        thread_events = None
        events = np.zeros(levels, dtype=np.int64)

    depth = 1
    try:
        if connectivity == 6:
            if parallel:
                for following in it:
                    _accumulate6_parallel(current, following, True, thread_events)
                    current = following
                    depth += 1
                _accumulate6_parallel(current, current, False, thread_events)
            else:
                for following in it:
                    fn6(current, following, True, events)
                    current = following
                    depth += 1
                fn6(current, current, False, events)
        else:
            if parallel:
                _accumulate26_parallel(current, current, False, True, thread_events, levels)
                for following in it:
                    _accumulate26_parallel(current, following, True, True, thread_events, levels)
                    current = following
                    depth += 1
                _accumulate26_parallel(current, current, True, False, thread_events, levels)
            else:
                fn26(current, current, False, True, events, levels)
                for following in it:
                    fn26(current, following, True, True, events, levels)
                    current = following
                    depth += 1
                fn26(current, current, True, False, events, levels)
    finally:
        if parallel:
            set_num_threads(original_threads)
    if parallel:
        events = thread_events.sum(axis=0, dtype=np.int64)

    nonzero = np.flatnonzero(events)
    values = np.cumsum(events[nonzero], dtype=np.int64)
    if len(values) and int(values[-1]) != 1:
        raise ArithmeticError(f'Expected chi(full rectangular domain)=1; got {values[-1]}')
    return EulerCurve(nonzero, values, events, connectivity, (depth, height, width))


def _read_sparse(path, name):
    points = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'threshold' not in reader.fieldnames or name not in reader.fieldnames:
            raise ValueError(f'{path} needs threshold,{name} columns')
        for row in reader:
            t = int(row['threshold'])
            v = int(row[name])
            if t in points:
                raise ValueError(f'duplicate threshold {t} in {path}')
            points[t] = v
    return points


def h1_curve_from_h0_h2(ecc: EulerCurve, h0_csv, h2_csv):
    """Join right-continuous H0/H2 curves with Euler curve to obtain beta1(t).

    Expects *sublevel* H0/H2 curves under the same 6/26 foreground
    connectivity convention and same source volume. Returns sparse steps.
    """
    b0 = _read_sparse(h0_csv, 'betti0')
    b2 = _read_sparse(h2_csv, 'betti2')
    ec = {int(t): int(v) for t, v in zip(ecc.thresholds, ecc.values)}
    thresholds = sorted(set(b0) | set(b2) | set(ec))
    v0 = v2 = chi = previous = 0
    result = []
    for t in thresholds:
        v0 = b0.get(t, v0)
        v2 = b2.get(t, v2)
        chi = ec.get(t, chi)
        b1 = v0 + v2 - chi
        if b1 < 0:
            raise ValueError(f'Negative beta1={b1} at t={t}: connectivity/filtration/input mismatch?')
        if b1 != previous:
            result.append((t, b1))
            previous = b1
    if result and result[-1][1] != 0:
        raise ValueError('beta1 not zero after full-volume sublevel filtration')
    return result



def h1_at_thresholds(ecc: EulerCurve, h0_counts_csv, h2_counts_csv):
    """Derive beta1 on an arbitrary sampled threshold grid from H0/H2 counts.

    Input CSVs are generated by ``--mode threshold-both`` in the existing CLI.
    This does NOT reconstruct the complete H1 curve between sample values.
    """
    def load(path, name):
        out = {}
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'threshold' not in reader.fieldnames or name not in reader.fieldnames:
                raise ValueError(f'{path} needs threshold,{name} columns')
            for row in reader:
                t = float(row['threshold'])
                if not np.isfinite(t):
                    raise ValueError('thresholds must be finite')
                if t in out:
                    raise ValueError(f'duplicate threshold {t} in {path}')
                out[t] = int(row[name])
        return out
    a = load(h0_counts_csv, 'betti0')
    b = load(h2_counts_csv, 'betti2')
    if set(a) != set(b):
        raise ValueError('H0 and H2 counts have different sampled thresholds')
    output = []
    for t in sorted(a):
        val = a[t] + b[t] - ecc.at(t)
        if val < 0:
            raise ValueError(f'negative H1 at t={t}: check connectivity, source and conventions')
        output.append((int(t) if t.is_integer() else t, int(val)))
    return output


def write_curve_csv(curve: EulerCurve, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'euler_characteristic'])
        writer.writerows(zip(curve.thresholds.tolist(), curve.values.tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Streaming 3-D Euler characteristic curve (U8/U16).')
    ap.add_argument('source', type=Path, help='TIFF slice directory or 3D .npy')
    ap.add_argument('--connectivity', type=int, choices=(6,26), default=6)
    ap.add_argument('--backend', choices=('auto','numba','parallel','python'), default='auto')
    ap.add_argument('--threads', type=int, default=None, help='Number of parallel Numba workers (default at most 4)')
    ap.add_argument('--output-dir', type=Path, default=Path('euler_results'))
    ap.add_argument('--h0-curve', type=Path, default=None, help='h0_betti_curve.csv from same volume')
    ap.add_argument('--h2-curve', type=Path, default=None, help='h2_betti_curve.csv from same volume')
    ap.add_argument('--h0-counts', type=Path, default=None, help='h0_threshold_counts.csv sampled thresholds')
    ap.add_argument('--h2-counts', type=Path, default=None, help='h2_threshold_counts.csv sampled thresholds')
    args = ap.parse_args(argv)
    if (args.h0_curve is None) != (args.h2_curve is None):
        ap.error('Supply both --h0-curve and --h2-curve to derive H1')
    if (args.h0_counts is None) != (args.h2_counts is None):
        ap.error('Supply both --h0-counts and --h2-counts for sampled H1')
    curve = euler_curve(args.source, connectivity=args.connectivity, backend=args.backend, threads=args.threads)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    p = args.output_dir / 'euler_curve.csv'
    write_curve_csv(curve, p)
    print(f'{curve.shape} connectivity={curve.connectivity}: wrote {p} ({len(curve.thresholds)} changes)')
    if args.h0_curve is not None:
        b1 = h1_curve_from_h0_h2(curve, args.h0_curve, args.h2_curve)
        dest = args.output_dir / 'h1_betti_curve.csv'
        with open(dest, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['threshold','betti1'])
            writer.writerows(b1)
        print(f'wrote {dest} ({len(b1)} changes)')
    if args.h0_counts is not None:
        values = h1_at_thresholds(curve, args.h0_counts, args.h2_counts)
        dest = args.output_dir / 'h1_threshold_counts.csv'
        with open(dest, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['threshold', 'betti1'])
            writer.writerows(values)
        print(f'wrote {dest} ({len(values)} sampled thresholds)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
