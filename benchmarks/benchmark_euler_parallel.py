#!/usr/bin/env python3
"""Reproducible warmed serial-vs-parallel Euler benchmark.

Run from an installed checkout, e.g.:
python benchmarks/benchmark_euler_parallel.py /path/to/tiff_stack --repeats 5 --threads 1 2 4
"""
import argparse
import json
import statistics
import time

import numpy as np
from betti_curves_py.euler_stream import euler_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='3D .npy or directory of 2D TIFF images')
    ap.add_argument('--connectivity', type=int, nargs='+', choices=(6, 26), default=[6, 26])
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--output-json', default=None)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats must be positive')
    results = []
    for connectivity in args.connectivity:
        expected = euler_curve(args.input, connectivity=connectivity, backend='numba')
        for label, backend, threads in [('serial', 'numba', None)] + [(f'{n}-thread', 'parallel', n) for n in args.threads]:
            # Warm the exact specialization so timed runs exclude compilation.
            first = euler_curve(args.input, connectivity=connectivity, backend=backend, threads=threads)
            if not np.array_equal(first.histogram, expected.histogram):
                raise AssertionError(f'Histogram mismatch on {label} connectivity={connectivity}')
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = euler_curve(args.input, connectivity=connectivity, backend=backend, threads=threads)
                samples.append(time.perf_counter() - started)
                if not np.array_equal(result.histogram, expected.histogram):
                    raise AssertionError('Histogram changed across runs')
            record = dict(connectivity=connectivity, method=label, threads=threads,
                          volume_shape=list(expected.shape), median_seconds=statistics.median(samples),
                          seconds=samples)
            results.append(record)
            print(f'{connectivity}-connected {label}: {record["median_seconds"]:.3f}s median', flush=True)
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
