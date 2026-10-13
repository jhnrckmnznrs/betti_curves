"""v0.15: parallel Euler kernels must match the serial event histogram exactly."""

import numpy as np
import pytest

from betti_curves_py.euler_stream import euler_curve

numba = pytest.importorskip('numba')


def compare(volume, connectivity, threads):
    serial = euler_curve(volume, connectivity=connectivity, backend='numba')
    parallel = euler_curve(volume, connectivity=connectivity, backend='parallel', threads=threads)
    assert np.array_equal(parallel.histogram, serial.histogram)
    assert np.array_equal(parallel.thresholds, serial.thresholds)
    assert np.array_equal(parallel.values, serial.values)
    assert parallel.shape == serial.shape
    assert np.sum(parallel.histogram) == 1


@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 5, 9), (2, 1, 4), (3, 6, 1), (3, 9, 12), (4, 71, 93)])
def test_nontrivial_shapes(connectivity, dtype, shape):
    rng = np.random.default_rng(100 + sum(shape))
    volume = rng.integers(0, 11, size=shape, dtype=dtype)
    compare(volume, connectivity, threads=min(4, numba.config.NUMBA_NUM_THREADS))


@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('value', [0, 1, 255])
def test_plateaus(connectivity, dtype, value):
    compare(np.full((3, 13, 17), value, dtype=dtype), connectivity, threads=2)


@pytest.mark.parametrize('connectivity', [6, 26])
def test_maximum_uint16_and_large_plane(connectivity):
    v = np.random.default_rng(101).integers(0, 65536, size=(3, 256, 273), dtype=np.uint16)
    v[0, 0, 0] = 65535
    v[1, 0, 0] = 0
    compare(v, connectivity, threads=3)


@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_exhaustive_binary_blocks(connectivity, dtype):
    for pattern in range(256):
        v = (np.arange(8).reshape((2, 2, 2)) & 0).astype(dtype)
        for i in range(8):
            v.flat[i] = (pattern >> i) & 1
        compare(v, connectivity, threads=2)


@pytest.mark.parametrize('connectivity', [6, 26])
def test_tiff_and_npy_input(tmp_path, connectivity):
    tifffile = pytest.importorskip('tifffile')
    volume = np.random.default_rng(17).integers(0, 65536, size=(4, 141, 137), dtype=np.uint16)
    folder = tmp_path / 'stack'
    folder.mkdir()
    for i, plane in enumerate(volume):
        tifffile.imwrite(folder / f'slice_{i:04d}.tif', plane)
    np.save(tmp_path / 'stack.npy', volume)
    expected = euler_curve(volume, backend='numba', connectivity=connectivity)
    for source in (folder, tmp_path / 'stack.npy'):
        actual = euler_curve(source, backend='parallel', connectivity=connectivity, threads=3)
        assert np.array_equal(actual.histogram, expected.histogram)


@pytest.mark.parametrize('connectivity', [6, 26])
def test_auto_for_large_plane(connectivity):
    rng = np.random.default_rng(67)
    volume = rng.integers(0, 256, size=(2, 513, 513), dtype=np.uint8)
    result = euler_curve(volume, connectivity=connectivity, backend='auto', threads=2)
    expected = euler_curve(volume, connectivity=connectivity, backend='numba')
    assert np.array_equal(result.histogram, expected.histogram)


def test_numba_thread_mask_restoration():
    before = numba.get_num_threads()
    v = np.zeros((2, 10, 10), dtype=np.uint8)
    euler_curve(v, backend='parallel', threads=2)
    assert numba.get_num_threads() == before


@pytest.mark.parametrize('bad', [0, -1, 1.3, True, '4'])
def test_bad_worker_count(bad):
    with pytest.raises(ValueError):
        euler_curve(np.zeros((2, 2, 2), dtype=np.uint8), backend='parallel', threads=bad)


def test_too_many_threads():
    with pytest.raises(ValueError):
        euler_curve(np.zeros((2, 2, 2), dtype=np.uint8), backend='parallel', threads=numba.config.NUMBA_NUM_THREADS + 1)


def test_explicit_parallel_fails_without_numba(monkeypatch):
    import betti_curves_py.euler_stream as mod
    monkeypatch.setattr(mod, 'njit', None)
    with pytest.raises(ImportError):
        mod.euler_curve(np.zeros((2, 2, 2), dtype=np.uint8), backend='parallel')
