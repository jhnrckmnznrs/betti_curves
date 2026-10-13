"""Independent digital Euler oracle and H1 integration tests."""
import csv
import numpy as np
import pytest

from betti_curves_py.euler_stream import (
    euler_curve, h1_curve_from_h0_h2, write_curve_csv,
)


@pytest.mark.parametrize('connectivity', [6, 26])
def test_exhaustive_binary_2x2x2(connectivity):
    from skimage.measure import euler_number
    c = 1 if connectivity == 6 else 3
    for pattern in range(256):
        binary = np.array([(pattern >> i) & 1 for i in range(8)], dtype=np.uint8).reshape(2, 2, 2)
        volume = np.where(binary, 0, 1).astype(np.uint8)
        curve = euler_curve(volume, connectivity=connectivity)
        assert curve.at(-1) == 0
        assert curve.at(0) == euler_number(binary, connectivity=c), (pattern, connectivity)
        assert curve.at(1) == 1


@pytest.mark.parametrize('connectivity', [6, 26])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_random_multilevel_vs_skimage(connectivity, dtype):
    from skimage.measure import euler_number
    for seed in range(30):
        rng = np.random.default_rng(seed + 400)
        volume = rng.integers(0, 8, size=(4, 5, 6), dtype=dtype)
        curve = euler_curve(volume, connectivity=connectivity)
        oracle = [euler_number(volume <= i, connectivity=(1 if connectivity == 6 else 3))
                  for i in range(-1, 9)]
        assert curve.at(np.arange(-1, 9)).tolist() == oracle
        assert int(curve.histogram.sum()) == 1


@pytest.mark.parametrize('conn', [6, 26])
def test_numpy_tiff_and_python_backends(tmp_path, conn):
    import tifffile
    rng = np.random.default_rng(72)
    volume = rng.integers(0, 65536, size=(4, 6, 7), dtype=np.uint16)
    np.save(tmp_path / 'v.npy', volume)
    folder = tmp_path / 'slices'
    folder.mkdir()
    for i, plane in enumerate(volume):
        tifffile.imwrite(folder / f'slice_{i:04d}.tif', plane)
    array_result = euler_curve(volume, connectivity=conn, backend='numba')
    file_result = euler_curve(tmp_path / 'v.npy', connectivity=conn)
    tiff_result = euler_curve(folder, connectivity=conn)
    python_result = euler_curve(volume, connectivity=conn, backend='python')
    for result in (file_result, tiff_result, python_result):
        assert np.array_equal(result.histogram, array_result.histogram)
        assert result.shape == volume.shape
    write_curve_csv(tiff_result, tmp_path / 'ecc.csv')
    assert (tmp_path / 'ecc.csv').read_text().startswith('threshold,euler_characteristic')


def test_known_tunnel_and_enclosed_cavity_h1(tmp_path):
    from betti_curves_py.core import both_persistence, betti_curve
    for conn in (6, 26):
        for case in ('tunnel', 'cavity'):
            v = np.full((7, 9, 9), 255, dtype=np.uint8)
            if case == 'tunnel':
                v[2:5, 2:7, 2:7] = 0
                v[2:5, 3:6, 3:6] = 255
                expected_chi, expected_b1 = 0, 1
            else:
                v[1:6, 1:8, 1:8] = 0
                v[2:5, 2:7, 2:7] = 255
                expected_chi, expected_b1 = 2, 0
            ecc = euler_curve(v, connectivity=conn)
            assert ecc.at(0) == expected_chi
            assert ecc.at(255) == 1
            h0, h2 = both_persistence(v, foreground_connectivity=conn, backend='numba')
            for name, points in [('h0', betti_curve(h0)), ('h2', betti_curve(h2))]:
                path = tmp_path / f'{name}.csv'
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['threshold', f'betti{name[1:]}'])
                    w.writerows(points)
            b1 = h1_curve_from_h0_h2(ecc, tmp_path / 'h0.csv', tmp_path / 'h2.csv')
            def b1_at(t):
                return next((v for x, v in reversed(b1) if x <= t), 0)
            assert b1_at(0) == expected_b1
            assert b1_at(255) == 0


def test_rejections():
    v = np.zeros((2, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError):
        euler_curve(v, connectivity=18)
    with pytest.raises(TypeError):
        euler_curve(v.astype(np.float32))
    with pytest.raises(ValueError):
        euler_curve(np.zeros((2, 2), dtype=np.uint8))


@pytest.mark.parametrize('conn', [6, 26])
def test_maximum_uint16_level(conn):
    v = np.full((2, 2, 2), 65535, dtype=np.uint16)
    ec = euler_curve(v, connectivity=conn)
    assert ec.at(65534) == 0
    assert ec.at(65535) == 1


def test_sampled_h1_threshold_counts(tmp_path):
    from betti_curves_py.euler_stream import h1_at_thresholds
    volume = np.full((7, 9, 9), 255, dtype=np.uint8)
    volume[2:5, 2:7, 2:7] = 0
    volume[2:5, 3:6, 3:6] = 255
    ecc = euler_curve(volume, connectivity=6)
    for name, counts in [('h0', [1, 1]), ('h2', [0, 0])]:
        with open(tmp_path / f'{name}_counts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['threshold', f'betti{name[1:]}'])
            writer.writerows(zip([0, 255], counts))
    got = h1_at_thresholds(ecc, tmp_path / 'h0_counts.csv', tmp_path / 'h2_counts.csv')
    assert got == [(0, 1), (255, 0)]
