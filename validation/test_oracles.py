from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

import reference_check as oracle


class BarcodeOracleTests(unittest.TestCase):
    def assert_curve_coverage_matches_components(
        self, volume: np.ndarray, foreground_connectivity: int
    ) -> None:
        background_connectivity = 26 if foreground_connectivity == 6 else 6
        values = sorted({int(value) for value in volume.ravel()})
        thresholds = sorted(
            {
                int(volume.min()) - 1,
                int(volume.max()),
                *values,
                *(value - 1 for value in values),
            }
        )
        h0 = oracle.h0_barcode_oracle(volume, foreground_connectivity)
        h2 = oracle.h2_barcode_oracle(volume, background_connectivity)
        self.assertEqual(
            oracle.curve_from_intervals(h0, thresholds),
            [
                oracle.beta0(volume, threshold, foreground_connectivity)
                for threshold in thresholds
            ],
        )
        self.assertEqual(
            oracle.curve_from_intervals(h2, thresholds),
            [
                oracle.beta2(volume, threshold, background_connectivity)
                for threshold in thresholds
            ],
        )

    def test_one_voxel_and_shell_have_expected_intervals(self) -> None:
        one = np.array([[[3]]], dtype=np.uint16)
        self.assertEqual(oracle.h0_barcode_oracle(one, 6), Counter({(3, None): 1}))
        self.assertFalse(oracle.h2_barcode_oracle(one, 26))

        shell = np.ones((3, 3, 3), dtype=np.uint16)
        shell[1, 1, 1] = 4
        self.assertEqual(oracle.h2_barcode_oracle(shell, 26), Counter({(1, 4): 1}))

    def test_random_oracle_intervals_cover_independent_scipy_counts(self) -> None:
        generator = np.random.default_rng(20260827)
        for _ in range(20):
            volume = generator.integers(0, 5, size=(3, 3, 4), dtype=np.uint16)
            for connectivity in (6, 26):
                self.assert_curve_coverage_matches_components(volume, connectivity)

    def test_float_signed_zero_is_normalized(self) -> None:
        volume = np.array([[[-0.0, 0.0, 2.5]]], dtype=np.float64)
        intervals = oracle.h0_barcode_oracle(volume, 6)
        self.assertEqual(intervals, Counter({(0.0, None): 1}))

    def test_branch_tree_validator_checks_barcode_and_parent_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodes.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["node", "parent", "birth_value", "death_value"])
                writer.writerow([0, "", 0, "inf"])
                writer.writerow([1, 0, 2, 5])
                writer.writerow([2, 1, 3, 5])
            oracle.validate_branch_tree_nodes(
                path,
                Counter({(0, None): 1, (2, 5): 1, (3, 5): 1}),
                dimension=0,
            )


if __name__ == "__main__":
    unittest.main()
