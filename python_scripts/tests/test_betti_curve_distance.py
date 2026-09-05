from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "betti_curve_distance.py"
SPEC = importlib.util.spec_from_file_location("betti_curve_distance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
distance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = distance
SPEC.loader.exec_module(distance)


class BettiCurveDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_curve(
        self,
        name: str,
        rows: list[tuple[float, float]],
        *,
        compressed: bool = False,
    ) -> Path:
        path = self.root / (f"{name}.csv.gz" if compressed else f"{name}.csv")
        text = "threshold,betti0\n" + "".join(
            f"{threshold},{value}\n" for threshold, value in rows
        )
        if compressed:
            with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
                output.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def pair(self, first: Path, second: Path) -> object:
        return distance.compute_distances(
            first,
            second,
            domain_min=-1.0,
            domain_max=4.0,
            volume_a=1.0,
            volume_b=1.0,
            initial_a=0.0,
            initial_b=0.0,
            threshold_column_a=None,
            threshold_column_b=None,
            value_column_a=None,
            value_column_b=None,
        )

    def test_pair_and_matrix_agree_for_every_metric(self) -> None:
        curves = [
            self.write_curve("a", [(-1.0, 0), (0.0, 2), (2.0, 1)]),
            self.write_curve("b", [(-0.5, 1), (1.0, 3), (3.0, 0)]),
            self.write_curve("c", [(0.0, 1), (1.0, 1), (2.5, 4)]),
        ]
        for metric in ("l1", "l2", "linf"):
            matrix, columns = distance.compute_pairwise_matrix(
                curves,
                metric=metric,
                domain_min=-1.0,
                domain_max=4.0,
                volumes=[1.0, 1.0, 1.0],
                initial_value=0.0,
                threshold_column=None,
                value_column=None,
            )
            self.assertEqual(columns, ["betti0", "betti0", "betti0"])
            for row in range(len(curves)):
                self.assertEqual(matrix.get(row, row), 0.0)
                for column in range(row + 1, len(curves)):
                    expected = getattr(self.pair(curves[row], curves[column]), metric)
                    self.assertAlmostEqual(matrix.get(row, column), expected)
                    self.assertAlmostEqual(matrix.get(column, row), expected)

    def test_gzip_and_plain_inputs_agree(self) -> None:
        rows = [(-2.0, 0), (0.0, 4), (2.0, 1)]
        plain = self.write_curve("plain", rows)
        compressed = self.write_curve("compressed", rows, compressed=True)
        result = self.pair(plain, compressed)
        self.assertEqual(result.l1, 0.0)
        self.assertEqual(result.l2, 0.0)
        self.assertEqual(result.linf, 0.0)

    def test_events_at_domain_endpoints_use_right_continuous_semantics(self) -> None:
        first = self.write_curve("first", [(0.0, 2), (1.0, 4), (2.0, 100)])
        second = self.write_curve("second", [(0.0, 1), (1.0, 3), (2.0, 0)])
        result = distance.compute_distances(
            first,
            second,
            domain_min=0.0,
            domain_max=2.0,
            volume_a=1.0,
            volume_b=1.0,
            initial_a=0.0,
            initial_b=0.0,
            threshold_column_a=None,
            threshold_column_b=None,
            value_column_a=None,
            value_column_b=None,
        )
        self.assertEqual(result.l1, 2.0)
        self.assertEqual(result.l2, math.sqrt(2.0))
        self.assertEqual(result.linf, 1.0)

    def test_observed_domain_is_deliberate_and_uses_union_of_ranges(self) -> None:
        first = self.write_curve("first", [(-3.0, 1), (2.0, 0)])
        second = self.write_curve("second", [(-1.0, 2), (7.0, 0)])
        infos = distance.inspect_curves(
            [first, second], threshold_column=None, value_column=None
        )
        self.assertEqual(
            distance.resolve_domain(
                domain_policy="observed",
                domain_min=None,
                domain_max=None,
                infos=infos,
            ),
            (-3.0, 7.0),
        )
        with self.assertRaisesRegex(ValueError, "requires both"):
            distance.resolve_domain(
                domain_policy="explicit",
                domain_min=None,
                domain_max=None,
                infos=infos,
            )

    def test_malformed_duplicate_threshold_is_rejected(self) -> None:
        malformed = self.write_curve("bad", [(0.0, 1), (0.0, 2)])
        with self.assertRaisesRegex(distance.CurveFormatError, "strictly increasing"):
            distance.inspect_curve(
                malformed, threshold_column=None, value_column=None
            )

    def test_atomic_text_output_preserves_previous_file_on_failure(self) -> None:
        output = self.root / "matrix.csv"
        output.write_text("complete\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            with distance.atomic_text_output(output) as temporary:
                temporary.write("partial\n")
                raise RuntimeError("simulated failure")
        self.assertEqual(output.read_text(encoding="utf-8"), "complete\n")

    def test_matrix_command_writes_scientific_metadata(self) -> None:
        curves = [
            self.write_curve("a", [(0.0, 1), (2.0, 0)]),
            self.write_curve("b", [(0.0, 2), (2.0, 0)]),
        ]
        matrix_path = self.root / "distances.csv"
        metadata_path = self.root / "metadata.json"
        status = distance.main(
            [
                *map(str, curves),
                "--matrix",
                "--domain-min",
                "0",
                "--domain-max",
                "2",
                "--no-heatmap",
                "--matrix-output",
                str(matrix_path),
                "--metadata-output",
                str(metadata_path),
            ]
        )
        self.assertEqual(status, 0)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["algorithm"]["maximum_open_curve_files"], 2)
        self.assertEqual(metadata["domain"], [0.0, 2.0])
        self.assertTrue(metadata["scientific_notes"])


if __name__ == "__main__":
    unittest.main()
