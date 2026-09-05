#!/usr/bin/env python3
"""Exact distances between sparse, right-continuous Betti step curves.

A row ``(t, b)`` means that the curve has value ``b`` from ``t`` up to, but
not including, its next change point. The value before the first row is given
by ``--initial*`` and defaults to zero.

The comparison interval is never guessed silently. Supply both
``--domain-min`` and ``--domain-max``, or request ``--domain-policy observed``.
Pair mode uses constant auxiliary memory. Matrix mode replays pairs, keeps at
most two inputs open, and stores one flat upper triangle of doubles.

These distances compare Betti counts, not persistence pairings. Different
barcodes can have distance zero. Results also depend on the finite domain,
threshold coordinate, and normalization, so raw axes should be compared only
when their units and calibration agree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
import tempfile
from array import array
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Iterator, Sequence


SCIENTIFIC_NOTES = (
    "Betti-curve distances do not determine persistence-barcode distances; "
    "different barcodes may have the same Betti curve.",
    "Distances depend on the finite comparison domain and threshold coordinate.",
    "Volume normalization changes counts to densities but does not align "
    "threshold axes or physical units.",
)


class CurveFormatError(ValueError):
    """Raised when a curve CSV is malformed."""


@dataclass(frozen=True, slots=True)
class CurveEvent:
    threshold: float
    value: float


@dataclass(frozen=True, slots=True)
class CurveInfo:
    value_column: str
    first_threshold: float | None
    last_threshold: float | None
    row_count: int


@dataclass(frozen=True, slots=True)
class DistanceResult:
    l1: float
    l1_mean: float
    l2: float
    l2_rms: float
    linf: float
    domain_min: float
    domain_max: float
    column_a: str
    column_b: str


class CompensatedSum:
    """Neumaier compensated summation with constant memory."""

    __slots__ = ("_sum", "_correction")

    def __init__(self) -> None:
        self._sum = 0.0
        self._correction = 0.0

    def add(self, value: float) -> None:
        updated = self._sum + value
        if abs(self._sum) >= abs(value):
            self._correction += (self._sum - updated) + value
        else:
            self._correction += (value - updated) + self._sum
        self._sum = updated

    @property
    def value(self) -> float:
        return self._sum + self._correction


class TriangularDistanceMatrix:
    """Symmetric zero-diagonal matrix stored as one flat upper triangle."""

    __slots__ = ("size", "_values")

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("matrix size must be nonnegative")
        self.size = size
        self._values = array("d", [0.0]) * (size * (size - 1) // 2)

    def _offset(self, row: int, column: int) -> int:
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise IndexError("matrix index out of range")
        if row == column:
            raise IndexError("the diagonal is implicit")
        if row > column:
            row, column = column, row
        return row * (2 * self.size - row - 1) // 2 + (column - row - 1)

    def set(self, row: int, column: int, value: float) -> None:
        if row == column:
            if value != 0.0:
                raise ValueError("the implicit diagonal must be zero")
            return
        self._values[self._offset(row, column)] = value

    def get(self, row: int, column: int) -> float:
        if row == column:
            if not 0 <= row < self.size:
                raise IndexError("matrix index out of range")
            return 0.0
        return self._values[self._offset(row, column)]

    def row(self, row: int) -> Iterator[float]:
        for column in range(self.size):
            yield self.get(row, column)

    def to_dense(self) -> list[list[float]]:
        return [list(self.row(row)) for row in range(self.size)]


def open_text(path: Path) -> IO[str]:
    """Open a plain or gzip-compressed UTF-8 CSV for streaming."""

    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(
        mode="rt", encoding="utf-8-sig", newline="", buffering=1024 * 1024
    )


def _sync_parent(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_text_output(path: Path) -> Iterator[IO[str]]:
    """Yield a text file and replace ``path`` only after a successful write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        _sync_parent(path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_render_path(path: Path) -> Iterator[Path]:
    """Yield a same-directory image path, then replace the result atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix or ".image"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        with temporary_path.open("rb") as rendered:
            os.fsync(rendered.fileno())
        os.replace(temporary_path, path)
        _sync_parent(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def normalized_header_name(name: str) -> str:
    return name.strip().lower()


def resolve_column(
    header: Sequence[str], requested: str | None, *, kind: str, path: Path
) -> tuple[int, str]:
    cleaned = [name.strip() for name in header]
    normalized = [normalized_header_name(name) for name in cleaned]
    if requested is not None:
        key = normalized_header_name(requested)
        matches = [index for index, name in enumerate(normalized) if name == key]
        if len(matches) != 1:
            raise CurveFormatError(
                f"{path}: expected exactly one column named {requested!r}; "
                f"header is {cleaned!r}"
            )
        index = matches[0]
        return index, cleaned[index]
    if kind == "threshold":
        aliases = {"threshold", "filtration", "filtration_value", "value"}
        matches = [index for index, name in enumerate(normalized) if name in aliases]
    else:
        matches = []
        for index, name in enumerate(normalized):
            compact = re.sub(r"[^a-z0-9]", "", name)
            if compact.startswith("betti") or compact.startswith("beta"):
                matches.append(index)
    if len(matches) != 1:
        description = "threshold" if kind == "threshold" else "Betti-value"
        raise CurveFormatError(
            f"{path}: could not identify exactly one {description} column from "
            f"header {cleaned!r}; specify it explicitly"
        )
    index = matches[0]
    return index, cleaned[index]


def betti_dimension(column: str) -> int | None:
    compact = re.sub(r"[^a-z0-9]", "", column.lower())
    for prefix in ("betti", "beta"):
        if compact.startswith(prefix):
            suffix = compact[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


class CurveEventReader:
    """Streaming reader that validates strictly increasing change points."""

    def __init__(
        self,
        path: Path,
        *,
        threshold_column: str | None,
        value_column: str | None,
    ) -> None:
        self.path = path
        self._threshold_column_request = threshold_column
        self._value_column_request = value_column
        self._file: IO[str] | None = None
        self._rows: Iterator[list[str]] | None = None
        self._threshold_index = -1
        self._value_index = -1
        self._line_number = 1
        self._previous_threshold: float | None = None
        self.threshold_column = ""
        self.value_column = ""

    def __enter__(self) -> CurveEventReader:
        try:
            self._file = open_text(self.path)
        except OSError as error:
            raise OSError(f"could not open {self.path}: {error}") from error
        try:
            self._rows = iter(csv.reader(self._file))
            try:
                header = next(self._rows)
            except StopIteration as error:
                raise CurveFormatError(f"{self.path}: file is empty") from error
            if not header:
                raise CurveFormatError(f"{self.path}: CSV header is empty")
            self._threshold_index, self.threshold_column = resolve_column(
                header,
                self._threshold_column_request,
                kind="threshold",
                path=self.path,
            )
            self._value_index, self.value_column = resolve_column(
                header,
                self._value_column_request,
                kind="value",
                path=self.path,
            )
            if self._threshold_index == self._value_index:
                raise CurveFormatError(
                    f"{self.path}: threshold and Betti value resolved to the same column"
                )
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def next_event(self) -> CurveEvent | None:
        if self._rows is None:
            raise RuntimeError("CurveEventReader must be used as a context manager")
        required_width = max(self._threshold_index, self._value_index) + 1
        for row in self._rows:
            self._line_number += 1
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < required_width:
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: expected at least "
                    f"{required_width} columns, found {len(row)}"
                )
            try:
                threshold = float(row[self._threshold_index].strip())
            except ValueError as error:
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: invalid threshold "
                    f"{row[self._threshold_index]!r}"
                ) from error
            try:
                value = float(row[self._value_index].strip())
            except ValueError as error:
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: invalid Betti value "
                    f"{row[self._value_index]!r}"
                ) from error
            if not math.isfinite(threshold):
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: threshold must be finite"
                )
            if not math.isfinite(value) or value < 0.0:
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: Betti value must be finite "
                    "and nonnegative"
                )
            if (
                self._previous_threshold is not None
                and threshold <= self._previous_threshold
            ):
                raise CurveFormatError(
                    f"{self.path}:{self._line_number}: thresholds must be strictly "
                    f"increasing; found {threshold!r} after "
                    f"{self._previous_threshold!r}"
                )
            self._previous_threshold = threshold
            return CurveEvent(threshold, value)
        return None


def inspect_curve(
    path: Path,
    *,
    threshold_column: str | None,
    value_column: str | None,
) -> CurveInfo:
    """Validate one file and return its observed threshold range."""

    with CurveEventReader(
        path, threshold_column=threshold_column, value_column=value_column
    ) as reader:
        first: float | None = None
        last: float | None = None
        row_count = 0
        while (event := reader.next_event()) is not None:
            if first is None:
                first = event.threshold
            last = event.threshold
            row_count += 1
        return CurveInfo(reader.value_column, first, last, row_count)


def inspect_curves(
    curves: Sequence[Path],
    *,
    threshold_column: str | None,
    value_column: str | None,
) -> list[CurveInfo]:
    infos = [
        inspect_curve(
            path, threshold_column=threshold_column, value_column=value_column
        )
        for path in curves
    ]
    dimensions = {
        dimension
        for info in infos
        if (dimension := betti_dimension(info.value_column)) is not None
    }
    if len(dimensions) > 1:
        columns = ", ".join(sorted({info.value_column for info in infos}))
        raise ValueError(
            f"input mixes homology dimensions ({columns}); "
            "process Betti-0 and Betti-2 separately"
        )
    return infos


def resolve_domain(
    *,
    domain_policy: str,
    domain_min: float | None,
    domain_max: float | None,
    infos: Sequence[CurveInfo],
) -> tuple[float, float]:
    """Resolve an explicit or deliberately requested observed domain."""

    if domain_policy == "explicit":
        if domain_min is None or domain_max is None:
            raise ValueError(
                "explicit domain policy requires both --domain-min and --domain-max; "
                "use --domain-policy observed only when the observed range is "
                "scientifically appropriate"
            )
        left, right = domain_min, domain_max
    elif domain_policy == "observed":
        if domain_min is not None or domain_max is not None:
            raise ValueError(
                "do not combine --domain-policy observed with explicit domain bounds"
            )
        firsts = [info.first_threshold for info in infos if info.first_threshold is not None]
        lasts = [info.last_threshold for info in infos if info.last_threshold is not None]
        if not firsts or not lasts:
            raise ValueError("observed domain is undefined because every curve is empty")
        left, right = min(firsts), max(lasts)
    else:
        raise ValueError(f"unknown domain policy {domain_policy!r}")
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("domain bounds must be finite")
    if right <= left:
        raise ValueError(
            "comparison domain must have positive width; supply explicit bounds "
            "when the observed range has only one change point"
        )
    return left, right


def consume_through(
    reader: CurveEventReader,
    event: CurveEvent | None,
    current_value: float,
    threshold: float,
) -> tuple[CurveEvent | None, float]:
    while event is not None and event.threshold <= threshold:
        current_value = event.value
        event = reader.next_event()
    return event, current_value


def compute_distances(
    curve_a: Path,
    curve_b: Path,
    *,
    domain_min: float,
    domain_max: float,
    volume_a: float,
    volume_b: float,
    initial_a: float,
    initial_b: float,
    threshold_column_a: str | None,
    threshold_column_b: str | None,
    value_column_a: str | None,
    value_column_b: str | None,
) -> DistanceResult:
    """Compute exact L1, L2, and L-infinity step-curve distances."""

    if not math.isfinite(domain_min) or not math.isfinite(domain_max):
        raise ValueError("domain bounds must be finite")
    if domain_max <= domain_min:
        raise ValueError("domain_max must be greater than domain_min")
    for label, volume in (("volume_a", volume_a), ("volume_b", volume_b)):
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    for label, value in (("initial_a", initial_a), ("initial_b", initial_b)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")

    with CurveEventReader(
        curve_a,
        threshold_column=threshold_column_a,
        value_column=value_column_a,
    ) as reader_a, CurveEventReader(
        curve_b,
        threshold_column=threshold_column_b,
        value_column=value_column_b,
    ) as reader_b:
        dimension_a = betti_dimension(reader_a.value_column)
        dimension_b = betti_dimension(reader_b.value_column)
        if (
            dimension_a is not None
            and dimension_b is not None
            and dimension_a != dimension_b
        ):
            raise ValueError(
                f"refusing to compare {reader_a.value_column!r} with "
                f"{reader_b.value_column!r}; compare matching dimensions"
            )
        event_a = reader_a.next_event()
        event_b = reader_b.next_event()
        current_a = initial_a
        current_b = initial_b
        event_a, current_a = consume_through(reader_a, event_a, current_a, domain_min)
        event_b, current_b = consume_through(reader_b, event_b, current_b, domain_min)

        l1_sum = CompensatedSum()
        l2_squared_sum = CompensatedSum()
        linf = 0.0
        left = domain_min
        while left < domain_max:
            right = domain_max
            if event_a is not None and event_a.threshold < right:
                right = event_a.threshold
            if event_b is not None and event_b.threshold < right:
                right = event_b.threshold
            width = right - left
            if width < 0.0:
                raise RuntimeError("internal error: change points became unsorted")
            if width > 0.0:
                difference = current_a / volume_a - current_b / volume_b
                absolute_difference = abs(difference)
                l1_sum.add(absolute_difference * width)
                l2_squared_sum.add(difference * difference * width)
                linf = max(linf, absolute_difference)
            left = right
            if left >= domain_max:
                break
            event_a, current_a = consume_through(reader_a, event_a, current_a, left)
            event_b, current_b = consume_through(reader_b, event_b, current_b, left)

        width = domain_max - domain_min
        l1 = l1_sum.value
        l2_squared = max(0.0, l2_squared_sum.value)
        return DistanceResult(
            l1=l1,
            l1_mean=l1 / width,
            l2=math.sqrt(l2_squared),
            l2_rms=math.sqrt(l2_squared / width),
            linf=linf,
            domain_min=domain_min,
            domain_max=domain_max,
            column_a=reader_a.value_column,
            column_b=reader_b.value_column,
        )


def compute_pairwise_matrix(
    curves: Sequence[Path],
    *,
    metric: str,
    domain_min: float,
    domain_max: float,
    volumes: Sequence[float],
    initial_value: float,
    threshold_column: str | None,
    value_column: str | None,
) -> tuple[TriangularDistanceMatrix, list[str]]:
    """Compute an exact matrix with two open inputs and triangular storage.

    Every pair is replayed independently. This bounds open files and live
    curve state regardless of cohort size. The tradeoff is repeated parsing,
    with ``Theta(N R + N^2)`` leading work for ``N`` curves and ``R`` rows.
    """

    curve_count = len(curves)
    if curve_count < 2:
        raise ValueError("at least two curves are required")
    if metric not in {"l1", "l2", "linf"}:
        raise ValueError("matrix mode requires metric l1, l2, or linf")
    if len(volumes) != curve_count:
        raise ValueError("the number of volumes must match the number of curves")
    if not math.isfinite(initial_value) or initial_value < 0.0:
        raise ValueError("initial value must be finite and nonnegative")
    for index, volume in enumerate(volumes):
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"volume for {curves[index]} must be finite and positive")

    infos = inspect_curves(
        curves, threshold_column=threshold_column, value_column=value_column
    )
    matrix = TriangularDistanceMatrix(curve_count)
    for row in range(curve_count):
        for column in range(row + 1, curve_count):
            result = compute_distances(
                curves[row],
                curves[column],
                domain_min=domain_min,
                domain_max=domain_max,
                volume_a=volumes[row],
                volume_b=volumes[column],
                initial_a=initial_value,
                initial_b=initial_value,
                threshold_column_a=threshold_column,
                threshold_column_b=threshold_column,
                value_column_a=value_column,
                value_column_b=value_column,
            )
            matrix.set(row, column, getattr(result, metric))
    return matrix, [info.value_column for info in infos]


def curve_label(path: Path, mode: str) -> str:
    if mode == "path":
        return str(path)
    if mode == "name":
        return path.name
    name = path.name
    return name[:-7] if name.lower().endswith(".csv.gz") else path.stem


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(path))
    ]


def has_betti_curve_header(
    path: Path,
    *,
    threshold_column: str | None,
    value_column: str | None,
) -> bool:
    try:
        with open_text(path) as input_file:
            header = next(csv.reader(input_file), None)
        if not header:
            return False
        resolve_column(header, threshold_column, kind="threshold", path=path)
        resolve_column(header, value_column, kind="value", path=path)
        return True
    except (CurveFormatError, OSError):
        return False


def collect_curves(
    positional: Sequence[Path],
    *,
    directory: Path | None,
    patterns: Sequence[str] | None,
    recursive: bool,
    excluded_paths: Sequence[Path],
    threshold_column: str | None,
    value_column: str | None,
) -> list[Path]:
    if directory is not None and positional:
        raise ValueError("provide either positional files or --directory, not both")
    if directory is None:
        curves = list(positional)
    else:
        if not directory.is_dir():
            raise ValueError(f"curve directory does not exist: {directory}")
        curves: list[Path] = []
        for pattern in patterns or ("*.csv", "*.csv.gz"):
            iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
            curves.extend(iterator)
        curves = sorted(set(curves), key=natural_sort_key)
        curves = [
            path
            for path in curves
            if path.is_file()
            and has_betti_curve_header(
                path,
                threshold_column=threshold_column,
                value_column=value_column,
            )
        ]
    excluded = {path.resolve(strict=False) for path in excluded_paths}
    curves = [path for path in curves if path.resolve(strict=False) not in excluded]
    if len(curves) < 2:
        raise ValueError(f"found {len(curves)} curve file(s); at least two are required")
    missing = [path for path in curves if not path.is_file()]
    if missing:
        raise ValueError(f"curve file does not exist: {missing[0]}")
    return curves


def load_volume_map(path: Path, curves: Sequence[Path], labels: Sequence[str]) -> list[float]:
    """Load per-curve normalizers from columns ``curve,volume``."""

    with open_text(path) as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise CurveFormatError(f"{path}: volume-map CSV has no header")
        normalized = {
            name.strip().lower(): name for name in reader.fieldnames if name is not None
        }
        if "curve" not in normalized or "volume" not in normalized:
            raise CurveFormatError(f"{path}: expected columns curve,volume")
        mapping: dict[str, float] = {}
        for line_number, row in enumerate(reader, start=2):
            key = (row.get(normalized["curve"]) or "").strip()
            raw_volume = (row.get(normalized["volume"]) or "").strip()
            if not key:
                raise CurveFormatError(f"{path}:{line_number}: curve is empty")
            try:
                volume = float(raw_volume)
            except ValueError as error:
                raise CurveFormatError(
                    f"{path}:{line_number}: invalid volume {raw_volume!r}"
                ) from error
            if not math.isfinite(volume) or volume <= 0.0:
                raise CurveFormatError(
                    f"{path}:{line_number}: volume must be finite and positive"
                )
            if key in mapping:
                raise CurveFormatError(f"{path}:{line_number}: duplicate key {key!r}")
            mapping[key] = volume

    volumes: list[float] = []
    for curve, label in zip(curves, labels, strict=True):
        candidates = (str(curve), str(curve.resolve()), curve.name, label)
        matches = {mapping[key] for key in candidates if key in mapping}
        if not matches:
            raise CurveFormatError(f"{path}: no volume found for {curve}")
        if len(matches) > 1:
            raise CurveFormatError(f"{path}: conflicting volumes found for {curve}")
        volumes.append(matches.pop())
    return volumes


def write_distance_matrix(
    path: Path, labels: Sequence[str], matrix: TriangularDistanceMatrix
) -> None:
    if len(labels) != matrix.size:
        raise ValueError("label count does not match matrix size")
    with atomic_text_output(path) as output:
        writer = csv.writer(output)
        writer.writerow(["curve", *labels])
        for label, row in zip(labels, range(matrix.size), strict=True):
            writer.writerow(
                [label, *(format(value, ".17g") for value in matrix.row(row))]
            )


def write_json(path: Path, payload: object) -> None:
    with atomic_text_output(path) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def metric_display_name(metric: str) -> str:
    return {"l1": "L1", "l2": "L2", "linf": "L-infinity"}[metric]


def plot_distance_heatmap(
    path: Path,
    labels: Sequence[str],
    matrix: TriangularDistanceMatrix,
    *,
    metric: str,
    title: str | None,
    annotate: bool | None,
    dpi: int,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(
            Path(tempfile.gettempdir()) / "betti-curve-matplotlib"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("heatmap output requires Matplotlib") from error

    dense = matrix.to_dense()
    count = len(labels)
    show_annotations = count <= 20 if annotate is None else annotate
    side = min(24.0, max(6.5, 3.5 + 0.48 * count))
    figure, axis = plt.subplots(figsize=(side, side))
    image = axis.imshow(dense, cmap="viridis", interpolation="nearest", vmin=0.0)
    axis.set_xticks(range(count), labels=labels)
    axis.set_yticks(range(count), labels=labels)
    tick_size = max(5.0, min(10.0, 130.0 / max(1, count)))
    axis.tick_params(axis="both", labelsize=tick_size)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axis.set_title(title or f"Betti-curve {metric_display_name(metric)} distances")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if show_annotations:
        maximum = max(max(row) for row in dense)
        for row in range(count):
            for column in range(count):
                value = dense[row][column]
                color = "white" if maximum == 0.0 or value / maximum < 0.55 else "black"
                axis.text(
                    column,
                    row,
                    "0" if value == 0.0 else format(value, ".3g"),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=max(5.0, tick_size - 1.0),
                )
    figure.tight_layout()
    with atomic_render_path(path) as temporary_path:
        figure.savefig(temporary_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact L1, L2, or L-infinity distances between sparse "
            "Betti step curves. A finite comparison domain is required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python betti_curve_distance.py a.csv b.csv \\
      --domain-min 0 --domain-max 65535 --threshold-unit intensity

  python betti_curve_distance.py a.csv b.csv --domain-policy observed

  python betti_curve_distance.py --directory ./curves --no-heatmap \\
      --domain-min 0 --domain-max 1 --volumes-csv volumes.csv \\
      --normalization-unit mm^3

Volume map: curve,volume. Process Betti-0 and Betti-2 separately.
""",
    )
    parser.add_argument("curves", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--metric", choices=("l1", "l2", "linf", "all"), default="l1"
    )
    parser.add_argument(
        "--domain-policy", choices=("explicit", "observed"), default="explicit"
    )
    parser.add_argument("--domain-min", type=float)
    parser.add_argument("--domain-max", type=float)
    parser.add_argument("--threshold-unit", default="unspecified")
    parser.add_argument("--volume-a", type=positive_float)
    parser.add_argument("--volume-b", type=positive_float)
    parser.add_argument("--normalization-unit")
    parser.add_argument("--initial-a", type=nonnegative_float, default=0.0)
    parser.add_argument("--initial-b", type=nonnegative_float, default=0.0)
    parser.add_argument("--initial", type=nonnegative_float, default=0.0)
    parser.add_argument("--threshold-column")
    parser.add_argument("--value-column")
    parser.add_argument("--threshold-column-a")
    parser.add_argument("--threshold-column-b")
    parser.add_argument("--value-column-a")
    parser.add_argument("--value-column-b")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--number-only", action="store_true")
    parser.add_argument("--volumes-csv", type=Path)
    parser.add_argument(
        "--label-mode", choices=("stem", "name", "path"), default="stem"
    )
    parser.add_argument(
        "--matrix-output", type=Path, default=Path("betti_pairwise_distances.csv")
    )
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--heatmap-output", type=Path, default=Path("betti_pairwise_heatmap.png")
    )
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--heatmap-max-curves", type=positive_int, default=200)
    annotation = parser.add_mutually_exclusive_group()
    annotation.add_argument("--annotate", dest="annotate", action="store_true")
    annotation.add_argument("--no-annotate", dest="annotate", action="store_false")
    parser.set_defaults(annotate=None)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=positive_int, default=300)
    return parser


def selected_metrics(result: DistanceResult, metric: str) -> dict[str, float]:
    values = {"l1": result.l1, "l2": result.l2, "linf": result.linf}
    return values if metric == "all" else {metric: values[metric]}


def run_pair_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if len(args.curves) != 2:
        parser.error("pair mode requires exactly two curve files")
    if args.number_only and args.metric == "all":
        parser.error("--number-only cannot be combined with --metric all")
    if args.number_only and args.format != "text":
        parser.error("--number-only cannot be combined with --format json")
    if (args.volume_a is None) != (args.volume_b is None):
        parser.error("--volume-a and --volume-b must be supplied together")
    if args.volumes_csv is not None:
        parser.error("use --volume-a and --volume-b in pair mode")
    if args.volume_a is not None and not args.normalization_unit:
        parser.error("normalization requires --normalization-unit")

    curve_a, curve_b = args.curves
    infos = [
        inspect_curve(
            curve_a,
            threshold_column=args.threshold_column_a or args.threshold_column,
            value_column=args.value_column_a or args.value_column,
        ),
        inspect_curve(
            curve_b,
            threshold_column=args.threshold_column_b or args.threshold_column,
            value_column=args.value_column_b or args.value_column,
        ),
    ]
    domain_min, domain_max = resolve_domain(
        domain_policy=args.domain_policy,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        infos=infos,
    )
    volume_a = 1.0 if args.volume_a is None else args.volume_a
    volume_b = 1.0 if args.volume_b is None else args.volume_b
    result = compute_distances(
        curve_a,
        curve_b,
        domain_min=domain_min,
        domain_max=domain_max,
        volume_a=volume_a,
        volume_b=volume_b,
        initial_a=args.initial_a,
        initial_b=args.initial_b,
        threshold_column_a=args.threshold_column_a or args.threshold_column,
        threshold_column_b=args.threshold_column_b or args.threshold_column,
        value_column_a=args.value_column_a or args.value_column,
        value_column_b=args.value_column_b or args.value_column,
    )
    metrics = selected_metrics(result, args.metric)
    if args.number_only:
        print(format(next(iter(metrics.values())), ".17g"))
        return 0
    normalization = "none" if args.volume_a is None else "supplied normalizer"
    payload = asdict(result) | {
        "curve_a": str(curve_a),
        "curve_b": str(curve_b),
        "domain_policy": args.domain_policy,
        "threshold_unit": args.threshold_unit,
        "normalization": normalization,
        "normalization_unit": args.normalization_unit,
        "volume_a": volume_a,
        "volume_b": volume_b,
        "reported_metric": args.metric,
        "scientific_notes": SCIENTIFIC_NOTES,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"curve A:       {curve_a} [{result.column_a}]")
    print(f"curve B:       {curve_b} [{result.column_b}]")
    print(f"domain:        [{domain_min:.17g}, {domain_max:.17g}] ({args.domain_policy})")
    print(f"threshold unit: {args.threshold_unit}")
    print(f"normalization: {normalization}")
    for name, value in metrics.items():
        print(f"{name} distance:   {value:.17g}")
    if args.metric in {"l1", "all"}:
        print(f"l1 mean:       {result.l1_mean:.17g}")
    if args.metric in {"l2", "all"}:
        print(f"l2 RMS:        {result.l2_rms:.17g}")
    print(f"note:          {SCIENTIFIC_NOTES[0]}")
    return 0


def run_matrix_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.metric == "all":
        parser.error("matrix mode requires one metric")
    if args.number_only or args.format != "text":
        parser.error("number-only and JSON format are pair-mode options")
    if args.volume_a is not None or args.volume_b is not None:
        parser.error("use --volumes-csv in matrix mode")
    if args.initial_a != 0.0 or args.initial_b != 0.0:
        parser.error("use --initial in matrix mode")
    if args.volumes_csv is not None and not args.normalization_unit:
        parser.error("normalization requires --normalization-unit")
    if any(
        value is not None
        for value in (
            args.threshold_column_a,
            args.threshold_column_b,
            args.value_column_a,
            args.value_column_b,
        )
    ):
        parser.error("use common column options in matrix mode")

    metadata_output = args.metadata_output or Path(f"{args.matrix_output}.metadata.json")
    curves = collect_curves(
        args.curves,
        directory=args.directory,
        patterns=args.pattern,
        recursive=args.recursive,
        excluded_paths=[
            args.matrix_output,
            metadata_output,
            *([] if args.volumes_csv is None else [args.volumes_csv]),
        ],
        threshold_column=args.threshold_column,
        value_column=args.value_column,
    )
    labels = [curve_label(path, args.label_mode) for path in curves]
    duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"duplicate matrix label {duplicates[0]!r}; choose another label mode"
        )
    infos = inspect_curves(
        curves, threshold_column=args.threshold_column, value_column=args.value_column
    )
    domain_min, domain_max = resolve_domain(
        domain_policy=args.domain_policy,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        infos=infos,
    )
    volumes = (
        [1.0] * len(curves)
        if args.volumes_csv is None
        else load_volume_map(args.volumes_csv, curves, labels)
    )
    matrix, columns = compute_pairwise_matrix(
        curves,
        metric=args.metric,
        domain_min=domain_min,
        domain_max=domain_max,
        volumes=volumes,
        initial_value=args.initial,
        threshold_column=args.threshold_column,
        value_column=args.value_column,
    )
    write_distance_matrix(args.matrix_output, labels, matrix)
    heatmap_written = False
    if not args.no_heatmap and len(curves) <= args.heatmap_max_curves:
        plot_distance_heatmap(
            args.heatmap_output,
            labels,
            matrix,
            metric=args.metric,
            title=args.title,
            annotate=args.annotate,
            dpi=args.dpi,
        )
        heatmap_written = True
    metadata = {
        "schema_version": 1,
        "curves": [str(path) for path in curves],
        "labels": labels,
        "value_columns": columns,
        "metric": args.metric,
        "domain": [domain_min, domain_max],
        "domain_policy": args.domain_policy,
        "threshold_unit": args.threshold_unit,
        "initial_value": args.initial,
        "normalization": "none" if args.volumes_csv is None else str(args.volumes_csv),
        "normalization_unit": args.normalization_unit,
        "matrix_output": str(args.matrix_output),
        "heatmap_output": str(args.heatmap_output) if heatmap_written else None,
        "algorithm": {
            "maximum_open_curve_files": 2,
            "matrix_storage": "flat upper triangle of float64 values",
            "pair_count": len(curves) * (len(curves) - 1) // 2,
        },
        "scientific_notes": SCIENTIFIC_NOTES,
    }
    write_json(metadata_output, metadata)
    print(f"curves:        {len(curves)} [{', '.join(sorted(set(columns)))}]")
    print(f"metric:        {metric_display_name(args.metric)}")
    print(f"domain:        [{domain_min:.17g}, {domain_max:.17g}] ({args.domain_policy})")
    print(f"matrix:        {args.matrix_output}")
    print(f"metadata:      {metadata_output}")
    if heatmap_written:
        print(f"heatmap:       {args.heatmap_output}")
    elif not args.no_heatmap:
        print(f"heatmap:       skipped above {args.heatmap_max_curves} curves")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix_mode = args.matrix or args.directory is not None or len(args.curves) > 2
    try:
        return run_matrix_mode(args, parser) if matrix_mode else run_pair_mode(args, parser)
    except (OSError, CurveFormatError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
