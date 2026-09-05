#!/usr/bin/env python3
"""Independent black-box validation for the betti_curves Rust executable."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[1]
_binary_setting = Path(
    os.environ.get("BETTI_CURVES_BINARY", "target/debug/betti_curves")
)
BINARY = (
    _binary_setting
    if _binary_setting.is_absolute()
    else PROJECT / _binary_setting
).resolve()


def structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    raise ValueError(connectivity)


def beta0(volume: np.ndarray, threshold: int, connectivity: int) -> int:
    _, count = ndimage.label(volume <= threshold, structure=structure(connectivity))
    return int(count)


def beta2(volume: np.ndarray, threshold: int, background_connectivity: int) -> int:
    labels, count = ndimage.label(
        volume > threshold, structure=structure(background_connectivity)
    )
    if count == 0:
        return 0

    boundary = np.concatenate(
        [
            labels[0, :, :].ravel(),
            labels[-1, :, :].ravel(),
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    outside = np.unique(boundary)
    outside = outside[outside != 0]
    return int(count - len(outside))


def neighbor_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    """Return voxel-neighbor offsets without using the Rust implementation."""

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                if connectivity == 6 and abs(dz) + abs(dy) + abs(dx) != 1:
                    continue
                if connectivity not in (6, 26):
                    raise ValueError(connectivity)
                offsets.append((dz, dy, dx))
    return tuple(offsets)


def exact_number(value: np.generic | int | float) -> int | float:
    """Convert a NumPy scalar and normalize the two floating zero encodings."""

    converted = value.item() if isinstance(value, np.generic) else value
    if isinstance(converted, float):
        if not math.isfinite(converted):
            raise ValueError("the oracle requires finite scalar values")
        return 0.0 if converted == 0.0 else converted
    return int(converted)


class OracleUnionFind:
    """Small, direct union-find used only by the independent pairing oracle."""

    def __init__(self, size: int) -> None:
        self.parent = [-1] * size

    def activate(self, item: int) -> None:
        if self.parent[item] != -1:
            raise AssertionError(f"oracle vertex {item} was activated twice")
        self.parent[item] = item

    def active(self, item: int) -> bool:
        return self.parent[item] != -1

    def find(self, item: int) -> int:
        if self.parent[item] == -1:
            raise AssertionError(f"oracle vertex {item} is inactive")
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root


def voxel_neighbors(
    vertex: int,
    shape: tuple[int, int, int],
    offsets: tuple[tuple[int, int, int], ...],
) -> list[int]:
    depth, height, width = shape
    slice_size = height * width
    z, remainder = divmod(vertex, slice_size)
    y, x = divmod(remainder, width)
    result = []
    for dz, dy, dx in offsets:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
            result.append(nz * slice_size + ny * width + nx)
    return result


def h0_barcode_oracle(
    volume: np.ndarray, connectivity: int
) -> Counter[tuple[int | float, int | float | None]]:
    """Compute the positive H0 barcode by a separately written graph sweep.

    Vertices enter in increasing value order. At a merge, the component with
    the smaller ``(birth value, birth vertex)`` pair survives. Zero-length
    intervals are omitted, matching the public CSV contract.
    """

    shape = tuple(int(length) for length in volume.shape)
    if len(shape) != 3:
        raise ValueError("the barcode oracle expects a three-dimensional array")
    values = [exact_number(value) for value in volume.ravel()]
    buckets: dict[int | float, list[int]] = {}
    for vertex, value in enumerate(values):
        buckets.setdefault(value, []).append(vertex)

    union_find = OracleUnionFind(len(values))
    births: list[int | float | None] = [None] * len(values)
    birth_vertices = [-1] * len(values)
    intervals: Counter[tuple[int | float, int | float | None]] = Counter()
    offsets = neighbor_offsets(connectivity)

    for value in sorted(buckets):
        for vertex in buckets[value]:
            union_find.activate(vertex)
            births[vertex] = value
            birth_vertices[vertex] = vertex
            for neighbor in voxel_neighbors(vertex, shape, offsets):
                if not union_find.active(neighbor):
                    continue
                root_a = union_find.find(vertex)
                root_b = union_find.find(neighbor)
                if root_a == root_b:
                    continue
                key_a = (births[root_a], birth_vertices[root_a])
                key_b = (births[root_b], birth_vertices[root_b])
                older, younger = (
                    (root_a, root_b) if key_a <= key_b else (root_b, root_a)
                )
                younger_birth = births[younger]
                assert younger_birth is not None
                if younger_birth != value:
                    intervals[(younger_birth, value)] += 1
                union_find.parent[younger] = older

    roots = {union_find.find(vertex) for vertex in range(len(values))}
    for root in roots:
        birth = births[root]
        assert birth is not None
        intervals[(birth, None)] += 1
    return intervals


def h2_barcode_oracle(
    volume: np.ndarray, background_connectivity: int
) -> Counter[tuple[int | float, int | float | None]]:
    """Compute foreground H2 intervals from decreasing background components.

    One explicit outside root is joined to every boundary voxel. A background
    branch born at ``death`` and merged at ``birth`` contributes the foreground
    interval ``[birth, death)``. The outside branch itself is not an interval.
    """

    shape = tuple(int(length) for length in volume.shape)
    if len(shape) != 3:
        raise ValueError("the barcode oracle expects a three-dimensional array")
    depth, height, width = shape
    values = [exact_number(value) for value in volume.ravel()]
    buckets: dict[int | float, list[int]] = {}
    for vertex, value in enumerate(values):
        buckets.setdefault(value, []).append(vertex)

    outside = len(values)
    union_find = OracleUnionFind(len(values) + 1)
    union_find.activate(outside)
    births: list[int | float | None] = [None] * (len(values) + 1)
    birth_vertices = [-1] * (len(values) + 1)
    intervals: Counter[tuple[int | float, int | float | None]] = Counter()
    offsets = neighbor_offsets(background_connectivity)

    def join(first: int, second: int, merge_value: int | float) -> None:
        root_a = union_find.find(first)
        root_b = union_find.find(second)
        if root_a == root_b:
            return
        if root_a == outside or root_b == outside:
            older = outside
            younger = root_b if root_a == outside else root_a
        else:
            birth_a = births[root_a]
            birth_b = births[root_b]
            assert birth_a is not None and birth_b is not None
            key_a = (birth_a, -birth_vertices[root_a])
            key_b = (birth_b, -birth_vertices[root_b])
            older, younger = (
                (root_a, root_b) if key_a >= key_b else (root_b, root_a)
            )
        younger_birth = births[younger]
        assert younger_birth is not None
        if merge_value != younger_birth:
            intervals[(merge_value, younger_birth)] += 1
        union_find.parent[younger] = older

    slice_size = height * width
    for value in sorted(buckets, reverse=True):
        for vertex in buckets[value]:
            union_find.activate(vertex)
            births[vertex] = value
            birth_vertices[vertex] = vertex
            z, remainder = divmod(vertex, slice_size)
            y, x = divmod(remainder, width)
            if (
                z in (0, depth - 1)
                or y in (0, height - 1)
                or x in (0, width - 1)
            ):
                join(vertex, outside, value)
            for neighbor in voxel_neighbors(vertex, shape, offsets):
                if union_find.active(neighbor):
                    join(vertex, neighbor, value)

    roots = {union_find.find(vertex) for vertex in range(len(values))}
    if roots != {outside}:
        raise AssertionError(f"background oracle left non-outside roots: {roots}")
    return intervals


def write_stack(path: Path, volume: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for z, image in enumerate(volume):
        Image.fromarray(image.astype(np.uint16)).save(path / f"slice_{z:04d}.tif")


def run_mode(
    stack: Path,
    work: Path,
    slab_depth: int,
    foreground_connectivity: int,
    mode: str,
    output: Path | None = None,
) -> str:
    work.mkdir(parents=True, exist_ok=True)
    command = [
        str(BINARY),
        str(stack),
        str(slab_depth),
        str(foreground_connectivity),
        mode,
    ]
    if output is not None:
        command.append(str(output))
    completed = subprocess.run(
        command,
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{mode} failed for {stack.name}, slab={slab_depth}, "
            f"fg={foreground_connectivity}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return completed.stdout


def read_curve(path: Path, field: str) -> list[tuple[int, int]]:
    with path.open(newline="") as handle:
        return [(int(row["threshold"]), int(row[field])) for row in csv.DictReader(handle)]


def read_float_curve(path: Path, field: str) -> list[tuple[float, int]]:
    with path.open(newline="") as handle:
        return [
            (float(row["threshold"]), int(row[field]))
            for row in csv.DictReader(handle)
        ]


def read_intervals(path: Path) -> Counter[tuple[int, int | None]]:
    with path.open(newline="") as handle:
        result: Counter[tuple[int, int | None]] = Counter()
        for row in csv.DictReader(handle):
            death = None if row["death"] == "inf" else int(float(row["death"]))
            result[(int(float(row["birth"])), death)] += 1
        return result


def read_float_intervals(path: Path) -> Counter[tuple[float, float | None]]:
    with path.open(newline="") as handle:
        result: Counter[tuple[float, float | None]] = Counter()
        for row in csv.DictReader(handle):
            death = None if row["death"] == "inf" else float(row["death"])
            result[(float(row["birth"]), death)] += 1
        return result


def validate_branch_tree_nodes(
    path: Path,
    expected_intervals: Counter[tuple[int, int | None]],
    *,
    dimension: int,
) -> None:
    """Check graph, order, and barcode invariants of a branch-node table."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"{path}: branch tree has no root row")

    nodes: dict[int, tuple[int | None, str, str]] = {}
    for row in rows:
        node = int(row["node"])
        parent = None if row["parent"] == "" else int(row["parent"])
        if node in nodes:
            raise AssertionError(f"{path}: duplicate node {node}")
        nodes[node] = (parent, row["birth_value"], row["death_value"])
    if set(nodes) != set(range(len(nodes))):
        raise AssertionError(f"{path}: node IDs are not dense from zero")

    roots = [node for node, (parent, _, _) in nodes.items() if parent is None]
    if len(roots) != 1:
        raise AssertionError(f"{path}: expected one root, found {roots}")
    root = roots[0]

    for node, (parent, _, _) in nodes.items():
        if parent is not None and parent not in nodes:
            raise AssertionError(f"{path}: node {node} has missing parent {parent}")
        seen: set[int] = set()
        current = node
        while current != root:
            if current in seen:
                raise AssertionError(f"{path}: parent cycle through node {current}")
            seen.add(current)
            next_parent = nodes[current][0]
            if next_parent is None:
                raise AssertionError(f"{path}: node {node} does not reach root {root}")
            current = next_parent

    actual_intervals: Counter[tuple[int, int | None]] = Counter()
    if dimension == 0:
        if nodes[root][2] != "inf":
            raise AssertionError(f"{path}: H0 root is not essential")
        for node, (parent, birth_text, death_text) in nodes.items():
            birth = int(birth_text)
            death = None if death_text == "inf" else int(death_text)
            if death is not None and birth >= death:
                raise AssertionError(f"{path}: nonpositive H0 branch at node {node}")
            actual_intervals[(birth, death)] += 1
            if parent is not None:
                parent_birth = int(nodes[parent][1])
                parent_death = (
                    None if nodes[parent][2] == "inf" else int(nodes[parent][2])
                )
                if parent_birth > birth:
                    raise AssertionError(f"{path}: parent is younger than child {node}")
                if parent_death is not None and (
                    death is None or parent_death < death
                ):
                    raise AssertionError(
                        f"{path}: parent interval does not contain child {node}"
                    )
    elif dimension == 2:
        if nodes[root][1:] != ("-inf", "inf"):
            raise AssertionError(f"{path}: H2 root is not the outside branch")
        for node, (parent, birth_text, death_text) in nodes.items():
            if node == root:
                continue
            birth = int(birth_text)
            death = int(death_text)
            if birth >= death:
                raise AssertionError(f"{path}: nonpositive H2 branch at node {node}")
            actual_intervals[(birth, death)] += 1
            if parent is not None and parent != root:
                parent_birth = int(nodes[parent][1])
                parent_death = int(nodes[parent][2])
                if parent_birth > birth or parent_death < death:
                    raise AssertionError(
                        f"{path}: parent interval does not contain child {node}"
                    )
    else:
        raise ValueError(dimension)

    if actual_intervals != expected_intervals:
        raise AssertionError(
            f"{path}: branch intervals differ from independent barcode\n"
            f"actual={actual_intervals}\nexpected={expected_intervals}"
        )


def expand_curve(curve: list[tuple[int, int]], thresholds: list[int]) -> list[int]:
    if not curve:
        raise AssertionError("curve is empty")
    position = 0
    current = curve[0][1]
    result = []
    for threshold in thresholds:
        while position + 1 < len(curve) and curve[position + 1][0] <= threshold:
            position += 1
            current = curve[position][1]
        if curve[0][0] > threshold:
            raise AssertionError(
                f"curve starts at {curve[0][0]}, after requested threshold {threshold}"
            )
        result.append(current)
    return result


def expand_float_curve(
    curve: list[tuple[float, int]], thresholds: list[float]
) -> list[int]:
    if not curve:
        raise AssertionError("curve is empty")
    position = 0
    current = curve[0][1]
    result = []
    for threshold in thresholds:
        while position + 1 < len(curve) and curve[position + 1][0] <= threshold:
            position += 1
            current = curve[position][1]
        if curve[0][0] > threshold:
            raise AssertionError(
                f"curve starts at {curve[0][0]}, after requested threshold {threshold}"
            )
        result.append(current)
    return result


def curve_from_intervals(
    intervals: Counter[tuple[int, int | None]], thresholds: list[int]
) -> list[int]:
    return [
        sum(
            multiplicity
            for (birth, death), multiplicity in intervals.items()
            if birth <= threshold and (death is None or threshold < death)
        )
        for threshold in thresholds
    ]


def expected_curves(
    volume: np.ndarray, foreground_connectivity: int
) -> tuple[list[int], list[int], list[int]]:
    max_value = int(volume.max())
    if max_value <= 255:
        thresholds = list(range(max_value + 1))
    else:
        values = {int(value) for value in np.unique(volume)}
        thresholds = sorted(
            {0, max_value, *values, *(value - 1 for value in values if value > 0)}
        )
    background_connectivity = 26 if foreground_connectivity == 6 else 6
    return (
        thresholds,
        [beta0(volume, t, foreground_connectivity) for t in thresholds],
        [beta2(volume, t, background_connectivity) for t in thresholds],
    )


def validate_case(
    root: Path,
    name: str,
    volume: np.ndarray,
    foreground_connectivity: int,
    slab_depth: int,
    extended: bool,
) -> None:
    stack = root / "stacks" / name
    write_stack(stack, volume)
    work = root / "runs" / f"{name}_fg{foreground_connectivity}_slab{slab_depth}"
    thresholds, expected_h0, expected_h2 = expected_curves(
        volume, foreground_connectivity
    )
    background_connectivity = 26 if foreground_connectivity == 6 else 6
    expected_h0_intervals = h0_barcode_oracle(volume, foreground_connectivity)
    expected_h2_intervals = h2_barcode_oracle(volume, background_connectivity)

    run_mode(stack, work / "baseline", slab_depth, foreground_connectivity, "both")
    baseline_h0 = read_curve(
        work / "baseline/global_betti0_curve_changes.csv", "betti0"
    )
    baseline_h2 = read_curve(
        work / "baseline/global_betti2_curve_changes.csv", "betti2"
    )
    assert expand_curve(baseline_h0, thresholds) == expected_h0, (
        name,
        foreground_connectivity,
        slab_depth,
        "baseline H0",
        baseline_h0,
        expected_h0,
    )
    assert expand_curve(baseline_h2, thresholds) == expected_h2, (
        name,
        foreground_connectivity,
        slab_depth,
        "baseline H2",
        baseline_h2,
        expected_h2,
    )

    curve_modes = [
        ("event-betti0", "event_global_betti0_curve_changes.csv", "betti0", expected_h0),
        (
            "event-betti0-stream",
            "event_global_betti0_stream_curve_changes.csv",
            "betti0",
            expected_h0,
        ),
        ("event-betti2", "event_global_betti2_curve_changes.csv", "betti2", expected_h2),
        (
            "event-betti2-stream",
            "event_global_betti2_stream_curve_changes.csv",
            "betti2",
            expected_h2,
        ),
    ]
    for mode, filename, field, expected in curve_modes:
        mode_work = work / mode
        run_mode(stack, mode_work, slab_depth, foreground_connectivity, mode)
        actual = read_curve(mode_work / filename, field)
        assert expand_curve(actual, thresholds) == expected, (
            name,
            foreground_connectivity,
            slab_depth,
            mode,
            actual,
            expected,
        )

    if not extended:
        return

    h0_work = work / "h0"
    run_mode(stack, h0_work, slab_depth, foreground_connectivity, "h0")
    h0_intervals = read_intervals(h0_work / "h0_persistence.csv")
    assert h0_intervals == expected_h0_intervals, (
        name,
        foreground_connectivity,
        slab_depth,
        "independent H0 barcode oracle",
        h0_intervals,
        expected_h0_intervals,
    )
    assert curve_from_intervals(h0_intervals, thresholds) == expected_h0

    h0_stream_work = work / "h0-stream"
    run_mode(stack, h0_stream_work, slab_depth, foreground_connectivity, "h0-stream")
    h0_stream_intervals = read_intervals(
        h0_stream_work / "h0_persistence_stream.csv"
    )
    assert h0_stream_intervals == h0_intervals

    h2_work = work / "h2"
    run_mode(stack, h2_work, slab_depth, foreground_connectivity, "h2")
    h2_intervals = read_intervals(h2_work / "h2_persistence.csv")
    assert h2_intervals == expected_h2_intervals, (
        name,
        foreground_connectivity,
        slab_depth,
        "independent H2 barcode oracle",
        h2_intervals,
        expected_h2_intervals,
    )
    assert curve_from_intervals(h2_intervals, thresholds) == expected_h2

    h2_stream_work = work / "h2-stream"
    run_mode(stack, h2_stream_work, slab_depth, foreground_connectivity, "h2-stream")
    h2_stream_intervals = read_intervals(
        h2_stream_work / "h2_persistence_stream.csv"
    )
    assert h2_stream_intervals == h2_intervals

    h0_scalar_work = work / "h0-scalar"
    h0_scalar_out = h0_scalar_work / "intervals.csv"
    run_mode(
        stack,
        h0_scalar_work,
        slab_depth,
        foreground_connectivity,
        "h0-scalar",
        h0_scalar_out,
    )
    h0_scalar_intervals = read_intervals(h0_scalar_out)
    assert h0_scalar_intervals == h0_intervals, (
        name,
        foreground_connectivity,
        slab_depth,
        "h0-scalar",
        h0_scalar_intervals,
        h0_intervals,
    )

    h0_scalar_stream_work = work / "h0-scalar-stream"
    h0_scalar_stream_out = h0_scalar_stream_work / "intervals.csv"
    run_mode(
        stack,
        h0_scalar_stream_work,
        slab_depth,
        foreground_connectivity,
        "h0-scalar-stream",
        h0_scalar_stream_out,
    )
    assert read_intervals(h0_scalar_stream_out) == h0_intervals

    h2_scalar_work = work / "h2-scalar"
    h2_scalar_out = h2_scalar_work / "intervals.csv"
    run_mode(
        stack,
        h2_scalar_work,
        slab_depth,
        foreground_connectivity,
        "h2-scalar",
        h2_scalar_out,
    )
    assert read_intervals(h2_scalar_out) == h2_intervals

    h2_scalar_stream_work = work / "h2-scalar-stream"
    h2_scalar_stream_out = h2_scalar_stream_work / "intervals.csv"
    run_mode(
        stack,
        h2_scalar_stream_work,
        slab_depth,
        foreground_connectivity,
        "h2-scalar-stream",
        h2_scalar_stream_out,
    )
    assert read_intervals(h2_scalar_stream_out) == h2_intervals

    for mode, edge_filename, node_filename, dimension, expected_intervals in [
        (
            "branch-tree-h0",
            "h0_merge_tree.csv",
            "h0_merge_tree_nodes.csv",
            0,
            expected_h0_intervals,
        ),
        (
            "branch-tree-h0-stream",
            "h0_merge_tree_stream.csv",
            "h0_merge_tree_stream_nodes.csv",
            0,
            expected_h0_intervals,
        ),
        (
            "branch-tree-h2",
            "h2_merge_tree.csv",
            "h2_merge_tree_nodes.csv",
            2,
            expected_h2_intervals,
        ),
        (
            "branch-tree-h2-stream",
            "h2_merge_tree_stream.csv",
            "h2_merge_tree_stream_nodes.csv",
            2,
            expected_h2_intervals,
        ),
    ]:
        mode_work = work / mode
        run_mode(stack, mode_work, slab_depth, foreground_connectivity, mode)
        assert (mode_work / edge_filename).is_file()
        assert (mode_work / node_filename).is_file()
        validate_branch_tree_nodes(
            mode_work / node_filename,
            expected_intervals,
            dimension=dimension,
        )

    assert (
        (work / "branch-tree-h0/h0_merge_tree.csv").read_text()
        == (work / "branch-tree-h0-stream/h0_merge_tree_stream.csv").read_text()
    )
    assert (
        (work / "branch-tree-h2/h2_merge_tree.csv").read_text()
        == (work / "branch-tree-h2-stream/h2_merge_tree_stream.csv").read_text()
    )
    assert (
        (work / "branch-tree-h0/h0_merge_tree_nodes.csv").read_text()
        == (
            work
            / "branch-tree-h0-stream/h0_merge_tree_stream_nodes.csv"
        ).read_text()
    )
    assert (
        (work / "branch-tree-h2/h2_merge_tree_nodes.csv").read_text()
        == (
            work
            / "branch-tree-h2-stream/h2_merge_tree_stream_nodes.csv"
        ).read_text()
    )


def cases() -> list[tuple[str, np.ndarray]]:
    shell = np.ones((3, 3, 3), dtype=np.uint16)
    shell[1, 1, 1] = 4

    two_cavities = np.ones((4, 5, 5), dtype=np.uint16)
    two_cavities[1, 1, 1] = 4
    two_cavities[2, 3, 3] = 5

    diagonal = np.full((3, 3, 3), 5, dtype=np.uint16)
    diagonal[0, 0, 0] = 1
    diagonal[1, 1, 1] = 1
    diagonal[2, 2, 2] = 1

    bridge = np.full((4, 4, 4), 5, dtype=np.uint16)
    bridge[1, 1, 0] = 1
    bridge[1, 1, 1] = 3
    bridge[1, 1, 2] = 2
    bridge[1, 1, 3] = 1

    crossing_cavity = np.ones((5, 5, 5), dtype=np.uint16)
    crossing_cavity[1:4, 1:4, 1:4] = 5

    diagonal_escape = np.ones((3, 3, 3), dtype=np.uint16)
    diagonal_escape[1, 1, 1] = 5
    diagonal_escape[0, 0, 0] = 5

    endpoint_shell = np.zeros((3, 3, 3), dtype=np.uint16)
    endpoint_shell[1, 1, 1] = np.iinfo(np.uint16).max

    rng = np.random.default_rng(20260730)
    return [
        ("one", np.array([[[3]]], dtype=np.uint16)),
        ("constant", np.full((2, 3, 4), 2, dtype=np.uint16)),
        ("shell", shell),
        ("two_cavities", two_cavities),
        ("diagonal", diagonal),
        ("bridge", bridge),
        ("crossing_cavity", crossing_cavity),
        ("diagonal_escape", diagonal_escape),
        ("u16_endpoints", endpoint_shell),
        ("random_a", rng.integers(0, 6, size=(4, 4, 5), dtype=np.uint16)),
        ("random_b", rng.integers(0, 4, size=(5, 3, 4), dtype=np.uint16)),
        ("random_binary", rng.integers(0, 2, size=(4, 5, 3), dtype=np.uint16)),
        ("flat_2d", rng.integers(0, 5, size=(1, 4, 5), dtype=np.uint16)),
    ]


def validate_natural_slice_order(root: Path) -> None:
    stack = root / "natural_order_stack"
    stack.mkdir(parents=True)
    slices = [
        ("slice1.tif", np.array([[1, 5]], dtype=np.uint16)),
        ("slice2.tif", np.array([[1, 1]], dtype=np.uint16)),
        ("slice10.tif", np.array([[5, 1]], dtype=np.uint16)),
    ]
    for name, image in slices:
        Image.fromarray(image).save(stack / name)

    work = root / "natural_order_run"
    run_mode(stack, work, 2, 6, "betti0")
    curve = read_curve(work / "global_betti0_curve_changes.csv", "betti0")
    assert expand_curve(curve, [1]) == [1], ("natural slice order", curve)
    print("ok natural numeric slice ordering", flush=True)


def validate_recursive_batches(root: Path) -> None:
    batch_root = root / "batch_input"
    datasets = {
        Path("ALN/CX09T1"): cases()[2][1],
        Path("PBL/CX10C"): cases()[5][1],
    }
    for relative, volume in datasets.items():
        write_stack(batch_root / relative, volume)

    modes = [
        (
            "h0-scalar-batch",
            "h0_persistence_scalar.csv",
            "h0",
        ),
        (
            "h0-scalar-batch-stream",
            "h0_persistence_scalar_stream.csv",
            "h0",
        ),
        (
            "h2-scalar-batch",
            "h2_persistence_scalar.csv",
            "h2",
        ),
        (
            "h2-scalar-batch-stream",
            "h2_persistence_scalar_stream.csv",
            "h2",
        ),
    ]

    for mode, filename, dimension in modes:
        mode_work = root / "batch_runs" / mode
        output_root = mode_work / "results"
        run_mode(batch_root, mode_work, 2, 26, mode, output_root)

        for relative, volume in datasets.items():
            output = output_root / relative / filename
            assert output.is_file(), (mode, output)
            intervals = read_intervals(output)
            thresholds, expected_h0, expected_h2 = expected_curves(volume, 26)
            expected = expected_h0 if dimension == "h0" else expected_h2
            assert curve_from_intervals(intervals, thresholds) == expected, (
                mode,
                relative,
                intervals,
                expected,
            )

        print(f"ok recursive hierarchy: {mode}", flush=True)


def validate_float_scalar_modes(root: Path) -> int:
    shell = np.full((3, 3, 3), -2.5, dtype=np.float32)
    shell[1, 1, 1] = 3.25
    rng = np.random.default_rng(20260731)
    random_field = rng.choice(
        np.array([-4.5, -0.25, 0.5, 3.125], dtype=np.float32),
        size=(4, 4, 5),
    )
    checked = 0

    for name, volume in [("float_shell", shell), ("float_random", random_field)]:
        stack = root / "float_stacks" / name
        stack.mkdir(parents=True)
        for z, image in enumerate(volume):
            Image.fromarray(image, mode="F").save(stack / f"slice_{z:04d}.tif")

        thresholds = [float(value) for value in np.unique(volume)]
        for foreground_connectivity in (6, 26):
            background_connectivity = (
                26 if foreground_connectivity == 6 else 6
            )
            expected_h0_intervals = h0_barcode_oracle(
                volume, foreground_connectivity
            )
            expected_h2_intervals = h2_barcode_oracle(
                volume, background_connectivity
            )
            expected_h0 = [
                beta0(volume, threshold, foreground_connectivity)
                for threshold in thresholds
            ]
            expected_h2 = [
                beta2(volume, threshold, background_connectivity)
                for threshold in thresholds
            ]

            for slab_depth in sorted({1, 2, int(volume.shape[0]) + 2}):
                work = (
                    root
                    / "float_runs"
                    / f"{name}_fg{foreground_connectivity}_slab{slab_depth}"
                )

                h0_event_output = work / "event-h0/curve.csv"
                run_mode(
                    stack,
                    work / "event-h0",
                    slab_depth,
                    foreground_connectivity,
                    "event-betti0-scalar-stream",
                    h0_event_output,
                )
                h0_event_curve = read_float_curve(h0_event_output, "betti0")
                assert (
                    expand_float_curve(h0_event_curve, thresholds) == expected_h0
                ), (
                    name,
                    foreground_connectivity,
                    slab_depth,
                    "event-betti0-scalar-stream",
                    h0_event_curve,
                    expected_h0,
                )

                h2_event_output = work / "event-h2/curve.csv"
                run_mode(
                    stack,
                    work / "event-h2",
                    slab_depth,
                    foreground_connectivity,
                    "event-betti2-scalar-stream",
                    h2_event_output,
                )
                h2_event_curve = read_float_curve(h2_event_output, "betti2")
                assert (
                    expand_float_curve(h2_event_curve, thresholds) == expected_h2
                ), (
                    name,
                    foreground_connectivity,
                    slab_depth,
                    "event-betti2-scalar-stream",
                    h2_event_curve,
                    expected_h2,
                )

                h0_output = work / "h0/intervals.csv"
                run_mode(
                    stack,
                    work / "h0",
                    slab_depth,
                    foreground_connectivity,
                    "h0-scalar",
                    h0_output,
                )
                h0_intervals = read_float_intervals(h0_output)
                assert h0_intervals == expected_h0_intervals, (
                    name,
                    foreground_connectivity,
                    slab_depth,
                    "independent scalar H0 barcode oracle",
                    h0_intervals,
                    expected_h0_intervals,
                )
                assert (
                    curve_from_intervals(h0_intervals, thresholds) == expected_h0
                )

                h0_stream_output = work / "h0-stream/intervals.csv"
                run_mode(
                    stack,
                    work / "h0-stream",
                    slab_depth,
                    foreground_connectivity,
                    "h0-scalar-stream",
                    h0_stream_output,
                )
                assert read_float_intervals(h0_stream_output) == h0_intervals

                h2_output = work / "h2/intervals.csv"
                run_mode(
                    stack,
                    work / "h2",
                    slab_depth,
                    foreground_connectivity,
                    "h2-scalar",
                    h2_output,
                )
                h2_intervals = read_float_intervals(h2_output)
                assert h2_intervals == expected_h2_intervals, (
                    name,
                    foreground_connectivity,
                    slab_depth,
                    "independent scalar H2 barcode oracle",
                    h2_intervals,
                    expected_h2_intervals,
                )
                assert (
                    curve_from_intervals(h2_intervals, thresholds) == expected_h2
                )

                h2_stream_output = work / "h2-stream/intervals.csv"
                run_mode(
                    stack,
                    work / "h2-stream",
                    slab_depth,
                    foreground_connectivity,
                    "h2-scalar-stream",
                    h2_stream_output,
                )
                assert read_float_intervals(h2_stream_output) == h2_intervals

                checked += 1
                print(
                    f"ok {name}: fg={foreground_connectivity}, slab={slab_depth}",
                    flush=True,
                )

    return checked


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(f"missing compiled executable: {BINARY}")

    root = Path(tempfile.mkdtemp(prefix="betti_reference_", dir="/tmp"))
    checked = 0
    try:
        for case_index, (name, volume) in enumerate(cases()):
            depths = sorted({1, 2, int(volume.shape[0]), int(volume.shape[0]) + 3})
            for foreground_connectivity in (6, 26):
                for slab_depth in depths:
                    validate_case(
                        root,
                        name,
                        volume,
                        foreground_connectivity,
                        slab_depth,
                        extended=(case_index < 10),
                    )
                    checked += 1
                    print(
                        f"ok {name}: fg={foreground_connectivity}, slab={slab_depth}",
                        flush=True,
                    )
        validate_natural_slice_order(root)
        validate_recursive_batches(root)
        float_checked = validate_float_scalar_modes(root)
        print(
            f"PASS: {checked} integer and {float_checked} float black-box configurations"
        )
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
