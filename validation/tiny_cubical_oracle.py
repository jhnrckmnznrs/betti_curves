#!/usr/bin/env python3
"""Direct cubical boundary-matrix oracle for tiny binary 3D images.

This module does not use connected-component formulas to compute homology. It
constructs every cubical cell, builds boundary matrices over the field with two
elements, and obtains Betti numbers from matrix ranks. It supports the two
digital image constructions used by the manuscript:

* ``vertex``: an image sample is a 0-cell; a higher cell is present only when
  all its vertices are foreground. Its H0 adjacency is 6-connectivity.
* ``top-cell``: an image sample is a unit 3-cell and all of its faces are
  included. Cubes touching at a face, edge, or vertex give 26-connectivity.

The exhaustive test covers every binary 2 x 2 x 2 image. A 3 x 3 x 3 shell
adds a nontrivial H2 check that cannot occur in a 2 x 2 x 2 box.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


# A cell is (x, y, z, direction_bits). Bit 0 is x, bit 1 is y, and bit 2 is z.
Cell = tuple[int, int, int, int]


def cell_dimension(cell: Cell) -> int:
    return cell[3].bit_count()


def cell_boundary(cell: Cell) -> Iterable[Cell]:
    """Yield the codimension-one faces; signs are irrelevant over F2."""

    coordinates = list(cell[:3])
    directions = cell[3]
    for axis in range(3):
        axis_bit = 1 << axis
        if not directions & axis_bit:
            continue
        face_directions = directions ^ axis_bit
        yield coordinates[0], coordinates[1], coordinates[2], face_directions
        upper = coordinates.copy()
        upper[axis] += 1
        yield upper[0], upper[1], upper[2], face_directions


def vertex_construction(foreground: np.ndarray) -> set[Cell]:
    """Build the max-on-vertices subcomplex for a binary foreground mask."""

    if foreground.ndim != 3:
        raise ValueError("expected a three-dimensional foreground mask")
    depth, height, width = map(int, foreground.shape)
    axis_sizes = (width, height, depth)
    cells: set[Cell] = set()

    for directions in range(8):
        coordinate_ranges = [
            range(size - 1 if directions & (1 << axis) else size)
            for axis, size in enumerate(axis_sizes)
        ]
        for x, y, z in product(*coordinate_ranges):
            present = True
            for vertex_mask in range(8):
                if vertex_mask & ~directions:
                    continue
                vx = x + ((vertex_mask >> 0) & 1)
                vy = y + ((vertex_mask >> 1) & 1)
                vz = z + ((vertex_mask >> 2) & 1)
                if not foreground[vz, vy, vx]:
                    present = False
                    break
            if present:
                cells.add((x, y, z, directions))
    return cells


def top_cell_construction(foreground: np.ndarray) -> set[Cell]:
    """Build the face closure of foreground unit cubes."""

    if foreground.ndim != 3:
        raise ValueError("expected a three-dimensional foreground mask")
    cells: set[Cell] = set()
    for z, y, x in np.argwhere(foreground):
        base = (int(x), int(y), int(z))
        for directions in range(8):
            for endpoint_mask in range(8):
                if endpoint_mask & directions:
                    continue
                coordinates = tuple(
                    base[axis]
                    + (0 if directions & (1 << axis) else (endpoint_mask >> axis) & 1)
                    for axis in range(3)
                )
                cells.add((*coordinates, directions))
    return cells


def rank_over_f2(columns: Iterable[int]) -> int:
    """Return the rank of bit-packed matrix columns over F2."""

    pivots: dict[int, int] = {}
    for column in columns:
        while column:
            pivot = column.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = column
                break
            column ^= previous
    return len(pivots)


def boundary_rank(cells_by_dimension: list[list[Cell]], dimension: int) -> int:
    if dimension == 0 or not cells_by_dimension[dimension]:
        return 0
    row_index = {
        cell: index for index, cell in enumerate(cells_by_dimension[dimension - 1])
    }
    columns = []
    for cell in cells_by_dimension[dimension]:
        packed = 0
        for face in cell_boundary(cell):
            packed ^= 1 << row_index[face]
        columns.append(packed)
    return rank_over_f2(columns)


def cubical_betti_numbers(
    foreground: np.ndarray, construction: str
) -> tuple[int, int, int, int]:
    """Compute beta_0 through beta_3 directly from cubical boundary ranks."""

    if construction == "vertex":
        cells = vertex_construction(foreground)
    elif construction == "top-cell":
        cells = top_cell_construction(foreground)
    else:
        raise ValueError("construction must be 'vertex' or 'top-cell'")
    cells_by_dimension = [
        sorted(cell for cell in cells if cell_dimension(cell) == dimension)
        for dimension in range(4)
    ]
    ranks = [0] + [
        boundary_rank(cells_by_dimension, dimension) for dimension in range(1, 4)
    ] + [0]
    return tuple(
        len(cells_by_dimension[dimension]) - ranks[dimension] - ranks[dimension + 1]
        for dimension in range(4)
    )


def mask_from_bits(mask: int) -> np.ndarray:
    values = np.zeros(8, dtype=bool)
    for bit in range(8):
        values[bit] = bool((mask >> bit) & 1)
    return values.reshape((2, 2, 2))


def component_count(foreground: np.ndarray, connectivity: int) -> int:
    """Simple graph traversal used only to compare the boundary-matrix H0."""

    offsets = []
    for dz, dy, dx in product((-1, 0, 1), repeat=3):
        if dz == dy == dx == 0:
            continue
        if connectivity == 6 and abs(dz) + abs(dy) + abs(dx) != 1:
            continue
        offsets.append((dz, dy, dx))
    visited: set[tuple[int, int, int]] = set()
    count = 0
    depth, height, width = foreground.shape
    for z, y, x in np.argwhere(foreground):
        start = (int(z), int(y), int(x))
        if start in visited:
            continue
        count += 1
        visited.add(start)
        stack = [start]
        while stack:
            cz, cy, cx = stack.pop()
            for dz, dy, dx in offsets:
                neighbor = (cz + dz, cy + dy, cx + dx)
                nz, ny, nx = neighbor
                if (
                    0 <= nz < depth
                    and 0 <= ny < height
                    and 0 <= nx < width
                    and foreground[neighbor]
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    stack.append(neighbor)
    return count


def validate_exhaustive_two_by_two_by_two() -> None:
    for mask in range(256):
        foreground = mask_from_bits(mask)
        vertex_betti = cubical_betti_numbers(foreground, "vertex")
        top_cell_betti = cubical_betti_numbers(foreground, "top-cell")
        if vertex_betti[0] != component_count(foreground, 6):
            raise AssertionError((mask, "vertex H0", vertex_betti))
        if top_cell_betti[0] != component_count(foreground, 26):
            raise AssertionError((mask, "top-cell H0", top_cell_betti))
        for betti in (vertex_betti, top_cell_betti):
            if betti[2:] != (0, 0):
                raise AssertionError((mask, "unexpected tiny-box H2/H3", betti))


def validate_shell_h2() -> None:
    shell = np.ones((3, 3, 3), dtype=bool)
    shell[1, 1, 1] = False
    for construction in ("vertex", "top-cell"):
        betti = cubical_betti_numbers(shell, construction)
        if betti[2] != 1:
            raise AssertionError((construction, "shell H2", betti))


def main() -> None:
    validate_exhaustive_two_by_two_by_two()
    validate_shell_h2()
    print("PASS: 512 tiny construction checks and two direct shell H2 checks")


if __name__ == "__main__":
    main()
