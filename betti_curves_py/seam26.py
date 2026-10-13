"""Bounded-memory 26-neighbor cross-Z candidate event generator.

For the ascending voxel filtration an inter-face edge appears when its later
endpoint activates (max of voxel intensities). In the descending filtration
it appears at min(endpoint intensities). Sorting ONLY the two face activation
orders suffices to produce all cross-face events in filtration order.

An edge is produced once, by its later active endpoint; equal plateaus use
left-face-before-right-face order, and then increasing row-major position.
There is no explicit seam spanning forest nor nine-edge-per-pixel array.

The parent persistence reducer filters cycles itself. An activation batch
is bounded by ``9 * activations`` connections and is recycled on the next
batch; this is a *bounded-buffer* algorithm, not a fixed-RSS guarantee.
"""
from __future__ import annotations

import numpy as np
from .hierarchical import _jit


@_jit
def generate_seam26_batch(left_values, right_values, left_order, right_order,
                          lp, rp, nl, width, descending,
                          pl, pr, batch_limit, out_u, out_v, out_weight):
    """Produce a sorted candidate-event batch; return next cursors and size.

    ``lp``/``rp`` are seam-face offsets inside the two child vertex arrays.
    The right child receives the extra ``nl`` offset in the joined parent.
    """
    n = len(left_values)
    count = 0
    processed = 0
    while processed < batch_limit and (pl < n or pr < n):
        if pl >= n:
            from_left = False
        elif pr >= n:
            from_left = True
        else:
            vl = left_values[left_order[pl]]
            vr = right_values[right_order[pr]]
            # Exactly equal activations favor the left face.
            from_left = vl >= vr if descending else vl <= vr
        if from_left:
            q = left_order[pl]
            pl += 1
            level = left_values[q]
            y = q // width
            x = q % width
            for dy in range(-1, 2):
                ny = y + dy
                if ny < 0 or ny >= n // width:
                    continue
                for dx in range(-1, 2):
                    nx = x + dx
                    if nx < 0 or nx >= width:
                        continue
                    v = ny*width + nx
                    w = right_values[v]
                    # On equal levels, right activates AFTER left.
                    if (w > level if descending else w < level):
                        out_u[count] = lp + q
                        out_v[count] = nl + rp + v
                        out_weight[count] = level
                        count += 1
        else:
            q = right_order[pr]
            pr += 1
            level = right_values[q]
            y = q // width
            x = q % width
            for dy in range(-1, 2):
                ny = y + dy
                if ny < 0 or ny >= n // width:
                    continue
                for dx in range(-1, 2):
                    nx = x + dx
                    if nx < 0 or nx >= width:
                        continue
                    v = ny*width + nx
                    w = left_values[v]
                    # On equal levels, left activates BEFORE right.
                    if (w >= level if descending else w <= level):
                        out_u[count] = lp + v
                        out_v[count] = nl + rp + q
                        out_weight[count] = level
                        count += 1
        processed += 1
    return pl, pr, count
