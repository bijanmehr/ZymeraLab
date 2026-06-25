"""Host-side visibility sensor (a numpy analysis tool, not on the jit path).

v1 is non-occluded: a cell is visible from ``pos`` iff it is within ``radius``
under the chosen metric and is not a wall. Occlusion (ray-casting) is a P2
extension when an experiment needs it. Positions are ``(row, col)``.
"""
from __future__ import annotations

import numpy as np


def visible_cells(pos, wall, radius: int, metric: str = "chebyshev"):
    """Return a ``(H, W)`` bool mask of cells visible from ``pos``."""
    wall = np.asarray(wall, dtype=bool)
    h, w = wall.shape
    rr, cc = np.mgrid[0:h, 0:w]
    pr, pc = int(pos[0]), int(pos[1])
    if metric == "chebyshev":
        dist = np.maximum(np.abs(rr - pr), np.abs(cc - pc))
    elif metric == "manhattan":
        dist = np.abs(rr - pr) + np.abs(cc - pc)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return (dist <= radius) & (~wall)
