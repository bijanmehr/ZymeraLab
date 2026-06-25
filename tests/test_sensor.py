import numpy as np
from zymera import sensor


def test_chebyshev_radius_and_walls():
    wall = np.zeros((5, 5), dtype=bool)
    wall[0, 0] = True                      # a wall cell is never visible
    vis = sensor.visible_cells((2, 2), wall, radius=1)   # pos = (row, col)
    assert vis.shape == (5, 5)
    assert vis[2, 2] and vis[1, 1] and vis[3, 3]         # within Chebyshev radius 1
    assert not vis[0, 0]                                 # wall
    assert not vis[4, 4]                                 # outside radius


def test_radius_zero_is_just_own_cell():
    wall = np.zeros((3, 3), dtype=bool)
    vis = sensor.visible_cells((1, 1), wall, radius=0)
    assert vis.sum() == 1 and vis[1, 1]
