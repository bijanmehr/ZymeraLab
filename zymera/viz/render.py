"""Top-down GIF rendering. Accepts stacked pytrees (scan output) or lists."""

import io
from typing import Iterable, List, Optional, Sequence

import jax
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from ..missions import Path as PathAnn, Point, Region


def iter_worlds(worlds) -> List:
    """Duck-type a trajectory: list of Worlds, or stacked pytree (leading T)."""
    if isinstance(worlds, (list, tuple)):
        return list(worlds)
    n = int(jax.tree_util.tree_leaves(worlds)[0].shape[0])
    return [jax.tree_util.tree_map(lambda x, t=t: x[t], worlds) for t in range(n)]


def _comm_edges(pos: np.ndarray, radius: int) -> Iterable:
    """Index pairs within Chebyshev ``radius`` (potential topology)."""
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(pos[i] - pos[j]).max() <= radius:
                yield i, j


def draw_frame(ax, world, *, comm_radius: Optional[int] = None,
               show_delivered: bool = True, annotations: Sequence = ()) -> None:
    """Draw one world: coverage heat, walls, comm edges, agents (by group)."""
    h, w = world.grid_h, world.grid_w
    explored = np.asarray(world.explored, dtype=float)
    covered = np.asarray(world.covered, dtype=float)
    wall = np.asarray(world.wall)
    pos = np.asarray(world.body.position)
    group = np.asarray(world.group)

    base = 0.25 * covered + 0.06 * np.minimum(explored, 8)
    base[wall] = np.nan                                   # walls drawn as ink
    ax.imshow(base, cmap="YlGn", vmin=0, vmax=1, origin="upper")
    ax.imshow(np.where(wall, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1,
              origin="upper")

    if comm_radius is not None:                           # potential: thin dotted
        for i, j in _comm_edges(pos, comm_radius):
            ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                    ls=":", lw=0.9, color="0.45", zorder=2)
    if show_delivered:                                    # delivered: solid
        adj = np.asarray(world.comm_graph)
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                            lw=1.6, color="#2e6e63", zorder=3)

    cmap = plt.get_cmap("tab10")
    multi_group = len(np.unique(group)) > 1
    colors = [cmap(int(group[i]) if multi_group else i % 10)
              for i in range(pos.shape[0])]
    ax.scatter(pos[:, 1], pos[:, 0], c=colors, s=120, edgecolors="black",
               linewidths=0.8, zorder=4)

    for a in annotations:                                 # mission overlays
        if isinstance(a, Point):
            p = np.asarray(a.pos)
            ax.scatter([p[1]], [p[0]], marker="*", s=180, color="#8e2c1f",
                       zorder=5)
        elif isinstance(a, PathAnn):
            cells = np.asarray(a.cells)
            ax.plot(cells[:, 1], cells[:, 0], lw=1.4, color="#8e2c1f",
                    alpha=0.8, zorder=5)
        elif isinstance(a, Region):
            mask = np.asarray(a.mask, dtype=float)
            ax.imshow(np.where(mask > 0, 0.8, np.nan), cmap="Reds", vmin=0,
                      vmax=1, alpha=0.3, origin="upper")

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlim(-0.5, w - 0.5), ax.set_ylim(h - 0.5, -0.5)


def render_frames(worlds, *, comm_radius=None, show_delivered=True,
                  annotations=None, figsize=3.2) -> List[Image.Image]:
    """Render each world to a PIL image (Agg, no display)."""
    frames = []
    for world in iter_worlds(worlds):
        ann = annotations(world) if callable(annotations) else (annotations or ())
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=110)
        draw_frame(ax, world, comm_radius=comm_radius,
                   show_delivered=show_delivered, annotations=ann)
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    return frames


def render_gif(worlds, path: str, *, fps: int = 6, comm_radius=None,
               show_delivered: bool = True, annotations=None) -> str:
    """Trajectory -> animated GIF at ``path``. Returns the path."""
    frames = render_frames(worlds, comm_radius=comm_radius,
                           show_delivered=show_delivered, annotations=annotations)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path


def render_comm_gif(worlds, path: str, *, comm_radius: int, fps: int = 6,
                    annotations=None) -> str:
    """GIF with the comm overlay: potential edges dotted, delivered solid."""
    return render_gif(worlds, path, fps=fps, comm_radius=comm_radius,
                      show_delivered=True, annotations=annotations)
