"""zymera.viz — drawing for trajectories and worlds (opt-in extra).

Imports matplotlib/pillow; never imported by the headless core. The viz
doctrine is RE-SIMULATION: a run is reproducible from (env spec, checkpoint,
key), so render from a fresh ``rollout(..., keep="all")`` rather than storing
heavy training trajectories.

Deferred (tracked in README status): isometric renderer and keyboard teleop —
port from zymera v0 when needed.
"""

from .render import draw_frame, render_comm_gif, render_gif
from .report import make_report

__all__ = ["draw_frame", "render_gif", "render_comm_gif", "make_report"]
