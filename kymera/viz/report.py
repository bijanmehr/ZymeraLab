"""Self-contained HTML report: embedded episode GIF + curves + heatmap."""

import base64
import html
import io
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt

from .render import render_frames


def _b64_gif(frames, fps: int) -> str:
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_report(traj, path: str, *, env=None, title: str = "kymera report",
                fps: int = 6, comm_radius: Optional[int] = None) -> str:
    """Write a single-file HTML report for a ``rollout(..., keep="all")`` dict.

    Sections: episode GIF, coverage-over-time, per-term reward curves (when the
    rollout used ``collect=("reward_terms",)``), final visit heatmap, and the
    env composition when ``env`` is given.
    """
    worlds = traj["world"]
    if comm_radius is None and env is not None:
        topo = getattr(getattr(env, "channel", None), "topology", None)
        comm_radius = getattr(topo, "radius", None)

    frames = render_frames(worlds, comm_radius=comm_radius)
    gif64 = _b64_gif(frames, fps)

    covered = np.asarray(worlds.seen_by).any(1)                # (T+1, H, W)
    cov_curve = covered.reshape(covered.shape[0], -1).mean(-1)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.plot(cov_curve, color="#8e2c1f")
    ax.set_xlabel("step"), ax.set_ylabel("coverage"), ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    cov64 = _b64_fig(fig)

    terms_html = ""
    terms = traj.get("info", {}).get("reward_terms")
    if terms:
        fig, ax = plt.subplots(figsize=(5, 2.4))
        for name, arr in sorted(terms.items()):
            ax.plot(np.asarray(arr).mean(-1), label=name, lw=1.2)
        ax.set_xlabel("step"), ax.set_ylabel("unweighted term mean")
        ax.legend(fontsize=7), ax.grid(alpha=0.25)
        terms_html = (
            "<h2>Reward terms</h2>"
            f'<img alt="per-term reward curves" src="data:image/png;base64,{_b64_fig(fig)}">'
        )

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(np.asarray(worlds.explored)[-1], cmap="YlGn", origin="upper")
    ax.set_xticks([]), ax.set_yticks([]), ax.set_title("visit counts", fontsize=9)
    heat64 = _b64_fig(fig)

    env_html = ""
    if env is not None:
        env_html = f"<h2>Env composition</h2><pre>{html.escape(repr(env))}</pre>"

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body {{ font-family: Charter, 'Iowan Old Style', Georgia, serif; margin: 2rem auto;
        max-width: 46rem; color: #1a1816; background: #f7f4ee; }}
 h1, h2 {{ font-family: 'Avenir Next', Avenir, 'Helvetica Neue', sans-serif; }}
 h1 {{ border-bottom: 2px solid #8e2c1f; padding-bottom: .3rem; }}
 img {{ max-width: 100%; border: 1px solid #d8d2c4; border-radius: 6px; }}
 pre {{ background: #211e1b; color: #f0ece2; padding: .8rem 1rem; border-radius: 6px;
       font: 12px 'SF Mono', Menlo, monospace; overflow-x: auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<h2>Episode</h2><img alt="episode gif" src="data:image/gif;base64,{gif64}">
<h2>Coverage over time</h2><img alt="coverage curve" src="data:image/png;base64,{cov64}">
{terms_html}
<h2>Final visit heatmap</h2><img alt="visit heatmap" src="data:image/png;base64,{heat64}">
{env_html}
</body></html>"""
    with open(path, "w") as f:
        f.write(doc)
    return path
