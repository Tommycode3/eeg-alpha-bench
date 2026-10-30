"""Figure style shared by reports and validation figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EO_COLOR = "#3d6fb6"   # eyes open: blue
EC_COLOR = "#c8553d"   # eyes closed: terracotta
NEUTRAL = "#4a4a4a"
ALPHA_SHADE = "#f2e3c6"

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.6,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def shade_alpha(ax, band=(8.0, 12.0)) -> None:  # noqa: ANN001
    ax.axvspan(*band, color=ALPHA_SHADE, zorder=0, lw=0)


def psd_figure(f: np.ndarray, eo: np.ndarray, ec: np.ndarray, title: str,
               band=(8.0, 12.0), fmax: float = 40.0):  # noqa: ANN001
    """PSD of EO vs EC (1-D arrays, uV^2/Hz), log y axis."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    m = (f >= 1) & (f <= fmax)
    shade_alpha(ax, band)
    ax.semilogy(f[m], eo[m], color=EO_COLOR, label="Eyes open")
    ax.semilogy(f[m], ec[m], color=EC_COLOR, label="Eyes closed")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title(title)
    ax.set_xlim(1, fmax)
    ax.legend(loc="upper right")
    ax.text(10, ax.get_ylim()[0], r"$\alpha$", ha="center", va="bottom", color="#8a6d3b")
    fig.tight_layout()
    return fig
