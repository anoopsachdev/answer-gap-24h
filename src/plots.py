"""Shared figure style."""

from pathlib import Path

import matplotlib.pyplot as plt

INK = "#1C2B33"
MUTED = "#5E6A71"
ACCENT = "#0F6E6E"
ACCENT_2 = "#D06B2B"
ACCENT_3 = "#3D5A80"
GRID = "#E6ECEC"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5DEDE",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
