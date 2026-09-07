"""Publication-quality plotting utilities for the MSc thesis.

The :class:`ThesisPlotter` class centralises all chart-generation logic so that
every figure produced by the project shares the same academic styling
(300 DPI, serif typography, colorblind-safe palette, white background,
tight layout). Captions are recorded to ``figures/captions.md`` so they can be
copy-pasted directly into the thesis document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .utils import FIGURES_DIR, ensure_dir, get_logger


_LOGGER = get_logger(__name__)


@dataclass
class ThesisPlotter:
    """Centralised, academic-style figure factory.

    Parameters
    ----------
    figures_dir:
        Directory where generated figures and ``captions.md`` are written.
        Defaults to the project-level ``figures/`` directory.
    dpi:
        Resolution for saved figures. The thesis requires 300 DPI.
    base_font_size:
        Body-text font size in points.
    title_font_size:
        Figure-title font size in points.
    palette_name:
        Seaborn palette name. ``"colorblind"`` and ``"deep"`` are the
        recommended academic defaults.
    """

    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)
    dpi: int = 300
    base_font_size: int = 11
    title_font_size: int = 13
    palette_name: str = "colorblind"
    captions_filename: str = "captions.md"

    def __post_init__(self) -> None:
        self.figures_dir = Path(self.figures_dir)
        ensure_dir(self.figures_dir)
        self.captions_path: Path = self.figures_dir / self.captions_filename
        self.set_style()

    # ------------------------------------------------------------------ #
    # Style
    # ------------------------------------------------------------------ #
    def set_style(self) -> None:
        """Enforce the academic style on the global matplotlib + seaborn state."""
        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette=self.palette_name,
            font="serif",
            font_scale=1.0,
        )

        mpl.rcParams.update(
            {
                # Typography
                "font.family": "serif",
                "font.serif": [
                    "DejaVu Serif",
                    "Times New Roman",
                    "Liberation Serif",
                    "serif",
                ],
                "font.size": self.base_font_size,
                "axes.titlesize": self.title_font_size,
                "axes.titleweight": "bold",
                "axes.labelsize": self.base_font_size,
                "axes.labelweight": "regular",
                "xtick.labelsize": self.base_font_size - 1,
                "ytick.labelsize": self.base_font_size - 1,
                "legend.fontsize": self.base_font_size - 1,
                "legend.title_fontsize": self.base_font_size,
                "figure.titlesize": self.title_font_size + 1,
                # Layout / resolution
                "figure.dpi": 110,
                "savefig.dpi": self.dpi,
                "figure.autolayout": False,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                # Lines, grid, spines
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#333333",
                "axes.linewidth": 0.8,
                # Misc
                "axes.titlepad": 12,
                "axes.labelpad": 8,
            }
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_figure(
        self,
        fig: Figure,
        filename: str,
        caption: str,
        *,
        title: Optional[str] = None,
    ) -> Path:
        """Save ``fig`` to ``figures/`` and append the caption to ``captions.md``.

        Parameters
        ----------
        fig:
            The matplotlib figure to persist.
        filename:
            Target filename inside ``figures_dir`` (e.g. ``fig_01_xyz.png``).
        caption:
            Thesis-ready caption text. Will be appended to ``captions.md``.
        title:
            Optional human-readable title for the captions file. Defaults to
            the filename stem.
        """
        if not filename.lower().endswith(".png"):
            filename = f"{filename}.png"

        target_path = self.figures_dir / filename
        fig.tight_layout()
        fig.savefig(
            target_path,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white",
            format="png",
        )
        plt.close(fig)
        _LOGGER.info("Saved figure %s (%d DPI)", target_path.name, self.dpi)

        self._append_caption(filename=filename, caption=caption, title=title)
        return target_path

    def _append_caption(
        self,
        *,
        filename: str,
        caption: str,
        title: Optional[str],
    ) -> None:
        """Write the caption for ``filename``, replacing any existing entry.

        This is an upsert rather than an append. The original implementation
        appended unconditionally, so regenerating a figure silently added a
        second entry for the same file; that is how ``captions.md`` came to
        carry two "Figure 10" sections describing different images. Keying on
        the ``**File:**`` line makes the captions file converge on one entry
        per figure however many times the figure is regenerated.
        """
        display_title = title or Path(filename).stem.replace("_", " ").title()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = (
            f"## {display_title}\n\n"
            f"- **File:** `{filename}`\n"
            f"- **Generated:** {timestamp}\n\n"
            f"> {caption}\n\n"
        )

        header = (
            "# Figure Captions\n\n"
            "Generated by `src/visualization.py`. Each entry is ready to paste "
            "directly into the thesis under the corresponding figure.\n\n"
        )

        if not self.captions_path.exists():
            self.captions_path.write_text(header + block, encoding="utf-8")
            return

        text = self.captions_path.read_text(encoding="utf-8")
        marker = f"- **File:** `{filename}`"

        if marker not in text:
            with self.captions_path.open("a", encoding="utf-8") as handle:
                handle.write(block)
            return

        # Replace the whole "## ..." section that owns this file marker.
        sections = text.split("\n## ")
        preamble, rest = sections[0], sections[1:]
        rebuilt = [
            block[3:] if marker in section else section for section in rest
        ]
        self.captions_path.write_text(
            preamble + "".join("\n## " + s for s in rebuilt), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #
    def new_figure(
        self,
        *,
        figsize: tuple[float, float] = (8.0, 5.0),
        nrows: int = 1,
        ncols: int = 1,
        **subplots_kwargs,
    ) -> tuple[Figure, plt.Axes]:
        """Return a ``(fig, axes)`` pair pre-styled for academic output."""
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            constrained_layout=False,
            **subplots_kwargs,
        )
        return fig, axes

    def palette(self, n_colors: int) -> Sequence[tuple[float, float, float]]:
        """Return ``n_colors`` from the configured colorblind-safe palette."""
        return sns.color_palette(self.palette_name, n_colors=n_colors)

    def severity_palette(
        self,
        order: Iterable[str] = ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    ) -> dict[str, tuple[float, float, float]]:
        """Green→red gradient mapping for severity levels.

        Uses :func:`seaborn.color_palette` with the ``"RdYlGn_r"`` (reversed)
        colormap so that LOW maps to a calm green and CRITICAL to a strong red.
        """
        ordered = list(order)
        colors = sns.color_palette("RdYlGn_r", n_colors=len(ordered))
        return dict(zip(ordered, colors))


__all__ = ["ThesisPlotter"]
