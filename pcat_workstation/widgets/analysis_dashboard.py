"""Collapsible analysis dashboard with matplotlib charts for PCAT results."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Vessel colors matching the workstation palette
_VESSEL_COLORS = {
    "LAD": "#ff453a",
    "LCx": "#0a84ff",
    "RCA": "#30d158",
}

# Craft-themed matplotlib styling
_MPL_STYLE = {
    "facecolor": "#1c1c1e",
    "text_color": "#e5e5e7",
    "grid_color": "#38383a",
    "spine_color": "#38383a",
    "font_size": 11,
}

_RISK_THRESHOLD = -70.1
_FAI_RANGE = (-190, -30)


class _ChartCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas pre-configured with Craft theme styling."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3.5), dpi=100)
        self.fig.set_facecolor(_MPL_STYLE["facecolor"])
        self.ax = self.fig.add_subplot(111)
        self._apply_theme()
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)

    def _apply_theme(self):
        ax = self.ax
        ax.set_facecolor(_MPL_STYLE["facecolor"])
        ax.tick_params(colors=_MPL_STYLE["text_color"], labelsize=_MPL_STYLE["font_size"])
        ax.title.set_color(_MPL_STYLE["text_color"])
        ax.xaxis.label.set_color(_MPL_STYLE["text_color"])
        ax.yaxis.label.set_color(_MPL_STYLE["text_color"])
        ax.grid(True, color=_MPL_STYLE["grid_color"], linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(_MPL_STYLE["spine_color"])

    def clear_plot(self):
        self.ax.cla()
        self._apply_theme()
        self.draw()


class _RingCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for the angular ring cross-section visualization."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.set_facecolor(_MPL_STYLE["facecolor"])
        self.ax = self.fig.add_subplot(111)
        self._apply_theme()
        self.fig.tight_layout(pad=1.0)
        super().__init__(self.fig)
        self.setParent(parent)

    def _apply_theme(self):
        ax = self.ax
        ax.set_facecolor(_MPL_STYLE["facecolor"])
        ax.set_aspect("equal")
        ax.tick_params(colors=_MPL_STYLE["text_color"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    def clear_plot(self):
        self.ax.cla()
        self._apply_theme()
        self.draw()


class AnalysisDashboard(QWidget):
    """Collapsible bottom panel with HU histogram and radial profile charts."""

    _EXPANDED_HEIGHT = 420

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed = True
        self._build_ui()
        self.set_collapsed(True)

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        self.setStyleSheet("background: #2c2c2e;")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setStyleSheet("background: #3a3a3c;")
        header.setFixedHeight(36)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12, 0, 12, 0)

        title = QLabel("Analysis")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; background: transparent;")
        h_lay.addWidget(title)
        h_lay.addStretch()

        self._toggle_btn = QPushButton("\u25b2")
        self._toggle_btn.setFixedSize(28, 28)
        self._toggle_btn.setStyleSheet(
            "QPushButton { border: none; color: #636366; font-size: 14pt; background: transparent; }"
        )
        self._toggle_btn.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        h_lay.addWidget(self._toggle_btn)
        root.addWidget(header)

        # Content area
        self._content = QWidget()
        c_lay = QVBoxLayout(self._content)
        c_lay.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._histogram_canvas = _ChartCanvas()
        self._profile_canvas = _ChartCanvas()
        self._tabs.addTab(self._histogram_canvas, "HU Histogram")
        self._tabs.addTab(self._profile_canvas, "Radial Profile")
        self._ring_canvas = _RingCanvas()
        self._tabs.addTab(self._ring_canvas, "Angular Asymmetry")
        c_lay.addWidget(self._tabs)
        root.addWidget(self._content)

    # -- Collapse / expand ---------------------------------------------------

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._toggle_btn.setText("\u25b2" if collapsed else "\u25bc")
        h = 36 if collapsed else self._EXPANDED_HEIGHT
        self.setFixedHeight(h)
        # Force the parent splitter to recompute sizes and repaint.
        # QSplitter caches child sizes — need to poke it to update.
        from PySide6.QtWidgets import QApplication, QSplitter
        p = self.parentWidget()
        while p is not None:
            if isinstance(p, QSplitter):
                # Recalculate splitter sizes
                sizes = p.sizes()
                p.setSizes(sizes)
                break
            p = p.parentWidget()
        QApplication.processEvents()
        self.update()
        if self.parentWidget():
            self.parentWidget().update()

    # -- Plotting API --------------------------------------------------------

    def plot_histogram(self, hu_values: np.ndarray, vessel_name: str):
        """Plot HU distribution histogram for a PCAT VOI."""
        canvas = self._histogram_canvas
        ax = canvas.ax
        ax.cla()
        canvas._apply_theme()

        color = _VESSEL_COLORS.get(vessel_name, "#666666")

        # Filter to FAI window and bin
        masked = hu_values[(hu_values >= _FAI_RANGE[0]) & (hu_values <= _FAI_RANGE[1])]
        ax.hist(masked, bins=40, range=_FAI_RANGE, color=color, edgecolor=color, alpha=0.8)

        # Risk threshold line
        ax.axvline(_RISK_THRESHOLD, linestyle="--", color="#888", linewidth=1)
        ax.text(
            _RISK_THRESHOLD + 2, ax.get_ylim()[1] * 0.9,
            f"{_RISK_THRESHOLD} HU", fontsize=9, color="#888",
        )

        ax.set_xlim(*_FAI_RANGE)
        ax.set_xlabel("HU", fontsize=_MPL_STYLE["font_size"])
        ax.set_ylabel("Count", fontsize=_MPL_STYLE["font_size"])
        ax.set_title(
            f"{vessel_name} PCAT HU Distribution",
            fontsize=_MPL_STYLE["font_size"] + 1, color=_MPL_STYLE["text_color"],
        )
        canvas.fig.tight_layout(pad=1.5)
        canvas.draw()
        self._tabs.setCurrentWidget(canvas)

    def plot_radial_profile(
        self,
        distances_mm: np.ndarray,
        mean_hu: np.ndarray,
        vessel_name: str,
        std_hu: np.ndarray | None = None,
    ):
        """Plot mean HU vs radial distance from vessel wall (reference style).

        Matches the left panel of pipeline/visualize.py plot_radial_hu_profile.
        """
        canvas = self._profile_canvas
        ax = canvas.ax
        ax.cla()
        canvas._apply_theme()

        valid = ~np.isnan(mean_hu)

        # Background: typical FAI range band
        ax.axhspan(-90, -65, alpha=0.12, color="lightblue",
                   label="Typical FAI range (-90 to -65 HU)")

        # FAI risk threshold dashed line
        ax.axhline(_RISK_THRESHOLD, color="#CC2200", linewidth=1.6,
                   linestyle=":", alpha=0.9,
                   label=f"FAI risk cut-off ({_RISK_THRESHOLD} HU)")

        if valid.any():
            # Mean HU line with circle markers
            ax.plot(
                distances_mm[valid],
                mean_hu[valid],
                marker="o",
                markersize=5,
                linewidth=2,
                color="#D94040",
                zorder=3,
                label="Mean FAI HU",
            )

            # +/- 1 SD shaded band (if std data available)
            if std_hu is not None:
                std_valid = ~np.isnan(std_hu)
                both_valid = valid & std_valid
                if both_valid.any():
                    ax.fill_between(
                        distances_mm[both_valid],
                        mean_hu[both_valid] - std_hu[both_valid],
                        mean_hu[both_valid] + std_hu[both_valid],
                        alpha=0.25,
                        color="#D94040",
                        label="\u00b11 SD",
                    )
        else:
            ax.text(
                0.5, 0.5, "No fat voxels found\nin this VOI",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )

        ax.set_xlabel("Distance from vessel wall (mm)", fontsize=_MPL_STYLE["font_size"])
        ax.set_ylabel("Mean HU (FAI range)", fontsize=_MPL_STYLE["font_size"])
        ax.set_xlim(0, 20)
        ax.set_ylim(-105, -50)
        ax.set_title(
            f"{vessel_name} Radial HU Profile",
            fontsize=_MPL_STYLE["font_size"] + 1, color=_MPL_STYLE["text_color"],
        )
        ax.legend(fontsize=8, facecolor=_MPL_STYLE["facecolor"],
                  edgecolor=_MPL_STYLE["grid_color"],
                  labelcolor=_MPL_STYLE["text_color"])
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(10))

        canvas.fig.tight_layout(pad=1.5)
        canvas.draw()
        self._tabs.setCurrentWidget(canvas)

    def plot_angular_asymmetry(self, octant_data: dict, vessel_name: str):
        """Plot angular asymmetry as a ring cross-section image.

        Renders a hollow artery cross-section with a colored ring around it.
        Each angular sector of the ring is colored by its mean HU value
        using a yellow (-190 HU) to red (-30 HU) colormap, showing the
        spatial distribution of pericoronary fat around the vessel.

        Parameters
        ----------
        octant_data : dict with "sectors" (list of dicts with "angle_deg",
                      "hu_mean") and "sector_labels" (list of str)
        vessel_name : e.g., "LAD"
        """
        canvas = self._ring_canvas
        ax = canvas.ax
        ax.cla()
        canvas._apply_theme()

        sectors = octant_data.get("sectors", [])
        labels = octant_data.get("sector_labels", [])
        if not sectors:
            canvas.draw()
            return

        n = len(sectors)
        values = np.array([s["hu_mean"] for s in sectors], dtype=float)

        # Build a ring image: for each pixel, determine angle → sector → HU → color
        size = 256
        center = size / 2
        r_inner = size * 0.20  # vessel lumen (hollow)
        r_outer = size * 0.45  # outer edge of PCAT ring

        # Create coordinate grids
        y, x = np.mgrid[:size, :size]
        dx = x - center
        dy = -(y - center)  # flip Y so anterior is up
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dx, dy)  # angle from top (anterior), clockwise
        theta = theta % (2 * np.pi)

        # Map angle to sector index
        sector_idx = (theta / (2 * np.pi) * n).astype(int) % n

        # Build RGBA image
        ring_mask = (r >= r_inner) & (r <= r_outer)
        lumen_mask = r < r_inner

        # FAI colormap matching Oikonomou et al. / pipeline/visualize.py:
        # yellow (#FFEE00) at -190 → orange (#FF8800) → red (#CC0000) at -30
        from matplotlib.colors import LinearSegmentedColormap
        fai_cmap = LinearSegmentedColormap.from_list("fai", [
            (0.0, "#FFEE00"), (0.4, "#FF8800"),
            (0.7, "#FF4400"), (1.0, "#CC0000"),
        ])

        # Map HU values to colors
        rgba = np.zeros((size, size, 4), dtype=float)

        # Background: dark
        rgba[:, :, :3] = 0.11  # match panel background
        rgba[:, :, 3] = 1.0

        # Lumen: black (hollow)
        rgba[lumen_mask, :3] = 0.05
        rgba[lumen_mask, 3] = 1.0

        # Vessel wall circle (white ring)
        wall_mask = (r >= r_inner - 2) & (r < r_inner + 1)
        rgba[wall_mask, :3] = 0.7
        rgba[wall_mask, 3] = 1.0

        # Ring sectors colored by HU
        for i in range(n):
            sec_mask = ring_mask & (sector_idx == i)
            if not sec_mask.any():
                continue
            hu = values[i]
            if np.isnan(hu):
                rgba[sec_mask, :3] = 0.2  # gray for no data
            else:
                t = np.clip((hu - _FAI_RANGE[0]) / (_FAI_RANGE[1] - _FAI_RANGE[0]), 0, 1)
                color = fai_cmap(t)
                rgba[sec_mask, 0] = color[0]
                rgba[sec_mask, 1] = color[1]
                rgba[sec_mask, 2] = color[2]
            rgba[sec_mask, 3] = 1.0

        # Draw sector boundaries (thin dark lines)
        for i in range(n):
            angle = 2 * np.pi * i / n
            for rr in np.linspace(r_inner, r_outer, 50):
                px = int(center + rr * np.sin(angle))
                py = int(center - rr * np.cos(angle))
                if 0 <= px < size and 0 <= py < size:
                    rgba[py, px, :3] = 0.2
                    rgba[py, px, 3] = 1.0

        ax.imshow(rgba, origin="upper")

        # Add sector labels around the outside
        label_r = r_outer + 15
        for i, label in enumerate(labels):
            angle = 2 * np.pi * i / n + np.pi / n  # center of sector
            lx = center + label_r * np.sin(angle)
            ly = center - label_r * np.cos(angle)
            ax.text(lx, ly, label, fontsize=7, ha="center", va="center",
                    color=_MPL_STYLE["text_color"])

        # Add HU values inside each sector
        hu_r = (r_inner + r_outer) / 2
        for i in range(n):
            hu = values[i]
            if np.isnan(hu):
                continue
            angle = 2 * np.pi * i / n + np.pi / n
            hx = center + hu_r * np.sin(angle)
            hy = center - hu_r * np.cos(angle)
            ax.text(hx, hy, f"{hu:.0f}", fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold")

        # Colorbar
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        sm = mcm.ScalarMappable(cmap=fai_cmap,
                                norm=mcolors.Normalize(vmin=_FAI_RANGE[0], vmax=_FAI_RANGE[1]))
        sm.set_array([])
        cbar = canvas.fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02,
                                   orientation="vertical")
        cbar.set_label("FAI (HU)", fontsize=9, color=_MPL_STYLE["text_color"])
        cbar.set_ticks([_FAI_RANGE[0], -150, -110, -70, _FAI_RANGE[1]])
        cbar.ax.tick_params(colors=_MPL_STYLE["text_color"], labelsize=7)

        ax.set_title(
            f"{vessel_name} Angular FAI Distribution",
            fontsize=_MPL_STYLE["font_size"] + 1,
            color=_MPL_STYLE["text_color"],
        )
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)

        canvas.fig.tight_layout(pad=1.0)
        canvas.draw()
        self._tabs.setCurrentWidget(canvas)

    def clear(self):
        """Clear all plots."""
        self._histogram_canvas.clear_plot()
        self._profile_canvas.clear_plot()
        self._ring_canvas.clear_plot()
