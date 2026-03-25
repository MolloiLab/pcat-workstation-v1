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
        self.fig = Figure(figsize=(6, 2.4), dpi=100)
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


class _PolarChartCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with a polar subplot for angular asymmetry."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.fig.set_facecolor(_MPL_STYLE["facecolor"])
        self.ax = self.fig.add_subplot(111, projection="polar")
        self._apply_theme()
        self.fig.tight_layout(pad=2.0)
        super().__init__(self.fig)
        self.setParent(parent)

    def _apply_theme(self):
        ax = self.ax
        ax.set_facecolor(_MPL_STYLE["facecolor"])
        ax.tick_params(colors=_MPL_STYLE["text_color"], labelsize=8)
        ax.grid(True, color=_MPL_STYLE["grid_color"], linewidth=0.5, alpha=0.4)

    def clear_plot(self):
        self.ax.cla()
        self._apply_theme()
        self.draw()


class AnalysisDashboard(QWidget):
    """Collapsible bottom panel with HU histogram and radial profile charts."""

    _EXPANDED_HEIGHT = 300

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
        self._polar_canvas = _PolarChartCanvas()
        self._tabs.addTab(self._polar_canvas, "Angular Asymmetry")
        c_lay.addWidget(self._tabs)
        root.addWidget(self._content)

    # -- Collapse / expand ---------------------------------------------------

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._toggle_btn.setText("\u25b2" if collapsed else "\u25bc")
        if collapsed:
            self.setFixedHeight(36)
        else:
            self.setFixedHeight(self._EXPANDED_HEIGHT)

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
        """Plot 8-spoke polar ring chart of per-octant FAI values.

        Values are shifted positive for polar rendering; tick labels show
        the real (unshifted) HU values.

        Parameters
        ----------
        octant_data : dict with "sectors" (list of dicts with "angle_deg",
                      "hu_mean", "fai_risk") and "sector_labels" (list of str)
        vessel_name : e.g., "LAD"
        """
        canvas = self._polar_canvas
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
        risks = [s.get("fai_risk", "LOW") for s in sectors]

        # Filter valid (non-NaN)
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            canvas.draw()
            return

        # Compute offset to shift all values positive for polar rendering
        min_val = float(np.nanmin(values))
        offset = abs(min_val) + 10.0  # ensure all shifted values > 0

        # Replace NaN with the minimum valid value for display
        display_vals = np.where(valid_mask, values + offset, 0.0)

        # Angles for each sector
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        width = 2 * np.pi / n * 0.85

        # Color each bar: red if HIGH risk, green if LOW
        colors = ["#ff453a" if r == "HIGH" else "#30d158" for r in risks]

        bars = ax.bar(
            angles, display_vals, width=width, bottom=0,
            color=colors, alpha=0.8, edgecolor="#555", linewidth=0.5,
        )

        # Draw risk threshold ring (shifted)
        threshold_shifted = _RISK_THRESHOLD + offset
        theta_ring = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta_ring, np.full_like(theta_ring, threshold_shifted),
                color="#CC2200", linewidth=1.5, linestyle="--", alpha=0.8,
                label=f"FAI risk ({_RISK_THRESHOLD} HU)")

        # Set sector labels around the ring
        ax.set_xticks(angles)
        if labels:
            ax.set_xticklabels(labels, fontsize=7, color=_MPL_STYLE["text_color"])
        ax.tick_params(axis="x", colors=_MPL_STYLE["text_color"], pad=8)

        # Set radial tick labels showing REAL (unshifted) HU values
        # Pick ~4 nice tick positions in shifted space, label with real values
        r_max = float(np.nanmax(display_vals)) * 1.15 if np.nanmax(display_vals) > 0 else 10
        ax.set_ylim(0, r_max)
        n_rticks = 4
        tick_positions = np.linspace(0, r_max, n_rticks + 1)[1:]  # skip 0
        real_values = tick_positions - offset
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"{v:.0f}" for v in real_values], fontsize=7,
                           color=_MPL_STYLE["text_color"])

        ax.set_title(
            f"{vessel_name} Angular FAI Asymmetry",
            fontsize=_MPL_STYLE["font_size"] + 1,
            color=_MPL_STYLE["text_color"],
            pad=15,
        )

        canvas.fig.tight_layout(pad=2.0)
        canvas.draw()
        self._tabs.setCurrentWidget(canvas)

    def clear(self):
        """Clear all plots."""
        self._histogram_canvas.clear_plot()
        self._profile_canvas.clear_plot()
        self._polar_canvas.clear_plot()
