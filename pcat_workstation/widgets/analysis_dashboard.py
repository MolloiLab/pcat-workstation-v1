"""Collapsible analysis dashboard with matplotlib charts for PCAT results."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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


class AnalysisDashboard(QWidget):
    """Collapsible bottom panel with HU histogram and radial profile charts."""

    _EXPANDED_HEIGHT = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed = True
        self._build_ui()
        self.set_collapsed(True)

    # ── UI construction ─────────────────────────────────────────────

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

        self._toggle_btn = QPushButton("▲")
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
        c_lay.addWidget(self._tabs)
        root.addWidget(self._content)

    # ── Collapse / expand ────────────────────────────────────────────

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._toggle_btn.setText("▲" if collapsed else "▼")
        if collapsed:
            self.setFixedHeight(36)
        else:
            self.setFixedHeight(self._EXPANDED_HEIGHT)

    # ── Plotting API ─────────────────────────────────────────────────

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
        self, distances_mm: np.ndarray, mean_hu: np.ndarray, vessel_name: str,
    ):
        """Plot mean HU vs radial distance from vessel wall."""
        canvas = self._profile_canvas
        ax = canvas.ax
        ax.cla()
        canvas._apply_theme()

        color = _VESSEL_COLORS.get(vessel_name, "#666666")
        ax.plot(distances_mm, mean_hu, color=color, linewidth=2)

        # Risk threshold line
        ax.axhline(_RISK_THRESHOLD, linestyle="--", color="#888", linewidth=1)
        ax.text(
            distances_mm[-1] * 0.7, _RISK_THRESHOLD + 2,
            f"{_RISK_THRESHOLD} HU", fontsize=9, color="#888",
        )

        ax.set_xlabel("Distance from vessel wall (mm)", fontsize=_MPL_STYLE["font_size"])
        ax.set_ylabel("Mean HU", fontsize=_MPL_STYLE["font_size"])
        ax.set_title(
            f"{vessel_name} Radial HU Profile",
            fontsize=_MPL_STYLE["font_size"] + 1, color=_MPL_STYLE["text_color"],
        )
        canvas.fig.tight_layout(pad=1.5)
        canvas.draw()
        self._tabs.setCurrentWidget(canvas)

    def clear(self):
        """Clear all plots."""
        self._histogram_canvas.clear_plot()
        self._profile_canvas.clear_plot()
