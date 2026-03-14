"""Pipeline progress panel for the right sidebar."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QGroupBox,
)
from PySide6.QtCore import Signal, Qt
from typing import Dict

from pcat_workstation.app.config import PIPELINE_STAGES, STAGE_LABELS, VESSEL_CONFIGS


# Status icon characters
_STATUS_ICONS = {
    "pending": "\u25CB",   # ○
    "running": "\u27F3",   # ⟳
    "complete": "\u2713",  # ✓
    "failed": "\u2717",    # ✗
    "skipped": "\u2298",   # ⊘
}

_STATUS_COLORS = {
    "pending": "#636366",
    "running": "#0a84ff",
    "complete": "#30d158",
    "failed": "#ff453a",
    "skipped": "#636366",
}

_VESSEL_COLORS = {
    "LAD": "#ff453a",
    "LCx": "#0a84ff",
    "RCA": "#30d158",
}


class _StageRow(QFrame):
    """Single pipeline stage row with icon, label, and elapsed time."""

    def __init__(self, stage_key: str, label: str, parent=None):
        super().__init__(parent)
        self.stage_key = stage_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.icon_label = QLabel(_STATUS_ICONS["pending"])
        self.icon_label.setFixedWidth(20)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("color: #636366; font-size: 15pt;")
        layout.addWidget(self.icon_label)

        self.name_label = QLabel(label)
        self.name_label.setStyleSheet("color: #e5e5e7; font-size: 13pt;")
        self.name_label.setSizePolicy(
            self.name_label.sizePolicy().horizontalPolicy(),
            self.name_label.sizePolicy().verticalPolicy(),
        )
        layout.addWidget(self.name_label, stretch=1)

        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet(
            "color: #98989d; font-family: 'Menlo', 'Courier New', monospace; font-size: 11pt;"
        )
        self.time_label.setFixedWidth(40)
        layout.addWidget(self.time_label)

    def set_status(self, status: str, elapsed_seconds: float = 0) -> None:
        icon = _STATUS_ICONS.get(status, "\u25CB")
        color = _STATUS_COLORS.get(status, "#666666")
        self.icon_label.setText(icon)
        self.icon_label.setStyleSheet(f"color: {color}; font-size: 15pt;")

        if status == "complete" and elapsed_seconds > 0:
            if elapsed_seconds < 60:
                self.time_label.setText(f"{int(elapsed_seconds)}s")
            else:
                mins = int(elapsed_seconds // 60)
                secs = int(elapsed_seconds % 60)
                self.time_label.setText(f"{mins}m{secs:02d}s")
        elif status == "running":
            self.time_label.setText("...")
        else:
            self.time_label.setText("")

    def reset(self) -> None:
        self.set_status("pending")


class ProgressPanel(QWidget):
    """Right sidebar showing pipeline stage status and run controls."""

    run_clicked = Signal()
    stage_action = Signal(str, str)  # (stage_name, action)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)

        self._stage_rows: Dict[str, _StageRow] = {}
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- Header ---
        header = QLabel("Pipeline")
        header.setStyleSheet("font-weight: bold; font-size: 18pt; color: #e5e5e7;")
        layout.addWidget(header)

        # --- Run button ---
        self._run_btn = QPushButton("\u25B6  Run All")
        self._run_btn.setFixedHeight(40)
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0a84ff;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                font-size: 15pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0070e0;
            }
            QPushButton:disabled {
                background-color: #3a3a3c;
                color: #636366;
            }
            """
        )
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self.run_clicked)
        layout.addWidget(self._run_btn)

        # --- Stage rows ---
        for stage_key in PIPELINE_STAGES:
            label = STAGE_LABELS.get(stage_key, stage_key)
            row = _StageRow(stage_key, label, parent=self)
            self._stage_rows[stage_key] = row
            layout.addWidget(row)

        # --- Overall progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, len(PIPELINE_STAGES))
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m stages")
        layout.addWidget(self._progress_bar)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #38383a;")
        sep.setFixedHeight(2)
        layout.addWidget(sep)

        # --- Vessel summary placeholder ---
        self._vessel_group = QGroupBox("Vessel Summary")
        self._vessel_layout = QVBoxLayout(self._vessel_group)
        self._vessel_layout.setContentsMargins(6, 6, 6, 6)
        self._vessel_layout.setSpacing(4)
        layout.addWidget(self._vessel_group)

        layout.addStretch(1)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_stage_status(
        self, stage: str, status: str, elapsed_seconds: float = 0
    ) -> None:
        """Update the status icon and elapsed time for a pipeline stage."""
        row = self._stage_rows.get(stage)
        if row is None:
            return
        row.set_status(status, elapsed_seconds)
        self._update_progress()

    def set_running(self, running: bool) -> None:
        """Toggle run button text and enabled state."""
        if running:
            self._run_btn.setText("\u23F9  Running...")
            self._run_btn.setEnabled(False)
        else:
            self._run_btn.setText("\u25B6  Run All")
            self._run_btn.setEnabled(True)

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable or disable the run button."""
        self._run_btn.setEnabled(enabled)

    def reset_stages(self) -> None:
        """Reset all stages to pending."""
        for row in self._stage_rows.values():
            row.reset()
        self._progress_bar.setValue(0)

    def set_vessel_summary(self, vessel_stats: Dict[str, Dict]) -> None:
        """Populate the vessel summary section with per-vessel FAI results."""
        # Clear existing items
        while self._vessel_layout.count():
            item = self._vessel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for vessel_name, stats in vessel_stats.items():
            color = _VESSEL_COLORS.get(vessel_name, "#e0e0e0")
            fai_value = stats.get("mean_fai", 0.0)
            risk = stats.get("risk", "LOW")

            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(2, 1, 2, 1)
            row_layout.setSpacing(6)

            name_lbl = QLabel(vessel_name)
            name_lbl.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 13pt;"
            )
            row_layout.addWidget(name_lbl)

            fai_lbl = QLabel(f"{fai_value:.1f} HU")
            fai_lbl.setStyleSheet(
                "color: #e5e5e7; font-family: 'Menlo', 'Courier New', monospace; font-size: 13pt;"
            )
            row_layout.addWidget(fai_lbl, stretch=1)

            risk_color = "#ff9f0a" if risk == "HIGH" else "#30d158"
            risk_lbl = QLabel(risk)
            risk_lbl.setStyleSheet(
                f"color: {risk_color}; font-weight: bold; font-size: 13pt;"
            )
            risk_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(risk_lbl)

            self._vessel_layout.addWidget(row)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _update_progress(self) -> None:
        """Recount completed stages and update progress bar."""
        completed = sum(
            1
            for row in self._stage_rows.values()
            if row.icon_label.text() == _STATUS_ICONS["complete"]
        )
        self._progress_bar.setValue(completed)
