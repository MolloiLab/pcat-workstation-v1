"""Main toolbar with vessel selector, mode, W/L presets, and action buttons."""

from PySide6.QtWidgets import (
    QToolBar, QWidget, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QToolButton, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction, QKeySequence

from pcat_workstation.app.config import VESSEL_CONFIGS


_WL_PRESETS = [
    ("CT Cardiac", 1500.0, 300.0),
    ("CT Soft Tissue", 400.0, 40.0),
    ("CT Lung", 1500.0, -500.0),
    ("CT Bone", 2000.0, 500.0),
    ("CT Vascular", 800.0, 200.0),
]


class MainToolBar(QToolBar):
    """Top toolbar for the PCAT Workstation main window."""

    vessel_changed = Signal(str)       # "LAD", "LCx", "RCA"
    mode_changed = Signal(str)         # "view", "edit"
    wl_preset_changed = Signal(float, float)  # (window, level)
    run_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)
        self.setFloatable(False)
        self.setFixedHeight(40)

        self._vessel_buttons: dict[str, QToolButton] = {}
        self._mode_buttons: dict[str, QToolButton] = {}

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # --- Vessel selector ---
        self._vessel_group = QButtonGroup(self)
        self._vessel_group.setExclusive(True)

        vessels = ["LAD", "LCx", "RCA"]
        shortcuts = {"LAD": "1", "LCx": "2", "RCA": "3"}

        for vessel in vessels:
            cfg = VESSEL_CONFIGS[vessel]
            color = cfg["color"]
            btn = QToolButton()
            btn.setText(vessel)
            btn.setCheckable(True)
            btn.setShortcut(QKeySequence(shortcuts[vessel]))
            btn.setToolTip(f"{vessel} (press {shortcuts[vessel]})")
            btn.setMinimumHeight(32)
            btn.setStyleSheet(self._vessel_button_style(color, checked=False))
            btn.toggled.connect(lambda checked, v=vessel: self._on_vessel_toggled(v, checked))
            self._vessel_group.addButton(btn)
            self._vessel_buttons[vessel] = btn
            self.addWidget(btn)

        self._vessel_buttons["LAD"].setChecked(True)

        self.addSeparator()

        # --- Mode selector ---
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        for mode_name, label in [("view", "View"), ("edit", "Edit")]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setStyleSheet(self._mode_button_style())
            btn.toggled.connect(lambda checked, m=mode_name: self._on_mode_toggled(m, checked))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode_name] = btn
            self.addWidget(btn)

        self._mode_buttons["view"].setChecked(True)

        self.addSeparator()

        # --- W/L presets ---
        wl_label = QLabel(" W/L: ")
        wl_label.setStyleSheet("color: #e5e5e7; font-size: 13pt;")
        self.addWidget(wl_label)

        self._wl_combo = QComboBox()
        self._wl_combo.setMinimumHeight(28)
        for name, w, l in _WL_PRESETS:
            self._wl_combo.addItem(name, (w, l))
        self._wl_combo.currentIndexChanged.connect(self._on_wl_changed)
        self.addWidget(self._wl_combo)

        # --- Spacer ---
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        self.addWidget(spacer)

        # --- Run button ---
        self._run_btn = QPushButton("\u25B6 Run")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.setShortcut(QKeySequence("Ctrl+R"))
        self._run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0a84ff;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 16px;
                font-size: 15pt;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0070e0; }
            QPushButton:disabled { background-color: #3a3a3c; color: #636366; }
            """
        )
        self._run_btn.clicked.connect(self.run_clicked)
        self.addWidget(self._run_btn)

        # --- Export button ---
        self._export_btn = QPushButton("Export")
        self._export_btn.setMinimumHeight(32)
        self._export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #e5e5e7;
                border: 1px solid #38383a;
                border-radius: 4px;
                padding: 4px 16px;
                font-size: 15pt;
            }
            QPushButton:hover { background-color: #3a3a3c; }
            """
        )
        self._export_btn.clicked.connect(self.export_clicked)
        self.addWidget(self._export_btn)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_vessel(self, vessel: str) -> None:
        """Programmatically select a vessel button."""
        btn = self._vessel_buttons.get(vessel)
        if btn:
            btn.setChecked(True)

    def set_mode(self, mode: str) -> None:
        """Programmatically select view/edit mode."""
        btn = self._mode_buttons.get(mode)
        if btn:
            btn.setChecked(True)

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable or disable the run button."""
        self._run_btn.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    #  Signal handlers
    # ------------------------------------------------------------------ #

    def _on_vessel_toggled(self, vessel: str, checked: bool) -> None:
        if checked:
            color = VESSEL_CONFIGS[vessel]["color"]
            # Update styles for all vessel buttons
            for v, btn in self._vessel_buttons.items():
                c = VESSEL_CONFIGS[v]["color"]
                btn.setStyleSheet(
                    self._vessel_button_style(c, checked=(v == vessel))
                )
            self.vessel_changed.emit(vessel)

    def _on_mode_toggled(self, mode: str, checked: bool) -> None:
        if checked:
            self.mode_changed.emit(mode)

    def _on_wl_changed(self, index: int) -> None:
        data = self._wl_combo.itemData(index)
        if data:
            w, l = data
            self.wl_preset_changed.emit(w, l)

    # ------------------------------------------------------------------ #
    #  Styling helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vessel_button_style(color: str, checked: bool = False) -> str:
        if checked:
            return f"""
                QToolButton {{
                    background-color: {color};
                    color: #ffffff;
                    border: none;
                    border-radius: 3px;
                    padding: 4px 12px;
                    font-weight: bold;
                    font-size: 13pt;
                }}
            """
        else:
            return f"""
                QToolButton {{
                    background-color: transparent;
                    color: {color};
                    border: 1px solid {color};
                    border-radius: 3px;
                    padding: 4px 12px;
                    font-weight: bold;
                    font-size: 13pt;
                }}
                QToolButton:hover {{
                    background-color: {color}20;
                }}
            """

    @staticmethod
    def _mode_button_style() -> str:
        return """
            QToolButton {
                background-color: transparent;
                color: #e5e5e7;
                border: 1px solid #38383a;
                border-radius: 3px;
                padding: 4px 12px;
                font-size: 13pt;
            }
            QToolButton:checked {
                background-color: #0a84ff;
                color: #ffffff;
                border: none;
            }
            QToolButton:hover:!checked {
                background-color: #3a3a3c;
            }
        """
