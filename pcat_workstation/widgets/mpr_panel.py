"""2x2 MPR panel with linked axial, coronal, sagittal viewers and a CPR placeholder."""

from PySide6.QtWidgets import QWidget, QGridLayout, QLabel
from PySide6.QtCore import Signal, Qt
import numpy as np

from pcat_workstation.widgets.vtk_slice_view import VTKSliceView


class MPRPanel(QWidget):
    """2x2 grid of slice viewers with linked crosshairs.

    Layout:
        Top-left: Axial       Top-right: Coronal
        Bottom-left: Sagittal  Bottom-right: CPR placeholder
    """

    window_level_changed = Signal(float, float)
    crosshair_moved = Signal(float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._linking = False
        self._build_ui()
        self._connect_signals()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._axial = VTKSliceView("axial")
        self._coronal = VTKSliceView("coronal")
        self._sagittal = VTKSliceView("sagittal")

        self._cpr_placeholder = QLabel("CPR — available after pipeline")
        self._cpr_placeholder.setAlignment(Qt.AlignCenter)
        self._cpr_placeholder.setStyleSheet(
            "QLabel { background-color: #0f0f0f; color: #6b6560; font-size: 15pt; }"
        )

        layout.addWidget(self._axial, 0, 0)
        layout.addWidget(self._coronal, 0, 1)
        layout.addWidget(self._sagittal, 1, 0)
        layout.addWidget(self._cpr_placeholder, 1, 1)

        for i in range(2):
            layout.setRowStretch(i, 1)
            layout.setColumnStretch(i, 1)

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.crosshair_moved.connect(self._on_crosshair_moved)
            viewer.window_level_changed.connect(self._on_window_level_changed)

    def _on_crosshair_moved(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        if self._linking:
            return
        self._linking = True
        try:
            sender = self.sender()
            for viewer in (self._axial, self._coronal, self._sagittal):
                if viewer is not sender:
                    viewer.set_crosshair(x_mm, y_mm, z_mm)
            self.crosshair_moved.emit(x_mm, y_mm, z_mm)
        finally:
            self._linking = False

    def _on_window_level_changed(self, window: float, level: float) -> None:
        if self._linking:
            return
        self._linking = True
        try:
            sender = self.sender()
            for viewer in (self._axial, self._coronal, self._sagittal):
                if viewer is not sender:
                    viewer.set_window_level(window, level)
            self.window_level_changed.emit(window, level)
        finally:
            self._linking = False

    # ── Public API ───────────────────────────────────────────────────

    def set_volume(self, volume: np.ndarray, spacing: list) -> None:
        """Pass volume and spacing to all three VTK viewers.

        Builds VTK image data once and shares it across viewers to avoid
        tripling memory usage for the numpy→VTK copy.
        """
        vtk_image = self._axial._numpy_to_vtk_image(volume, spacing)
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_volume_from_vtk(volume, spacing, vtk_image)

    def set_window_level(self, window: float, level: float) -> None:
        """Sync window/level across all viewers."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_window_level(window, level)

    def get_viewers(self) -> dict:
        """Return dict of named viewers."""
        return {
            "axial": self._axial,
            "coronal": self._coronal,
            "sagittal": self._sagittal,
        }

    def start_interactors(self) -> None:
        """Initialize VTK interactors on all viewers. Call after widget is shown."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.start_interactor()
