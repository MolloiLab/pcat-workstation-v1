"""2x2 MPR panel with linked axial, coronal, sagittal viewers and interactive CPR."""

from PySide6.QtWidgets import QWidget, QGridLayout
from PySide6.QtCore import Signal
import numpy as np

from pcat_workstation.widgets.vtk_slice_view import VTKSliceView
from pcat_workstation.widgets.cpr_view import CPRView


class MPRPanel(QWidget):
    """2x2 grid of slice viewers with linked crosshairs.

    Layout:
        Top-left: Axial       Top-right: Coronal
        Bottom-left: Sagittal  Bottom-right: Interactive CPR
    """

    window_level_changed = Signal(float, float)
    crosshair_moved = Signal(float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._linking = False
        self._volume = None
        self._spacing = None
        self._contour_results: dict = {}  # vessel -> ContourResult
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

        self._cpr_view = CPRView()

        layout.addWidget(self._axial, 0, 0)
        layout.addWidget(self._coronal, 0, 1)
        layout.addWidget(self._sagittal, 1, 0)
        layout.addWidget(self._cpr_view, 1, 1)

        for i in range(2):
            layout.setRowStretch(i, 1)
            layout.setColumnStretch(i, 1)

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.crosshair_moved.connect(self._on_crosshair_moved)
            viewer.window_level_changed.connect(self._on_window_level_changed)
            viewer.slice_changed.connect(self._on_slice_changed)

        # CPR → MPR sync: when needle moves, jump MPR viewers to that 3D location
        self._cpr_view.needle_moved.connect(self._on_cpr_needle_moved)
        self._cpr_view.window_level_changed.connect(self._on_window_level_changed)

    def _on_slice_changed(self, _index: int) -> None:
        """When any viewer scrolls, update crosshair lines on ALL viewers."""
        if self._linking or self._spacing is None:
            return
        self._linking = True
        try:
            # Build the current 3D position from all viewers' slices
            sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]
            x_mm = self._sagittal.get_slice() * sx
            y_mm = self._coronal.get_slice() * sy
            z_mm = self._axial.get_slice() * sz

            for viewer in (self._axial, self._coronal, self._sagittal):
                viewer.update_crosshair_lines(x_mm, y_mm, z_mm)
        finally:
            self._linking = False

    def _on_crosshair_moved(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        if self._linking:
            return
        self._linking = True
        try:
            sender = self.sender()
            for viewer in (self._axial, self._coronal, self._sagittal):
                if viewer is not sender:
                    viewer.set_crosshair(x_mm, y_mm, z_mm)
                else:
                    # Update crosshair lines on the sender too (it changed position)
                    viewer.update_crosshair_lines(x_mm, y_mm, z_mm)
            self.crosshair_moved.emit(x_mm, y_mm, z_mm)
        finally:
            self._linking = False

    def _on_cpr_needle_moved(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        """CPR needle moved — sync all MPR viewers to the 3D position."""
        if self._linking:
            return
        self._linking = True
        try:
            for viewer in (self._axial, self._coronal, self._sagittal):
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
            if sender is not self._cpr_view:
                self._cpr_view.set_window_level(window, level)
            self.window_level_changed.emit(window, level)
        finally:
            self._linking = False

    # ── Public API ───────────────────────────────────────────────────

    def set_volume(self, volume: np.ndarray, spacing: list) -> None:
        """Pass volume and spacing to all three VTK viewers and CPR view."""
        self._volume = volume
        self._spacing = list(spacing)
        self._vtk_flat = np.ascontiguousarray(volume, dtype=np.float32).ravel()
        vtk_image = VTKSliceView.build_vtk_image_data(volume, spacing, self._vtk_flat)
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_volume_from_vtk(volume, spacing, vtk_image)

        # Initialize crosshairs at center of volume
        sx, sy, sz = spacing[2], spacing[1], spacing[0]
        nz, ny, nx = volume.shape
        cx = (nx // 2) * sx
        cy = (ny // 2) * sy
        cz = (nz // 2) * sz
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.update_crosshair_lines(cx, cy, cz)

    def set_window_level(self, window: float, level: float) -> None:
        """Sync window/level across all viewers including CPR."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_window_level(window, level)
        self._cpr_view.set_window_level(window, level)

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

    def clear_overlays(self) -> None:
        """Remove all overlays from all viewers."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.clear_overlays()

    def set_seed_overlay(self, seeds_dict: dict, spacing: list) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_seed_overlay(seeds_dict, spacing)

    def set_centerline_overlay(self, centerlines_dict: dict, spacing: list) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_centerline_overlay(centerlines_dict, spacing)

    def set_contour_overlay(self, contour_results_dict: dict) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_contour_overlay(contour_results_dict)

    def set_voi_overlay(self, voi_masks_dict: dict, spacing: list) -> None:
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_voi_overlay(voi_masks_dict, spacing)

    def set_contour_data(self, contour_results_dict: dict) -> None:
        """Store ContourResult objects and pass to CPR view for cross-section rendering."""
        self._contour_results = contour_results_dict
        if self._volume is not None and self._spacing is not None:
            for vessel, cr in contour_results_dict.items():
                self._cpr_view.set_contour_data(
                    vessel, cr, self._volume, self._spacing,
                )

    def set_cpr_data(self, vessel: str, cpr_image: np.ndarray, row_extent_mm: float = 25.0) -> None:
        """Store a CPR image for a vessel in the CPR view."""
        self._cpr_view.set_cpr_data(vessel, cpr_image, row_extent_mm)

    def set_cpr_frame(self, vessel: str, frame_data: dict) -> None:
        """Pass CPR Bishop frame data to the CPR view for cross-section sampling."""
        self._cpr_view.set_cpr_frame(vessel, frame_data)

    def set_cpr_vessel(self, vessel: str) -> None:
        """Switch which vessel's CPR is displayed."""
        self._cpr_view.set_vessel(vessel)

    def set_edit_mode(self, enabled: bool) -> None:
        """Toggle edit mode on all VTK slice views."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_edit_mode(enabled)

    def set_edit_controller(self, controller) -> None:
        """Set the edit controller on all VTK slice views."""
        for viewer in (self._axial, self._coronal, self._sagittal):
            viewer.set_edit_controller(controller)

    def refresh_seed_overlay(self, state) -> None:
        """Rebuild seed overlays from SeedEditState."""
        if self._spacing is not None:
            for viewer in (self._axial, self._coronal, self._sagittal):
                viewer.set_seed_overlay_extended(state, self._spacing)

    def clear_cpr(self) -> None:
        """Clear CPR data."""
        self._contour_results.clear()
        self._cpr_view.clear()
