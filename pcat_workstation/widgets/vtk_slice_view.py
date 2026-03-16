"""VTK-based 2D medical image slice viewer for axial, coronal, and sagittal orientations."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Signal, Qt
import numpy as np
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - needed for VTK rendering
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkImageSlice,
    vtkImageSliceMapper,
    vtkImageProperty,
)
from vtk.util.numpy_support import numpy_to_vtk
from typing import Optional


class _SafeVTKWidget(QVTKRenderWindowInteractor):
    """QVTKRenderWindowInteractor subclass safe for macOS.

    On macOS the stock widget enters an infinite paint loop after the
    interactor is initialized/enabled:
      paintEvent → Render() → resizeEvent → update() → paintEvent …
    This starves the Qt event loop so timers and user events never fire.

    Fix: override paintEvent and resizeEvent with reentrancy-safe versions
    that break the cycle, and handle mouse/scroll events via Qt signals
    instead of VTK's observer system (which requires Initialize()).
    """

    # Signals for the owning VTKSliceView to connect to
    scroll_event = Signal(int)           # delta: +1 forward, -1 backward
    right_drag_event = Signal(int, int)  # dx, dy from drag start
    right_press_event = Signal()
    right_release_event = Signal()
    left_click_event = Signal()
    ctrl_scroll_event = Signal(int)      # +1 zoom in, -1 zoom out

    def __init__(self, parent=None, **kw):
        super().__init__(parent, **kw)
        self._render_pending = False
        self._in_render = False
        self._right_dragging = False
        self._right_start = (0, 0)

    def request_render(self):
        """Schedule a VTK render after a short delay.

        Uses singleShot(16) (~60fps) to coalesce rapid render requests
        and give the macOS event loop time between renders.  Without this
        delay, back-to-back Render() calls across multiple VTK widgets
        starve the Qt event loop.
        """
        if self._render_pending:
            return
        self._render_pending = True
        from PySide6.QtCore import QTimer as _QTimer
        _QTimer.singleShot(16, self._do_render)

    def _do_render(self):
        self._render_pending = False
        if self._in_render:
            return
        self._in_render = True
        try:
            self._RenderWindow.Render()
        finally:
            self._in_render = False

    def CreateTimer(self, obj, evt):
        pass

    def paintEvent(self, ev):
        # No-op: rendering is driven by _render_timer, not paint events
        pass

    def resizeEvent(self, ev):
        if self._in_render:
            return
        scale = self._getPixelRatio()
        w = int(round(scale * self.width()))
        h = int(round(scale * self.height()))
        self._RenderWindow.SetDPI(int(round(72 * scale)))
        from vtkmodules.vtkRenderingCore import vtkRenderWindow
        vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self.request_render()

    # ── Qt event handlers (bypass VTK interactor) ────────────────

    def wheelEvent(self, ev):
        from PySide6.QtCore import Qt as _Qt
        delta = ev.angleDelta().y()
        if ev.modifiers() & _Qt.ControlModifier:
            self.ctrl_scroll_event.emit(1 if delta > 0 else -1)
        elif delta > 0:
            self.scroll_event.emit(1)
        elif delta < 0:
            self.scroll_event.emit(-1)
        ev.accept()

    def mousePressEvent(self, ev):
        from PySide6.QtCore import Qt as _Qt
        if ev.button() == _Qt.RightButton:
            self._right_dragging = True
            self._right_start = (ev.position().x(), ev.position().y())
            self.right_press_event.emit()
        elif ev.button() == _Qt.LeftButton:
            self.left_click_event.emit()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        from PySide6.QtCore import Qt as _Qt
        if ev.button() == _Qt.RightButton:
            self._right_dragging = False
            self.right_release_event.emit()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._right_dragging:
            x, y = ev.position().x(), ev.position().y()
            dx = int(x - self._right_start[0])
            dy = int(y - self._right_start[1])
            self.right_drag_event.emit(dx, dy)
        ev.accept()


class VTKSliceView(QWidget):
    """A 2D medical image slice viewer using VTK.

    Supports axial, coronal, and sagittal orientations with interactive
    window/level, scrolling, zoom, and pan.
    """

    slice_changed = Signal(int)
    crosshair_moved = Signal(float, float, float)
    window_level_changed = Signal(float, float)

    _ORIENTATION_LABELS = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}

    def __init__(self, orientation: str = "axial", parent=None) -> None:
        super().__init__(parent)
        self._orientation = orientation.lower()
        assert self._orientation in ("axial", "coronal", "sagittal")

        self._volume: Optional[np.ndarray] = None
        self._spacing: list = [1.0, 1.0, 1.0]
        self._shape: tuple = (0, 0, 0)  # (Z, Y, X) numpy ordering
        self._current_slice: int = 0
        self._window: float = 1500.0
        self._level: float = 300.0

        self._build_ui()
        self._setup_vtk()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFrameShape = QFrame.Box
        self.setStyleSheet(
            "VTKSliceView { border: 1px solid #2a2a2a; background-color: #0f0f0f; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header label
        self._header = QLabel(self._ORIENTATION_LABELS[self._orientation])
        self._header.setStyleSheet(
            "QLabel { color: #e0e0e0; background-color: transparent; "
            "padding: 4px 8px; font-size: 13pt; font-weight: bold; }"
        )
        self._header.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self._header)

        # VTK render widget (guarded against reentrant render on macOS)
        self._vtk_widget = _SafeVTKWidget(self)
        layout.addWidget(self._vtk_widget, stretch=1)

    # ── VTK pipeline ────────────────────────────────────────────────

    def _setup_vtk(self) -> None:
        self._vtk_renderer = vtkRenderer()
        self._vtk_renderer.SetBackground(0.059, 0.059, 0.059)

        render_window = self._vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._vtk_renderer)

        # Mapper
        self._mapper = vtkImageSliceMapper()
        self._mapper.SliceFacesCameraOff()
        self._mapper.SliceAtFocalPointOff()

        # Image property (window/level, interpolation)
        self._image_property = vtkImageProperty()
        self._image_property.SetColorWindow(self._window)
        self._image_property.SetColorLevel(self._level)
        self._image_property.SetInterpolationTypeToLinear()

        # Image slice actor (added to renderer only when volume is loaded)
        self._image_slice = vtkImageSlice()
        self._image_slice.SetMapper(self._mapper)
        self._image_slice.SetProperty(self._image_property)
        self._actor_added = False

        # Connect Qt-level mouse/scroll signals from the safe widget
        self._wl_start = (self._window, self._level)
        self._vtk_widget.scroll_event.connect(self._on_scroll)
        self._vtk_widget.ctrl_scroll_event.connect(self._on_ctrl_scroll)
        self._vtk_widget.right_press_event.connect(self._on_right_press)
        self._vtk_widget.right_drag_event.connect(self._on_right_drag)
        self._vtk_widget.left_click_event.connect(self._emit_crosshair_at_cursor)

    def start_interactor(self) -> None:
        """No-op — events are handled via Qt signals, not VTK interactor."""
        pass

    def _on_ctrl_scroll(self, direction: int) -> None:
        factor = 1.1 if direction > 0 else 0.9
        self._vtk_renderer.GetActiveCamera().Zoom(factor)
        self._render()

    def _on_right_press(self) -> None:
        self._wl_start = (self._window, self._level)

    def _on_right_drag(self, dx: int, dy: int) -> None:
        w0, l0 = self._wl_start
        self.set_window_level(max(1.0, w0 + dx * 2.0), l0 + dy * 2.0)

    # ── Volume loading ──────────────────────────────────────────────

    def set_volume(self, volume: np.ndarray, spacing: list) -> None:
        """Load a numpy volume (Z, Y, X) float32 with given spacing [sz, sy, sx]."""
        self._vtk_flat = np.ascontiguousarray(volume, dtype=np.float32).ravel()
        vtk_image = VTKSliceView.build_vtk_image_data(volume, spacing, self._vtk_flat)
        self.set_volume_from_vtk(volume, spacing, vtk_image)

    def set_volume_from_vtk(
        self, volume: np.ndarray, spacing: list, vtk_image: vtkImageData
    ) -> None:
        """Load from a pre-built vtkImageData (avoids redundant copies)."""
        self._volume = volume
        self._spacing = list(spacing)
        self._shape = volume.shape  # (Z, Y, X)

        self._mapper.SetInputData(vtk_image)

        if not self._actor_added:
            self._vtk_renderer.AddViewProp(self._image_slice)
            self._actor_added = True

        # Set initial orientation and go to middle slice
        mid = self._max_slice() // 2
        self.set_slice(mid)
        self.reset_camera()

    @staticmethod
    def build_vtk_image_data(
        volume: np.ndarray, spacing: list, flat_array: np.ndarray
    ) -> vtkImageData:
        """Build vtkImageData from a pre-flattened numpy array.

        *flat_array* must be a contiguous float32 ravel of *volume*.
        The caller is responsible for keeping *flat_array* alive (deep=False).
        """
        nz, ny, nx = volume.shape

        vtk_image = vtkImageData()
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(spacing[2], spacing[1], spacing[0])  # sx, sy, sz
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        vtk_arr = numpy_to_vtk(flat_array, deep=False, array_type=10)  # VTK_FLOAT = 10
        vtk_arr.SetNumberOfComponents(1)
        vtk_image.GetPointData().SetScalars(vtk_arr)

        return vtk_image

    # ── Slice navigation ────────────────────────────────────────────

    def _max_slice(self) -> int:
        """Return the maximum slice index for the current orientation."""
        if self._volume is None:
            return 0
        nz, ny, nx = self._shape
        if self._orientation == "axial":
            return nz - 1
        elif self._orientation == "coronal":
            return ny - 1
        else:  # sagittal
            return nx - 1

    def set_slice(self, index: int) -> None:
        """Set the displayed slice, clamped to valid range."""
        if self._volume is None:
            return

        index = max(0, min(index, self._max_slice()))
        self._current_slice = index

        nz, ny, nx = self._shape

        if self._orientation == "axial":
            # Fix Z, show full X, Y
            self._mapper.SetSliceNumber(index)
            self._mapper.SetOrientationToZ()
        elif self._orientation == "coronal":
            # Fix Y, show full X, Z
            self._mapper.SetSliceNumber(index)
            self._mapper.SetOrientationToY()
        else:  # sagittal
            # Fix X, show full Y, Z
            self._mapper.SetSliceNumber(index)
            self._mapper.SetOrientationToX()

        self._update_header()
        self._render()
        self.slice_changed.emit(self._current_slice)

    def get_slice(self) -> int:
        """Return current slice index."""
        return self._current_slice

    def _update_header(self) -> None:
        label = self._ORIENTATION_LABELS[self._orientation]
        total = self._max_slice() + 1
        self._header.setText(f"{label}: {self._current_slice + 1}/{total}")

    def _on_scroll(self, delta: int) -> None:
        """Handle scroll: move slice by delta."""
        self.set_slice(self._current_slice + delta)

    # ── Window / Level ──────────────────────────────────────────────

    def set_window_level(self, window: float, level: float) -> None:
        """Set display window width and level."""
        self._window = max(1.0, window)
        self._level = level
        self._image_property.SetColorWindow(self._window)
        self._image_property.SetColorLevel(self._level)
        self._render()
        self.window_level_changed.emit(self._window, self._level)

    def get_window_level(self) -> tuple:
        """Return (window, level)."""
        return (self._window, self._level)

    # ── Crosshair ───────────────────────────────────────────────────

    def set_crosshair(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        """Set slice from patient coordinates (mm). Crosshair drawing deferred to Phase 2."""
        if self._volume is None:
            return

        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]

        if self._orientation == "axial":
            voxel_idx = int(round(z_mm / sz)) if sz > 0 else 0
        elif self._orientation == "coronal":
            voxel_idx = int(round(y_mm / sy)) if sy > 0 else 0
        else:  # sagittal
            voxel_idx = int(round(x_mm / sx)) if sx > 0 else 0

        self.set_slice(voxel_idx)

    def _emit_crosshair_at_cursor(self) -> None:
        """Convert current cursor position to patient coords and emit crosshair_moved."""
        if self._volume is None:
            return

        interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        event_x, event_y = interactor.GetEventPosition()

        # Pick the world coordinate at cursor
        self._vtk_renderer.SetDisplayPoint(event_x, event_y, 0)
        self._vtk_renderer.DisplayToWorld()
        world = self._vtk_renderer.GetWorldPoint()

        if world[3] != 0.0:
            wx = world[0] / world[3]
            wy = world[1] / world[3]
            wz = world[2] / world[3]
        else:
            wx, wy, wz = world[0], world[1], world[2]

        # Fill in the fixed axis from current slice position
        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]

        if self._orientation == "axial":
            wz = self._current_slice * sz
        elif self._orientation == "coronal":
            wy = self._current_slice * sy
        else:  # sagittal
            wx = self._current_slice * sx

        self.crosshair_moved.emit(wx, wy, wz)

    # ── Camera ──────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Orient camera for the current slice plane and fill the viewport.

        Follows ImageJ / radiology conventions:
        - Axial:    look from superior, row 0 at top (ViewUp = 0,-1,0)
        - Coronal:  look from anterior, superior at top (ViewUp = 0,0,1)
        - Sagittal: look from right,    superior at top (ViewUp = 0,0,1)
        Uses parallel projection so the image fills the widget like ImageJ.
        """
        if self._volume is None:
            return

        nz, ny, nx = self._shape
        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]

        # Physical extents (mm)
        wx, wy, wz = nx * sx, ny * sy, nz * sz
        cx, cy, cz = wx / 2, wy / 2, wz / 2
        dist = max(wx, wy, wz) * 2  # far enough to see everything

        cam = self._vtk_renderer.GetActiveCamera()
        cam.ParallelProjectionOn()

        if self._orientation == "axial":
            # Camera above, looking down -Z
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, -1, 0)  # flip Y so row 0 = top (ImageJ)
            half_w, half_h = wx / 2, wy / 2
        elif self._orientation == "coronal":
            # Camera in front, looking down -Y
            cam.SetPosition(cx, cy + dist, cz)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, 0, 1)  # Z up (superior at top)
            half_w, half_h = wx / 2, wz / 2
        else:  # sagittal
            # Camera on right side, looking down -X
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, 0, 1)  # Z up (superior at top)
            half_w, half_h = wy / 2, wz / 2

        # Compute parallel scale to fill viewport (like ImageJ's "Fit")
        widget_w = max(self._vtk_widget.width(), 1)
        widget_h = max(self._vtk_widget.height(), 1)
        aspect = widget_w / widget_h

        # ParallelScale = half the viewport height in world coords.
        # Pick whichever dimension is the limiting factor.
        scale_by_height = half_h
        scale_by_width = half_w / aspect
        cam.SetParallelScale(max(scale_by_height, scale_by_width))

        self._vtk_renderer.ResetCameraClippingRange()
        self._render()

    # ── Render helper ───────────────────────────────────────────────

    def _render(self) -> None:
        self._vtk_widget.request_render()

    # ── Cleanup ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Ensure VTK resources are cleaned up."""
        self._vtk_widget.GetRenderWindow().Finalize()
        super().closeEvent(event)
