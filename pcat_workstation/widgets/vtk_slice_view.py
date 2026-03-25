"""VTK-based 2D medical image slice viewer for axial, coronal, and sagittal orientations."""

from __future__ import annotations

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
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
from vtk.util.numpy_support import numpy_to_vtk
from typing import Optional

from pcat_workstation.widgets.overlay_painter import OverlayPainter


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
    left_click_event = Signal(int, int)   # (x_pixel, y_pixel)
    ctrl_scroll_event = Signal(int)      # +1 zoom in, -1 zoom out
    left_press_event = Signal(int, int)     # (x_pixel, y_pixel)
    left_drag_event = Signal(int, int)      # (x_pixel, y_pixel) - emitted after 3px deadzone
    left_release_event = Signal(int, int)   # (x_pixel, y_pixel)
    key_press_event = Signal(int, int)      # (key_code, modifiers)

    def __init__(self, parent=None, **kw):
        super().__init__(parent, **kw)
        self._render_pending = False
        self._in_render = False
        self._right_dragging = False
        self._right_start = (0, 0)
        self._left_pressed = False
        self._left_start = (0, 0)
        self._left_dragging = False  # True once 3px deadzone exceeded
        self._DRAG_DEADZONE = 3  # pixels
        self.setFocusPolicy(Qt.StrongFocus)
        self.grabGesture(Qt.PinchGesture)

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
        # Use pixelDelta for trackpad gestures, angleDelta for mouse wheel
        delta_y = ev.pixelDelta().y() if ev.pixelDelta().y() != 0 else ev.angleDelta().y()
        # Ctrl+scroll OR Cmd+scroll = zoom (macOS pinch sends one of these)
        mods = ev.modifiers()
        ctrl_meta = _Qt.ControlModifier | _Qt.MetaModifier
        if mods & ctrl_meta:
            self.ctrl_scroll_event.emit(1 if delta_y > 0 else -1)
        elif delta_y > 0:
            self.scroll_event.emit(1)
        elif delta_y < 0:
            self.scroll_event.emit(-1)
        ev.accept()

    def event(self, ev):
        """Handle gesture events (macOS trackpad pinch-to-zoom)."""
        from PySide6.QtCore import QEvent
        # Method 1: Native gesture (macOS Cocoa-level)
        if ev.type() == QEvent.NativeGesture:
            try:
                from PySide6.QtGui import QNativeGestureEvent
                if isinstance(ev, QNativeGestureEvent):
                    gt = ev.gestureType()
                    if gt == Qt.ZoomNativeGesture:
                        delta = ev.value()
                        if abs(delta) > 0.001:
                            self.ctrl_scroll_event.emit(1 if delta > 0 else -1)
                        ev.accept()
                        return True
            except (ImportError, AttributeError):
                pass
        # Method 2: Qt gesture framework (QPinchGesture)
        if ev.type() == QEvent.Gesture:
            pinch = ev.gesture(Qt.PinchGesture)
            if pinch is not None:
                scale = pinch.scaleFactor()
                if scale > 1.01:
                    self.ctrl_scroll_event.emit(1)
                elif scale < 0.99:
                    self.ctrl_scroll_event.emit(-1)
                ev.accept()
                return True
        return super().event(ev)

    def mousePressEvent(self, ev):
        from PySide6.QtCore import Qt as _Qt
        if ev.button() == _Qt.RightButton:
            self._right_dragging = True
            self._right_start = (ev.position().x(), ev.position().y())
            self.right_press_event.emit()
        elif ev.button() == _Qt.LeftButton:
            self._left_pressed = True
            self._left_start = (ev.position().x(), ev.position().y())
            self._left_dragging = False
            self.left_press_event.emit(
                int(ev.position().x()), int(ev.position().y())
            )
        ev.accept()

    def mouseReleaseEvent(self, ev):
        from PySide6.QtCore import Qt as _Qt
        if ev.button() == _Qt.RightButton:
            self._right_dragging = False
            self.right_release_event.emit()
        elif ev.button() == _Qt.LeftButton:
            x, y = int(ev.position().x()), int(ev.position().y())
            if self._left_dragging:
                self.left_release_event.emit(x, y)
            else:
                self.left_click_event.emit(x, y)
            self._left_pressed = False
            self._left_dragging = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._right_dragging:
            x, y = ev.position().x(), ev.position().y()
            dx = int(x - self._right_start[0])
            dy = int(y - self._right_start[1])
            self.right_drag_event.emit(dx, dy)
        elif self._left_pressed:
            x, y = ev.position().x(), ev.position().y()
            if not self._left_dragging:
                dist = ((x - self._left_start[0]) ** 2
                        + (y - self._left_start[1]) ** 2) ** 0.5
                if dist > self._DRAG_DEADZONE:
                    self._left_dragging = True
            if self._left_dragging:
                self.left_drag_event.emit(int(x), int(y))
        ev.accept()

    def keyPressEvent(self, ev):
        mods = ev.modifiers()
        mod_val = mods.value if hasattr(mods, "value") else int(mods)
        self.key_press_event.emit(ev.key(), mod_val)
        ev.accept()



class VTKSliceView(QWidget):
    """A 2D medical image slice viewer using VTK.

    Supports axial, coronal, and sagittal orientations with interactive
    window/level, scrolling, zoom, and pan.
    """

    slice_changed = Signal(int)
    crosshair_moved = Signal(float, float, float)
    window_level_changed = Signal(float, float)
    zoom_changed = Signal(float)

    _ORIENTATION_LABELS = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}

    def __init__(self, orientation: str = "axial", parent=None) -> None:
        super().__init__(parent)
        self._orientation = orientation.lower()
        assert self._orientation in ("axial", "coronal", "sagittal")

        self._volume: Optional[np.ndarray] = None
        self._spacing: list = [1.0, 1.0, 1.0]
        self._shape: tuple = (0, 0, 0)  # (Z, Y, X) numpy ordering
        self._current_slice: int = 0
        from pcat_workstation.app.config import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_LEVEL
        self._window: float = float(DEFAULT_WINDOW_WIDTH)
        self._level: float = float(DEFAULT_WINDOW_LEVEL)
        self._overlay_actors: list = []
        self._voi_slice = None
        self._voi_mapper = None
        self._crosshair_actors: list = []  # [h_line_actor, v_line_actor]
        self._crosshair_pos: Optional[tuple] = None  # (x_mm, y_mm, z_mm)

        # SeedEditor reference (new unified model+controller)
        self._seed_editor = None

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

        # QPainter overlay for seeds/centerlines (sits on top of VTK widget)
        self._overlay = OverlayPainter(self)
        self._overlay.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._overlay.setGeometry(self._vtk_widget.geometry())

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
        """Zoom toward/away from cursor position. Clamp at fit-to-window."""
        cam = self._vtk_renderer.GetActiveCamera()
        old_scale = cam.GetParallelScale()

        factor = 1.15 if direction > 0 else 1 / 1.15

        # Don't zoom out past the fit-to-window scale
        new_scale = old_scale / factor
        if hasattr(self, "_fit_scale") and new_scale > self._fit_scale:
            new_scale = self._fit_scale

        cam.SetParallelScale(new_scale)

        self._vtk_renderer.ResetCameraClippingRange()
        self._update_overlay_params()
        self._render()
        self.zoom_changed.emit(new_scale)

    def set_parallel_scale(self, scale: float) -> None:
        """Set camera parallel scale without emitting zoom_changed signal."""
        self._vtk_renderer.GetActiveCamera().SetParallelScale(scale)
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

        # Keep VOI overlay in sync
        if self._voi_mapper is not None:
            self._voi_mapper.SetSliceNumber(index)
            if self._orientation == "axial":
                self._voi_mapper.SetOrientationToZ()
            elif self._orientation == "coronal":
                self._voi_mapper.SetOrientationToY()
            else:
                self._voi_mapper.SetOrientationToX()

        self._update_overlay_params()
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
        """Set slice from patient coordinates (mm) and draw crosshair lines."""
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
        self.update_crosshair_lines(x_mm, y_mm, z_mm)

    def update_crosshair_lines(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        """Draw crosshair reference lines without changing the slice.

        Each view shows two lines indicating where the other two views'
        current slices intersect this plane.
        """
        if self._volume is None:
            return

        self._crosshair_pos = (x_mm, y_mm, z_mm)

        # Remove old crosshair actors
        for actor in self._crosshair_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._crosshair_actors.clear()

        nz, ny, nx = self._shape
        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]
        # Physical extents
        wx, wy, wz = nx * sx, ny * sy, nz * sz

        # Lines must lie ON the current slice plane (fixed axis = current slice position)
        if self._orientation == "axial":
            # Fixed Z plane at current slice; show Y (coronal) and X (sagittal) positions
            fixed_z = self._current_slice * sz
            h_pts = [(0, y_mm, fixed_z), (wx, y_mm, fixed_z)]
            v_pts = [(x_mm, 0, fixed_z), (x_mm, wy, fixed_z)]
        elif self._orientation == "coronal":
            # Fixed Y plane at current slice; show Z (axial) and X (sagittal) positions
            fixed_y = self._current_slice * sy
            h_pts = [(0, fixed_y, z_mm), (wx, fixed_y, z_mm)]
            v_pts = [(x_mm, fixed_y, 0), (x_mm, fixed_y, wz)]
        else:  # sagittal
            # Fixed X plane at current slice; show Z (axial) and Y (coronal) positions
            fixed_x = self._current_slice * sx
            h_pts = [(fixed_x, 0, z_mm), (fixed_x, wy, z_mm)]
            v_pts = [(fixed_x, y_mm, 0), (fixed_x, y_mm, wz)]

        for pts, color in [(h_pts, (0.25, 0.75, 1.0)), (v_pts, (1.0, 0.85, 0.15))]:
            points = vtkPoints()
            points.InsertNextPoint(*pts[0])
            points.InsertNextPoint(*pts[1])
            lines = vtkCellArray()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            pd = vtkPolyData()
            pd.SetPoints(points)
            pd.SetLines(lines)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(pd)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(1.0)
            actor.GetProperty().SetOpacity(0.5)
            self._vtk_renderer.AddActor(actor)
            self._crosshair_actors.append(actor)

        self._render()

    def pixel_to_world(self, qt_x: int, qt_y: int) -> tuple:
        """Convert Qt pixel coordinates to patient/world coordinates (mm).

        Returns ``(x_mm, y_mm, z_mm)`` where x=LR, y=AP, z=SI.
        The fixed axis for this orientation is filled from the current slice.
        """
        if self._volume is None:
            return (0.0, 0.0, 0.0)

        # Convert Qt widget coords to VTK display coords (Y is flipped)
        scale = self._vtk_widget._getPixelRatio()
        display_x = int(qt_x * scale)
        display_y = int((self._vtk_widget.height() - qt_y) * scale)

        self._vtk_renderer.SetDisplayPoint(display_x, display_y, 0)
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

        return (wx, wy, wz)

    def _emit_crosshair_at_cursor(self, qt_x: int, qt_y: int) -> None:
        """Convert Qt click position to patient coords and emit crosshair_moved."""
        if self._volume is None:
            return
        wx, wy, wz = self.pixel_to_world(qt_x, qt_y)
        self.crosshair_moved.emit(wx, wy, wz)

    # ── SeedEditor integration ───────────────────────────────────

    def set_seed_editor(self, editor) -> None:
        """Set the SeedEditor for overlay rendering and mouse interaction.

        Connects editor signals to trigger overlay repaints and wires
        mouse/key events from the VTK widget to the SeedEditor.
        """
        self._seed_editor = editor
        self._overlay.set_seed_editor(editor)

        # Repaint overlay when seeds/centerlines/selection change
        editor.seeds_changed.connect(lambda _v: self._overlay.update())
        editor.centerline_changed.connect(lambda _v: self._overlay.update())
        editor.selection_changed.connect(self._overlay.update)

        # Wire mouse events to SeedEditor
        self._vtk_widget.left_press_event.connect(self._on_left_press)
        self._vtk_widget.left_drag_event.connect(self._on_left_drag)
        self._vtk_widget.left_release_event.connect(self._on_left_release)
        self._vtk_widget.key_press_event.connect(self._on_key_press)

    def _on_left_press(self, qt_x: int, qt_y: int) -> None:
        """Forward left-press to SeedEditor with voxel coordinates."""
        if self._seed_editor is None:
            return
        voxel = self._pixel_to_voxel(qt_x, qt_y)
        if voxel is not None:
            self._seed_editor.on_left_press(voxel)
            self._overlay.update()

    def _on_left_drag(self, qt_x: int, qt_y: int) -> None:
        """Forward left-drag to SeedEditor with voxel coordinates."""
        if self._seed_editor is None:
            return
        voxel = self._pixel_to_voxel(qt_x, qt_y)
        if voxel is not None:
            self._seed_editor.on_left_drag(voxel)
            self._overlay.update()

    def _on_left_release(self, qt_x: int, qt_y: int) -> None:
        """Forward left-release to SeedEditor."""
        if self._seed_editor is None:
            return
        self._seed_editor.on_left_release()
        self._overlay.update()

    def _on_key_press(self, key: int, modifiers: int) -> None:
        """Forward key events to SeedEditor.

        Enter/Return adds a waypoint at the crosshair position (not mouse).
        Other keys (Delete, Ctrl+Z, arrows, etc.) are forwarded directly.
        """
        if self._seed_editor is None:
            return

        _ctrl = Qt.ControlModifier.value if hasattr(Qt.ControlModifier, "value") else int(Qt.ControlModifier)
        _shift = Qt.ShiftModifier.value if hasattr(Qt.ShiftModifier, "value") else int(Qt.ShiftModifier)
        ctrl = bool(modifiers & _ctrl)
        shift = bool(modifiers & _shift)

        if key in (Qt.Key_Return, Qt.Key_Enter):
            # Add waypoint at crosshair position (not mouse position)
            if self._crosshair_pos is not None:
                x_mm, y_mm, z_mm = self._crosshair_pos
                sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]
                voxel = [
                    z_mm / sz if sz > 0 else 0.0,
                    y_mm / sy if sy > 0 else 0.0,
                    x_mm / sx if sx > 0 else 0.0,
                ]
                self._seed_editor.add_waypoint_at(voxel)

        elif key in (Qt.Key_Backspace, Qt.Key_Delete):
            self._seed_editor.delete_selected()

        elif key == Qt.Key_Z and ctrl and shift:
            self._seed_editor.redo()

        elif key == Qt.Key_Z and ctrl:
            self._seed_editor.undo()

        elif key == Qt.Key_Y and ctrl:
            self._seed_editor.redo()

        elif key == Qt.Key_S and ctrl:
            self._seed_editor.save_requested.emit()

        elif key == Qt.Key_Left:
            self._seed_editor.cycle_selection(-1)

        elif key == Qt.Key_Right:
            self._seed_editor.cycle_selection(1)

        self._overlay.update()

    def _pixel_to_voxel(self, qt_x: int, qt_y: int) -> Optional[list]:
        """Convert Qt pixel coordinates to voxel [z, y, x].

        Reverses the voxel_to_screen conversion from OverlayPainter,
        using the VTK camera parameters.
        """
        if self._volume is None:
            return None

        cam = self._vtk_renderer.GetActiveCamera()
        ps = cam.GetParallelScale()
        if ps <= 0:
            return None

        fx, fy, fz = cam.GetFocalPoint()
        w = self._vtk_widget.width()
        h = self._vtk_widget.height()
        if h <= 0:
            return None

        scale = h / (2.0 * ps)  # pixels per mm
        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]

        if self._orientation == "axial":
            # screen_x = (wx - fx) * scale + w/2  =>  wx = (qt_x - w/2) / scale + fx
            # screen_y = (wy - fy) * scale + h/2  =>  wy = (qt_y - h/2) / scale + fy
            world_x = (qt_x - w / 2.0) / scale + fx
            world_y = (qt_y - h / 2.0) / scale + fy
            vox_x = world_x / sx if sx > 0 else 0.0
            vox_y = world_y / sy if sy > 0 else 0.0
            vox_z = float(self._current_slice)
            return [vox_z, vox_y, vox_x]

        elif self._orientation == "coronal":
            # screen_x = (wx - fx) * scale + w/2
            # screen_y = (fz - wz) * scale + h/2  =>  wz = fz - (qt_y - h/2) / scale
            world_x = (qt_x - w / 2.0) / scale + fx
            world_z = fz - (qt_y - h / 2.0) / scale
            vox_x = world_x / sx if sx > 0 else 0.0
            vox_y = float(self._current_slice)
            vox_z = world_z / sz if sz > 0 else 0.0
            return [vox_z, vox_y, vox_x]

        elif self._orientation == "sagittal":
            # screen_x = (wy - fy) * scale + w/2
            # screen_y = (fz - wz) * scale + h/2
            world_y = (qt_x - w / 2.0) / scale + fy
            world_z = fz - (qt_y - h / 2.0) / scale
            vox_x = float(self._current_slice)
            vox_y = world_y / sy if sy > 0 else 0.0
            vox_z = world_z / sz if sz > 0 else 0.0
            return [vox_z, vox_y, vox_x]

        return None

    def _update_overlay_params(self) -> None:
        """Push current view parameters to the QPainter overlay."""
        cam = self._vtk_renderer.GetActiveCamera()
        self._overlay.set_view_params(
            orientation=self._orientation,
            current_slice=self._current_slice,
            spacing=self._spacing,
            volume_shape=self._shape,
            parallel_scale=cam.GetParallelScale(),
            focal_point=cam.GetFocalPoint(),
        )
        self._overlay.update()

    # ── Camera ──────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Orient camera for the current slice plane and fill the viewport.

        Follows ImageJ / radiology conventions:
        - Axial:    look from inferior (feet), anterior at top (ViewUp = 0,-1,0)
        - Coronal:  look from posterior (facing patient), superior at top (ViewUp = 0,0,1)
        - Sagittal: look from right, superior at top (ViewUp = 0,0,1)
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
            # Camera below (at feet), looking up +Z — ImageJ/radiology convention
            # Screen right = +X (patient left), screen up = -Y (anterior)
            cam.SetPosition(cx, cy, cz - dist)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, -1, 0)  # anterior at top
            half_w, half_h = wx / 2, wy / 2
        elif self._orientation == "coronal":
            # Camera behind (posterior), looking +Y — facing the patient
            # Screen right = +X (patient left), screen up = +Z (superior)
            cam.SetPosition(cx, cy - dist, cz)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, 0, 1)  # superior at top
            half_w, half_h = wx / 2, wz / 2
        else:  # sagittal
            # Camera on right side, looking -X
            # Screen right = +Y (posterior), screen up = +Z (superior)
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetViewUp(0, 0, 1)  # superior at top
            half_w, half_h = wy / 2, wz / 2

        # Compute parallel scale to fill viewport (like ImageJ's "Fit")
        widget_w = max(self._vtk_widget.width(), 1)
        widget_h = max(self._vtk_widget.height(), 1)
        aspect = widget_w / widget_h

        # ParallelScale = half the viewport height in world coords.
        # Pick whichever dimension is the limiting factor.
        scale_by_height = half_h
        scale_by_width = half_w / aspect
        fit_scale = max(scale_by_height, scale_by_width)
        cam.SetParallelScale(fit_scale)
        self._fit_scale = fit_scale  # minimum zoom (can't zoom out past this)

        self._vtk_renderer.ResetCameraClippingRange()
        self._update_overlay_params()
        self._render()

    # ── Overlay rendering ─────────────────────────────────────────

    def clear_overlays(self) -> None:
        """Remove all overlay actors."""
        for actor in self._overlay_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._overlay_actors.clear()
        for actor in self._crosshair_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._crosshair_actors.clear()
        self._crosshair_pos = None
        if self._voi_slice is not None:
            self._vtk_renderer.RemoveViewProp(self._voi_slice)
            self._voi_slice = None
            self._voi_mapper = None
        self._overlay.update()
        self._render()

    def set_voi_overlay(self, voi_masks_dict: dict, spacing: list) -> None:
        """Show semi-transparent colored VOI mask overlay."""
        if self._volume is None:
            return

        nz, ny, nx = self._shape
        combined = np.zeros((nz, ny, nx), dtype=np.uint8)

        vessel_ids = {"LAD": 1, "LCx": 2, "LCX": 2, "RCA": 3}
        for vessel, mask in voi_masks_dict.items():
            vid = vessel_ids.get(vessel, 0)
            if vid and mask.shape == (nz, ny, nx):
                combined[mask] = vid

        if combined.max() == 0:
            return

        # Build RGBA image (4 components) — subtle tint, Horos style
        rgba = np.zeros((nz, ny, nx, 4), dtype=np.uint8)
        color_map = {
            1: (255, 180, 50, 128),  # LAD — warm yellow-orange, visible
            2: (100, 170, 240, 128), # LCx — blue, visible
            3: (80, 220, 140, 128),  # RCA — green, visible
        }
        for vid, color in color_map.items():
            mask = combined == vid
            rgba[mask] = color

        flat = rgba.ravel()
        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=3)  # VTK_UNSIGNED_CHAR
        vtk_arr.SetNumberOfComponents(4)

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(spacing[2], spacing[1], spacing[0])
        vtk_img.SetOrigin(0.0, 0.0, 0.0)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        self._voi_mapper = vtkImageSliceMapper()
        self._voi_mapper.SetInputData(vtk_img)
        # Match current orientation and slice
        self._voi_mapper.SetSliceNumber(self._current_slice)
        if self._orientation == "axial":
            self._voi_mapper.SetOrientationToZ()
        elif self._orientation == "coronal":
            self._voi_mapper.SetOrientationToY()
        else:
            self._voi_mapper.SetOrientationToX()

        voi_prop = vtkImageProperty()
        voi_prop.SetInterpolationTypeToNearest()

        self._voi_slice = vtkImageSlice()
        self._voi_slice.SetMapper(self._voi_mapper)
        self._voi_slice.SetProperty(voi_prop)

        self._vtk_renderer.AddViewProp(self._voi_slice)
        self._render()

    # ── Render helper ───────────────────────────────────────────────

    def _render(self) -> None:
        self._vtk_widget.request_render()

    # ── Cleanup ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Ensure VTK resources are cleaned up."""
        self._vtk_widget.GetRenderWindow().Finalize()
        super().closeEvent(event)
