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
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource, vtkSphereSource, vtkCubeSource
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtk.util.numpy_support import numpy_to_vtk
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pcat_workstation.models.seed_edit_state import SeedEditState


# Horos-style vessel colors (from pipeline/visualize.py)
_VESSEL_COLORS_RGB = {
    "LAD": (232, 83, 58),    # #E8533A — red-orange
    "LCx": (74, 144, 217),   # #4A90D9 — blue
    "LCX": (74, 144, 217),
    "RCA": (46, 204, 113),   # #2ECC71 — green
}


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
        self._seed_actor_info: list = []  # [{"actors": [...], "world_pos": (cx, cy, cz)}]
        self._voi_slice = None
        self._voi_mapper = None
        self._crosshair_actors: list = []  # [h_line_actor, v_line_actor]
        self._crosshair_pos: Optional[tuple] = None  # (x_mm, y_mm, z_mm)

        # Edit mode state
        self._edit_mode: bool = False
        self._edit_controller = None  # SeedEditController reference
        self._highlight_actors: list = []  # yellow selection highlight actors

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
        scale = self._vtk_renderer.GetActiveCamera().GetParallelScale()
        self.zoom_changed.emit(scale)

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

        self._update_seed_visibility()
        self._update_centerline_clip()
        self._update_header()
        self._render()
        self.slice_changed.emit(self._current_slice)

    def get_slice(self) -> int:
        """Return current slice index."""
        return self._current_slice

    def _update_seed_visibility(self) -> None:
        """Show seed markers only when the current slice is within ±2 mm."""
        if not self._seed_actor_info or self._volume is None:
            return
        sx, sy, sz = self._spacing[2], self._spacing[1], self._spacing[0]
        # Current slice position in mm along the slicing axis
        if self._orientation == "axial":
            slice_mm = self._current_slice * sz
        elif self._orientation == "coronal":
            slice_mm = self._current_slice * sy
        else:  # sagittal
            slice_mm = self._current_slice * sx

        for info in self._seed_actor_info:
            cx, cy, cz = info["world_pos"]
            if self._orientation == "axial":
                seed_mm = cz
            elif self._orientation == "coronal":
                seed_mm = cy
            else:
                seed_mm = cx
            visible = abs(seed_mm - slice_mm) <= 2.0
            for actor in info["actors"]:
                actor.SetVisibility(visible)

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

    # ── Edit mode ──────────────────────────────────────────────────

    def set_edit_mode(self, enabled: bool) -> None:
        """Toggle edit mode on/off.

        When enabled, mouse events are forwarded to the attached
        ``SeedEditController`` in addition to crosshair navigation.
        Crosshair on click remains active so clicking empty space
        still navigates.
        """
        if enabled == self._edit_mode:
            return
        self._edit_mode = enabled

        if not enabled:
            # Leaving edit mode — clear selection highlights
            self.clear_selection_highlight()

    def set_edit_controller(self, controller) -> None:
        """Attach a :class:`SeedEditController` and wire signals.

        The controller's event handlers receive ``(view, x, y)`` so we
        use ``functools.partial`` to curry *self* as the first argument.
        """
        from functools import partial

        # Disconnect previous controller if any
        if self._edit_controller is not None:
            self._disconnect_edit_signals()

        self._edit_controller = controller

        if controller is not None:
            v = self  # captured by lambdas
            self._vtk_widget.left_press_event.connect(
                lambda x, y, _v=v: controller.on_left_press(_v, x, y)
            )
            self._vtk_widget.left_drag_event.connect(
                lambda x, y, _v=v: controller.on_left_drag(_v, x, y)
            )
            self._vtk_widget.left_release_event.connect(
                lambda x, y, _v=v: controller.on_left_release(_v, x, y)
            )
            self._vtk_widget.key_press_event.connect(
                lambda key, mods, _v=v: controller.on_key_press(_v, key, mods)
            )

    def _disconnect_edit_signals(self) -> None:
        """Disconnect controller signals from the VTK widget."""
        for sig in (
            self._vtk_widget.left_press_event,
            self._vtk_widget.left_drag_event,
            self._vtk_widget.left_release_event,
            self._vtk_widget.key_press_event,
        ):
            try:
                sig.disconnect()
            except RuntimeError:
                pass

        # Re-add non-edit connections that were on these signals
        # (left_press/drag/release have no default connections besides edit)

    def set_seed_overlay_extended(
        self, state: SeedEditState, spacing: list
    ) -> None:
        """Render seeds with distinct markers: squares for ostia, circles for waypoints.

        Stores ``{"vessel", "type", "index"}`` metadata alongside each
        seed's actors in ``_seed_actor_info`` so the controller can map
        pick results back to the data model.
        """
        # Remove existing seed overlay actors
        for info in self._seed_actor_info:
            for actor in info["actors"]:
                self._vtk_renderer.RemoveActor(actor)
        # Also remove from _overlay_actors list
        actor_set = set()
        for info in self._seed_actor_info:
            for actor in info["actors"]:
                actor_set.add(id(actor))
        self._overlay_actors = [
            a for a in self._overlay_actors if id(a) not in actor_set
        ]
        self._seed_actor_info.clear()

        sx, sy, sz = spacing[2], spacing[1], spacing[0]

        for vessel, entry in state.seeds.items():
            rgb = _VESSEL_COLORS_RGB.get(vessel, (232, 83, 58))

            # --- Ostium (cube marker) ---
            if entry["ostium"] is not None:
                ijk = entry["ostium"]
                z, y, x = float(ijk[0]), float(ijk[1]), float(ijk[2])
                cx, cy, cz = x * sx, y * sy, z * sz
                actors = self._create_cube_marker(cx, cy, cz, 5.0, rgb)
                self._seed_actor_info.append({
                    "actors": actors,
                    "world_pos": (cx, cy, cz),
                    "vessel": vessel,
                    "type": "ostium",
                    "index": 0,
                })

            # --- Waypoints (sphere markers) ---
            for idx, wp in enumerate(entry["waypoints"]):
                z, y, x = float(wp[0]), float(wp[1]), float(wp[2])
                cx, cy, cz = x * sx, y * sy, z * sz
                actors = self._create_sphere_marker(cx, cy, cz, 2.5, rgb)
                self._seed_actor_info.append({
                    "actors": actors,
                    "world_pos": (cx, cy, cz),
                    "vessel": vessel,
                    "type": "waypoint",
                    "index": idx,
                })

        self._update_seed_visibility()
        self._render()

    def _create_sphere_marker(
        self,
        cx: float, cy: float, cz: float,
        radius: float,
        rgb: tuple,
    ) -> list:
        """Create a flat 2D disc marker for a waypoint seed (aligned to slice plane)."""
        from vtkmodules.vtkFiltersSources import vtkDiskSource
        disk = vtkDiskSource()
        disk.SetInnerRadius(0)
        disk.SetOuterRadius(radius)
        disk.SetRadialResolution(1)
        disk.SetCircumferentialResolution(24)
        disk.Update()

        # Transform to position and align to slice plane
        from vtkmodules.vtkCommonTransforms import vtkTransform
        from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
        t = vtkTransform()
        t.Translate(cx, cy, cz)
        if self._orientation == "coronal":
            t.RotateX(90)
        elif self._orientation == "sagittal":
            t.RotateY(90)
        # axial: disk already in XY plane (default)
        tf = vtkTransformPolyDataFilter()
        tf.SetInputData(disk.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        actors = []
        # White outline
        m1 = vtkPolyDataMapper()
        m1.SetInputData(tf.GetOutput())
        a1 = vtkActor()
        a1.SetMapper(m1)
        a1.GetProperty().SetColor(1.0, 1.0, 1.0)
        a1.GetProperty().SetRepresentationToWireframe()
        a1.GetProperty().SetLineWidth(1.0)
        a1.GetProperty().SetOpacity(0.7)
        self._vtk_renderer.AddActor(a1)
        self._overlay_actors.append(a1)
        actors.append(a1)

        # Colored fill
        m2 = vtkPolyDataMapper()
        m2.SetInputData(tf.GetOutput())
        a2 = vtkActor()
        a2.SetMapper(m2)
        a2.GetProperty().SetColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        a2.GetProperty().SetRepresentationToSurface()
        a2.GetProperty().SetOpacity(0.85)
        self._vtk_renderer.AddActor(a2)
        self._overlay_actors.append(a2)
        actors.append(a2)

        return actors

    def _create_cube_marker(
        self,
        cx: float, cy: float, cz: float,
        size: float,
        rgb: tuple,
    ) -> list:
        """Create a flat 2D square marker for an ostium seed (aligned to slice plane)."""
        from vtkmodules.vtkFiltersSources import vtkPlaneSource
        plane = vtkPlaneSource()
        hs = size / 2.0  # half-size
        # Align to slice orientation
        if self._orientation == "axial":
            plane.SetOrigin(cx - hs, cy - hs, cz)
            plane.SetPoint1(cx + hs, cy - hs, cz)
            plane.SetPoint2(cx - hs, cy + hs, cz)
        elif self._orientation == "coronal":
            plane.SetOrigin(cx - hs, cy, cz - hs)
            plane.SetPoint1(cx + hs, cy, cz - hs)
            plane.SetPoint2(cx - hs, cy, cz + hs)
        else:  # sagittal
            plane.SetOrigin(cx, cy - hs, cz - hs)
            plane.SetPoint1(cx, cy + hs, cz - hs)
            plane.SetPoint2(cx, cy - hs, cz + hs)
        plane.Update()

        actors = []
        # White outline
        m1 = vtkPolyDataMapper()
        m1.SetInputData(plane.GetOutput())
        a1 = vtkActor()
        a1.SetMapper(m1)
        a1.GetProperty().SetColor(1.0, 1.0, 1.0)
        a1.GetProperty().SetRepresentationToWireframe()
        a1.GetProperty().SetLineWidth(2.0)
        a1.GetProperty().SetOpacity(0.8)
        self._vtk_renderer.AddActor(a1)
        self._overlay_actors.append(a1)
        actors.append(a1)

        # Colored fill
        m2 = vtkPolyDataMapper()
        m2.SetInputData(plane.GetOutput())
        a2 = vtkActor()
        a2.SetMapper(m2)
        a2.GetProperty().SetColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        a2.GetProperty().SetRepresentationToSurface()
        a2.GetProperty().SetOpacity(0.85)
        self._vtk_renderer.AddActor(a2)
        self._overlay_actors.append(a2)
        actors.append(a2)

        return actors

    def update_single_seed_position(
        self,
        vessel: str,
        seed_type: str,
        index: int,
        world_pos: tuple,
    ) -> None:
        """Reposition one seed's VTK actors without rebuilding all overlays.

        Computes the delta from old position and translates each actor.
        """
        for info in self._seed_actor_info:
            if (
                info["vessel"] == vessel
                and info["type"] == seed_type
                and info["index"] == index
            ):
                old_x, old_y, old_z = info["world_pos"]
                new_x, new_y, new_z = world_pos
                dx = new_x - old_x
                dy = new_y - old_y
                dz = new_z - old_z

                for actor in info["actors"]:
                    pos = actor.GetPosition()
                    actor.SetPosition(pos[0] + dx, pos[1] + dy, pos[2] + dz)

                info["world_pos"] = (new_x, new_y, new_z)
                self._update_seed_visibility()
                self._render()
                return

    def set_selection_highlight(
        self, vessel: str, seed_type: str, index: int
    ) -> None:
        """Add a yellow ring/border around the selected seed's actors."""
        self.clear_selection_highlight()

        for info in self._seed_actor_info:
            if (
                info["vessel"] == vessel
                and info["type"] == seed_type
                and info["index"] == index
            ):
                cx, cy, cz = info["world_pos"]
                highlight_radius = 3.0  # slightly larger than seed markers

                # Build highlight ring in 3 planes
                appender = vtkAppendPolyData()
                for normal in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
                    ring = vtkRegularPolygonSource()
                    ring.SetNumberOfSides(32)
                    ring.SetRadius(highlight_radius)
                    ring.SetCenter(cx, cy, cz)
                    ring.SetNormal(*normal)
                    ring.GeneratePolygonOff()  # ring only, no fill
                    ring.Update()
                    appender.AddInputData(ring.GetOutput())
                appender.Update()

                mapper = vtkPolyDataMapper()
                mapper.SetInputData(appender.GetOutput())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # yellow
                actor.GetProperty().SetLineWidth(2.5)
                actor.GetProperty().SetOpacity(0.9)
                self._vtk_renderer.AddActor(actor)
                self._highlight_actors.append(actor)
                self._render()
                return

    def clear_selection_highlight(self) -> None:
        """Remove yellow selection highlight actors."""
        for actor in self._highlight_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._highlight_actors.clear()

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
        cam.SetParallelScale(max(scale_by_height, scale_by_width))

        self._vtk_renderer.ResetCameraClippingRange()
        self._render()

    # ── Overlay rendering ─────────────────────────────────────────

    def clear_overlays(self) -> None:
        """Remove all overlay actors."""
        for actor in self._overlay_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._overlay_actors.clear()
        self._seed_actor_info.clear()
        for actor in self._crosshair_actors:
            self._vtk_renderer.RemoveActor(actor)
        self._crosshair_actors.clear()
        self._crosshair_pos = None
        if self._voi_slice is not None:
            self._vtk_renderer.RemoveViewProp(self._voi_slice)
            self._voi_slice = None
            self._voi_mapper = None
        self._render()

    def set_seed_overlay(self, seeds_dict: dict, spacing: list) -> None:
        """Show seed/ostium points as filled sphere dots.

        Each seed renders as a small solid sphere visible from any angle.
        Two-layer rendering: white wireframe outline underneath,
        vessel-colored solid fill on top.
        """
        sx, sy, sz = spacing[2], spacing[1], spacing[0]

        for vessel, ijk in seeds_dict.items():
            z, y, x = float(ijk[0]), float(ijk[1]), float(ijk[2])
            cx, cy, cz = x * sx, y * sy, z * sz
            rgb = _VESSEL_COLORS_RGB.get(vessel, (232, 83, 58))

            seed_actors = self._create_sphere_marker(cx, cy, cz, 2.5, rgb)

            self._seed_actor_info.append({
                "actors": seed_actors,
                "world_pos": (cx, cy, cz),
            })

        self._update_seed_visibility()
        self._render()

    def set_centerline_overlay(self, centerlines_dict: dict, spacing: list) -> None:
        """Show vessel centerlines as colored lines with white outline.

        Centerlines are clipped to a ±2mm slab around the current slice
        so they only appear where the vessel is visible.
        """
        from vtkmodules.vtkCommonDataModel import vtkPlane

        sx, sy, sz = spacing[2], spacing[1], spacing[0]
        slab_mm = 2.0  # show centerline within ±2mm of current slice

        # Remove old centerline clipping planes
        self._centerline_mappers = []

        for vessel, cl_ijk in centerlines_dict.items():
            if cl_ijk is None or len(cl_ijk) < 2:
                continue

            rgb = _VESSEL_COLORS_RGB.get(vessel, (232, 83, 58))

            points = vtkPoints()
            for pt in cl_ijk:
                z, y, x = float(pt[0]), float(pt[1]), float(pt[2])
                points.InsertNextPoint(x * sx, y * sy, z * sz)

            lines = vtkCellArray()
            lines.InsertNextCell(len(cl_ijk))
            for i in range(len(cl_ijk)):
                lines.InsertCellPoint(i)

            pd = vtkPolyData()
            pd.SetPoints(points)
            pd.SetLines(lines)

            # Clipping planes: two parallel planes forming a slab
            plane1 = vtkPlane()
            plane2 = vtkPlane()
            if self._orientation == "axial":
                plane1.SetNormal(0, 0, 1)
                plane2.SetNormal(0, 0, -1)
            elif self._orientation == "coronal":
                plane1.SetNormal(0, 1, 0)
                plane2.SetNormal(0, -1, 0)
            else:  # sagittal
                plane1.SetNormal(1, 0, 0)
                plane2.SetNormal(-1, 0, 0)

            # White outline (wider, behind)
            mapper_bg = vtkPolyDataMapper()
            mapper_bg.SetInputData(pd)
            mapper_bg.AddClippingPlane(plane1)
            mapper_bg.AddClippingPlane(plane2)
            actor_bg = vtkActor()
            actor_bg.SetMapper(mapper_bg)
            actor_bg.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor_bg.GetProperty().SetLineWidth(4.0)
            actor_bg.GetProperty().SetOpacity(0.5)
            self._vtk_renderer.AddActor(actor_bg)
            self._overlay_actors.append(actor_bg)

            # Vessel-colored line (on top)
            mapper_fg = vtkPolyDataMapper()
            mapper_fg.SetInputData(pd)
            mapper_fg.AddClippingPlane(plane1)
            mapper_fg.AddClippingPlane(plane2)
            actor_fg = vtkActor()
            actor_fg.SetMapper(mapper_fg)
            actor_fg.GetProperty().SetColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
            actor_fg.GetProperty().SetLineWidth(2.5)
            actor_fg.GetProperty().SetOpacity(0.9)
            self._vtk_renderer.AddActor(actor_fg)
            self._overlay_actors.append(actor_fg)

            self._centerline_mappers.extend([
                (mapper_bg, plane1, plane2),
                (mapper_fg, plane1, plane2),
            ])

        self._update_centerline_clip()
        self._render()

    def _update_centerline_clip(self) -> None:
        """Update centerline clipping planes to match the current slice."""
        if not hasattr(self, "_centerline_mappers") or not self._centerline_mappers:
            return
        slab_mm = 2.0
        if self._orientation == "axial":
            pos_mm = self._current_slice * self._spacing[0]
        elif self._orientation == "coronal":
            pos_mm = self._current_slice * self._spacing[1]
        else:
            pos_mm = self._current_slice * self._spacing[2]

        for _mapper, p1, p2 in self._centerline_mappers:
            if self._orientation == "axial":
                p1.SetOrigin(0, 0, pos_mm - slab_mm)
                p2.SetOrigin(0, 0, pos_mm + slab_mm)
            elif self._orientation == "coronal":
                p1.SetOrigin(0, pos_mm - slab_mm, 0)
                p2.SetOrigin(0, pos_mm + slab_mm, 0)
            else:
                p1.SetOrigin(pos_mm - slab_mm, 0, 0)
                p2.SetOrigin(pos_mm + slab_mm, 0, 0)

    def set_contour_overlay(self, contour_results_dict: dict) -> None:
        """Show vessel wall contours in white (Horos style).

        Horos convention: vessel lumen boundary in white (lw 2.0, alpha 0.85),
        shown every ~2mm along the vessel for clean visualization.
        """
        for vessel, cr in contour_results_dict.items():
            points = vtkPoints()
            lines = vtkCellArray()
            pt_offset = 0

            # Show every ~2mm along vessel (use arclengths if available)
            step = max(1, len(cr.contours) // 20)
            for i in range(0, len(cr.contours), step):
                contour = cr.contours[i]
                n = len(contour)
                if n < 3:
                    continue

                for pt in contour:
                    z, y, x = float(pt[0]), float(pt[1]), float(pt[2])
                    points.InsertNextPoint(x, y, z)  # already in mm

                # Closed polyline
                lines.InsertNextCell(n + 1)
                for j in range(n):
                    lines.InsertCellPoint(pt_offset + j)
                lines.InsertCellPoint(pt_offset)
                pt_offset += n

            if points.GetNumberOfPoints() == 0:
                continue

            pd = vtkPolyData()
            pd.SetPoints(points)
            pd.SetLines(lines)

            # White vessel wall contour (Horos style)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(pd)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetProperty().SetLineWidth(1.8)
            actor.GetProperty().SetOpacity(0.85)
            self._vtk_renderer.AddActor(actor)
            self._overlay_actors.append(actor)

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
            1: (232, 83, 58, 40),    # LAD — red-orange, very subtle
            2: (74, 144, 217, 40),   # LCx — blue, very subtle
            3: (46, 204, 113, 40),   # RCA — green, very subtle
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
