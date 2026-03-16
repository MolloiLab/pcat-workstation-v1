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
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtk.util.numpy_support import numpy_to_vtk
from typing import Optional


class SliceInteractorStyle(vtkInteractorStyleImage):
    """Custom interactor style for medical image slice viewing.

    - Right-drag: window/level (horizontal=width, vertical=level)
    - Scroll: change slice
    - Ctrl+scroll: zoom
    - Middle-drag: pan
    - Left-click: emit crosshair position
    """

    def __init__(self, viewer: "VTKSliceView") -> None:
        super().__init__()
        self._viewer = viewer
        self._right_dragging = False
        self._right_drag_start = (0, 0)
        self._wl_start = (1500.0, 300.0)

        self.AddObserver("MouseWheelForwardEvent", self._on_scroll_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_scroll_backward)
        self.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        self.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        self.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        self.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.AddObserver("LeftButtonReleaseEvent", self._on_left_release)

    # ── Scroll ──────────────────────────────────────────────────────

    def _on_scroll_forward(self, obj, event) -> None:
        interactor = self.GetInteractor()
        if interactor.GetControlKey():
            # Ctrl+scroll: zoom in
            self._viewer._vtk_renderer.GetActiveCamera().Zoom(1.1)
            self._viewer._render()
        else:
            self._viewer._on_scroll(delta=-1)

    def _on_scroll_backward(self, obj, event) -> None:
        interactor = self.GetInteractor()
        if interactor.GetControlKey():
            # Ctrl+scroll: zoom out
            self._viewer._vtk_renderer.GetActiveCamera().Zoom(0.9)
            self._viewer._render()
        else:
            self._viewer._on_scroll(delta=1)

    # ── Right-drag: window/level ────────────────────────────────────

    def _on_right_press(self, obj, event) -> None:
        self._right_dragging = True
        interactor = self.GetInteractor()
        self._right_drag_start = interactor.GetEventPosition()
        self._wl_start = self._viewer.get_window_level()

    def _on_right_release(self, obj, event) -> None:
        self._right_dragging = False

    # ── Middle-drag: pan ────────────────────────────────────────────

    def _on_middle_press(self, obj, event) -> None:
        self.StartPan()
        self.OnMiddleButtonDown()

    def _on_middle_release(self, obj, event) -> None:
        self.EndPan()
        self.OnMiddleButtonUp()

    # ── Left-click: crosshair ──────────────────────────────────────

    def _on_left_press(self, obj, event) -> None:
        self._viewer._emit_crosshair_at_cursor()

    def _on_left_release(self, obj, event) -> None:
        pass

    # ── Mouse move ──────────────────────────────────────────────────

    def _on_mouse_move(self, obj, event) -> None:
        if self._right_dragging:
            self._handle_right_drag()
        else:
            self.OnMouseMove()

    def _handle_right_drag(self) -> None:
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        dx = x - self._right_drag_start[0]
        dy = y - self._right_drag_start[1]

        w0, l0 = self._wl_start
        new_window = max(1.0, w0 + dx * 2.0)
        new_level = l0 + dy * 2.0
        self._viewer.set_window_level(new_window, new_level)


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

        # VTK render widget
        self._vtk_widget = QVTKRenderWindowInteractor(self)
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

        # Interactor style
        self._interactor_style = SliceInteractorStyle(self)
        interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(self._interactor_style)

    def start_interactor(self) -> None:
        """Initialize the VTK interactor. Call after the widget is shown."""
        interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        interactor.Initialize()
        interactor.SetEnableRender(True)

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
        """Reset zoom/pan to fit the current slice."""
        self._vtk_renderer.ResetCamera()
        self._render()

    # ── Render helper ───────────────────────────────────────────────

    def _render(self) -> None:
        # Use Qt's update() instead of direct vtkRenderWindow.Render() to
        # avoid blocking the event loop on macOS.  paintEvent already calls
        # _Iren.Render().  update() coalesces rapid calls (e.g. during W/L
        # drag) into a single paint, which is both safe and efficient.
        self._vtk_widget.update()

    # ── Cleanup ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Ensure VTK resources are cleaned up."""
        self._vtk_widget.GetRenderWindow().Finalize()
        super().closeEvent(event)
