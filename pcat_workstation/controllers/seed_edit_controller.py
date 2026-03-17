"""Controller mediating between VTK mouse events and SeedEditState.

Translates low-level mouse/keyboard events from :class:`VTKSliceView`
into high-level mutations on :class:`SeedEditState` (select, move, add,
delete, undo/redo) and keeps VTK overlays in sync.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Qt

from pcat_workstation.models.seed_edit_state import SeedEditState

# Avoid circular import at module level — VTKSliceView is only needed for
# type hints and method calls which happen at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcat_workstation.widgets.vtk_slice_view import VTKSliceView


class SeedEditController(QObject):
    """Mediates between VTK mouse events and SeedEditState.

    Responsibilities:
    * Translate pixel coordinates to voxel coordinates.
    * Hit-test seeds on left-press, initiate drag.
    * Move seeds during drag, push history on release.
    * Handle keyboard shortcuts (add/delete waypoint, undo/redo, vessel cycle).
    * Refresh VTK overlays on all attached views after mutations.
    """

    request_pipeline_rerun = Signal()  # emitted when user wants to re-run pipeline

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        state: SeedEditState,
        views: List[VTKSliceView],
        spacing: list,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        self._state = state
        self._views: List[VTKSliceView] = list(views)
        self._spacing: List[float] = list(spacing)  # [sz, sy, sx]

        # Drag state
        self._dragging: bool = False
        self._drag_vessel: str = ""
        self._drag_type: str = ""
        self._drag_index: int = 0

        # Wire state signals → overlay refresh
        self._state.seeds_changed.connect(self._on_seeds_changed)
        self._state.selection_changed.connect(self._on_selection_changed)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _world_to_voxel(
        self, x_mm: float, y_mm: float, z_mm: float
    ) -> np.ndarray:
        """Convert VTK world coords (mm) to voxel [z, y, x]."""
        sx = self._spacing[2]
        sy = self._spacing[1]
        sz = self._spacing[0]
        return np.array(
            [
                z_mm / sz if sz > 0 else 0.0,
                y_mm / sy if sy > 0 else 0.0,
                x_mm / sx if sx > 0 else 0.0,
            ],
            dtype=np.float64,
        )

    def _voxel_to_world(self, ijk: np.ndarray) -> Tuple[float, float, float]:
        """Convert voxel [z, y, x] to world (x_mm, y_mm, z_mm)."""
        sz, sy, sx = self._spacing[0], self._spacing[1], self._spacing[2]
        return (float(ijk[2]) * sx, float(ijk[1]) * sy, float(ijk[0]) * sz)

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def on_left_press(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Handle left mouse press: select nearest seed or deselect."""
        world = view.pixel_to_world(qt_x, qt_y)
        voxel = self._world_to_voxel(*world)

        vessel = self._state.current_vessel
        if not vessel:
            return

        hit = self._state.find_nearest_seed(vessel, voxel, max_dist_vox=10.0)

        if hit is not None:
            seed_type, index = hit
            self._state.select(vessel, seed_type, index)
            # Prepare for potential drag
            self._dragging = False  # will become True on first drag event
            self._drag_vessel = vessel
            self._drag_type = seed_type
            self._drag_index = index
        else:
            self._state.clear_selection()
            self._dragging = False

    def on_left_drag(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Handle left mouse drag: move selected seed in real-time."""
        if not self._drag_vessel:
            return

        self._dragging = True
        world = view.pixel_to_world(qt_x, qt_y)
        voxel = self._world_to_voxel(*world)

        # Move in model (no history push during drag)
        self._state.move_seed(
            self._drag_vessel,
            self._drag_type,
            self._drag_index,
            voxel.tolist(),
        )

        # Update actor positions on all views for responsiveness
        for v in self._views:
            v.update_single_seed_position(
                self._drag_vessel,
                self._drag_type,
                self._drag_index,
                world,
            )

    def on_left_release(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Handle left mouse release: finalise drag with history push."""
        if self._dragging:
            # Push history now that the drag is complete
            self._state.push_history()
            self.refresh_all_views()

        self._dragging = False
        self._drag_vessel = ""
        self._drag_type = ""
        self._drag_index = 0

    def on_right_click(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Handle right click: delete nearest waypoint."""
        world = view.pixel_to_world(qt_x, qt_y)
        voxel = self._world_to_voxel(*world)

        vessel = self._state.current_vessel
        if not vessel:
            return

        hit = self._state.find_nearest_seed(vessel, voxel, max_dist_vox=10.0)
        if hit is not None:
            seed_type, index = hit
            self._state.select(vessel, seed_type, index)
            self._state.delete_selected()

    def on_key_press(
        self, view: VTKSliceView, key: int, modifiers: int
    ) -> None:
        """Handle keyboard shortcuts for seed editing.

        Supported keys:
        * Enter/Return — add waypoint at current crosshair position
        * Backspace/Delete — delete selected waypoint
        * Ctrl+Z — undo
        * Ctrl+Shift+Z / Ctrl+Y — redo
        * Left/Right arrows — cycle through vessels
        """
        ctrl = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._add_waypoint_at_crosshair(view)

        elif key in (Qt.Key_Backspace, Qt.Key_Delete):
            self._state.delete_selected()

        elif key == Qt.Key_Z and ctrl and shift:
            self._state.redo()

        elif key == Qt.Key_Z and ctrl:
            self._state.undo()

        elif key == Qt.Key_Y and ctrl:
            self._state.redo()

        elif key == Qt.Key_Left:
            self._cycle_vessel(-1)

        elif key == Qt.Key_Right:
            self._cycle_vessel(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_waypoint_at_crosshair(self, view: VTKSliceView) -> None:
        """Add a waypoint at the view's current crosshair position."""
        pos = getattr(view, "_crosshair_pos", None)
        if pos is None:
            return
        x_mm, y_mm, z_mm = pos
        voxel = self._world_to_voxel(x_mm, y_mm, z_mm)
        vessel = self._state.current_vessel
        if vessel:
            self._state.add_waypoint(vessel, voxel.tolist())

    def _cycle_vessel(self, direction: int) -> None:
        """Cycle through vessels in the seed dict."""
        vessels = list(self._state.seeds.keys())
        if not vessels:
            return
        try:
            idx = vessels.index(self._state.current_vessel)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(vessels)
        self._state.current_vessel = vessels[idx]
        self._state.clear_selection()

    def refresh_all_views(self) -> None:
        """Rebuild seed overlays on all views from state."""
        for view in self._views:
            view.set_seed_overlay_extended(self._state, self._spacing)

    # ------------------------------------------------------------------
    # State signal handlers
    # ------------------------------------------------------------------

    def _on_seeds_changed(self, vessel: str) -> None:
        """Refresh overlays when seeds change."""
        self.refresh_all_views()

    def _on_selection_changed(self) -> None:
        """Update selection highlight on all views."""
        vessel = self._state._selected_vessel
        seed_type = self._state._selected_type
        index = (
            self._state.selected_idx
            if self._state.selected_idx is not None
            else 0
        )

        for view in self._views:
            if vessel and seed_type:
                view.set_selection_highlight(vessel, seed_type, index)
            else:
                view.clear_selection_highlight()
