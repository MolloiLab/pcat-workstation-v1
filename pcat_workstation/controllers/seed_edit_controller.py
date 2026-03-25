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
    save_requested = Signal()  # Ctrl+S — save seeds to session

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
            # Check if this seed is ALREADY selected (select-then-drag model)
            if (self._state.selected_vessel == vessel
                    and self._state.selected_type == seed_type
                    and self._state.selected_idx == (index if seed_type == "waypoint" else -1)):
                # Already selected → prepare for drag
                self._dragging = False
                self._drag_vessel = vessel
                self._drag_type = seed_type
                self._drag_index = index
            else:
                # Not selected → just select it (no drag on first click)
                self._state.select(vessel, seed_type, index)
                self._dragging = False
                self._drag_vessel = ""  # no drag on first click
        else:
            # Empty space: place ostium if none exists, otherwise just deselect.
            # Waypoints are added via Enter key (like the old seed_editor).
            entry = self._state.seeds.get(vessel, {})
            if entry.get("ostium") is None:
                # No ostium yet → place ostium here
                self._state.push_history()
                self._state.seeds[vessel]["ostium"] = voxel.tolist()
                self._state.recompute_centerline(vessel)
                self._state.seeds_changed.emit(vessel)
                self.refresh_all_views()
            else:
                self._state.clear_selection()
            self._dragging = False
            self._drag_vessel = ""

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

        # Move the VTK actor directly (fast — no full rebuild)
        for v in self._views:
            v.update_single_seed_position(
                self._drag_vessel,
                self._drag_type,
                self._drag_index,
                world,
            )

        # Recompute spline and update centerline overlay (no CPR regen)
        self._state.recompute_centerline(self._drag_vessel, emit=False)
        cl_dict = {v: c for v, c in self._state.centerlines.items() if c is not None}
        if cl_dict:
            for view in self._views:
                view.set_centerline_overlay(cl_dict, self._spacing)

    def on_left_release(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Handle left mouse release: emit signals to trigger CPR regeneration."""
        if self._dragging and self._drag_vessel:
            # Centerline was already updated during drag. Now emit signals
            # so CPR regenerates (only on release, not every drag pixel).
            self._state.centerline_changed.emit(self._drag_vessel)
            self._state.seeds_changed.emit(self._drag_vessel)
            self._state.push_history()
            self.refresh_all_views()

        self._dragging = False
        self._drag_vessel = ""
        self._drag_type = ""
        self._drag_index = 0

    def on_right_click(
        self, view: VTKSliceView, qt_x: int, qt_y: int
    ) -> None:
        """Right-click: no-op (use Backspace to delete seeds)."""
        pass

    def on_key_press(
        self, view: VTKSliceView, key: int, modifiers: int
    ) -> None:
        """Handle keyboard shortcuts for seed editing.

        Supported keys:
        * Enter/Return — add waypoint at current crosshair position
        * Backspace/Delete — delete selected waypoint
        * Ctrl+Z — undo
        * Ctrl+Shift+Z / Ctrl+Y — redo
        * Left/Right arrows — cycle through seeds in current vessel
        """
        # modifiers is int (.value), Qt enums may also be int or enum
        _ctrl = Qt.ControlModifier.value if hasattr(Qt.ControlModifier, "value") else int(Qt.ControlModifier)
        _shift = Qt.ShiftModifier.value if hasattr(Qt.ShiftModifier, "value") else int(Qt.ShiftModifier)
        ctrl = bool(modifiers & _ctrl)
        shift = bool(modifiers & _shift)

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

        elif key == Qt.Key_S and ctrl:
            self.save_requested.emit()

        elif key == Qt.Key_Left:
            self._cycle_seed(-1)

        elif key == Qt.Key_Right:
            self._cycle_seed(1)

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

    def _cycle_seed(self, direction: int) -> None:
        """Cycle selection through seeds (ostium + waypoints) in the current vessel."""
        vessel = self._state.current_vessel
        if not vessel:
            return
        all_seeds = self._get_all_seeds(vessel)
        if not all_seeds:
            return

        # Determine current position in the flat list
        current_flat = self._selected_to_flat_index()
        if current_flat is None:
            new_flat = 0
        else:
            new_flat = max(0, min(len(all_seeds) - 1, current_flat + direction))

        # Convert flat index back to (type, index)
        has_ostium = self._state.seeds[vessel]["ostium"] is not None
        if has_ostium and new_flat == 0:
            self._state.select(vessel, "ostium", 0)
        else:
            wp_idx = new_flat - (1 if has_ostium else 0)
            self._state.select(vessel, "waypoint", wp_idx)

        self.refresh_all_views()

    def _get_all_seeds(self, vessel: str) -> list:
        """Return flat list of all seed positions [ostium, wp0, wp1, ...] for vessel."""
        entry = self._state.seeds.get(vessel, {})
        pts = []
        if entry.get("ostium") is not None:
            pts.append(entry["ostium"])
        pts.extend(entry.get("waypoints", []))
        return pts

    def _selected_to_flat_index(self) -> int | None:
        """Convert current selection to flat index in all_seeds list."""
        vessel = self._state.current_vessel
        if not self._state.selected_vessel or self._state.selected_vessel != vessel:
            return None
        has_ostium = self._state.seeds[vessel]["ostium"] is not None
        if self._state.selected_type == "ostium":
            return 0
        elif self._state.selected_type == "waypoint" and self._state.selected_idx is not None:
            return self._state.selected_idx + (1 if has_ostium else 0)
        return None

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
        vessel = self._state.selected_vessel
        seed_type = self._state.selected_type
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
