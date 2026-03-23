"""Data model for interactive seed editing workflow.

Owns seeds, selection state, undo/redo history, and centerline
recomputation.  Consumed by ``SeedEditController`` and ``VTKSliceView``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal
from scipy.interpolate import CubicSpline


# Maximum number of undo snapshots to keep.
_MAX_HISTORY = 50


def _fit_spline_centerline(
    seeds_ijk: List[List[float]],
    spacing_mm: List[float],
    volume_shape: Tuple[int, int, int],
    step_mm: float = 0.5,
) -> Optional[np.ndarray]:
    """Fit a cubic spline through seed points and sample densely.

    Parameters
    ----------
    seeds_ijk : list of [z, y, x] seed points in voxel coordinates.
    spacing_mm : [sz, sy, sx] voxel spacing.
    volume_shape : (Z, Y, X) volume dimensions.
    step_mm : arc-length step for dense sampling.

    Returns
    -------
    dense_ijk : (M, 3) float64 array, or *None* if insufficient points.
    """
    if len(seeds_ijk) < 2:
        return None

    pts_ijk = np.array(seeds_ijk, dtype=np.float64)
    pts_mm = pts_ijk * np.array(spacing_mm)

    # Remove duplicate points (zero-length segments).
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-8])
    pts_mm = pts_mm[keep]

    if len(pts_mm) < 2:
        return None

    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]

    if total < 1e-6:
        return None

    if len(pts_mm) >= 3:
        cs = CubicSpline(arc, pts_mm, bc_type="not-a-knot")
    else:
        cs = CubicSpline(arc, pts_mm, bc_type="natural")

    n_out = max(10, int(total / step_mm))
    s_vals = np.linspace(0, total, n_out)
    dense_mm = cs(s_vals)

    dense_ijk = dense_mm / np.array(spacing_mm)
    dense_ijk = np.clip(dense_ijk, 0, np.array(volume_shape) - 1)
    return dense_ijk


class SeedEditState(QObject):
    """Owns seed positions, selection, undo history, and centerlines.

    Seed storage format (extended)::

        {vessel: {"ostium": [z, y, x] | None, "waypoints": [[z,y,x], ...]}}

    The model is UI-agnostic; it emits signals so that views and
    controllers can react to mutations.
    """

    seeds_changed = Signal(str)       # vessel name
    selection_changed = Signal()
    centerline_changed = Signal(str)  # vessel name

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        initial_seeds: dict,
        spacing_mm: list,
        volume_shape: tuple,
        parent=None,
    ):
        super().__init__(parent)

        self.spacing_mm: List[float] = list(spacing_mm)
        self.volume_shape: Tuple[int, int, int] = tuple(volume_shape)  # type: ignore[assignment]

        # Convert to extended format.
        self.seeds: Dict[str, Dict] = self._normalise_seeds(initial_seeds)

        self.current_vessel: str = next(iter(self.seeds), "")
        self.selected_idx: Optional[int] = None  # waypoint idx, or -1 for ostium
        self._selected_vessel: str = ""
        self._selected_type: str = ""  # "ostium" | "waypoint"

        self.centerlines: Dict[str, Optional[np.ndarray]] = {}
        self.history: List[dict] = []
        self.redo_stack: List[dict] = []

        # Compute initial centerlines for every vessel.
        for vessel in self.seeds:
            self.recompute_centerline(vessel)

    # ------------------------------------------------------------------
    # Seed format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_seeds(raw: dict) -> Dict[str, Dict]:
        """Normalise seeds to extended format.

        Accepts either:
        - Simple: ``{vessel: [z, y, x]}``
        - Extended: ``{vessel: {"ostium": [...], "waypoints": [...]}}``
        """
        out: Dict[str, Dict] = {}
        for vessel, value in raw.items():
            if isinstance(value, dict):
                out[vessel] = {
                    "ostium": value.get("ostium"),
                    "waypoints": list(value.get("waypoints", [])),
                }
            else:
                # Simple format — treat as ostium seed.
                out[vessel] = {
                    "ostium": list(value) if value is not None else None,
                    "waypoints": [],
                }
        return out

    # ------------------------------------------------------------------
    # Proximity search
    # ------------------------------------------------------------------

    def find_nearest_seed(
        self,
        vessel: str,
        pos_ijk: np.ndarray,
        max_dist_vox: float = 10.0,
    ) -> Optional[Tuple[str, int]]:
        """Find the closest seed to *pos_ijk* within *max_dist_vox*.

        Parameters
        ----------
        vessel : vessel key to search.
        pos_ijk : (3,) array — query position in voxel coordinates.
        max_dist_vox : maximum Euclidean distance in voxels.

        Returns
        -------
        ``(seed_type, index)`` — *seed_type* is ``"ostium"`` or
        ``"waypoint"``; *index* is the waypoint list index (0 for
        ostium).  Returns *None* if nothing is within range.
        """
        if vessel not in self.seeds:
            return None

        entry = self.seeds[vessel]
        pos = np.asarray(pos_ijk, dtype=np.float64)

        best_dist = max_dist_vox
        best: Optional[Tuple[str, int]] = None

        # Check ostium.
        if entry["ostium"] is not None:
            d = np.linalg.norm(np.asarray(entry["ostium"], dtype=np.float64) - pos)
            if d < best_dist:
                best_dist = d
                best = ("ostium", 0)

        # Check waypoints.
        for idx, wp in enumerate(entry["waypoints"]):
            d = np.linalg.norm(np.asarray(wp, dtype=np.float64) - pos)
            if d < best_dist:
                best_dist = d
                best = ("waypoint", idx)

        return best

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def move_seed(
        self,
        vessel: str,
        seed_type: str,
        index: int,
        new_pos_ijk: list,
    ) -> None:
        """Move a seed to *new_pos_ijk* without pushing history.

        History should be pushed on mouse-release, not during dragging.
        """
        entry = self.seeds[vessel]
        if seed_type == "ostium":
            entry["ostium"] = list(new_pos_ijk)
        else:
            entry["waypoints"][index] = list(new_pos_ijk)
        self.recompute_centerline(vessel)
        self.seeds_changed.emit(vessel)

    def add_waypoint(self, vessel: str, pos_ijk: list) -> None:
        """Append a waypoint to *vessel* and recompute its centerline."""
        self.push_history()
        self.seeds[vessel]["waypoints"].append(list(pos_ijk))
        self.recompute_centerline(vessel)
        self.seeds_changed.emit(vessel)

    def delete_selected(self) -> None:
        """Delete the currently selected waypoint (ostium cannot be deleted)."""
        if not self._selected_vessel or self._selected_type != "waypoint":
            return
        if self.selected_idx is None or self.selected_idx < 0:
            return

        vessel = self._selected_vessel
        wps = self.seeds[vessel]["waypoints"]
        if self.selected_idx >= len(wps):
            return

        self.push_history()
        wps.pop(self.selected_idx)
        self.clear_selection()
        self.recompute_centerline(vessel)
        self.seeds_changed.emit(vessel)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, vessel: str, seed_type: str, index: int) -> None:
        """Set the active selection."""
        self._selected_vessel = vessel
        self._selected_type = seed_type
        self.selected_idx = index if seed_type == "waypoint" else -1
        self.current_vessel = vessel
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        """Clear any active selection."""
        self._selected_vessel = ""
        self._selected_type = ""
        self.selected_idx = None
        self.selection_changed.emit()

    @property
    def selected_vessel(self) -> str:
        """Currently selected vessel name (empty if none)."""
        return self._selected_vessel

    @property
    def selected_type(self) -> str:
        """Selected seed type: ``"ostium"``, ``"waypoint"``, or ``""``."""
        return self._selected_type

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def push_history(self) -> None:
        """Snapshot the current seeds dict onto the undo stack."""
        self.history.append(copy.deepcopy(self.seeds))
        if len(self.history) > _MAX_HISTORY:
            self.history.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        """Restore the previous seeds state."""
        if not self.history:
            return
        self.redo_stack.append(copy.deepcopy(self.seeds))
        self.seeds = self.history.pop()
        self._recompute_all_centerlines()
        for vessel in self.seeds:
            self.seeds_changed.emit(vessel)

    def redo(self) -> None:
        """Re-apply an undone change."""
        if not self.redo_stack:
            return
        self.history.append(copy.deepcopy(self.seeds))
        self.seeds = self.redo_stack.pop()
        self._recompute_all_centerlines()
        for vessel in self.seeds:
            self.seeds_changed.emit(vessel)

    # ------------------------------------------------------------------
    # Centerline fitting
    # ------------------------------------------------------------------

    def recompute_centerline(self, vessel: str) -> None:
        """Refit the spline centerline for *vessel* from its seeds."""
        entry = self.seeds.get(vessel)
        if entry is None:
            self.centerlines[vessel] = None
            self.centerline_changed.emit(vessel)
            return

        ordered: List[list] = []
        if entry["ostium"] is not None:
            ordered.append(entry["ostium"])
        ordered.extend(entry["waypoints"])

        self.centerlines[vessel] = _fit_spline_centerline(
            ordered, self.spacing_mm, self.volume_shape
        )
        self.centerline_changed.emit(vessel)

    def _recompute_all_centerlines(self) -> None:
        """Recompute centerlines for every vessel."""
        for vessel in self.seeds:
            self.recompute_centerline(vessel)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def get_all_seeds_flat(self) -> dict:
        """Return ``{vessel: [z, y, x]}`` (ostium only) for pipeline use."""
        out: dict = {}
        for vessel, entry in self.seeds.items():
            out[vessel] = list(entry["ostium"]) if entry["ostium"] is not None else None
        return out

    def export_path(self, filepath) -> None:
        """Export seeds and centerlines to a JSON file."""
        import json
        from pathlib import Path as _Path

        data = {
            "seeds": copy.deepcopy(self.seeds),
            "centerlines": {
                v: cl.tolist() if cl is not None else None
                for v, cl in self.centerlines.items()
            },
            "current_vessel": self.current_vessel,
        }
        _Path(filepath).write_text(json.dumps(data, indent=2))

    @classmethod
    def import_path(cls, filepath, spacing_mm, volume_shape):
        """Load seeds and centerlines from a JSON file."""
        import json
        from pathlib import Path as _Path

        data = json.loads(_Path(filepath).read_text())
        state = cls(data["seeds"], spacing_mm, volume_shape)
        state.current_vessel = data.get("current_vessel", "")
        return state

    def save_to_session(self, session) -> None:
        """Persist extended seeds into a :class:`PatientSession`.

        Stores the full extended dict in ``session.seeds_data`` and a flat
        ostium-only dict for pipeline compatibility.
        """
        session.seeds_data = {
            "flat": self.get_all_seeds_flat(),
            "extended": copy.deepcopy(self.seeds),
        }
        session.save()
