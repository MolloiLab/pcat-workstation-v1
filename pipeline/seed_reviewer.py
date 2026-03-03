"""
seed_reviewer.py
Interactive matplotlib tool for reviewing and correcting auto-generated coronary artery seed points.

Usage:
    python seed_reviewer.py --dicom /path/to/patient/dir --seeds input_seeds.json --output corrected_seeds.json

Purpose:
    - Review auto-generated seeds from TotalSegmentator (auto_seeds.py)
    - Display clinical warnings with color-coded confidence levels
    - Allow dragging seeds to correct positions
    - Add, delete, or clear waypoints as needed
    - Preserve warning information while allowing user confirmation

Interface:
  - Shows three orthogonal slices (axial, coronal, sagittal) side by side
  - Displays seeds with confidence indicators (green/yellow/red)
  - Click near seeds to drag them (within 10 pixels)
  - Click elsewhere to add waypoints
  - Scroll wheel changes the displayed slice
  - Switch vessels with number keys
  - Press 'c' to clear warnings for current vessel
  - Press 'd' to delete nearest waypoint
  - Press 's' to save, 'u' to undo, 'r' to reset vessel, 'q' to quit

Output JSON format (same as input):
{
  "LAD": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z, y, x], ...],
    "segment_length_mm": 40.0,
    "_warnings": ["Sub-voxel radius detected"]  // optional list
  },
  "LCX": { ... },
  "RCA": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z, y, x], ...],
    "segment_start_mm": 10.0,
    "segment_length_mm": 40.0
  }
}

Clinical warning system:
  - GREEN (✓ HIGH): ostium set, ≥1 waypoint, no _warnings
  - YELLOW (⚠ REVIEW): ostium set but _warnings non-empty OR only 0 waypoints
  - RED (✗ CRITICAL): ostium missing OR _warnings includes "only N vessels found"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend
import matplotlib.pyplot as plt
# Disable matplotlib's default 's' = save-figure keybinding so our 's' = save-seeds works
plt.rcParams['keymap.save'] = []
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dicom_loader import load_dicom_series


# ─────────────────────────────────────────────
# Seed state
# ─────────────────────────────────────────────

VESSEL_CONFIGS = {
    "LAD": {"segment_length_mm": 40.0},
    "LCX": {"segment_length_mm": 40.0},
    "RCA": {"segment_start_mm": 10.0, "segment_length_mm": 40.0},
}

VESSEL_COLORS = {
    "LAD": "#E8533A",
    "LCX": "#4A90D9",
    "RCA": "#2ECC71",
}

VESSEL_KEYS = ["LAD", "LCX", "RCA"]

# Warning severity classification
RED_WARNINGS = ["only 1 vessel found", "only 2 vessels found", "no vessels found"]
YELLOW_WARNINGS = [
    "Sub-voxel radius", "RCA found via fallback", "watershed split", 
    "aorta exclusion triggered"
]


class SeedReviewer:
    """
    Interactive 3-plane seed reviewer for coronary artery seeds with clinical warnings.
    
    Displays axial/coronal/sagittal MPR views with auto-generated seeds overlaid.
    Allows reviewing, correcting, and adjusting seed positions with drag-and-drop.
    """

    def __init__(
        self, 
        volume: np.ndarray, 
        spacing_mm: List[float], 
        seeds_data: Dict[str, Any], 
        output_path: str | Path,
        warnings_data: Optional[Dict[str, List[str]]] = None
    ):
        self.volume = volume
        self.spacing_mm = spacing_mm  # [sz, sy, sx]
        self.output_path = Path(output_path)
        self.shape = volume.shape  # (Z, Y, X)
        
        # Convert input seeds to internal format
        self.seeds: Dict[str, Dict] = {}
        self.warnings: Dict[str, List[str]] = {}
        for v in VESSEL_KEYS:
            self.seeds[v] = {"ostium": None, "waypoints": []}
            self.warnings[v] = []
        
        self._load_seeds_from_data(seeds_data, warnings_data or {})
        
        # Current view state
        self.z_slice = self.shape[0] // 2
        self.y_slice = self.shape[1] // 2
        self.x_slice = self.shape[2] // 2
        
        # UI state
        self.current_vessel = "LAD"
        self.active_view = "axial"  # which view was last clicked
        self.dragging_seed = None  # (vessel, type, original_pos)
        self.drag_start = None
        self.history: List[Dict] = []  # For undo functionality
        
        # Window/level
        self.ww = 600   # window width
        self.wl = 50    # window level (center)
        
        self._save_state()
        self._build_figure()
        self._connect_events()
    
    def _load_seeds_from_data(self, seeds_data: Dict[str, Any], warnings_data: Dict[str, List[str]]):
        """Load seeds from input JSON data."""
        for v in VESSEL_KEYS:
            if v in seeds_data:
                vessel_data = seeds_data[v]
                self.seeds[v]["ostium"] = vessel_data.get("ostium_ijk")
                self.seeds[v]["waypoints"] = vessel_data.get("waypoints_ijk", [])
                self.warnings[v] = vessel_data.get("_warnings", [])
    
    # ─────────────────────────────────────────
    # Figure construction
    # ─────────────────────────────────────────
    
    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(
            "PCAT Seed Reviewer — Drag seeds to correct | Click to add waypoints | Scroll to change slice\n"
            "Keys: 1=LAD  2=LCX  3=RCA  |  c=Clear warnings  d=Delete waypoint |  u=Undo  r=Reset vessel  s=Save & Continue  q=Quit without saving",
            fontsize=10,
        )
        
        # Layout: 3 image axes + 1 info panel
        gs = self.fig.add_gridspec(
            2, 4,
            width_ratios=[1, 1, 1, 0.6],
            height_ratios=[5, 1],
            hspace=0.05,
            wspace=0.1,
        )
        
        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_coronal = self.fig.add_subplot(gs[0, 1])
        self.ax_sagittal = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[0, 3])
        self.ax_status = self.fig.add_subplot(gs[1, :3])
        
        self.ax_axial.set_title("Axial (scroll Z)", fontsize=9)
        self.ax_coronal.set_title("Coronal (scroll Y)", fontsize=9)
        self.ax_sagittal.set_title("Sagittal (scroll X)", fontsize=9)
        self.ax_info.set_title("Seeds Review", fontsize=9)
        self.ax_status.axis("off")
        
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.axis("off")
        
        self.ax_info.axis("off")
        
        # Initialize image objects
        clim = self._clim()
        self.im_axial = self.ax_axial.imshow(
            self._axial_slice(), cmap="gray", aspect="equal",
            vmin=clim[0], vmax=clim[1], origin="upper"
        )
        self.im_coronal = self.ax_coronal.imshow(
            self._coronal_slice(), cmap="gray", aspect="auto",
            vmin=clim[0], vmax=clim[1], origin="upper"
        )
        self.im_sagittal = self.ax_sagittal.imshow(
            self._sagittal_slice(), cmap="gray", aspect="auto",
            vmin=clim[0], vmax=clim[1], origin="upper"
        )
        
        # Crosshair lines
        self._draw_crosshairs()
        
        # Scatter plots for seed overlay (one per vessel per view)
        self.scatter_axial = {}
        self.scatter_coronal = {}
        self.scatter_sagittal = {}
        for v in VESSEL_KEYS:
            c = VESSEL_COLORS[v]
            self.scatter_axial[v] = {
                "ostium": self.ax_axial.plot([], [], "s", color=c, markersize=8, markeredgewidth=1.5, markeredgecolor="white")[0],
                "waypoints": self.ax_axial.plot([], [], "o", color=c, markersize=5, alpha=0.8)[0],
            }
            self.scatter_coronal[v] = {
                "ostium": self.ax_coronal.plot([], [], "s", color=c, markersize=8, markeredgewidth=1.5, markeredgecolor="white")[0],
                "waypoints": self.ax_coronal.plot([], [], "o", color=c, markersize=5, alpha=0.8)[0],
            }
            self.scatter_sagittal[v] = {
                "ostium": self.ax_sagittal.plot([], [], "s", color=c, markersize=8, markeredgewidth=1.5, markeredgecolor="white")[0],
                "waypoints": self.ax_sagittal.plot([], [], "o", color=c, markersize=5, alpha=0.8)[0],
            }
        
        self._update_status_bar()
        self._update_info_panel()
    
    def _clim(self):
        lo = self.wl - self.ww / 2
        hi = self.wl + self.ww / 2
        return lo, hi
    
    # ─────────────────────────────────────────
    # Slice extraction
    # ─────────────────────────────────────────
    
    def _axial_slice(self):
        return self.volume[self.z_slice, :, :]   # (Y, X)
    
    def _coronal_slice(self):
        # FIX: Flip Z axis so superior (head) is at top
        return np.flipud(self.volume[:, self.y_slice, :])   # (Z, X) flipped
    
    def _sagittal_slice(self):
        # FIX: Flip Z axis so superior (head) is at top
        return np.flipud(self.volume[:, :, self.x_slice])   # (Z, Y) flipped
    
    # ─────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────
    
    def _draw_crosshairs(self):
        """Draw crosshair lines at current cursor position."""
        # Remove old lines
        for attr in ["_ch_ax", "_ch_co", "_ch_sa"]:
            if hasattr(self, attr):
                for line in getattr(self, attr):
                    line.remove()
        
        kw = dict(color="yellow", linewidth=0.8, alpha=0.6, linestyle="--")
        self._ch_ax = [
            self.ax_axial.axhline(self.y_slice, **kw),
            self.ax_axial.axvline(self.x_slice, **kw),
        ]
        # FIX: Use flipped Z coordinate for coronal/sagittal crosshairs
        self._ch_co = [
            self.ax_coronal.axhline(self.shape[0] - 1 - self.z_slice, **kw),
            self.ax_coronal.axvline(self.x_slice, **kw),
        ]
        self._ch_sa = [
            self.ax_sagittal.axhline(self.shape[0] - 1 - self.z_slice, **kw),
            self.ax_sagittal.axvline(self.y_slice, **kw),
        ]
    
    def _refresh_images(self):
        clim = self._clim()
        self.im_axial.set_data(self._axial_slice())
        self.im_axial.set_clim(clim)
        self.im_coronal.set_data(self._coronal_slice())
        self.im_coronal.set_clim(clim)
        self.im_sagittal.set_data(self._sagittal_slice())
        self.im_sagittal.set_clim(clim)
        self._draw_crosshairs()
        self._refresh_seed_markers()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _refresh_seed_markers(self):
        """Update scatter plots to show seeds projected onto current slices."""
        Z, Y, X = self.z_slice, self.y_slice, self.x_slice
        
        for v in VESSEL_KEYS:
            sd = self.seeds[v]
            all_pts = []
            if sd["ostium"] is not None:
                all_pts.append(("ostium", sd["ostium"]))
            for wp in sd["waypoints"]:
                all_pts.append(("waypoint", wp))
            
            # Axial: show points at current Z slice (±1)
            ax_o_x, ax_o_y, ax_w_x, ax_w_y = [], [], [], []
            co_o_x, co_o_z, co_w_x, co_w_z = [], [], [], []
            sa_o_y, sa_o_z, sa_w_y, sa_w_z = [], [], [], []
            
            for ptype, pt in all_pts:
                pz, py, px = pt
                # Axial: only if within 2 slices of current Z
                if abs(pz - Z) <= 2:
                    if ptype == "ostium":
                        ax_o_x.append(px); ax_o_y.append(py)
                    else:
                        ax_w_x.append(px); ax_w_y.append(py)
                # Coronal: within 2 slices of Y — map pz to flipped pixel row
                if abs(py - Y) <= 2:
                    flipped_pz = self.shape[0] - 1 - pz
                    if ptype == "ostium":
                        co_o_x.append(px); co_o_z.append(flipped_pz)
                    else:
                        co_w_x.append(px); co_w_z.append(flipped_pz)
                # Sagittal: within 2 slices of X — map pz to flipped pixel row
                if abs(px - X) <= 2:
                    flipped_pz = self.shape[0] - 1 - pz
                    if ptype == "ostium":
                        sa_o_y.append(py); sa_o_z.append(flipped_pz)
                    else:
                        sa_w_y.append(py); sa_w_z.append(flipped_pz)
            
            self.scatter_axial[v]["ostium"].set_data(ax_o_x, ax_o_y)
            self.scatter_axial[v]["waypoints"].set_data(ax_w_x, ax_w_y)
            self.scatter_coronal[v]["ostium"].set_data(co_o_x, co_o_z)
            self.scatter_coronal[v]["waypoints"].set_data(co_w_x, co_w_z)
            self.scatter_sagittal[v]["ostium"].set_data(sa_o_y, sa_o_z)
            self.scatter_sagittal[v]["waypoints"].set_data(sa_w_y, sa_w_z)
    
    def _get_confidence_level(self, vessel: str) -> str:
        """Get confidence level for a vessel: HIGH, REVIEW, or CRITICAL."""
        sd = self.seeds[vessel]
        warnings = self.warnings[vessel]
        
        # Check for critical warnings
        for w in warnings:
            for red in RED_WARNINGS:
                if red in w:
                    return "CRITICAL"
        
        # Check for critical condition: missing ostium
        if sd["ostium"] is None:
            return "CRITICAL"
        
        # Check for review condition: warnings exist or no waypoints
        if warnings or len(sd["waypoints"]) == 0:
            return "REVIEW"
        
        # Otherwise high confidence
        return "HIGH"
    
    def _get_confidence_symbol(self, level: str) -> str:
        """Get symbol for confidence level."""
        if level == "HIGH":
            return "✓"
        elif level == "REVIEW":
            return "⚠"
        else:  # CRITICAL
            return "✗"
    
    def _get_confidence_color(self, level: str) -> str:
        """Get color for confidence level."""
        if level == "HIGH":
            return "green"
        elif level == "REVIEW":
            return "orange"
        else:  # CRITICAL
            return "red"
    
    def _update_status_bar(self):
        self.ax_status.cla()
        self.ax_status.axis("off")
        v = self.current_vessel
        sd = self.seeds[v]
        n_wp = len(sd["waypoints"])
        has_ostium = sd["ostium"] is not None
        confidence = self._get_confidence_level(v)
        symbol = self._get_confidence_symbol(confidence)
        color = self._get_confidence_color(confidence)
        
        msg = (
            f"  Active vessel: {v} (color: {VESSEL_COLORS[v]})  |  "
            f"Confidence: {symbol} {confidence}  |  "
            f"Ostium: {'SET' if has_ostium else 'NOT SET'}  |  "
            f"Waypoints: {n_wp}  |  "
            f"Slice Z={self.z_slice} Y={self.y_slice} X={self.x_slice}  |  "
            f"W/L: {self.ww}/{self.wl}"
        )
        self.ax_status.text(
            0.01, 0.5, msg,
            ha="left", va="center",
            transform=self.ax_status.transAxes,
            fontsize=9,
            color=VESSEL_COLORS[v],
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=3),
        )
    
    def _update_info_panel(self):
        self.ax_info.cla()
        self.ax_info.axis("off")
        lines = ["Seeds Review:", ""]
        
        for v in VESSEL_KEYS:
            sd = self.seeds[v]
            c = VESSEL_COLORS[v]
            has_o = sd["ostium"] is not None
            n_w = len(sd["waypoints"])
            
            confidence = self._get_confidence_level(v)
            symbol = self._get_confidence_symbol(confidence)
            color = self._get_confidence_color(confidence)
            
            lines.append(f"[{symbol}] {v}")
            if has_o:
                z, y, x = sd["ostium"]
                lines.append(f"   ostium: ({z},{y},{x})")
            else:
                lines.append(f"   ostium: —")
            lines.append(f"   waypoints: {n_w}")
            
            # Show warnings if any
            if v in self.warnings and self.warnings[v]:
                for w in self.warnings[v]:
                    lines.append(f"   ⚠ {w}")
            
            lines.append("")
        
        self.ax_info.text(
            0.05, 0.95, "\n".join(lines),
            ha="left", va="top",
            transform=self.ax_info.transAxes,
            fontsize=8,
            family="monospace",
        )
    
    def _save_state(self):
        """Save current state for undo functionality."""
        import copy
        self.history.append({
            "seeds": copy.deepcopy(self.seeds),
            "warnings": copy.deepcopy(self.warnings),
        })
        # Keep only last 50 states
        if len(self.history) > 50:
            self.history.pop(0)
    
    def _find_nearest_seed(self, z: int, y: int, x: int, vessel: Optional[str] = None) -> Optional[Tuple[str, str, List[int]]]:
        """Find the nearest seed to the given coordinates."""
        if vessel is None:
            vessel = self.current_vessel
        
        nearest = None
        min_dist = float("inf")
        
        # Check ostium
        if self.seeds[vessel]["ostium"] is not None:
            oz, oy, ox = self.seeds[vessel]["ostium"]
            dist = (oz - z) ** 2 + (oy - y) ** 2 + (ox - x) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = (vessel, "ostium", self.seeds[vessel]["ostium"])
        
        # Check waypoints
        for i, wp in enumerate(self.seeds[vessel]["waypoints"]):
            wz, wy, wx = wp
            dist = (wz - z) ** 2 + (wy - y) ** 2 + (wx - x) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = (vessel, f"waypoint_{i}", wp)
        
        # Return None if too far (10 pixels in display space ~ 10 voxels)
        if min_dist > 100:  # 10^2
            return None
        
        return nearest
    
    def _delete_nearest_waypoint(self, z: int, y: int, x: int, vessel: Optional[str] = None):
        """Delete the nearest waypoint to the given coordinates."""
        if vessel is None:
            vessel = self.current_vessel
        
        nearest_idx = None
        min_dist = float("inf")
        
        # Check waypoints only (not ostium)
        for i, wp in enumerate(self.seeds[vessel]["waypoints"]):
            wz, wy, wx = wp
            dist = (wz - z) ** 2 + (wy - y) ** 2 + (wx - x) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Delete if close enough (10 voxels)
        if min_dist <= 100 and nearest_idx is not None:
            removed = self.seeds[vessel]["waypoints"].pop(nearest_idx)
            print(f"[seed_reviewer] Deleted waypoint from {vessel}: {removed}")
            self._save_state()
            return True
        
        return False
    
    # ─────────────────────────────────────────
    # Events
    # ─────────────────────────────────────────
    
    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    
    def _on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        
        ax = event.inaxes
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        # Determine which view was clicked and derive the (z, y, x) seed point
        if ax == self.ax_axial:
            z, y, x = self.z_slice, iy, ix
            self.active_view = "axial"
        elif ax == self.ax_coronal:
            # FIX: Un-flip Z coordinate since coronal view is flipped
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, self.y_slice, ix
            self.active_view = "coronal"
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        elif ax == self.ax_sagittal:
            actual_z = self.shape[0] - 1 - iy  # Un-flip Z (sagittal is flipud)
            z, y, x = actual_z, ix, self.x_slice
            self.active_view = "sagittal"
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        else:
            return
        
        # Clamp to volume
        z = int(np.clip(z, 0, self.shape[0] - 1))
        y = int(np.clip(y, 0, self.shape[1] - 1))
        x = int(np.clip(x, 0, self.shape[2] - 1))
        
        # Check if clicking near a seed for dragging
        nearest = self._find_nearest_seed(z, y, x)
        if nearest:
            vessel, seed_type, pos = nearest
            self.dragging_seed = (vessel, seed_type, pos)
            self.drag_start = (z, y, x)
            print(f"[seed_reviewer] Started dragging {vessel} {seed_type}")
        else:
            # Add new waypoint
            self.seeds[self.current_vessel]["waypoints"].append([z, y, x])
            print(f"[seed_reviewer] {self.current_vessel} waypoint {len(self.seeds[self.current_vessel]['waypoints'])}: ({z}, {y}, {x})")
            self._save_state()
        
        # Update crosshair cursor
        self.y_slice = y
        self.x_slice = x
        
        self._refresh_images()
        self._update_info_panel()
        self.fig.canvas.draw_idle()
    
    def _on_release(self, event):
        if self.dragging_seed:
            vessel, seed_type, original_pos = self.dragging_seed
            print(f"[seed_reviewer] Released {vessel} {seed_type}")
            self.dragging_seed = None
            self.drag_start = None
            self._save_state()
    
    def _on_motion(self, event):
        if self.dragging_seed is None or event.inaxes is None:
            return
        
        ax = event.inaxes
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        # Determine new position based on active view
        if ax == self.ax_axial:
            z, y, x = self.z_slice, iy, ix
        elif ax == self.ax_coronal:
            # FIX: Un-flip Z coordinate since coronal view is flipped
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, self.y_slice, ix
            self.active_view = "coronal"
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        elif ax == self.ax_sagittal:
            actual_z = self.shape[0] - 1 - iy  # Un-flip Z (sagittal is flipud)
            z, y, x = actual_z, ix, self.x_slice
        else:
            return
        
        # Clamp to volume
        z = int(np.clip(z, 0, self.shape[0] - 1))
        y = int(np.clip(y, 0, self.shape[1] - 1))
        x = int(np.clip(x, 0, self.shape[2] - 1))
        
        vessel, seed_type, _ = self.dragging_seed
        
        # Update seed position
        if seed_type == "ostium":
            self.seeds[vessel]["ostium"] = [z, y, x]
        elif seed_type.startswith("waypoint_"):
            idx = int(seed_type.split("_")[1])
            if 0 <= idx < len(self.seeds[vessel]["waypoints"]):
                self.seeds[vessel]["waypoints"][idx] = [z, y, x]
        
        # Update crosshair
        self.z_slice = z
        self.y_slice = y
        self.x_slice = x
        
        self._refresh_images()
        self.fig.canvas.draw_idle()
    
    def _on_scroll(self, event):
        delta = 1 if event.button == "up" else -1
        
        if event.inaxes == self.ax_axial:
            self.z_slice = int(np.clip(self.z_slice + delta, 0, self.shape[0] - 1))
        elif event.inaxes == self.ax_coronal:
            self.y_slice = int(np.clip(self.y_slice + delta, 0, self.shape[1] - 1))
        elif event.inaxes == self.ax_sagittal:
            self.x_slice = int(np.clip(self.x_slice + delta, 0, self.shape[2] - 1))
        
        self._refresh_images()
    
    def _on_key(self, event):
        key = event.key
        
        if key == "1":
            self.current_vessel = "LAD"
            print("[seed_reviewer] Switched to LAD")
        elif key == "2":
            self.current_vessel = "LCX"
            print("[seed_reviewer] Switched to LCX")
        elif key == "3":
            self.current_vessel = "RCA"
            print("[seed_reviewer] Switched to RCA")
        elif key == "u":
            # Undo last action
            if len(self.history) > 1:
                self.history.pop()  # Remove current state
                prev_state = self.history[-1]
                self.seeds = prev_state["seeds"]
                self.warnings = prev_state["warnings"]
                print("[seed_reviewer] Undid last action")
            else:
                print("[seed_reviewer] Nothing to undo")
        elif key == "r":
            # Reset current vessel
            self.seeds[self.current_vessel] = {"ostium": None, "waypoints": []}
            self.warnings[self.current_vessel] = []
            print(f"[seed_reviewer] Reset {self.current_vessel}")
            self._save_state()
        elif key == "c":
            # Clear warnings for current vessel
            self.warnings[self.current_vessel] = []
            print(f"[seed_reviewer] Cleared warnings for {self.current_vessel}")
            self._save_state()
        elif key == "d":
            # Delete nearest waypoint to cursor
            if self._delete_nearest_waypoint(self.z_slice, self.y_slice, self.x_slice):
                print(f"[seed_reviewer] Deleted waypoint from {self.current_vessel}")
        elif key == "s":
            self._save_and_close()
            return
        elif key == "q":
            print("[seed_reviewer] Quitting without saving.")
            plt.close(self.fig)
            return
        elif key == "w":
            self.ww = min(self.ww + 50, 3000)
        elif key == "W":
            self.ww = max(self.ww - 50, 50)
        elif key == "l":
            self.wl += 20
        elif key == "L":
            self.wl -= 20
        
        self._refresh_images()
        self._update_info_panel()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _save(self):
        output = {}
        for v in VESSEL_KEYS:
            sd = self.seeds[v]
            if sd["ostium"] is None:
                print(f"[seed_reviewer] WARNING: {v} has no ostium — still saving in output")
            
            entry = {
                "ostium_ijk": sd["ostium"],
                "waypoints_ijk": sd["waypoints"],
            }
            entry.update(VESSEL_CONFIGS[v])
            
            # Include warnings if any
            if v in self.warnings and self.warnings[v]:
                entry["_warnings"] = self.warnings[v]
            
            output[v] = entry
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[seed_reviewer] Seeds saved to {self.output_path}")
        self.fig.suptitle(
            f"✓ Saved to {self.output_path.name}  —  Seed Reviewer | s=Save & Continue  q=Quit without saving",
            fontsize=10, color="lightgreen"
        )
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    def _save_and_close(self):
        """Save seeds to JSON and close the window (signals pipeline to continue)."""
        self._save()
        # Write a signal file so the pipeline knows the reviewer is done
        signal_path = self.output_path.with_suffix(".done")
        signal_path.write_text("done")
        print(f"[seed_reviewer] Signal written: {signal_path}")
        plt.close(self.fig)
    
    def run(self):
        print("\n=== PCAT Seed Reviewer ===")
        print("Keys:  1=LAD  2=LCX  3=RCA  |  c=Clear warnings  d=Delete waypoint")
        print("       u=Undo  r=Reset  s=Save  q=Quit")
        print("       w/W=wider/narrower window  l/L=brighter/darker level")
        print("Click near seeds to drag them; click elsewhere to add waypoints.\n")
        plt.show()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    epilog = """
HOW TO USE
----------
This tool opens a 3-panel MPR viewer (Axial / Coronal / Sagittal) for reviewing
auto-generated coronary artery seeds from TotalSegmentator.

CLINICAL WARNING SYSTEM
  - GREEN (✓ HIGH): ostium set, ≥1 waypoint, no _warnings
  - YELLOW (⚠ REVIEW): ostium set but _warnings non-empty OR only 0 waypoints
  - RED (✗ CRITICAL): ostium missing OR _warnings includes "only N vessels found"

STEP-BY-STEP
  1. The tool loads DICOM series and auto-generated seeds with warnings.
     Seeds are displayed with confidence indicators in the info panel.

  2. Review each vessel:
     - GREEN (✓): Seeds are likely correct
     - YELLOW (⚠): Seeds need review due to warnings or missing waypoints
     - RED (✗): Critical issues that must be addressed

  3. Correct seeds as needed:
     - Click near a seed (within 10 pixels) and drag to reposition
     - Click elsewhere to add waypoints
     - Press 'd' to delete the nearest waypoint to cursor
     - Press 'c' to clear all warnings for current vessel after review

  4. Navigate slices with scroll wheel in any view panel.
     Crosshairs show the current position across all three views.

  5. Press 'S' to save the corrected seeds to the output JSON.
     All warning information is preserved in the output unless cleared.

KEYBOARD SHORTCUTS
  1 / 2 / 3   Switch active vessel (LAD / LCX / RCA)
  c           Clear warnings for current vessel
  d           Delete nearest waypoint to cursor
  u           Undo last action
  r           Reset all points and warnings for current vessel
  s           Save seeds to JSON and continue
  q           Quit without saving
  w / W       Window width wider / narrower  (adjust contrast)
  l / L       Window level brighter / darker (adjust brightness)

TIPS
  - Seeds are color-coded: LAD=orange, LCX=blue, RCA=green.
  - Square markers indicate ostium, circles indicate waypoints.
  - Drag seeds instead of deleting and re-adding to preserve warning context.
  - The confidence level automatically updates based on warnings and waypoints.
  - All warnings from auto_seeds.py are preserved unless explicitly cleared.
"""
    parser = argparse.ArgumentParser(
        description="Interactive seed point reviewer for PCAT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory for one patient"
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Input seed JSON file (from auto_seeds or seed_picker)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for corrected seed JSON file"
    )
    args = parser.parse_args()
    
    print(f"[seed_reviewer] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[seed_reviewer] Volume shape: {volume.shape}, spacing: {spacing_mm}")
    print(f"[seed_reviewer] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    print(f"[seed_reviewer] Loading seeds from {args.seeds} ...")
    with open(args.seeds, "r") as f:
        seeds_data = json.load(f)
    print(f"[seed_reviewer] Loaded seeds for {list(seeds_data.keys())}")
    
    reviewer = SeedReviewer(
        volume, spacing_mm, seeds_data, output_path=args.output
    )
    reviewer.run()


if __name__ == "__main__":
    main()