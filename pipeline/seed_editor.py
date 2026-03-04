#!/usr/bin/env python3
"""
seed_editor.py — Interactive seed placement with real-time CPR visualization.

A unified tool that merges seed placement and centerline visualization.
Replaces both seed_reviewer.py and centerline_editor.py.

Features:
  - Two linked MIP views (coronal + axial) for seed placement
  - Cubic spline centerline fitted through seeds, overlaid on MIP views
  - CPR panel that updates on mouse release (~100-200ms)
  - Per-vessel editing (LAD/LCX/RCA)
  - Saves seeds as JSON + centerlines as NPZ + .done signal file

Layout (figsize=(20, 10)):
┌──────────────┬──────────────┬──────────────┐
│  Coronal MIP │  Axial MIP   │     CPR      │
│  (scroll Y)  │  (scroll Z)  │              │
│              │              │              │
│  Seeds shown │  Seeds shown │  Auto-       │
│  as markers  │  as markers  │  generated   │
│  Spline      │  Spline      │  from spline │
│  overlaid    │  overlaid    │              │
├──────────────┴──────────────┴──────────────┤
│  Status bar: vessel, seed count, etc.      │
└────────────────────────────────────────────┘

Interaction:
  - Left click on seed → select it (yellow ring)
  - Left click on selected seed → drag to reposition
  - Left click on empty space → deselect
  - ← → arrow keys → cycle selection prev/next
  - Enter → add new waypoint at cursor position (inserted after selected seed)
  - Backspace → delete selected seed
  - Right click → delete nearest waypoint
  - Scroll wheel: adjust MIP slab center
  - Shift+scroll: adjust slab thickness (±2mm, range 5-50mm)
  - 1/2/3: switch vessel (resets selection)
  - u: undo | r: reset vessel | s: save | q: quit

Usage:
    python pipeline/seed_editor.py \
        --dicom /path/to/patient/dir \
        --seeds input_seeds.json \
        --output output/patient \
        --prefix patient
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

# Interactive backend — MUST be set before pyplot is imported
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Disable matplotlib's default 's' = save-figure keybinding
plt.rcParams['keymap.save'] = []

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.dicom_loader import load_dicom_series
from pipeline.visualize import (
    _bezier_fit_centerline,
    _sample_bezier_frame,
    _build_cpr_image_fast,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

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

# Window/Level for CT display (contrast-enhanced coronary CT)
WW = 800   # window width
WL = 300   # window level

# Default MIP slab thickness
DEFAULT_SLAB_MM = 20.0
MIN_SLAB_MM = 5.0
MAX_SLAB_MM = 50.0

# Seed proximity threshold (voxels)
PROXIMITY_THRESHOLD = 10  # distance² < 100


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mip_slab(
    volume: np.ndarray,
    center_idx: int,
    slab_voxels: int,
    axis: int,
) -> np.ndarray:
    """
    Compute thin-slab MIP centered on a slice index along a given axis.
    
    Parameters
    ----------
    volume : (Z, Y, X) float32
    center_idx : center slice index along the axis
    slab_voxels : half-thickness of the slab in voxels
    axis : 0=Z (axial), 1=Y (coronal)
    
    Returns
    -------
    mip : 2D array
    """
    shape = volume.shape
    lo = max(0, center_idx - slab_voxels)
    hi = min(shape[axis], center_idx + slab_voxels + 1)
    
    if axis == 0:
        slab = volume[lo:hi, :, :]
        mip = np.max(slab, axis=0)  # (Y, X)
    else:  # axis == 1
        slab = volume[:, lo:hi, :]
        mip = np.max(slab, axis=1)  # (Z, X)
    
    return mip


def _fit_spline_centerline(
    seeds_ijk: List[List[int]],
    spacing_mm: List[float],
    volume_shape: Tuple[int, int, int],
    step_mm: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Fit cubic spline through seed points, sample densely.
    
    Parameters
    ----------
    seeds_ijk : list of [z, y, x] seed points
    spacing_mm : [sz, sy, sx]
    volume_shape : (Z, Y, X)
    step_mm : arc-length step for dense sampling
    
    Returns
    -------
    dense_ijk : (M, 3) int array, or None if insufficient points
    """
    if len(seeds_ijk) < 2:
        return None
    
    pts_ijk = np.array(seeds_ijk, dtype=np.float64)
    pts_mm = pts_ijk * np.array(spacing_mm)
    
    # Compute cumulative arc-length along pts_mm
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    
    # Remove duplicate points (zero-length segments) — CRITICAL
    keep = np.concatenate([[True], seg > 1e-8])
    pts_mm = pts_mm[keep]
    
    if len(pts_mm) < 2:
        return None
    
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    
    if total < 1e-6:
        return None
    
    # Fit cubic spline
    if len(pts_mm) >= 3:
        cs = CubicSpline(arc, pts_mm, bc_type='not-a-knot')
    else:
        # Only 2 points — use linear interpolation
        cs = CubicSpline(arc, pts_mm, bc_type='natural')
    
    # Sample densely
    n_out = max(10, int(total / step_mm))
    s_vals = np.linspace(0, total, n_out)
    dense_mm = cs(s_vals)
    
    # Convert back to voxel indices and clip to volume
    dense_ijk = np.round(dense_mm / np.array(spacing_mm)).astype(int)
    dense_ijk = np.clip(dense_ijk, 0, np.array(volume_shape) - 1)
    
    return dense_ijk


def _compute_cpr_from_centerline(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    n_pixels: int = 256,
    row_extent_mm: float = 15.0,
) -> Optional[np.ndarray]:
    """
    Compute CPR image from dense centerline using fast linear interpolation.
    
    Parameters
    ----------
    volume : (Z, Y, X) float32 HU
    centerline_ijk : (N, 3) dense centerline points
    spacing_mm : [sz, sy, sx]
    n_pixels : output image size
    row_extent_mm : half-width of CPR in mm
    
    Returns
    -------
    cpr_img : (n_pixels, n_pixels) float32, or None on failure
    """
    if centerline_ijk is None or len(centerline_ijk) < 3:
        return None
    
    vox_size = np.array(spacing_mm, dtype=np.float64)
    cl_mm = centerline_ijk.astype(np.float64) * vox_size
    
    try:
        cs, total_len = _bezier_fit_centerline(cl_mm)
    except ValueError:
        return None
    
    s, positions, tangents, normals, binormals = _sample_bezier_frame(
        cs, total_len, n_pixels
    )
    
    cpr_img = _build_cpr_image_fast(
        volume, vox_size, positions, normals, binormals,
        n_rows=n_pixels, row_extent_mm=row_extent_mm, slab_mm=0.0
    )
    
    return cpr_img


# ─────────────────────────────────────────────────────────────────────────────
# Main Editor Class
# ─────────────────────────────────────────────────────────────────────────────

class SeedEditor:
    """
    Interactive seed editor with real-time CPR visualization.
    
    Parameters
    ----------
    volume : (Z, Y, X) float32
        CT HU volume
    spacing_mm : [sz, sy, sx]
        Voxel spacing in mm
    seeds_data : dict
        Seeds dictionary from JSON file
    output_dir : Path
        Directory for saving results
    prefix : str
        Filename prefix for output files
    """
    
    def __init__(
        self,
        volume: np.ndarray,
        spacing_mm: List[float],
        seeds_data: Dict[str, Any],
        output_dir: Path,
        prefix: str,
    ):
        self.volume = volume
        self.spacing_mm = spacing_mm
        self.vox_size = np.array(spacing_mm, dtype=np.float64)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.volume_shape = volume.shape  # (Z, Y, X)
        
        # Per-vessel data: seeds[vessel] = {"ostium": [z,y,x]|None, "waypoints": [[z,y,x], ...]}
        self.seeds: Dict[str, Dict] = {}
        self._load_seeds_from_data(seeds_data)
        
        # Current state
        self.current_vessel = "LAD"
        
        # MIP slab settings
        self.slab_mm = DEFAULT_SLAB_MM
        
        # MIP slab centers (in voxel indices)
        self.y_center: Dict[str, int] = {}  # For coronal MIP (Y-axis collapsed)
        self.z_center: Dict[str, int] = {}  # For axial MIP (Z-axis collapsed)
        
        for v in VESSEL_KEYS:
            # Initialize slab centers based on existing seeds or volume center
            all_pts = []
            if self.seeds[v]["ostium"] is not None:
                all_pts.append(self.seeds[v]["ostium"])
            all_pts.extend(self.seeds[v]["waypoints"])
            
            if all_pts:
                pts_arr = np.array(all_pts)
                self.y_center[v] = int(np.mean(pts_arr[:, 1]))
                self.z_center[v] = int(np.mean(pts_arr[:, 0]))
            else:
                self.y_center[v] = self.volume_shape[1] // 2
                self.z_center[v] = self.volume_shape[0] // 2
        
        # Computed centerlines (dense) for each vessel
        self.centerlines: Dict[str, Optional[np.ndarray]] = {}
        self._recompute_all_centerlines()
        
        # Interaction state
        self.dragging_seed: Optional[Tuple[str, str, List[int]]] = None  # (vessel, type, original_pos)
        self._selected_idx: Optional[int] = None  # index into _get_all_seeds_for_vessel(current_vessel)
        self._coronal_selection_ring = None  # scatter artist for yellow highlight ring
        self._axial_selection_ring = None    # scatter artist for yellow highlight ring
        
        # History for undo
        self.history: List[Dict] = []
        self._save_state()
        
        # Matplotlib handles
        self.fig = None
        self.ax_coronal = None
        self.ax_axial = None
        self.ax_cpr = None
        self.ax_status = None
        
        # Image handles
        self._coronal_im = None
        self._axial_im = None
        self._cpr_im = None
        
        # Line handles for splines
        self._coronal_spline_lines: Dict[str, Any] = {}
        self._axial_spline_lines: Dict[str, Any] = {}
        
        # Scatter handles for seeds
        self._coronal_seed_scatters: Dict[str, Dict[str, Any]] = {}
        self._axial_seed_scatters: Dict[str, Dict[str, Any]] = {}
        
        # MIP caches: (center, slab_vox, mip_data)
        self._cached_coronal_mip: Optional[Tuple[int, int, np.ndarray]] = None
        self._cached_axial_mip: Optional[Tuple[int, int, np.ndarray]] = None
        
        # Text handles for CPR panel
        self._cpr_text_no_seeds = None
        self._cpr_text_failed = None
        
        # Text handles for status bar
        self._status_vessel_texts: List[Any] = []
        self._status_info_text = None
        
        # Build GUI
        self._build_gui()
        self._update_display()
    
    def _load_seeds_from_data(self, seeds_data: Dict[str, Any]) -> None:
        """Load seeds from input JSON data."""
        for v in VESSEL_KEYS:
            self.seeds[v] = {"ostium": None, "waypoints": []}
            
            if v in seeds_data:
                vessel_data = seeds_data[v]
                ostium = vessel_data.get("ostium_ijk")
                waypoints = vessel_data.get("waypoints_ijk", [])
                
                if ostium:
                    self.seeds[v]["ostium"] = list(ostium)
                self.seeds[v]["waypoints"] = [list(wp) for wp in waypoints]
    
    def _save_state(self) -> None:
        """Save current state for undo functionality."""
        self.history.append({
            "seeds": copy.deepcopy(self.seeds),
            "y_center": copy.deepcopy(self.y_center),
            "z_center": copy.deepcopy(self.z_center),
        })
        # Keep only last 50 states
        if len(self.history) > 50:
            self.history.pop(0)
    
    def _undo(self) -> None:
        """Undo last action."""
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            prev_state = self.history[-1]
            self.seeds = prev_state["seeds"]
            self.y_center = prev_state["y_center"]
            self.z_center = prev_state["z_center"]
            self._recompute_all_centerlines()
            self._selected_idx = None
            print("[seed_editor] Undo")
            self._update_display()
        else:
            print("[seed_editor] Nothing to undo")
    
    def _reset_vessel(self) -> None:
        """Reset current vessel (clear all seeds)."""
        v = self.current_vessel
        self.seeds[v] = {"ostium": None, "waypoints": []}
        self.centerlines[v] = None
        self._selected_idx = None
        self._save_state()
        print(f"[seed_editor] Reset {v}")
        self._update_display()
    
    def _switch_vessel(self, vessel_name: str) -> None:
        """Switch to a different vessel."""
        if vessel_name in VESSEL_KEYS:
            self.current_vessel = vessel_name
            self._selected_idx = None
            print(f"[seed_editor] Switched to {vessel_name}")
            self._update_display()
    
    def _get_all_seeds_for_vessel(self, vessel: str) -> List[List[int]]:
        """Get all seeds (ostium + waypoints) for a vessel."""
        pts = []
        if self.seeds[vessel]["ostium"] is not None:
            pts.append(self.seeds[vessel]["ostium"])
        pts.extend(self.seeds[vessel]["waypoints"])
        return pts
    
    def _get_seed_index_from_type(self, vessel: str, seed_type_str: str) -> int:
        """Convert seed_type string ("ostium" or "waypoint_N") to index in all_seeds list."""
        if seed_type_str == "ostium":
            return 0
        # "waypoint_N" → index is N + (1 if ostium exists else 0)
        wp_idx = int(seed_type_str.split("_")[1])
        offset = 1 if self.seeds[vessel]["ostium"] is not None else 0
        return wp_idx + offset

    def _get_selected_seed_info(self) -> Optional[Tuple[str, List[int]]]:
        """Get (seed_type_str, position) for currently selected seed, or None."""
        if self._selected_idx is None:
            return None
        v = self.current_vessel
        all_seeds = self._get_all_seeds_for_vessel(v)
        if not all_seeds or self._selected_idx >= len(all_seeds):
            self._selected_idx = None
            return None
        pos = all_seeds[self._selected_idx]
        # Determine type string
        has_ostium = self.seeds[v]["ostium"] is not None
        if has_ostium and self._selected_idx == 0:
            return ("ostium", pos)
        wp_idx = self._selected_idx - (1 if has_ostium else 0)
        return (f"waypoint_{wp_idx}", pos)
    
    def _recompute_all_centerlines(self) -> None:
        """Recompute centerlines for all vessels."""
        for v in VESSEL_KEYS:
            all_seeds = self._get_all_seeds_for_vessel(v)
            if len(all_seeds) >= 2:
                self.centerlines[v] = _fit_spline_centerline(
                    all_seeds, self.spacing_mm, self.volume_shape
                )
            else:
                self.centerlines[v] = None
    
    def _recompute_current_centerline(self) -> None:
        """Recompute centerline for current vessel only."""
        v = self.current_vessel
        all_seeds = self._get_all_seeds_for_vessel(v)
        if len(all_seeds) >= 2:
            self.centerlines[v] = _fit_spline_centerline(
                all_seeds, self.spacing_mm, self.volume_shape
            )
        else:
            self.centerlines[v] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # GUI Building
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_gui(self) -> None:
        """Build the matplotlib figure (default theme)."""
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.canvas.manager.set_window_title(
            f"PCAT Seed Editor — {self.prefix}"
        )
        
        # Title with controls hint
        self.fig.suptitle(
            "Left-click: select/drag  |  Enter: add seed  |  Backspace: delete selected  |  Right-click: delete nearest\n"
            "←→ arrows: cycle selection  |  Scroll: slab center  |  Shift+scroll: slab thickness  |  1/2/3: Vessel  |  u: Undo  |  r: Reset  |  s: Save  |  q: Quit",
            fontsize=9,
        )
        
        # Layout: 3 panels + status bar
        gs = self.fig.add_gridspec(
            2, 3,
            height_ratios=[5, 1],
            width_ratios=[1, 1, 1],
            left=0.04, right=0.98,
            bottom=0.06, top=0.90,
            hspace=0.12, wspace=0.08,
        )
        
        # Main panels
        self.ax_coronal = self.fig.add_subplot(gs[0, 0])
        self.ax_axial = self.fig.add_subplot(gs[0, 1])
        self.ax_cpr = self.fig.add_subplot(gs[0, 2])
        self.ax_status = self.fig.add_subplot(gs[1, :])
        
        # Style axes (default matplotlib theme)
        for ax in [self.ax_coronal, self.ax_axial, self.ax_cpr]:
            ax.tick_params(labelsize=8)
        
        self.ax_status.axis("off")
        
        self.ax_coronal.set_title("Coronal MIP (scroll Y)", fontsize=10)
        self.ax_axial.set_title("Axial MIP (scroll Z)", fontsize=10)
        self.ax_cpr.set_title("CPR (updates on release)", fontsize=10)
        
        # Initialize scatter and line handles
        for v in VESSEL_KEYS:
            self._coronal_seed_scatters[v] = {"ostium": None, "waypoints": None}
            self._axial_seed_scatters[v] = {"ostium": None, "waypoints": None}
            self._coronal_spline_lines[v] = None
            self._axial_spline_lines[v] = None
        
        # Event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
    
    def _clim(self) -> Tuple[float, float]:
        """Get color limits for window/level."""
        lo = WL - WW / 2
        hi = WL + WW / 2
        return lo, hi
    
    # ─────────────────────────────────────────────────────────────────────────
    # Display Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_display(self) -> None:
        """Update all display panels."""
        self._draw_coronal_mip()
        self._draw_axial_mip()
        self._draw_cpr()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _draw_coronal_mip(self) -> None:
        """Draw coronal MIP with seeds and spline overlay."""
        v = self.current_vessel

        
        # Compute MIP (project Y axis)
        y_center = self.y_center[v]
        slab_vox = int(np.round(self.slab_mm / self.spacing_mm[1] / 2))
        slab_vox = max(1, slab_vox)
        
        # Check if we can use cached MIP
        mip = None
        if (self._cached_coronal_mip is not None and
            self._cached_coronal_mip[0] == y_center and
            self._cached_coronal_mip[1] == slab_vox):
            mip = self._cached_coronal_mip[2]
        else:
            mip = _compute_mip_slab(self.volume, y_center, slab_vox, axis=1)  # (Z, X)
            self._cached_coronal_mip = (y_center, slab_vox, mip)
        
        # Normalize for display
        lo, hi = self._clim()
        mip_norm = np.clip(mip, lo, hi)
        mip_norm = (mip_norm - lo) / (hi - lo)
        
        # Display with origin='upper' and flip Z so superior at top
        mip_display = np.flipud(mip_norm)
        
        # First call: create artists
        if self._coronal_im is None:
            self._coronal_im = self.ax_coronal.imshow(
                mip_display,
                aspect='auto',
                origin='upper',
                cmap='gray',
                vmin=0, vmax=1,
            )
            
            # Create spline lines and scatter plots for each vessel
            for vessel_name in VESSEL_KEYS:
                vc = VESSEL_COLORS[vessel_name]
                # Spline line
                self._coronal_spline_lines[vessel_name], = self.ax_coronal.plot(
                    [], [], color=vc, linewidth=1.5, alpha=0.8,
                )
                # Ostium scatter
                self._coronal_seed_scatters[vessel_name]["ostium"] = self.ax_coronal.scatter(
                    [], [], c=vc, s=80, marker='s',
                    edgecolors='white', linewidths=1.5,
                    alpha=1.0, zorder=5,
                )
                # Waypoints scatter
                self._coronal_seed_scatters[vessel_name]["waypoints"] = self.ax_coronal.scatter(
                    [], [], c=vc, s=40, marker='o',
                    edgecolors='white', linewidths=0.8,
                    alpha=1.0, zorder=4,
                )
            
            # Selection ring
            self._coronal_selection_ring = self.ax_coronal.scatter(
                [], [], c='none', s=200, marker='o',
                edgecolors='yellow', linewidths=3.0,
                alpha=1.0, zorder=10,
            )
        else:
            # Update existing artists
            self._coronal_im.set_data(mip_display)
            self._coronal_im.set_clim(0, 1)
        
        # Update splines for all vessels (with slab filtering)
        for vessel_name in VESSEL_KEYS:
            cl = self.centerlines[vessel_name]
            alpha = 0.8 if vessel_name == v else 0.3
            
            if cl is not None and len(cl) > 1:
                # Filter by Y-proximity: abs(cl[:, 1] - y_center) <= slab_vox
                # Insert NaN at breaks for disjoint segments
                x_coords = cl[:, 2].astype(np.float64)
                z_coords_flipped = (self.volume_shape[0] - 1 - cl[:, 0]).astype(np.float64)
                
                # Create mask for points within slab
                in_slab = np.abs(cl[:, 1] - y_center) <= slab_vox
                
                # Set coords to NaN where not in slab (breaks the line)
                x_filtered = x_coords.copy()
                z_filtered = z_coords_flipped.copy()
                x_filtered[~in_slab] = np.nan
                z_filtered[~in_slab] = np.nan
                
                self._coronal_spline_lines[vessel_name].set_data(x_filtered, z_filtered)
                self._coronal_spline_lines[vessel_name].set_alpha(alpha)
            else:
                # No centerline: clear the line
                self._coronal_spline_lines[vessel_name].set_data([], [])
        
        # Update seeds for all vessels
        for vessel_name in VESSEL_KEYS:
            sd = self.seeds[vessel_name]
            is_active = vessel_name == v
            alpha = 1.0 if is_active else 0.3
            
            # Ostium
            if sd["ostium"] is not None:
                oz, oy, ox = sd["ostium"]
                # Check if within slab (tightened to slab_vox)
                if abs(oy - y_center) <= slab_vox:
                    flipped_z = self.volume_shape[0] - 1 - oz
                    self._coronal_seed_scatters[vessel_name]["ostium"].set_offsets(
                        np.array([[ox, flipped_z]], dtype=np.float64)
                    )
                else:
                    self._coronal_seed_scatters[vessel_name]["ostium"].set_offsets(
                        np.empty((0, 2), dtype=np.float64)
                    )
            else:
                self._coronal_seed_scatters[vessel_name]["ostium"].set_offsets(
                    np.empty((0, 2), dtype=np.float64)
                )
            
            # Waypoints
            waypoint_coords = []
            for wp in sd["waypoints"]:
                wz, wy, wx = wp
                if abs(wy - y_center) <= slab_vox:
                    waypoint_coords.append([wx, self.volume_shape[0] - 1 - wz])
            
            if waypoint_coords:
                self._coronal_seed_scatters[vessel_name]["waypoints"].set_offsets(
                    np.array(waypoint_coords, dtype=np.float64)
                )
            else:
                self._coronal_seed_scatters[vessel_name]["waypoints"].set_offsets(
                    np.empty((0, 2), dtype=np.float64)
                )
            
            # Update alpha for both scatter plots
            self._coronal_seed_scatters[vessel_name]["ostium"].set_alpha(alpha)
            self._coronal_seed_scatters[vessel_name]["waypoints"].set_alpha(alpha)
        
        # Update selection ring
        if self._selected_idx is not None:
            info = self._get_selected_seed_info()
            if info is not None:
                _, pos = info
                sz, sy, sx = pos
                y_center = self.y_center[self.current_vessel]
                slab_vox_y = int(np.round(self.slab_mm / self.spacing_mm[1] / 2))
                slab_vox_y = max(1, slab_vox_y)
                if abs(sy - y_center) <= slab_vox_y:
                    flipped_z = self.volume_shape[0] - 1 - sz
                    self._coronal_selection_ring.set_offsets(
                        np.array([[sx, flipped_z]], dtype=np.float64)
                    )
                else:
                    self._coronal_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
            else:
                self._coronal_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
        else:
            if self._coronal_selection_ring is not None:
                self._coronal_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
        
        self.ax_coronal.set_title(
            f"Coronal MIP (Y={y_center}, slab={self.slab_mm:.0f}mm)",
            fontsize=10
        )
    
    def _draw_axial_mip(self) -> None:
        """Draw axial MIP with seeds and spline overlay."""
        v = self.current_vessel

        
        # Compute MIP (project Z axis)
        z_center = self.z_center[v]
        slab_vox = int(np.round(self.slab_mm / self.spacing_mm[0] / 2))
        slab_vox = max(1, slab_vox)
        
        # Check if we can use cached MIP
        mip = None
        if (self._cached_axial_mip is not None and
            self._cached_axial_mip[0] == z_center and
            self._cached_axial_mip[1] == slab_vox):
            mip = self._cached_axial_mip[2]
        else:
            mip = _compute_mip_slab(self.volume, z_center, slab_vox, axis=0)  # (Y, X)
            self._cached_axial_mip = (z_center, slab_vox, mip)
        
        # Normalize for display
        lo, hi = self._clim()
        mip_norm = np.clip(mip, lo, hi)
        mip_norm = (mip_norm - lo) / (hi - lo)
        
        # First call: create artists
        if self._axial_im is None:
            self._axial_im = self.ax_axial.imshow(
                mip_norm,
                aspect='auto',
                origin='upper',
                cmap='gray',
                vmin=0, vmax=1,
            )
            
            # Create spline lines and scatter plots for each vessel
            for vessel_name in VESSEL_KEYS:
                vc = VESSEL_COLORS[vessel_name]
                # Spline line
                self._axial_spline_lines[vessel_name], = self.ax_axial.plot(
                    [], [], color=vc, linewidth=1.5, alpha=0.8,
                )
                # Ostium scatter
                self._axial_seed_scatters[vessel_name]["ostium"] = self.ax_axial.scatter(
                    [], [], c=vc, s=80, marker='s',
                    edgecolors='white', linewidths=1.5,
                    alpha=1.0, zorder=5,
                )
                # Waypoints scatter
                self._axial_seed_scatters[vessel_name]["waypoints"] = self.ax_axial.scatter(
                    [], [], c=vc, s=40, marker='o',
                    edgecolors='white', linewidths=0.8,
                    alpha=1.0, zorder=4,
                )
            
            # Selection ring
            self._axial_selection_ring = self.ax_axial.scatter(
                [], [], c='none', s=200, marker='o',
                edgecolors='yellow', linewidths=3.0,
                alpha=1.0, zorder=10,
            )
        else:
            # Update existing artists
            self._axial_im.set_data(mip_norm)
            self._axial_im.set_clim(0, 1)
        
        # Update splines for all vessels (with slab filtering)
        for vessel_name in VESSEL_KEYS:
            cl = self.centerlines[vessel_name]
            alpha = 0.8 if vessel_name == v else 0.3
            
            if cl is not None and len(cl) > 1:
                # Filter by Z-proximity: abs(cl[:, 0] - z_center) <= slab_vox
                # Insert NaN at breaks for disjoint segments
                x_coords = cl[:, 2].astype(np.float64)
                y_coords = cl[:, 1].astype(np.float64)
                
                # Create mask for points within slab
                in_slab = np.abs(cl[:, 0] - z_center) <= slab_vox
                
                # Set coords to NaN where not in slab (breaks the line)
                x_filtered = x_coords.copy()
                y_filtered = y_coords.copy()
                x_filtered[~in_slab] = np.nan
                y_filtered[~in_slab] = np.nan
                
                self._axial_spline_lines[vessel_name].set_data(x_filtered, y_filtered)
                self._axial_spline_lines[vessel_name].set_alpha(alpha)
            else:
                # No centerline: clear the line
                self._axial_spline_lines[vessel_name].set_data([], [])
        
        # Update seeds for all vessels
        for vessel_name in VESSEL_KEYS:
            sd = self.seeds[vessel_name]
            is_active = vessel_name == v
            alpha = 1.0 if is_active else 0.3
            
            # Ostium
            if sd["ostium"] is not None:
                oz, oy, ox = sd["ostium"]
                # Check if within slab (tightened to slab_vox)
                if abs(oz - z_center) <= slab_vox:
                    self._axial_seed_scatters[vessel_name]["ostium"].set_offsets(
                        np.array([[ox, oy]], dtype=np.float64)
                    )
                else:
                    self._axial_seed_scatters[vessel_name]["ostium"].set_offsets(
                        np.empty((0, 2), dtype=np.float64)
                    )
            else:
                self._axial_seed_scatters[vessel_name]["ostium"].set_offsets(
                    np.empty((0, 2), dtype=np.float64)
                )
            
            # Waypoints
            waypoint_coords = []
            for wp in sd["waypoints"]:
                wz, wy, wx = wp
                if abs(wz - z_center) <= slab_vox:
                    waypoint_coords.append([wx, wy])
            
            if waypoint_coords:
                self._axial_seed_scatters[vessel_name]["waypoints"].set_offsets(
                    np.array(waypoint_coords, dtype=np.float64)
                )
            else:
                self._axial_seed_scatters[vessel_name]["waypoints"].set_offsets(
                    np.empty((0, 2), dtype=np.float64)
                )
            
            # Update alpha for both scatter plots
            self._axial_seed_scatters[vessel_name]["ostium"].set_alpha(alpha)
            self._axial_seed_scatters[vessel_name]["waypoints"].set_alpha(alpha)
        
        # Update selection ring
        if self._selected_idx is not None:
            info = self._get_selected_seed_info()
            if info is not None:
                _, pos = info
                sz, sy, sx = pos
                z_center = self.z_center[self.current_vessel]
                slab_vox_z = int(np.round(self.slab_mm / self.spacing_mm[0] / 2))
                slab_vox_z = max(1, slab_vox_z)
                if abs(sz - z_center) <= slab_vox_z:
                    self._axial_selection_ring.set_offsets(
                        np.array([[sx, sy]], dtype=np.float64)
                    )
                else:
                    self._axial_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
            else:
                self._axial_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
        else:
            if self._axial_selection_ring is not None:
                self._axial_selection_ring.set_offsets(np.empty((0, 2), dtype=np.float64))
        
        self.ax_axial.set_title(
            f"Axial MIP (Z={z_center}, slab={self.slab_mm:.0f}mm)",
            fontsize=10
        )
    
    def _draw_cpr(self) -> None:
        """Draw CPR panel."""
        v = self.current_vessel
        cl = self.centerlines[v]
        
        # First call: create text artists
        if self._cpr_text_no_seeds is None:
            self._cpr_text_no_seeds = self.ax_cpr.text(
                0.5, 0.5,
                "Place ≥3 seeds to see CPR",
                ha='center', va='center',
                transform=self.ax_cpr.transAxes,
                fontsize=12, color='gray',
            )
            self._cpr_text_failed = self.ax_cpr.text(
                0.5, 0.5,
                "CPR computation failed",
                ha='center', va='center',
                transform=self.ax_cpr.transAxes,
                fontsize=12, color='red',
            )
        
        # Check if we have enough seeds
        if cl is None or len(cl) < 3:
            # Show "no seeds" text, hide failed text and image
            self._cpr_text_no_seeds.set_visible(True)
            self._cpr_text_failed.set_visible(False)
            if self._cpr_im is not None:
                self._cpr_im.set_visible(False)
            self.ax_cpr.set_title(f"CPR — {v}", fontsize=10)
            return
        
        # Compute CPR
        t0 = time.perf_counter()
        cpr_img = _compute_cpr_from_centerline(
            self.volume, cl, self.spacing_mm,
            n_pixels=256, row_extent_mm=15.0
        )
        dt = time.perf_counter() - t0
        print(f"[seed_editor] CPR: {dt*1000:.0f}ms")
        
        if cpr_img is None:
            # Show "failed" text, hide no seeds text and image
            self._cpr_text_no_seeds.set_visible(False)
            self._cpr_text_failed.set_visible(True)
            if self._cpr_im is not None:
                self._cpr_im.set_visible(False)
            return
        
        # Normalize
        lo, hi = self._clim()
        cpr_norm = np.clip(cpr_img, lo, hi)
        cpr_norm = (cpr_norm - lo) / (hi - lo)
        cpr_norm = np.nan_to_num(cpr_norm, nan=0.5)
        
        # Hide text, show image
        self._cpr_text_no_seeds.set_visible(False)
        self._cpr_text_failed.set_visible(False)
        
        # First call: create image artist
        if self._cpr_im is None:
            # Display: transpose so cols=arc-length, rows=lateral
            self._cpr_im = self.ax_cpr.imshow(
                cpr_norm.T,
                aspect='auto',
                origin='upper',
                cmap='gray',
                vmin=0, vmax=1,
                interpolation='bilinear',
            )
        else:
            # Update existing image
            self._cpr_im.set_data(cpr_norm.T)
            self._cpr_im.set_clim(0, 1)
            self._cpr_im.set_visible(True)
        
        self.ax_cpr.set_title(f"CPR — {v}", fontsize=10)
    
    def _update_status_bar(self) -> None:
        """Update status bar with current state."""
        v = self.current_vessel
        sd = self.seeds[v]
        color = VESSEL_COLORS[v]
        
        n_waypoints = len(sd["waypoints"])
        has_ostium = sd["ostium"] is not None
        n_seeds = n_waypoints + (1 if has_ostium else 0)
        
        # First call: create text artists
        if not self._status_vessel_texts:
            self.ax_status.axis("off")
            button_x = 0.02
            for i, vn in enumerate(VESSEL_KEYS):
                vc = VESSEL_COLORS[vn]
                is_current = vn == v
                weight = "bold" if is_current else "normal"
                alpha = 1.0 if is_current else 0.5
                txt = self.ax_status.text(
                    button_x + i * 0.08, 0.5,
                    f"[{i+1}] {vn}",
                    ha='left', va='center',
                    transform=self.ax_status.transAxes,
                    fontsize=9, fontweight=weight, color=vc, alpha=alpha,
                )
                self._status_vessel_texts.append(txt)
            
            msg = (
                f"  |  Vessel: {v}  |  "
                f"Ostium: {'SET' if has_ostium else 'NOT SET'}  |  "
                f"Waypoints: {n_waypoints}  |  "
                f"Total seeds: {n_seeds}  |  "
                f"Slab: {self.slab_mm:.0f}mm  |  "
                f"W/L: {WW}/{WL}  |  "
                f"Selected: #{self._selected_idx if self._selected_idx is not None else 'none'}"
            )
            self._status_info_text = self.ax_status.text(
                0.30, 0.5, msg,
                ha='left', va='center',
                transform=self.ax_status.transAxes,
                fontsize=9, color=color,
            )
        else:
            # Update existing text artists
            for i, vn in enumerate(VESSEL_KEYS):
                is_current = vn == v
                self._status_vessel_texts[i].set_fontweight(
                    "bold" if is_current else "normal"
                )
                self._status_vessel_texts[i].set_alpha(
                    1.0 if is_current else 0.5
                )
            
            msg = (
                f"  |  Vessel: {v}  |  "
                f"Ostium: {'SET' if has_ostium else 'NOT SET'}  |  "
                f"Waypoints: {n_waypoints}  |  "
                f"Total seeds: {n_seeds}  |  "
                f"Slab: {self.slab_mm:.0f}mm  |  "
                f"W/L: {WW}/{WL}  |  "
                f"Selected: #{self._selected_idx if self._selected_idx is not None else 'none'}"
            )
            self._status_info_text.set_text(msg)
            self._status_info_text.set_color(color)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Seed Finding
    # ─────────────────────────────────────────────────────────────────────────
    
    def _find_nearest_seed(
        self,
        z: int,
        y: int,
        x: int,
        vessel: Optional[str] = None,
        view: Optional[str] = None,
    ) -> Optional[Tuple[str, str, List[int]]]:
        """
        Find the nearest seed to the given coordinates.
        Parameters
        ----------
        view : 'coronal' or 'axial' or None
            If set, uses 2D projected distance (ignoring the depth axis)
            so seeds visible in the slab can be clicked.
        Returns (vessel, seed_type, position) or None if too far.
        seed_type is "ostium" or "waypoint_N"
        """
        if vessel is None:
            vessel = self.current_vessel
        min_dist = float("inf")
        def _dist2(sz, sy, sx):
            if view == 'coronal':
                # Coronal shows (Z, X) — ignore Y depth
                return (sz - z) ** 2 + (sx - x) ** 2
            elif view == 'axial':
                # Axial shows (Y, X) — ignore Z depth
                return (sy - y) ** 2 + (sx - x) ** 2
            else:
                return (sz - z) ** 2 + (sy - y) ** 2 + (sx - x) ** 2
        # Check ostium
        if self.seeds[vessel]["ostium"] is not None:
            oz, oy, ox = self.seeds[vessel]["ostium"]
            dist = _dist2(oz, oy, ox)
            if dist < min_dist:
                min_dist = dist
                nearest = (vessel, "ostium", self.seeds[vessel]["ostium"])
        for i, wp in enumerate(self.seeds[vessel]["waypoints"]):
            wz, wy, wx = wp
            dist = _dist2(wz, wy, wx)
            if dist < min_dist:
                min_dist = dist
                nearest = (vessel, f"waypoint_{i}", wp)
        if min_dist > PROXIMITY_THRESHOLD ** 2:
            return None
    
    def _delete_nearest_waypoint(
        self,
        z: int,
        y: int,
        x: int,
        vessel: Optional[str] = None,
        view: Optional[str] = None,
    ) -> bool:
        """Delete the nearest waypoint to the given coordinates."""
        if vessel is None:
            vessel = self.current_vessel
        min_dist = float("inf")
        for i, wp in enumerate(self.seeds[vessel]["waypoints"]):
            wz, wy, wx = wp
            if view == 'coronal':
                dist = (wz - z) ** 2 + (wx - x) ** 2
            elif view == 'axial':
                dist = (wy - y) ** 2 + (wx - x) ** 2
            else:
                dist = (wz - z) ** 2 + (wy - y) ** 2 + (wx - x) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        if min_dist <= PROXIMITY_THRESHOLD ** 2 and nearest_idx is not None:
            removed = self.seeds[vessel]["waypoints"].pop(nearest_idx)
            print(f"[seed_editor] Deleted waypoint from {vessel}: {removed}")
            self._recompute_current_centerline()
            self._save_state()
            return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press: select, drag, or add seeds."""
        if event.inaxes not in [self.ax_coronal, self.ax_axial]:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        # Determine view type and (z, y, x) seed position
        if ax == self.ax_coronal:
            view = 'coronal'
            actual_z = self.volume_shape[0] - 1 - iy
            z, y, x = actual_z, self.y_center[self.current_vessel], ix
        else:  # axial
            view = 'axial'
            z, y, x = self.z_center[self.current_vessel], iy, ix
        z = int(np.clip(z, 0, self.volume_shape[0] - 1))
        y = int(np.clip(y, 0, self.volume_shape[1] - 1))
        x = int(np.clip(x, 0, self.volume_shape[2] - 1))
        v = self.current_vessel
        
        # Left click: select, drag, or add
        if event.button == 1:
            nearest = self._find_nearest_seed(z, y, x, v, view=view)
            if nearest:
                vessel, seed_type, pos = nearest
                clicked_idx = self._get_seed_index_from_type(vessel, seed_type)
                if clicked_idx == self._selected_idx:
                    # Already selected → start drag
                    self.dragging_seed = (vessel, seed_type, list(pos))
                    print(f"[seed_editor] Dragging {vessel} {seed_type}")
                else:
                    # Select this seed
                    self._selected_idx = clicked_idx
                    print(f"[seed_editor] Selected seed #{clicked_idx} ({seed_type})")
                    self._update_display()
            else:
                # Clicked empty space → deselect
                self._selected_idx = None
                self._update_display()
        
        # Right click — delete nearest waypoint
        elif event.button == 3:
            if self._delete_nearest_waypoint(z, y, x, v, view=view):
                # Adjust selection index after deletion
                all_seeds = self._get_all_seeds_for_vessel(v)
                if self._selected_idx is not None:
                    if self._selected_idx >= len(all_seeds):
                        self._selected_idx = len(all_seeds) - 1 if all_seeds else None
                self._update_display()
    
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dragging seeds (preserves depth axis)."""
        if self.dragging_seed is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        vessel, seed_type, _ = self.dragging_seed
        
        # Get current seed position to preserve depth axis
        if seed_type == "ostium":
            old_pos = self.seeds[vessel]["ostium"]
        else:
            wp_idx = int(seed_type.split("_")[1])
            old_pos = self.seeds[vessel]["waypoints"][wp_idx]
        
        # Update only the two visible axes, preserve the depth axis
        if ax == self.ax_coronal:
            # Coronal shows (Z, X) — Y is depth, preserve it
            new_z = int(np.clip(self.volume_shape[0] - 1 - iy, 0, self.volume_shape[0] - 1))
            new_x = int(np.clip(ix, 0, self.volume_shape[2] - 1))
            new_pos = [new_z, old_pos[1], new_x]
        else:  # axial
            # Axial shows (Y, X) — Z is depth, preserve it
            new_y = int(np.clip(iy, 0, self.volume_shape[1] - 1))
            new_x = int(np.clip(ix, 0, self.volume_shape[2] - 1))
            new_pos = [old_pos[0], new_y, new_x]
        # Update seed position
        if seed_type == "ostium":
            self.seeds[vessel]["ostium"] = new_pos
        elif seed_type.startswith("waypoint_"):
            wp_idx = int(seed_type.split("_")[1])
            if 0 <= wp_idx < len(self.seeds[vessel]["waypoints"]):
                self.seeds[vessel]["waypoints"][wp_idx] = new_pos
        
        # Recompute centerline (fast)
        self._recompute_current_centerline()
        
        # Update MIP views only (NOT CPR during drag)
        self._draw_coronal_mip()
        self._draw_axial_mip()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release: compute CPR."""
        if self.dragging_seed:
            vessel, seed_type, _ = self.dragging_seed
            print(f"[seed_editor] Released {vessel} {seed_type}")
            self.dragging_seed = None
            self._save_state()
            
            # Now compute CPR
            self._draw_cpr()
            self.fig.canvas.draw_idle()
    
    def _on_scroll(self, event) -> None:
        """Handle scroll wheel for MIP slab control."""
        if event.inaxes not in [self.ax_coronal, self.ax_axial]:
            return
        
        v = self.current_vessel
        
        if event.key == 'shift':
            # Shift+scroll: adjust slab thickness
            delta = 2.0 if event.button == 'up' else -2.0
            self.slab_mm = np.clip(
                self.slab_mm + delta,
                MIN_SLAB_MM,
                MAX_SLAB_MM
            )
            print(f"[seed_editor] Slab thickness: {self.slab_mm:.0f}mm")
        else:
            # Regular scroll: adjust slab center
            delta = 2 if event.button == 'up' else -2
            
            if event.inaxes == self.ax_coronal:
                # Scroll on coronal: adjust Y center
                new_y = self.y_center[v] + delta
                self.y_center[v] = int(np.clip(new_y, 0, self.volume_shape[1] - 1))
            else:  # axial
                # Scroll on axial: adjust Z center
                new_z = self.z_center[v] + delta
                self.z_center[v] = int(np.clip(new_z, 0, self.volume_shape[0] - 1))
        
        self._update_display()
    
    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        key = event.key
        
        if key == 'q':
            print("[seed_editor] Quit without saving")
            plt.close(self.fig)
        elif key == 's':
            self._save_and_close()
        elif key == 'u':
            self._undo()
        elif key == 'r':
            self._reset_vessel()
        elif key == 'backspace':
            # Delete selected seed
            v = self.current_vessel
            info = self._get_selected_seed_info()
            if info is None:
                print("[seed_editor] No seed selected — press left-click to select one")
                return
            seed_type, pos = info
            if seed_type == "ostium":
                self.seeds[v]["ostium"] = None
                print(f"[seed_editor] Deleted {v} ostium")
            else:
                wp_idx = int(seed_type.split("_")[1])
                if 0 <= wp_idx < len(self.seeds[v]["waypoints"]):
                    removed = self.seeds[v]["waypoints"].pop(wp_idx)
                    print(f"[seed_editor] Deleted {v} {seed_type}: {removed}")
            # Clamp selection
            all_seeds = self._get_all_seeds_for_vessel(v)
            if not all_seeds:
                self._selected_idx = None
            elif self._selected_idx is not None and self._selected_idx >= len(all_seeds):
                self._selected_idx = len(all_seeds) - 1
            self._recompute_current_centerline()
            self._save_state()
            self._update_display()
        elif key == 'return':
            # Add new waypoint at cursor position
            if event.inaxes not in [self.ax_coronal, self.ax_axial]:
                print("[seed_editor] Hover mouse over a MIP view, then press Enter")
                return
            if event.xdata is None or event.ydata is None:
                return
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            v = self.current_vessel
            if event.inaxes == self.ax_coronal:
                actual_z = self.volume_shape[0] - 1 - iy  # flipud
                z, y, x = actual_z, self.y_center[v], ix
            else:
                z, y, x = self.z_center[v], iy, ix
            z = int(np.clip(z, 0, self.volume_shape[0] - 1))
            y = int(np.clip(y, 0, self.volume_shape[1] - 1))
            x = int(np.clip(x, 0, self.volume_shape[2] - 1))
            # Insert after selected seed, or append at end
            has_ostium = self.seeds[v]["ostium"] is not None
            all_seeds = self._get_all_seeds_for_vessel(v)
            if self._selected_idx is not None and self._selected_idx < len(all_seeds):
                # Insert after selected
                wp_insert_idx = (self._selected_idx + 1) - (1 if has_ostium else 0)
                wp_insert_idx = max(0, min(wp_insert_idx, len(self.seeds[v]["waypoints"])))
                self.seeds[v]["waypoints"].insert(wp_insert_idx, [z, y, x])
                self._selected_idx += 1  # Select the newly inserted seed
            else:
                # No selection — append at end
                self.seeds[v]["waypoints"].append([z, y, x])
                all_seeds = self._get_all_seeds_for_vessel(v)
                self._selected_idx = len(all_seeds) - 1
            n = len(self.seeds[v]["waypoints"])
            print(f"[seed_editor] Added {v} waypoint ({z}, {y}, {x}), total {n}")
            self._recompute_current_centerline()
            self._save_state()
            self._update_display()
        elif key == 'left':
            # Cycle selection backward
            v = self.current_vessel
            all_seeds = self._get_all_seeds_for_vessel(v)
            if not all_seeds:
                return
            if self._selected_idx is None:
                self._selected_idx = 0
            else:
                self._selected_idx = max(0, self._selected_idx - 1)
            print(f"[seed_editor] Selected seed #{self._selected_idx}")
            self._update_display()
        elif key == 'right':
            # Cycle selection forward
            v = self.current_vessel
            all_seeds = self._get_all_seeds_for_vessel(v)
            if not all_seeds:
                return
            if self._selected_idx is None:
                self._selected_idx = 0
            else:
                self._selected_idx = min(len(all_seeds) - 1, self._selected_idx + 1)
            print(f"[seed_editor] Selected seed #{self._selected_idx}")
            self._update_display()
        elif key == '1':
            self._switch_vessel('LAD')
        elif key == '2':
            self._switch_vessel('LCX')
        elif key == '3':
            self._switch_vessel('RCA')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_and_close(self) -> None:
        """Save seeds, centerlines, and close."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ── Save seeds JSON ───────────────────────────────────────────────────
        output_json = {}
        for v in VESSEL_KEYS:
            sd = self.seeds[v]
            if sd["ostium"] is None:
                print(f"[seed_editor] WARNING: {v} has no ostium — still saving")
            
            entry = {
                "ostium_ijk": sd["ostium"],
                "waypoints_ijk": sd["waypoints"],
            }
            entry.update(VESSEL_CONFIGS[v])
            output_json[v] = entry
        
        json_path = self.output_dir / f"{self.prefix}_seeds.json"
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"[seed_editor] Seeds saved to {json_path}")
        
        # ── Save centerlines NPZ ──────────────────────────────────────────────
        npz_data = {}
        for v in VESSEL_KEYS:
            # Save dense centerline if available
            cl = self.centerlines[v]
            if cl is not None and len(cl) > 0:
                npz_data[f"{v}_centerline_ijk"] = cl
            
            # Save raw seeds
            all_seeds = self._get_all_seeds_for_vessel(v)
            if all_seeds:
                npz_data[f"{v}_seeds_ijk"] = np.array(all_seeds, dtype=int)
        
        npz_path = self.output_dir / f"{self.prefix}_centerlines.npz"
        np.savez(str(npz_path), **npz_data)
        print(f"[seed_editor] Centerlines saved to {npz_path}")
        
        # ── Write signal file ─────────────────────────────────────────────────
        signal_path = self.output_dir / f"{self.prefix}_seeds.done"
        signal_path.write_text("done")
        print(f"[seed_editor] Signal written: {signal_path}")
        
        plt.close(self.fig)
    
    def run(self) -> None:
        """Run the interactive editor (blocks until window is closed)."""
        print("\n=== PCAT Seed Editor ===")
        print("Left click → select/drag seed | Enter → add seed at cursor")
        print("← → arrows → cycle selection | Backspace → delete selected")
        print("Backspace → delete selected | Right-click → delete nearest")
        print("Scroll → slab | Shift+scroll → thickness")
        print("1/2/3 → vessel | u → undo | r → reset | s → save | q → quit\n")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive seed editor with real-time CPR visualization. "
            "Place/edit seed points for coronary artery centerline generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls:\n"
            "  Left-click on seed      Select it (yellow ring)\n"
            "  Left-click selected     Drag to reposition\n"
            "  Left-click empty        Deselect\n"
            "  Right-click             Delete nearest waypoint\n"
            "  ← → arrows              Cycle through seeds\n"
            "  Enter                   Add new waypoint at cursor position\n"
            "  Backspace               Delete selected seed\n"
            "  Scroll                  Adjust MIP slab center\n"
            "  Shift+scroll            Adjust slab thickness (5-50mm)\n"
            "  1/2/3                   Switch vessel (LAD/LCX/RCA)\n"
            "  u                       Undo last action\n"
            "  r                       Reset current vessel (clear all seeds)\n"
            "  s                       Save and close\n"
            "  q                       Quit without saving\n"
        ),
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory for one patient"
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Input seed JSON file (from auto_seeds)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for seeds JSON, centerlines NPZ, and .done signal"
    )
    parser.add_argument(
        "--prefix", default="patient",
        help="Filename prefix for output files (default: patient)"
    )
    
    args = parser.parse_args()
    
    # Load DICOM volume
    print(f"[seed_editor] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[seed_editor] Volume shape: {volume.shape}, spacing: {spacing_mm}")
    print(f"[seed_editor] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Load seeds
    print(f"[seed_editor] Loading seeds from {args.seeds} ...")
    with open(args.seeds, "r") as f:
        seeds_data = json.load(f)
    print(f"[seed_editor] Loaded seeds for {list(seeds_data.keys())}")
    
    # Create and run editor
    editor = SeedEditor(
        volume=volume,
        spacing_mm=spacing_mm,
        seeds_data=seeds_data,
        output_dir=Path(args.output),
        prefix=args.prefix,
    )
    editor.run()


if __name__ == "__main__":
    main()
