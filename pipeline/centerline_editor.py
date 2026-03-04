#!/usr/bin/env python3
"""
centerline_editor.py — Interactive centerline editor with real-time CPR visualization.

A critical clinical tool for reviewing/correcting vessel centerlines BEFORE contour
extraction in the PCAT pipeline.

Features:
  - Two linked MIP views (coronal + axial) with draggable centerline control points
  - CPR panel that updates in real-time during point dragging
  - Per-vessel editing (LAD/LCX/RCA)
  - Saves modified centerlines to .npz format

Layout (figsize=(20, 10)):
+------------------+------------------+-----------------+
|                  |                  |                 |
|  Coronal MIP     |  Axial MIP       |    CPR          |
|  (drag Z-X)      |  (drag Y-X)      |  (live update)  |
|                  |                  |                 |
+------------------+------------------+-----------------+
| [vessel buttons]                    | [status bar]    |
+-------------------------------------+-----------------+

Interaction:
  - Left click + drag on control point: move it. CPR updates in real-time.
  - Double-click near centerline path: insert new control point
  - Right-click on control point: delete it (min 3 control points)
  - Scroll wheel on MIP views: adjust MIP slab center depth
  - 1/2/3 keys: switch vessel (LAD/LCX/RCA)
  - R key: reset current vessel to original centerline
  - U key: undo last edit
  - S key: save all corrections and close
  - Q key: quit without saving

Usage:
    python pipeline/centerline_editor.py \
        --dicom Rahaf_Patients/1200.2 \
        --seeds seeds/1200_2_reviewed.json \
        --output output/patient_1200/raw \
        --prefix patient1200
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
from scipy.ndimage import map_coordinates

# Interactive backend — MUST be set before pyplot is imported
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Disable matplotlib's default 's' = save-figure keybinding
plt.rcParams['keymap.save'] = []

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from pipeline modules
try:
    from centerline import (
        compute_vesselness,
        extract_centerline_seeds,
        clip_centerline_by_arclength,
        estimate_vessel_radii,
        load_seeds,
        VESSEL_CONFIGS,
    )
    from visualize import (
        _compute_cpr_data,
        _bezier_fit_centerline,
        _sample_bezier_frame,
        _build_cpr_image,
    )
    from dicom_loader import load_dicom_series
except ImportError:
    from pipeline.centerline import (
        compute_vesselness,
        extract_centerline_seeds,
        clip_centerline_by_arclength,
        estimate_vessel_radii,
        load_seeds,
        VESSEL_CONFIGS,
    )
    from pipeline.visualize import (
        _compute_cpr_data,
        _bezier_fit_centerline,
        _sample_bezier_frame,
        _build_cpr_image,
    )
    from pipeline.dicom_loader import load_dicom_series


# ─────────────────────────────────────────────────────────────────────────────
# Color Theme (Dark clinical style)
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_FACECOLOR = "#1a1a2e"
AXES_FACECOLOR = "#0d0d1a"
TICK_COLORS = "#aaaacc"
SPINE_COLORS = "#3a3a5a"

VESSEL_COLORS = {
    "LAD": "#E8533A",
    "LCX": "#4A90D9",
    "RCA": "#2ECC71",
}

VESSEL_KEYS = ["LAD", "LCX", "RCA"]

# Window/Level for CT display
WW_MIP = 1500  # window width for MIP
WL_MIP = 300   # window level for MIP
WW_CPR = 600   # window width for CPR
WL_CPR = 100   # window level for CPR


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _subsample_control_points(centerline_ijk: np.ndarray, n_target: int = 10) -> np.ndarray:
    """
    Subsample a dense centerline to n_target control points uniformly spaced along arc-length.
    
    Parameters
    ----------
    centerline_ijk : (N, 3) array [z, y, x]
    n_target : target number of control points
    
    Returns
    -------
    control_points : (M, 3) array where M = min(n_target, N)
    """
    n_pts = len(centerline_ijk)
    if n_pts <= n_target:
        return centerline_ijk.copy()
    
    # Uniform arc-length sampling
    indices = np.linspace(0, n_pts - 1, n_target, dtype=int)
    return centerline_ijk[indices].copy()


def _reconstruct_dense_centerline(
    control_points_ijk: np.ndarray,
    spacing_mm: List[float],
    volume_shape: Tuple[int, int, int],
    step_mm: float = 0.5,
) -> np.ndarray:
    """
    Fit cubic spline through control points, sample densely.
    
    Parameters
    ----------
    control_points_ijk : (N, 3) array [z, y, x]
    spacing_mm : [sz, sy, sx]
    volume_shape : (Z, Y, X)
    step_mm : arc-length step for dense sampling
    
    Returns
    -------
    dense_ijk : (M, 3) int array
    """
    if len(control_points_ijk) < 2:
        return control_points_ijk.copy()
    
    pts_mm = control_points_ijk.astype(np.float64) * np.array(spacing_mm)
    
    # Compute arc-length and remove duplicate points (zero-length segments)
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-8])
    pts_mm = pts_mm[keep]
    seg = seg[keep[1:]]
    if len(pts_mm) < 3:
        return control_points_ijk.copy()
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total < 1e-6:
        return control_points_ijk.copy()
    
    # Fit cubic spline
    cs = CubicSpline(arc, pts_mm, bc_type='not-a-knot')
    
    # Sample densely
    n_out = max(10, int(total / step_mm))
    s_vals = np.linspace(0, total, n_out)
    dense_mm = cs(s_vals)
    
    # Convert back to voxel indices and clip to volume
    dense_ijk = np.round(dense_mm / np.array(spacing_mm)).astype(int)
    dense_ijk = np.clip(dense_ijk, 0, np.array(volume_shape) - 1)
    
    return dense_ijk


def _compute_cpr_preview(
    volume: np.ndarray,
    control_points_ijk: np.ndarray,
    spacing_mm: List[float],
    pixels: int = 192,
) -> np.ndarray:
    """
    Lightweight CPR preview during drag (low-res, no slab).
    
    Parameters
    ----------
    volume : (Z, Y, X) float32 HU
    control_points_ijk : (N, 3) array [z, y, x]
    spacing_mm : [sz, sy, sx]
    pixels : output image size
    
    Returns
    -------
    cpr_img : (pixels, pixels) float32
    """
    if len(control_points_ijk) < 3:
        return np.zeros((pixels, pixels), dtype=np.float32)
    
    vox_size = np.array(spacing_mm, dtype=np.float64)
    cl_mm = control_points_ijk.astype(np.float64) * vox_size
    
    try:
        cs, total_len = _bezier_fit_centerline(cl_mm)
    except ValueError:
        return np.zeros((pixels, pixels), dtype=np.float32)
    
    s, positions, tangents, normals, binormals = _sample_bezier_frame(
        cs, total_len, pixels
    )
    
    cpr_img = _build_cpr_image(
        volume, vox_size, positions, normals, binormals,
        n_rows=pixels, row_extent_mm=15.0, slab_mm=0.0
    )
    
    return cpr_img  # (pixels, pixels) — rows=lateral, cols=arc-length


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
    axis : 0=Z (axial), 1=Y (coronal), 2=X (sagittal)
    
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
    elif axis == 1:
        slab = volume[:, lo:hi, :]
        mip = np.max(slab, axis=1)  # (Z, X)
    else:  # axis == 2
        slab = volume[:, :, lo:hi]
        mip = np.max(slab, axis=2)  # (Z, Y)
    
    return mip


# ─────────────────────────────────────────────────────────────────────────────
# Main Editor Class
# ─────────────────────────────────────────────────────────────────────────────

class CenterlineEditor:
    """
    Interactive centerline editor with real-time CPR visualization.
    
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
        vessel_filter: Optional[str] = None,
    ):
        self.volume = volume
        self.spacing_mm = spacing_mm
        self.vox_size = np.array(spacing_mm, dtype=np.float64)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.volume_shape = volume.shape
        
        # Per-vessel data
        self.vessel_data: Dict[str, Dict] = {}
        self._extract_centerlines_from_seeds(seeds_data, vessel_filter)
        
        if not self.vessel_data:
            raise ValueError("No valid vessel data could be extracted from seeds")
        
        # Current state
        self.vessel_names = list(self.vessel_data.keys())
        self.current_vessel = self.vessel_names[0]
        
        # MIP slab depths (center indices for the collapsed dimension)
        self.mip_y_center: Dict[str, int] = {}  # For coronal MIP (Y-axis collapsed)
        self.mip_z_center: Dict[str, int] = {}  # For axial MIP (Z-axis collapsed)
        
        for v in self.vessel_names:
            cl = self.vessel_data[v]["control_points"]
            if len(cl) > 0:
                self.mip_y_center[v] = int(np.mean(cl[:, 1]))
                self.mip_z_center[v] = int(np.mean(cl[:, 0]))
            else:
                self.mip_y_center[v] = self.volume_shape[1] // 2
                self.mip_z_center[v] = self.volume_shape[0] // 2
        
        # Interaction state
        self._dragging = False
        self._drag_point_idx: Optional[int] = None
        self._drag_view: Optional[str] = None  # 'coronal' or 'axial'
        self._last_click_time: float = 0.0
        self._last_click_pos: Optional[Tuple[int, int]] = None
        
        # History for undo
        self.history: List[Dict[str, np.ndarray]] = []
        self._save_history()
        
        # Store original centerlines for reset
        self.original_centerlines: Dict[str, np.ndarray] = {}
        for v in self.vessel_names:
            self.original_centerlines[v] = self.vessel_data[v]["centerline"].copy()
        
        # Track modifications
        self.modified: Dict[str, bool] = {v: False for v in self.vessel_names}
        
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
        self._coronal_path_line = None
        self._axial_path_line = None
        self._coronal_points = None
        self._axial_points = None
        self._highlight_point = None
        
        # Build GUI
        self._build_gui()
        self._update_display()
    
    def _extract_centerlines_from_seeds(
        self,
        seeds_data: Dict[str, Any],
        vessel_filter: Optional[str] = None,
    ) -> None:
        """Extract centerlines from seed points for each vessel."""
        all_seed_points = []
        
        # First pass: collect all seed points for vesselness ROI
        for v in VESSEL_KEYS:
            if v not in seeds_data:
                continue
            if vessel_filter and v != vessel_filter:
                continue
            
            vd = seeds_data[v]
            ostium = vd.get("ostium_ijk")
            waypoints = vd.get("waypoints_ijk", [])
            
            if ostium:
                all_seed_points.append(ostium)
            for wp in waypoints:
                all_seed_points.append(wp)
        
        if not all_seed_points:
            print("[centerline_editor] No seed points found in seeds file")
            return
        
        # Compute vesselness on ROI around seeds
        print("[centerline_editor] Computing vesselness filter...")
        vesselness = compute_vesselness(
            self.volume,
            self.spacing_mm,
            sigmas=[0.5, 1.0, 2.0],
            seed_points=all_seed_points,
            roi_margin_mm=25.0,
        )
        
        # Extract centerline for each vessel
        for v in VESSEL_KEYS:
            if v not in seeds_data:
                continue
            if vessel_filter and v != vessel_filter:
                continue
            
            vd = seeds_data[v]
            ostium = vd.get("ostium_ijk")
            waypoints = vd.get("waypoints_ijk", [])
            
            if not ostium:
                print(f"[centerline_editor] No ostium for {v}, skipping")
                continue
            
            # Extract full centerline
            try:
                centerline = extract_centerline_seeds(
                    self.volume,
                    vesselness,
                    self.spacing_mm,
                    ostium,
                    waypoints,
                    roi_radius_mm=35.0,
                )
            except Exception as e:
                print(f"[centerline_editor] Failed to extract centerline for {v}: {e}")
                continue
            
            if len(centerline) < 3:
                print(f"[centerline_editor] Too few centerline points for {v}, skipping")
                continue
            
            # Clip by arc-length
            config = VESSEL_CONFIGS.get(v, {"start_mm": 0.0, "length_mm": 50.0})
            start_mm = vd.get("segment_start_mm", config.get("start_mm", 0.0))
            length_mm = vd.get("segment_length_mm", config.get("length_mm", 50.0))
            
            clipped = clip_centerline_by_arclength(
                centerline, self.spacing_mm, start_mm, length_mm
            )
            
            if len(clipped) < 3:
                print(f"[centerline_editor] Clipped centerline too short for {v}, using full")
                clipped = centerline
            
            # Subsample control points
            control_points = _subsample_control_points(clipped, n_target=10)
            
            # Reconstruct dense path from control points
            dense_path = _reconstruct_dense_centerline(
                control_points, self.spacing_mm, self.volume_shape
            )
            
            self.vessel_data[v] = {
                "centerline": dense_path,
                "control_points": control_points,
                "radii": estimate_vessel_radii(
                    self.volume, dense_path, self.spacing_mm
                ),
            }
            
            print(f"[centerline_editor] {v}: {len(dense_path)} dense pts, {len(control_points)} control pts")
    
    def _build_gui(self) -> None:
        """Build the matplotlib figure with dark theme."""
        self.fig = plt.figure(figsize=(20, 10), facecolor=FIGURE_FACECOLOR)
        self.fig.canvas.manager.set_window_title(
            f"PCAT Centerline Editor — {self.prefix}"
        )
        
        # Title with controls hint
        self.fig.suptitle(
            "Drag control points to adjust centerline. CPR updates in real-time.\n"
            "1/2/3 Vessel | R Reset | U Undo | S Save | Q Quit | Scroll Adjust slab depth",
            fontsize=10,
            color="#cccccc",
        )
        
        # Layout: 3 panels + status bar
        gs = self.fig.add_gridspec(
            2, 3,
            height_ratios=[5, 0.4],
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
        
        # Style axes
        for ax in [self.ax_coronal, self.ax_axial, self.ax_cpr]:
            ax.set_facecolor(AXES_FACECOLOR)
            ax.tick_params(colors=TICK_COLORS, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(SPINE_COLORS)
        
        self.ax_status.set_facecolor(FIGURE_FACECOLOR)
        self.ax_status.axis("off")
        
        self.ax_coronal.set_title("Coronal MIP (drag Z-X)", fontsize=10, color="#cccccc")
        self.ax_axial.set_title("Axial MIP (drag Y-X)", fontsize=10, color="#cccccc")
        self.ax_cpr.set_title("CPR (live update)", fontsize=10, color="#cccccc")
        
        # Event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
    
    def _save_history(self) -> None:
        """Save current control points state for undo."""
        state = {}
        for v in self.vessel_names:
            state[v] = self.vessel_data[v]["control_points"].copy()
        self.history.append(state)
        if len(self.history) > 50:
            self.history.pop(0)
    
    def _undo(self) -> None:
        """Undo last edit."""
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            prev_state = self.history[-1]
            for v in self.vessel_names:
                self.vessel_data[v]["control_points"] = prev_state[v].copy()
                # Reconstruct dense path
                self.vessel_data[v]["centerline"] = _reconstruct_dense_centerline(
                    prev_state[v], self.spacing_mm, self.volume_shape
                )
            print("[centerline_editor] Undo")
            self._update_display()
        else:
            print("[centerline_editor] Nothing to undo")
    
    def _reset_vessel(self) -> None:
        """Reset current vessel to original centerline."""
        v = self.current_vessel
        original = self.original_centerlines[v]
        control_points = _subsample_control_points(original, n_target=10)
        self.vessel_data[v]["control_points"] = control_points
        self.vessel_data[v]["centerline"] = _reconstruct_dense_centerline(
            control_points, self.spacing_mm, self.volume_shape
        )
        self.modified[v] = False
        self._save_history()
        print(f"[centerline_editor] Reset {v} to original")
        self._update_display()
    
    def _switch_vessel(self, vessel_name: str) -> None:
        """Switch to a different vessel."""
        if vessel_name in self.vessel_names:
            self.current_vessel = vessel_name
            print(f"[centerline_editor] Switched to {vessel_name}")
            self._update_display()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Display Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _update_display(self) -> None:
        """Update all display panels."""
        self._draw_coronal_mip()
        self._draw_axial_mip()
        self._draw_cpr_full()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _draw_coronal_mip(self) -> None:
        """Draw coronal MIP with centerline overlay."""
        self.ax_coronal.cla()
        self.ax_coronal.set_facecolor(AXES_FACECOLOR)
        
        v = self.current_vessel
        data = self.vessel_data[v]
        control_points = data["control_points"]
        dense_path = data["centerline"]
        color = VESSEL_COLORS[v]
        
        # Compute MIP (project Y axis)
        y_center = self.mip_y_center[v]
        slab_vox = 15  # ~10mm slab
        mip = _compute_mip_slab(self.volume, y_center, slab_vox, axis=1)  # (Z, X)
        
        # Normalize for display
        lo = WL_MIP - WW_MIP / 2
        hi = WL_MIP + WW_MIP / 2
        mip_norm = np.clip(mip, lo, hi)
        mip_norm = (mip_norm - lo) / (hi - lo)
        
        # Display with origin='upper' so Z=0 (superior) at top
        self._coronal_im = self.ax_coronal.imshow(
            mip_norm,
            aspect='auto',
            origin='upper',
            cmap='gray',
            vmin=0, vmax=1,
        )
        
        # Draw dense path as thin line
        if len(dense_path) > 1:
            # Coronal view: Z vs X
            self.ax_coronal.plot(
                dense_path[:, 2], dense_path[:, 0],  # x, z
                color=color, linewidth=1.0, alpha=0.6,
            )
        
        # Draw control points as circles
        if len(control_points) > 0:
            self._coronal_points = self.ax_coronal.scatter(
                control_points[:, 2], control_points[:, 0],  # x, z
                c=color, s=80, edgecolors='white', linewidths=1.5,
                alpha=0.9, zorder=5,
            )
        
        self.ax_coronal.set_title(
            f"Coronal MIP (Y={y_center})",
            fontsize=10, color="#cccccc"
        )
        self.ax_coronal.tick_params(colors=TICK_COLORS, labelsize=8)
        for spine in self.ax_coronal.spines.values():
            spine.set_color(SPINE_COLORS)
    
    def _draw_axial_mip(self) -> None:
        """Draw axial MIP with centerline overlay."""
        self.ax_axial.cla()
        self.ax_axial.set_facecolor(AXES_FACECOLOR)
        
        v = self.current_vessel
        data = self.vessel_data[v]
        control_points = data["control_points"]
        dense_path = data["centerline"]
        color = VESSEL_COLORS[v]
        
        # Compute MIP (project Z axis)
        z_center = self.mip_z_center[v]
        slab_vox = 15
        mip = _compute_mip_slab(self.volume, z_center, slab_vox, axis=0)  # (Y, X)
        
        # Normalize
        lo = WL_MIP - WW_MIP / 2
        hi = WL_MIP + WW_MIP / 2
        mip_norm = np.clip(mip, lo, hi)
        mip_norm = (mip_norm - lo) / (hi - lo)
        
        self._axial_im = self.ax_axial.imshow(
            mip_norm,
            aspect='auto',
            origin='upper',
            cmap='gray',
            vmin=0, vmax=1,
        )
        
        # Draw dense path
        if len(dense_path) > 1:
            # Axial view: Y vs X
            self.ax_axial.plot(
                dense_path[:, 2], dense_path[:, 1],  # x, y
                color=color, linewidth=1.0, alpha=0.6,
            )
        
        # Draw control points
        if len(control_points) > 0:
            self._axial_points = self.ax_axial.scatter(
                control_points[:, 2], control_points[:, 1],  # x, y
                c=color, s=80, edgecolors='white', linewidths=1.5,
                alpha=0.9, zorder=5,
            )
        
        self.ax_axial.set_title(
            f"Axial MIP (Z={z_center})",
            fontsize=10, color="#cccccc"
        )
        self.ax_axial.tick_params(colors=TICK_COLORS, labelsize=8)
        for spine in self.ax_axial.spines.values():
            spine.set_color(SPINE_COLORS)
    
    def _draw_cpr_preview(self) -> None:
        """Draw low-res CPR preview during drag."""
        self.ax_cpr.cla()
        self.ax_cpr.set_facecolor(AXES_FACECOLOR)
        
        v = self.current_vessel
        control_points = self.vessel_data[v]["control_points"]
        
        # Compute preview CPR
        t0 = time.perf_counter()
        cpr_img = _compute_cpr_preview(
            self.volume, control_points, self.spacing_mm, pixels=192
        )
        dt = time.perf_counter() - t0
        print(f"[centerline_editor] CPR preview: {dt*1000:.0f}ms")
        
        # Normalize
        lo = WL_CPR - WW_CPR / 2
        hi = WL_CPR + WW_CPR / 2
        cpr_norm = np.clip(cpr_img, lo, hi)
        cpr_norm = (cpr_norm - lo) / (hi - lo)
        cpr_norm = np.nan_to_num(cpr_norm, nan=0.5)
        
        self._cpr_im = self.ax_cpr.imshow(
            cpr_norm,
            aspect='auto',
            origin='upper',
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='bilinear',
        )
        
        self.ax_cpr.set_title("CPR (preview)", fontsize=10, color="#cccccc")
        self.ax_cpr.tick_params(colors=TICK_COLORS, labelsize=8)
        for spine in self.ax_cpr.spines.values():
            spine.set_color(SPINE_COLORS)
    
    def _draw_cpr_full(self) -> None:
        """Draw full-res CPR after drag release."""
        self.ax_cpr.cla()
        self.ax_cpr.set_facecolor(AXES_FACECOLOR)
        
        v = self.current_vessel
        data = self.vessel_data[v]
        centerline = data["centerline"]
        
        if len(centerline) < 3:
            self.ax_cpr.text(0.5, 0.5, "Insufficient\ncenterline points",
                           ha='center', va='center', transform=self.ax_cpr.transAxes,
                           color="#888888", fontsize=12)
            return
        
        # Compute full CPR
        t0 = time.perf_counter()
        try:
            cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_h, n_w = _compute_cpr_data(
                self.volume, centerline, self.spacing_mm,
                slab_thickness_mm=3.0, width_mm=15.0,
                pixels_wide=512, pixels_high=512,
            )
            cpr_img = cpr_volume.T  # (n_h, n_w)
        except Exception as e:
            print(f"[centerline_editor] CPR computation failed: {e}")
            self.ax_cpr.text(0.5, 0.5, f"CPR error:\n{e}",
                           ha='center', va='center', transform=self.ax_cpr.transAxes,
                           color="#ff6666", fontsize=10)
            return
        
        dt = time.perf_counter() - t0
        print(f"[centerline_editor] CPR full: {dt*1000:.0f}ms")
        
        # Normalize
        lo = WL_CPR - WW_CPR / 2
        hi = WL_CPR + WW_CPR / 2
        cpr_norm = np.clip(cpr_img, lo, hi)
        cpr_norm = (cpr_norm - lo) / (hi - lo)
        cpr_norm = np.nan_to_num(cpr_norm, nan=0.5)
        
        self._cpr_im = self.ax_cpr.imshow(
            cpr_norm,
            aspect='auto',
            origin='upper',
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='bilinear',
        )
        
        # Add arc-length ticks
        total_mm = float(arclengths[-1]) if len(arclengths) > 0 else 0.0
        if total_mm > 0:
            x_ticks_mm = np.arange(0, total_mm + 1, 10.0)
            x_tick_idxs = ((x_ticks_mm / total_mm) * (n_w - 1)).astype(int)
            self.ax_cpr.set_xticks(np.clip(x_tick_idxs, 0, n_w - 1))
            self.ax_cpr.set_xticklabels([f"{t:.0f}" for t in x_ticks_mm], fontsize=7)
            self.ax_cpr.set_xlabel("Arc-length (mm)", fontsize=8, color=TICK_COLORS)
        
        self.ax_cpr.set_title(f"CPR — {v}", fontsize=10, color="#cccccc")
        self.ax_cpr.tick_params(colors=TICK_COLORS, labelsize=7)
        for spine in self.ax_cpr.spines.values():
            spine.set_color(SPINE_COLORS)
    
    def _update_status_bar(self) -> None:
        """Update status bar with current state."""
        self.ax_status.cla()
        self.ax_status.set_facecolor(FIGURE_FACECOLOR)
        self.ax_status.axis("off")
        
        v = self.current_vessel
        data = self.vessel_data[v]
        n_control = len(data["control_points"])
        n_dense = len(data["centerline"])
        modified_str = " [MODIFIED]" if self.modified[v] else ""
        color = VESSEL_COLORS[v]
        
        # Vessel buttons
        button_x = 0.02
        for i, vn in enumerate(self.vessel_names):
            vc = VESSEL_COLORS[vn]
            is_current = vn == v
            weight = "bold" if is_current else "normal"
            alpha = 1.0 if is_current else 0.5
            self.ax_status.text(
                button_x + i * 0.08, 0.5,
                f"[{i+1}] {vn}",
                ha='left', va='center',
                transform=self.ax_status.transAxes,
                fontsize=9, fontweight=weight, color=vc, alpha=alpha,
            )
        
        # Status text
        msg = (
            f"  |  Vessel: {v}  |  "
            f"Control points: {n_control}  |  "
            f"Dense points: {n_dense}{modified_str}"
        )
        self.ax_status.text(
            0.30, 0.5, msg,
            ha='left', va='center',
            transform=self.ax_status.transAxes,
            fontsize=9, color=color,
        )
    
    def _highlight_drag_point(self, view: str, idx: int) -> None:
        """Highlight the currently dragged point."""
        if self._highlight_point is not None:
            self._highlight_point.remove()
            self._highlight_point = None
        
        control_points = self.vessel_data[self.current_vessel]["control_points"]
        if idx < 0 or idx >= len(control_points):
            return
        
        pt = control_points[idx]
        ax = self.ax_coronal if view == 'coronal' else self.ax_axial
        
        if view == 'coronal':
            x, y = pt[2], pt[0]  # X, Z
        else:
            x, y = pt[2], pt[1]  # X, Y
        
        self._highlight_point = ax.scatter(
            [x], [y],
            c='white', s=150, edgecolors='yellow', linewidths=2.5,
            alpha=1.0, zorder=10, marker='o',
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _find_nearest_control_point(
        self,
        ax,
        x: float,
        y: float,
        threshold_px: float = 15.0,
    ) -> Optional[int]:
        """
        Find the nearest control point to screen coordinates.
        
        Returns the index of the nearest point within threshold, or None.
        """
        control_points = self.vessel_data[self.current_vessel]["control_points"]
        if len(control_points) == 0:
            return None
        
        # Convert data coordinates to display coordinates for distance check
        min_dist = float('inf')
        nearest_idx = None
        
        for i, pt in enumerate(control_points):
            if ax == self.ax_coronal:
                px, py = pt[2], pt[0]  # X, Z
            else:  # axial
                px, py = pt[2], pt[1]  # X, Y
            
            # Transform to display coordinates
            disp = ax.transData.transform([(px, py)])[0]
            click_disp = ax.transData.transform([(x, y)])[0]
            
            dist = np.linalg.norm(disp - click_disp)
            if dist < min_dist and dist < threshold_px:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for dragging or double-click insert."""
        if event.inaxes not in [self.ax_coronal, self.ax_axial]:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        view = 'coronal' if ax == self.ax_coronal else 'axial'
        
        # Check for double-click (insert new point)
        current_time = time.time()
        current_pos = (int(round(x)), int(round(y)))
        
        if (self._last_click_pos == current_pos and
            current_time - self._last_click_time < 0.35):
            # Double-click: try to insert point
            self._insert_control_point_near(view, x, y)
            self._last_click_time = 0.0
            self._last_click_pos = None
            return
        
        self._last_click_time = current_time
        self._last_click_pos = current_pos
        
        # Left click: check if near a control point
        if event.button == 1:
            nearest_idx = self._find_nearest_control_point(ax, x, y)
            if nearest_idx is not None:
                self._dragging = True
                self._drag_point_idx = nearest_idx
                self._drag_view = view
                self._highlight_drag_point(view, nearest_idx)
                self.fig.canvas.draw_idle()
        
        # Right click: delete control point
        elif event.button == 3:
            nearest_idx = self._find_nearest_control_point(ax, x, y)
            if nearest_idx is not None:
                self._delete_control_point(nearest_idx)
    
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dragging control points."""
        if not self._dragging or self._drag_point_idx is None:
            return
        if event.inaxes not in [self.ax_coronal, self.ax_axial]:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        ax = event.inaxes
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        
        v = self.current_vessel
        control_points = self.vessel_data[v]["control_points"]
        idx = self._drag_point_idx
        
        # Update control point position based on which view we're dragging in
        if ax == self.ax_coronal:
            # Coronal: update Z and X
            new_z = int(np.clip(y, 0, self.volume_shape[0] - 1))
            new_x = int(np.clip(x, 0, self.volume_shape[2] - 1))
            control_points[idx, 0] = new_z
            control_points[idx, 2] = new_x
        else:  # axial
            # Axial: update Y and X
            new_y = int(np.clip(y, 0, self.volume_shape[1] - 1))
            new_x = int(np.clip(x, 0, self.volume_shape[2] - 1))
            control_points[idx, 1] = new_y
            control_points[idx, 2] = new_x
        
        # Reconstruct dense path
        self.vessel_data[v]["centerline"] = _reconstruct_dense_centerline(
            control_points, self.spacing_mm, self.volume_shape
        )
        
        self.modified[v] = True
        
        # Update MIP overlays and CPR preview
        self._draw_coronal_mip()
        self._draw_axial_mip()
        self._highlight_drag_point(self._drag_view, idx)
        self._draw_cpr_preview()
        self._update_status_bar()
        self.fig.canvas.draw_idle()
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release: compute full-res CPR."""
        if self._dragging:
            self._dragging = False
            self._drag_point_idx = None
            
            if self._highlight_point is not None:
                self._highlight_point.remove()
                self._highlight_point = None
            
            self._save_history()
            
            # Compute full-res CPR
            self._draw_cpr_full()
            self.fig.canvas.draw_idle()
    
    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        key = event.key
        
        if key == 'q':
            print("[centerline_editor] Quit without saving")
            plt.close(self.fig)
        elif key == 's':
            self._save_and_close()
        elif key == 'u':
            self._undo()
        elif key == 'r':
            self._reset_vessel()
        elif key == '1':
            self._switch_vessel('LAD')
        elif key == '2':
            self._switch_vessel('LCX')
        elif key == '3':
            self._switch_vessel('RCA')
    
    def _on_scroll(self, event) -> None:
        """Handle scroll wheel to adjust MIP slab depth."""
        if event.inaxes not in [self.ax_coronal, self.ax_axial]:
            return
        
        delta = 2 if event.button == 'up' else -2
        v = self.current_vessel
        
        if event.inaxes == self.ax_coronal:
            # Scroll on coronal: adjust Y center
            new_y = self.mip_y_center[v] + delta
            self.mip_y_center[v] = int(np.clip(new_y, 0, self.volume_shape[1] - 1))
            self._draw_coronal_mip()
        else:  # axial
            # Scroll on axial: adjust Z center
            new_z = self.mip_z_center[v] + delta
            self.mip_z_center[v] = int(np.clip(new_z, 0, self.volume_shape[0] - 1))
            self._draw_axial_mip()
        
        self.fig.canvas.draw_idle()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Control Point Editing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _insert_control_point_near(self, view: str, x: float, y: float) -> None:
        """Insert a new control point near the click position on the path."""
        v = self.current_vessel
        dense_path = self.vessel_data[v]["centerline"]
        control_points = self.vessel_data[v]["control_points"]
        
        if len(dense_path) < 2:
            return
        
        # Find closest point on dense path
        min_dist = float('inf')
        best_idx = 0
        
        for i, pt in enumerate(dense_path):
            if view == 'coronal':
                dist = (pt[2] - x)**2 + (pt[0] - y)**2
            else:
                dist = (pt[2] - x)**2 + (pt[1] - y)**2
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # Find where to insert in control points (by arc-length)
        arc_at_click = best_idx / len(dense_path)
        
        # Compute arc-length ratios for existing control points
        arcs = [i / (len(control_points) - 1) if len(control_points) > 1 else 0
                for i in range(len(control_points))]
        
        # Find insertion position
        insert_pos = 0
        for i, arc in enumerate(arcs):
            if arc_at_click > arc:
                insert_pos = i + 1
        
        # Insert the point
        new_point = dense_path[best_idx].copy()
        new_control = np.insert(control_points, insert_pos, new_point, axis=0)
        
        self.vessel_data[v]["control_points"] = new_control
        self.vessel_data[v]["centerline"] = _reconstruct_dense_centerline(
            new_control, self.spacing_mm, self.volume_shape
        )
        self.modified[v] = True
        self._save_history()
        
        print(f"[centerline_editor] Inserted control point at index {insert_pos}")
        self._update_display()
    
    def _delete_control_point(self, idx: int) -> None:
        """Delete a control point (minimum 3 points required)."""
        v = self.current_vessel
        control_points = self.vessel_data[v]["control_points"]
        
        if len(control_points) <= 3:
            print("[centerline_editor] Cannot delete: minimum 3 control points required")
            return
        
        new_control = np.delete(control_points, idx, axis=0)
        self.vessel_data[v]["control_points"] = new_control
        self.vessel_data[v]["centerline"] = _reconstruct_dense_centerline(
            new_control, self.spacing_mm, self.volume_shape
        )
        self.modified[v] = True
        self._save_history()
        
        print(f"[centerline_editor] Deleted control point {idx}")
        self._update_display()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_and_close(self) -> None:
        """Save corrected centerlines and close."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build save dictionary
        save_data = {}
        
        for v in self.vessel_names:
            # Reconstruct dense centerline from control points
            control_points = self.vessel_data[v]["control_points"]
            dense_centerline = _reconstruct_dense_centerline(
                control_points, self.spacing_mm, self.volume_shape
            )
            save_data[f"{v}_centerline"] = dense_centerline
        
        # Save to .npz
        output_file = self.output_dir / f"{self.prefix}_centerlines.npz"
        np.savez(str(output_file), **save_data)
        print(f"[centerline_editor] Saved centerlines to {output_file}")
        
        # Write signal file
        signal_file = self.output_dir / f"{self.prefix}_centerline_editor.done"
        signal_file.write_text("done")
        print(f"[centerline_editor] Wrote signal file: {signal_file}")
        
        plt.close(self.fig)
    
    def run(self) -> None:
        """Run the interactive editor (blocks until window is closed)."""
        print("\n=== PCAT Centerline Editor ===")
        print("Keys:  1/2/3 = Switch vessel  |  R = Reset  |  U = Undo")
        print("       S = Save & close  |  Q = Quit without saving")
        print("Interaction:")
        print("  - Drag control points to adjust centerline")
        print("  - Double-click near path to insert point")
        print("  - Right-click on point to delete (min 3 points)")
        print("  - Scroll on MIP views to adjust slab depth\n")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive centerline editor with real-time CPR visualization. "
            "Review and correct vessel centerlines before contour extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls:\n"
            "  Drag control points to adjust centerline\n"
            "  Double-click near path to insert new control point\n"
            "  Right-click on point to delete (minimum 3 points)\n"
            "  Scroll wheel on MIP views to adjust slab depth\n"
            "  1/2/3         Switch vessel (LAD/LCX/RCA)\n"
            "  R             Reset current vessel to original\n"
            "  U             Undo last edit\n"
            "  S             Save all corrections and close\n"
            "  Q             Quit without saving\n"
        ),
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory for one patient"
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Input seed JSON file (from seed_reviewer or auto_seeds)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for corrected centerlines"
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Filename prefix for output files"
    )
    parser.add_argument(
        "--vessel", default=None,
        choices=["LAD", "LCX", "RCA"],
        help="Edit only this vessel (default: all vessels)"
    )
    
    args = parser.parse_args()
    
    # Load DICOM volume
    print(f"[centerline_editor] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[centerline_editor] Volume shape: {volume.shape}, spacing: {spacing_mm}")
    print(f"[centerline_editor] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Load seeds
    print(f"[centerline_editor] Loading seeds from {args.seeds} ...")
    seeds_data = load_seeds(args.seeds)
    print(f"[centerline_editor] Loaded seeds for {list(seeds_data.keys())}")
    
    # Create and run editor
    editor = CenterlineEditor(
        volume=volume,
        spacing_mm=spacing_mm,
        seeds_data=seeds_data,
        output_dir=Path(args.output),
        prefix=args.prefix,
        vessel_filter=args.vessel,
    )
    editor.run()


if __name__ == "__main__":
    main()
