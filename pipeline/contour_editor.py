#!/usr/bin/env python3
"""
contour_editor.py — Interactive vessel contour editor for correcting contours
before PCAT VOI computation.
Draw a freehand lasso around the region you want to fill or erase, then release.
Features:
  - Clean clinical interface (no dark theme, no neon colors)
  - CPR cross-section viewer with polar contour overlay
  - Scissors editing: left-click drag freehand lasso to fill/erase contour regions
  - Auto-snap (A key): re-detects vessel boundary using gradient analysis
  - Auto-smoothing after every manual edit using Gaussian wrap-around
  - Fill-between-slices interpolation for efficient editing
  - Navigation through centerline positions (slider/arrow keys)
  - Per-vessel switching (LAD/LCX/RCA)
  - 3D pyvista visualization with vessel meshes and semi-transparent fat
- Save corrected contours to .npz
  Left-drag   Draw freehand lasso to fill/erase contour region
  Right-drag  Same as left-drag (backward compat)
  A            Auto-snap boundary (re-detect using gradient analysis)
  E            Toggle fill/erase mode
  Left/Right   Navigate ±1 position
  Up/Down      Navigate ±5 positions
  1/2/3        Switch vessel (LAD/LCX/RCA)
  R            Reset current contour to auto-detected values
  I            Fill between slices (interpolate)
  S            Save all corrections and close
  Q            Quit without saving
  Space        Toggle contour visibility
  V            Toggle VOI ring visibility
  Scroll       Zoom cross-section view
  python contour_editor.py \\
      --dicom path/to/dicom \\
      --contour-data path/to/contour_data.npz \\
      --output path/to/output \\
      --prefix patient
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d

# Interactive backend — MUST be set before pyplot is imported
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Disable matplotlib's default 's' = save-figure keybinding
plt.rcParams['keymap.save'] = []
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

# Pyvista for 3D visualization
import pyvista as pv

# Import boundary detection functions from contour_extraction
try:
    from contour_extraction import _polar_transform_cross_section, _detect_adventitial_boundary
except ImportError:
    from pipeline.contour_extraction import _polar_transform_cross_section, _detect_adventitial_boundary

# ─────────────────────────────────────────────────────────────────────────────
# Color Theme (Clinical professional style)
# ─────────────────────────────────────────────────────────────────────────────

# Vessel colors (matching existing pipeline)
VESSEL_COLORS = {
    "LAD": "#E8533A",
    "LCX": "#4A90D9",
    "RCA": "#2ECC71",
}

# RGB colors for pyvista (normalized 0-1)
VESSEL_COLORS_RGB = {
    "LAD": (0.91, 0.33, 0.23),
    "LCX": (0.29, 0.56, 0.85),
    "RCA": (0.18, 0.80, 0.44),
}

# Window/Level for CT display
WW = 600  # window width
WL = 50   # window level (center)

# ─────────────────────────────────────────────────────────────────────────────
# Volume Sampling Functions
# ─────────────────────────────────────────────────────────────────────────────

def _sample_volume_linear(
    volume: np.ndarray,
    vox_size: np.ndarray,
    pts_mm: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation of CT volume at arbitrary mm positions.

    Parameters
    ----------
    volume : (Z, Y, X) float32
        CT HU volume
    vox_size : (3,) [sz, sy, sx]
        Voxel spacing in mm
    pts_mm : (..., 3) float64
        Sample points in DICOM mm [z, y, x]

    Returns
    -------
    vals : (...) float32
        HU values; NaN for out-of-bounds points
    """
    shape_in = pts_mm.shape[:-1]
    pts_flat = pts_mm.reshape(-1, 3)

    # Convert mm → voxel coordinates
    pts_vox = pts_flat / vox_size[np.newaxis, :]
    z_v, y_v, x_v = pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2]

    vol_shape = np.array(volume.shape, dtype=np.float64)
    valid = (
        (z_v >= 0) & (z_v <= vol_shape[0] - 1) &
        (y_v >= 0) & (y_v <= vol_shape[1] - 1) &
        (x_v >= 0) & (x_v <= vol_shape[2] - 1)
    )

    vals = map_coordinates(
        volume, [z_v, y_v, x_v], order=1, mode='nearest', cval=0.0,
    ).astype(np.float32)

    vals[~valid] = np.nan
    return vals.reshape(shape_in)


# ─────────────────────────────────────────────────────────────────────────────
# Main Editor Class
# ─────────────────────────────────────────────────────────────────────────────

class ContourEditor:
    """
    Professional clinical contour editor for vessel wall correction.

    Parameters
    ----------
    volume : (Z, Y, X) float32
        CT HU volume
    spacing_mm : [sz, sy, sx]
        Voxel spacing in mm
    contour_data : dict
        Dictionary containing vessel contour data from .npz file
    output_dir : Path
        Directory for saving results
    prefix : str
        Filename prefix for output files
    """

    def __init__(
        self,
        volume: np.ndarray,
        spacing_mm: List[float],
        contour_data: Dict[str, np.ndarray],
        output_dir: Path,
        prefix: str,
    ):
        self.volume = volume
        self.spacing_mm = spacing_mm
        self.vox_size = np.array(spacing_mm, dtype=np.float64)
        self.output_dir = output_dir
        self.prefix = prefix

        # Parse vessel data from contour_data dict
        self.vessel_data: Dict[str, Dict] = {}
        self._parse_vessel_data(contour_data)

        if not self.vessel_data:
            raise ValueError("No vessel data found in contour data file")

        # Current state
        self.vessel_names = list(self.vessel_data.keys())
        self.current_vessel = self.vessel_names[0]
        self.current_position = 0
        self.zoom_level = 1.0
        self.contour_visible = True
        self.contour_visible = True
        self.voi_visible = True

        # Control points: 8 cardinal directions (every 45°)
        self.n_control_points = 8
        self.control_angles = np.linspace(0, 2 * np.pi, self.n_control_points, endpoint=False)


        # Lasso/scissors tool state
        self.lasso_active = False
        self.lasso_points: List[Tuple[float, float]] = []
        self.fill_mode = True  # True = fill, False = erase
        self.lasso_line = None

        # Track which positions have been manually modified
        self.modified_mask: Dict[str, np.ndarray] = {}
        for vessel_name in self.vessel_names:
            n_pos = len(self.vessel_data[vessel_name]["positions_mm"])
            self.modified_mask[vessel_name] = np.zeros(n_pos, dtype=bool)

        # Store original r_theta for reset functionality
        self.original_r_theta: Dict[str, np.ndarray] = {}
        for vessel_name, data in self.vessel_data.items():
            self.original_r_theta[vessel_name] = data["r_theta"].copy()

        # Matplotlib handles
        self.fig = None
        self.ax_cross = None
        self.ax_long = None
        self.ax_status = None
        self.ax_slider = None

        # Image handles
        self._cross_im = None
        self._contour_line = None
        self._voi_ring = None
        self._control_points = []
        self._crosshairs = []

        # Slider
        self._slider = None

        # 3D Pyvista visualization
        self.plotter: Optional[pv.Plotter] = None
        self.vessel_meshes: Dict[str, pv.PolyData] = {}
        self.fat_meshes: Dict[str, pv.PolyData] = {}
        self.vessel_actors: Dict[str, Any] = {}
        self.fat_actors: Dict[str, Any] = {}

        # Build GUI
        self._build_gui()
        self._init_pyvista()
        self._update_display()

    def _parse_vessel_data(self, contour_data: Dict[str, np.ndarray]) -> None:
        """Parse vessel data from the loaded .npz dictionary."""
        # Group keys by vessel name
        vessels = set()
        for key in contour_data.keys():
            for vessel in ["LAD", "LCX", "RCA"]:
                if key.startswith(f"{vessel}_"):
                    vessels.add(vessel)
                    break

        for vessel in vessels:
            data = {
                "r_theta": contour_data.get(f"{vessel}_r_theta"),
                "positions_mm": contour_data.get(f"{vessel}_positions_mm"),
                "N_frame": contour_data.get(f"{vessel}_N_frame"),
                "B_frame": contour_data.get(f"{vessel}_B_frame"),
                "r_eq": contour_data.get(f"{vessel}_r_eq"),
                "arclengths": contour_data.get(f"{vessel}_arclengths"),
                "fallback_mask": contour_data.get(f"{vessel}_fallback_mask"),
                "centerline": contour_data.get(f"{vessel}_centerline"),
            }

            # Check required fields
            if data["r_theta"] is not None and data["positions_mm"] is not None:
                self.vessel_data[vessel] = data
                print(f"[contour_editor] Loaded {vessel}: {len(data['positions_mm'])} positions")

    def _build_gui(self) -> None:
        """Build the matplotlib figure with clinical professional UI."""
        # Create figure with default matplotlib background
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title(
            f"PCAT Contour Editor — {self.prefix}"
        )

        # Title with controls hint
        self.fig.suptitle(
            f"PCAT Contour Editor — {self.prefix}\n"
            f"Draw lasso to edit | E Fill/Erase | A Auto-snap | ←/→ Navigate | 1/2/3 Vessel | R Reset | I Interpolate | S Save | Q Quit",
            fontsize=10,
        )

        # Layout: Cross-section (left) | Vessel overview (right) | Status/slider (bottom)
        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=[5, 1],
            width_ratios=[1.2, 1],
            left=0.06, right=0.98,
            bottom=0.08, top=0.90,
            hspace=0.15, wspace=0.12,
        )

        # Main panels
        self.ax_cross = self.fig.add_subplot(gs[0, 0])
        self.ax_long = self.fig.add_subplot(gs[0, 1])
        self.ax_slider = self.fig.add_subplot(gs[1, 0])
        self.ax_status = self.fig.add_subplot(gs[1, 1])

        # Style axes
        self.ax_cross.set_title("Cross-Section View", fontsize=10)
        self.ax_cross.axis('off')
        
        self.ax_long.set_title("Vessel Overview", fontsize=10)
        
        self.ax_status.axis('off')

        # ── Setup Slider ────────────────────────────────────────────────────
        n_positions = len(self.vessel_data[self.current_vessel]["positions_mm"])

        self._slider = Slider(
            ax=self.ax_slider,
            label="Position",
            valmin=0,
            valmax=max(n_positions - 1, 0),
            valinit=0,
            valstep=1,
            color=VESSEL_COLORS.get(self.current_vessel, "#4A90D9"),
        )
        self._slider.label.set_fontsize(9)
        self._slider.valtext.set_fontsize(9)

        self._slider.on_changed(self._on_slider_change)

        # ── Event Handlers ──────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _init_pyvista(self) -> None:
        """Initialize the 3D pyvista visualization window."""
        try:
            self.plotter = pv.Plotter(
                title=f"3D Vessel View — {self.prefix}",
                window_size=[800, 600],
            )
            self.plotter.add_axes()
            self._build_vessel_meshes()
            # Open non-blocking window
            self.plotter.show(interactive_update=True)
        except Exception as e:
            print(f"[contour_editor] Warning: Could not initialize 3D view: {e}")
            self.plotter = None

    def _build_vessel_meshes(self) -> None:
        """Build vessel surface meshes for 3D visualization."""
        if self.plotter is None:
            return

        # Clear existing actors
        for actor in self.vessel_actors.values():
            self.plotter.remove_actor(actor)
        for actor in self.fat_actors.values():
            self.plotter.remove_actor(actor)
        self.vessel_actors.clear()
        self.fat_actors.clear()

        for vessel_name, data in self.vessel_data.items():
            try:
                # Build vessel surface mesh from contour rings
                mesh = self._create_vessel_surface_mesh(vessel_name, data)
                if mesh is not None:
                    self.vessel_meshes[vessel_name] = mesh
                    actor = self.plotter.add_mesh(
                        mesh,
                        color=VESSEL_COLORS_RGB.get(vessel_name, (0.5, 0.5, 0.5)),
                        opacity=0.9,
                        name=f"vessel_{vessel_name}",
                    )
                    self.vessel_actors[vessel_name] = actor

                # Build semi-transparent fat volume (3× r_eq)
                fat_mesh = self._create_fat_volume_mesh(vessel_name, data)
                if fat_mesh is not None:
                    self.fat_meshes[vessel_name] = fat_mesh
                    actor = self.plotter.add_mesh(
                        fat_mesh,
                        color=(1.0, 0.9, 0.5),  # Yellow
                        opacity=0.15,
                        name=f"fat_{vessel_name}",
                    )
                    self.fat_actors[vessel_name] = actor
            except Exception as e:
                print(f"[contour_editor] Warning: Could not build mesh for {vessel_name}: {e}")

        if self.plotter:
            self.plotter.reset_camera()

    def _create_vessel_surface_mesh(self, vessel_name: str, data: Dict) -> Optional[pv.PolyData]:
        """Create a surface mesh from contour rings using quad faces."""
        r_theta = data["r_theta"]  # (n_positions, n_angles)
        positions_mm = data["positions_mm"]  # (n_positions, 3)
        N_frame = data["N_frame"]  # (n_positions, 3)
        B_frame = data["B_frame"]  # (n_positions, 3)

        n_positions, n_angles = r_theta.shape
        if n_positions < 2 or n_angles < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Build all contour points
        all_points = []
        for i in range(n_positions):
            center = positions_mm[i]
            N = N_frame[i]
            B = B_frame[i]
            r = r_theta[i]

            # Contour points: center + r[j] * (cos(θ) * N + sin(θ) * B)
            for j in range(n_angles):
                pt = center + r[j] * (np.cos(angles[j]) * N + np.sin(angles[j]) * B)
                all_points.append(pt)

        points = np.array(all_points)  # (n_positions * n_angles, 3)

        # Build quad faces connecting adjacent rings
        faces = []
        for i in range(n_positions - 1):
            for j in range(n_angles):
                j_next = (j + 1) % n_angles
                # Quad: (i, j), (i, j+1), (i+1, j+1), (i+1, j)
                v0 = i * n_angles + j
                v1 = i * n_angles + j_next
                v2 = (i + 1) * n_angles + j_next
                v3 = (i + 1) * n_angles + j
                # Pyvista quad face format: [4, v0, v1, v2, v3]
                faces.extend([4, v0, v1, v2, v3])

        faces = np.array(faces)
        mesh = pv.PolyData(points, faces)
        return mesh

    def _create_fat_volume_mesh(self, vessel_name: str, data: Dict) -> Optional[pv.PolyData]:
        """Create a semi-transparent fat volume mesh (3× r_eq)."""
        r_eq = data["r_eq"]  # (n_positions,)
        positions_mm = data["positions_mm"]  # (n_positions, 3)
        N_frame = data["N_frame"]
        B_frame = data["B_frame"]

        n_positions = len(positions_mm)
        n_angles = 36  # Use fewer angles for the fat tube
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        if n_positions < 2:
            return None

        # Build fat tube points (3× r_eq radius)
        all_points = []
        for i in range(n_positions):
            center = positions_mm[i]
            N = N_frame[i]
            B = B_frame[i]
            r_fat = r_eq[i] * 3.0

            for j in range(n_angles):
                pt = center + r_fat * (np.cos(angles[j]) * N + np.sin(angles[j]) * B)
                all_points.append(pt)

        points = np.array(all_points)

        # Build quad faces
        faces = []
        for i in range(n_positions - 1):
            for j in range(n_angles):
                j_next = (j + 1) % n_angles
                v0 = i * n_angles + j
                v1 = i * n_angles + j_next
                v2 = (i + 1) * n_angles + j_next
                v3 = (i + 1) * n_angles + j
                faces.extend([4, v0, v1, v2, v3])

        faces = np.array(faces)
        mesh = pv.PolyData(points, faces)
        return mesh

    def _update_pyvista_meshes(self) -> None:
        """Schedule 3D mesh update on the Tk event loop (avoids GIL crash)."""
        if self.plotter is None:
            return
        try:
            tk_widget = self.fig.canvas.get_tk_widget()
            tk_widget.after_idle(self._do_update_pyvista)
        except Exception:
            # Fallback: try direct update if Tk widget unavailable
            self._do_update_pyvista()

    def _do_update_pyvista(self) -> None:
        """Actually rebuild and render pyvista meshes (called from Tk idle)."""
        if self.plotter is None:
            return
        try:
            self._build_vessel_meshes()
            self.plotter.update()
        except Exception as e:
            print(f"[contour_editor] Warning: Could not update 3D view: {e}")

    def _update_display(self) -> None:
        """Update all display panels."""
        self._draw_crosssection()
        self._draw_longitudinal()
        self._update_status_bar()
        self._update_slider()
        self.fig.canvas.draw_idle()

    def _draw_crosssection(self) -> None:
        """Draw the cross-section view with contour overlay."""
        vessel = self.current_vessel
        data = self.vessel_data[vessel]
        pos = self.current_position

        # Get current position data
        center_mm = data["positions_mm"][pos]
        N = data["N_frame"][pos]
        B = data["B_frame"][pos]
        r_theta = data["r_theta"][pos]
        r_eq = data["r_eq"][pos]

        # Determine cross-section extent based on zoom and max radius
        max_radius = max(float(np.max(r_theta)), r_eq * 2) * 1.5
        extent = max_radius / self.zoom_level

        # Sample cross-section grid
        n_pixels = 300
        offsets = np.linspace(-extent, extent, n_pixels)
        nn, bb = np.meshgrid(offsets, offsets)

        # Sample positions: center + n*N + b*B
        pts = (
            center_mm[np.newaxis, np.newaxis, :]
            + nn[:, :, np.newaxis] * N[np.newaxis, np.newaxis, :]
            + bb[:, :, np.newaxis] * B[np.newaxis, np.newaxis, :]
        )

        # Sample volume
        cs_img = _sample_volume_linear(self.volume, self.vox_size, pts)

        # Grayscale normalization with W/L
        lo = WL - WW / 2
        hi = WL + WW / 2
        gray = np.clip(cs_img, lo, hi)
        gray = (gray - lo) / (hi - lo)
        gray = np.nan_to_num(gray, nan=0.5)

        # Draw or update image
        if self._cross_im is None:
            self._cross_im = self.ax_cross.imshow(
                gray,
                aspect='equal',
                origin='upper',
                cmap='gray',
                vmin=0.0, vmax=1.0,
                interpolation='bilinear',
                extent=(-extent, extent, extent, -extent),
            )
        else:
            self._cross_im.set_data(gray)
            self._cross_im.set_extent((-extent, extent, extent, -extent))

        # Clear old contour elements
        if self._contour_line is not None:
            self._contour_line.remove()
            self._contour_line = None
        if self._voi_ring is not None:
            self._voi_ring.remove()
            self._voi_ring = None
        for cp in self._control_points:
            cp.remove()
        self._control_points.clear()
        for ch in self._crosshairs:
            ch.remove()
        self._crosshairs.clear()

        # Draw crosshairs
        kw = dict(color="yellow", linewidth=0.8, alpha=0.6, linestyle="--")
        self._crosshairs.append(self.ax_cross.axhline(0, **kw))
        self._crosshairs.append(self.ax_cross.axvline(0, **kw))

        if self.contour_visible:
            # Convert r_theta to Cartesian contour
            n_angles = len(r_theta)
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            x_contour = r_theta * np.cos(angles)
            y_contour = r_theta * np.sin(angles)

            # Close the contour
            x_closed = np.append(x_contour, x_contour[0])
            y_closed = np.append(y_contour, y_contour[0])

            # Get vessel color
            vessel_color = VESSEL_COLORS.get(vessel, "#4A90D9")

            # Draw main contour line
            self._contour_line, = self.ax_cross.plot(
                x_closed, y_closed,
                color=vessel_color,
                linewidth=2.0,
                alpha=0.9,
            )

            # Draw control points (8 cardinal directions)
            for i, angle in enumerate(self.control_angles):
                # Interpolate r at this angle
                idx_continuous = angle / (2 * np.pi) * n_angles
                idx_lo = int(idx_continuous) % n_angles
                idx_hi = (idx_lo + 1) % n_angles
                frac = idx_continuous - int(idx_continuous)
                r_interp = r_theta[idx_lo] * (1 - frac) + r_theta[idx_hi] * frac

                x_cp = r_interp * np.cos(angle)
                y_cp = r_interp * np.sin(angle)

                # Control point (white dot with vessel-colored edge)
                cp = self.ax_cross.plot(
                    x_cp, y_cp, 'o',
                    color="white",
                    markersize=7,
                    markeredgecolor=vessel_color,
                    markeredgewidth=2,
                    alpha=0.9,
                )[0]
                self._control_points.append(cp)

        # Draw PCAT VOI ring (3× r_eq)
        if self.voi_visible:
            r_voi = r_eq * 3.0
            self._voi_ring = mpatches.Circle(
                (0, 0), radius=r_voi,
                fill=False,
                edgecolor="gold",
                linewidth=1.5,
                linestyle='--',
                alpha=0.7,
            )
            self.ax_cross.add_patch(self._voi_ring)

        # Set axis limits
        self.ax_cross.set_xlim(-extent, extent)
        self.ax_cross.set_ylim(extent, -extent)

        # Update title
        self.ax_cross.set_title(
            f"Cross-Section — {vessel} — Pos {pos + 1}",
            fontsize=10,
        )

    def _draw_longitudinal(self) -> None:
        """Draw vessel overview panel with radius profile and stats."""
        self.ax_long.cla()
        vessel = self.current_vessel
        data = self.vessel_data[vessel]
        vessel_color = VESSEL_COLORS.get(vessel, '#4A90D9')

        arclengths = data['arclengths']
        r_eq = data['r_eq']
        n_positions = len(arclengths)
        # ── Radius profile (r_eq vs arc-length) ────────────────────────
        self.ax_long.plot(
            arclengths, r_eq,
            color=vessel_color, linewidth=1.8, label=f'{vessel} r_eq',
        )

        # Shade modified positions
        mask = self.modified_mask[vessel]
        if np.any(mask):
            mod_arcs = arclengths[mask]
            mod_r = r_eq[mask]
            self.ax_long.scatter(
                mod_arcs, mod_r,
                color=vessel_color, s=8, alpha=0.6, zorder=3,
                label='modified',
            )

        # Current position marker
        self.ax_long.axvline(
            arclengths[self.current_position],
            color='gold', linewidth=1.5, linestyle='--', alpha=0.8,
        )
        self.ax_long.plot(
            arclengths[self.current_position],
            r_eq[self.current_position],
            'o', color='gold', markersize=6, zorder=4,
        )

        # ── Stats text ─────────────────────────────────────────────────
        n_modified = int(np.sum(mask))
        stats = (
            f"Positions: {n_positions}\n"
            f"Arc length: {arclengths[-1]:.1f} mm\n"
            f"Mean r_eq: {np.mean(r_eq):.2f} mm\n"
            f"Modified: {n_modified}/{n_positions}"
        )
        self.ax_long.text(
            0.98, 0.98, stats,
            ha='right', va='top',
            transform=self.ax_long.transAxes,
            fontsize=8, family='monospace',
            color=vessel_color,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=vessel_color, alpha=0.8),
        )

        # ── Vessel legend (all vessels) ────────────────────────────────
        for vname in self.vessel_names:
            if vname != vessel:
                vd = self.vessel_data[vname]
                vc = VESSEL_COLORS.get(vname, '#888888')
                self.ax_long.plot(
                    vd['arclengths'], vd['r_eq'],
                    color=vc, linewidth=0.8, alpha=0.4, label=vname,
                )

        self.ax_long.set_xlabel('Arc-length (mm)', fontsize=8)
        self.ax_long.set_ylabel('r_eq (mm)', fontsize=8)
        self.ax_long.set_title(f'Vessel Overview — {vessel}', fontsize=10)
        self.ax_long.legend(fontsize=7, loc='lower right')
        self.ax_long.tick_params(labelsize=7)

    def _update_status_bar(self) -> None:
        """Update status bar with current state."""
        self.ax_status.cla()
        self.ax_status.axis("off")

        vessel = self.current_vessel
        data = self.vessel_data[vessel]
        pos = self.current_position
        n_total = len(data["positions_mm"])
        vessel_color = VESSEL_COLORS.get(vessel, "#4A90D9")

        arc_mm = data["arclengths"][pos]
        r_eq = data["r_eq"][pos]
        fallback = data["fallback_mask"][pos] if data["fallback_mask"] is not None else False

        mode_str = "FILL" if self.fill_mode else "ERASE"

        modified = self.modified_mask[vessel][pos]

        msg = (
            f"  Vessel: {vessel}  |  "
            f"Pos: {pos + 1}/{n_total}  |  "
            f"Arc: {arc_mm:.1f} mm  |  "
            f"R_eq: {r_eq:.2f} mm  |  "
            f"{'FALLBACK' if fallback else 'GRADIENT'}  |  "
            f"{mode_str}  |  "
            f"{'[MODIFIED]' if modified else ''}"
        )


        self.ax_status.text(
            0.01, 0.5, msg,
            ha="left", va="center",
            transform=self.ax_status.transAxes,
            fontsize=9,
            family="monospace",
            color=vessel_color,
        )

    def _update_slider(self) -> None:
        """Update slider to match current vessel."""
        data = self.vessel_data[self.current_vessel]
        n_positions = len(data["positions_mm"])

        # Update slider range
        self._slider.valmin = 0
        self._slider.valmax = max(n_positions - 1, 0)
        self._slider.ax.set_xlim(0, max(n_positions - 1, 0))

        # Set current value
        self._slider.set_val(self.current_position)

        # Update color based on vessel
        vessel_color = VESSEL_COLORS.get(self.current_vessel, "#4A90D9")
        self._slider.poly.set_facecolor(vessel_color)

    # ─────────────────────────────────────────────────────────────────────────
    # Scissors/Lasso Tool
    # ─────────────────────────────────────────────────────────────────────────

    def _ray_polygon_intersections(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        polygon_points: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Find all intersection distances along a ray with a polygon.

        Uses cross-product form: t = cross(P1, P2-P1) / cross(d, P2-P1)

        Parameters
        ----------
        ray_origin : (2,) array
            Origin point of the ray
        ray_dir : (2,) array
            Direction vector of the ray (unit vector)
        polygon_points : list of (x, y) tuples
            Polygon vertices

        Returns
        -------
        intersections : list of float
            Distances t along the ray where intersections occur (t > 0)
        """
        intersections = []
        n_pts = len(polygon_points)

        if n_pts < 3:
            return intersections

        d = ray_dir
        ox, oy = ray_origin

        for i in range(n_pts):
            p1 = np.array(polygon_points[i])
            p2 = np.array(polygon_points[(i + 1) % n_pts])

            # Edge vector
            edge = p2 - p1

            # cross(d, edge) = d_x * edge_y - d_y * edge_x
            denom = d[0] * edge[1] - d[1] * edge[0]

            if abs(denom) < 1e-10:
                # Ray is parallel to edge
                continue

            # P1 relative to ray origin
            p1_rel = p1 - ray_origin

            # t = cross(P1_rel, edge) / cross(d, edge)
            # cross(P1_rel, edge) = p1_rel_x * edge_y - p1_rel_y * edge_x
            cross_p1_edge = p1_rel[0] * edge[1] - p1_rel[1] * edge[0]
            t = cross_p1_edge / denom

            # u = cross(P1_rel, d) / cross(d, edge)
            cross_p1_d = p1_rel[0] * d[1] - p1_rel[1] * d[0]
            u = cross_p1_d / denom

            # Valid intersection: t > 0 and 0 <= u <= 1
            if t > 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
                intersections.append(t)

        return intersections

    def _apply_lasso_to_contour(self) -> None:
        """Apply lasso polygon to current contour using ray-polygon intersection."""
        if len(self.lasso_points) < 3:
            return

        vessel = self.current_vessel
        pos = self.current_position
        data = self.vessel_data[vessel]
        r_theta = data["r_theta"][pos]
        n_angles = len(r_theta)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        for i in range(n_angles):
            theta = angles[i]
            ray_dir = np.array([np.cos(theta), np.sin(theta)])
            ray_origin = np.array([0.0, 0.0])

            intersections = self._ray_polygon_intersections(
                ray_origin, ray_dir, self.lasso_points
            )

            if not intersections:
                continue

            current_r = r_theta[i]

            if self.fill_mode:
                # FILL mode: expand to max intersection
                max_r = max(intersections)
                r_theta[i] = max(current_r, max_r)
            else:
                # ERASE mode: shrink to innermost intersection if inside
                # Check if current contour point is inside polygon
                pt_x = current_r * np.cos(theta)
                pt_y = current_r * np.sin(theta)
                if self._point_in_polygon(pt_x, pt_y, self.lasso_points):
                    # Find innermost intersection
                    innermost = min(intersections)
                    r_theta[i] = min(current_r, innermost)

        # Ensure minimum radius
        r_theta[:] = np.maximum(r_theta, 0.3)

        # Mark as modified
        self.modified_mask[vessel][pos] = True

        # Recalculate r_eq
        self._recalculate_r_eq(vessel, pos)

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point (x, y) is inside polygon using ray casting."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside

            j = i

        return inside

    # ─────────────────────────────────────────────────────────────────────────
    # Fill Between Slices
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_between_slices(self) -> None:
        """Interpolate contours between manually modified positions."""
        vessel = self.current_vessel
        data = self.vessel_data[vessel]
        r_theta = data["r_theta"]  # (n_positions, n_angles)
        mask = self.modified_mask[vessel]

        # Find modified positions
        modified_indices = np.where(mask)[0]

        if len(modified_indices) < 2:
            print(f"[contour_editor] Need at least 2 modified positions for fill-between")
            return

        n_positions, n_angles = r_theta.shape

        # Sort modified indices
        modified_indices = np.sort(modified_indices)

        # Interpolate between consecutive modified positions
        for i in range(len(modified_indices) - 1):
            start_idx = modified_indices[i]
            end_idx = modified_indices[i + 1]

            if end_idx - start_idx <= 1:
                continue

            # Linear interpolation for each angle
            for j in range(n_angles):
                start_val = r_theta[start_idx, j]
                end_val = r_theta[end_idx, j]

                for k in range(start_idx + 1, end_idx):
                    t = (k - start_idx) / (end_idx - start_idx)
                    r_theta[k, j] = start_val * (1 - t) + end_val * t

        # Before first modified: copy first modified's r_theta
        first_modified = modified_indices[0]
        for i in range(first_modified):
            r_theta[i] = r_theta[first_modified].copy()

        # After last modified: copy last modified's r_theta
        last_modified = modified_indices[-1]
        for i in range(last_modified + 1, n_positions):
            r_theta[i] = r_theta[last_modified].copy()

        # Smooth along position axis with gaussian filter
        for j in range(n_angles):
            r_theta[:, j] = gaussian_filter1d(r_theta[:, j], sigma=1.0)

        # Mark all positions as modified
        mask[:] = True

        # Recalculate r_eq for all positions
        for i in range(n_positions):
            self._recalculate_r_eq(vessel, i)

        n_filled = n_positions - len(modified_indices)
        n_key = len(modified_indices)
        print(f"[contour_editor] Filled {n_filled} positions between {n_key} key slices")

        self._update_display()
        self._update_pyvista_meshes()

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_slider_change(self, val: float) -> None:
        """Handle slider value change."""
        new_pos = int(val)
        if new_pos != self.current_position:
            self.current_position = new_pos
            self._update_display()

    def _on_mouse_press(self, event) -> None:
        """Handle mouse press for lasso or contour deformation."""
        if event.inaxes != self.ax_cross:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Right-click: always start scissors (backward compat)
        if event.button == 3:
            self.lasso_active = True
            self.lasso_points = [(event.xdata, event.ydata)]
            return
        if event.button != 1:
            return

        # Left-click: start lasso
        self.lasso_active = True
        self.lasso_points = [(event.xdata, event.ydata)]

    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dragging or lasso drawing."""
        if event.inaxes != self.ax_cross:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Lasso drawing (scissors mode or right-click)
        if self.lasso_active:
            self.lasso_points.append((event.xdata, event.ydata))
            if self.lasso_line is not None:
                self.lasso_line.remove()
            xs = [p[0] for p in self.lasso_points] + [self.lasso_points[0][0]]
            ys = [p[1] for p in self.lasso_points] + [self.lasso_points[0][1]]
            color = "green" if self.fill_mode else "red"
            self.lasso_line, = self.ax_cross.plot(
                xs, ys,
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.8,
            )
            self.fig.canvas.draw_idle()
            return


    def _on_mouse_release(self, event) -> None:
        """Handle mouse release to finish dragging or lasso."""
        # Lasso release (scissors mode or right-click)
        if self.lasso_active:
            self.lasso_active = False
            if len(self.lasso_points) >= 3:
                self._apply_lasso_to_contour()
                # Auto-smooth after scissors
                self._smooth_contour_inplace()

            # Clear lasso visual
            if self.lasso_line is not None:
                self.lasso_line.remove()
                self.lasso_line = None

            self.lasso_points = []
            self._update_display()
            self._update_pyvista_meshes()
            return


    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        if event.key == 'q':
            # Quit without saving
            if self.plotter is not None:
                self.plotter.close()
            plt.close(self.fig)
        elif event.key == 's':
            # Save and close
            self._save_and_close()
        elif event.key == 'r':
            # Reset current contour
            self._reset_current_contour()
        elif event.key == 'e':
            # Toggle fill/erase mode
            self.fill_mode = not self.fill_mode
            mode_str = "FILL" if self.fill_mode else "ERASE"
            print(f"[contour_editor] Scissors mode: {mode_str}")
            self._update_status_bar()
            self.fig.canvas.draw_idle()
        elif event.key == 'a':
            # Auto-snap boundary
            self._auto_snap_boundary()
        elif event.key == 'i':
            # Fill between slices
            self._fill_between_slices()
        elif event.key == ' ':
            # Toggle contour visibility
            self.contour_visible = not self.contour_visible
            self._update_display()
        elif event.key == 'v':
            # Toggle VOI visibility
            self.voi_visible = not self.voi_visible
            self._update_display()
        elif event.key == '1':
            # Switch to LAD
            if 'LAD' in self.vessel_names:
                self._switch_vessel('LAD')
        elif event.key == '2':
            # Switch to LCX
            if 'LCX' in self.vessel_names:
                self._switch_vessel('LCX')
        elif event.key == '3':
            # Switch to RCA
            if 'RCA' in self.vessel_names:
                self._switch_vessel('RCA')
        elif event.key == 'left':
            # Navigate to previous position
            self._navigate_position(-1)
        elif event.key == 'right':
            # Navigate to next position
            self._navigate_position(1)
        elif event.key == 'up':
            # Navigate by -5 positions
            self._navigate_position(-5)
        elif event.key == 'down':
            # Navigate by +5 positions
            self._navigate_position(5)

    def _on_scroll(self, event) -> None:
        """Handle scroll wheel for zoom."""
        if event.inaxes != self.ax_cross:
            return

        if event.button == 'up':
            self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        elif event.button == 'down':
            self.zoom_level = max(self.zoom_level / 1.2, 0.5)

        self._draw_crosssection()
        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _switch_vessel(self, vessel_name: str) -> None:
        """Switch to a different vessel."""
        if vessel_name in self.vessel_names:
            self.current_vessel = vessel_name
            self.current_position = 0
            self.zoom_level = 1.0
            self._update_display()

    def _navigate_position(self, delta: int) -> None:
        """Navigate to a different position."""
        data = self.vessel_data[self.current_vessel]
        n_positions = len(data["positions_mm"])
        new_pos = max(0, min(self.current_position + delta, n_positions - 1))
        if new_pos != self.current_position:
            self.current_position = new_pos
            self._update_display()

    def _reset_current_contour(self) -> None:
        """Reset current contour to original auto-detected values."""
        vessel = self.current_vessel
        pos = self.current_position
        self.vessel_data[vessel]["r_theta"][pos] = self.original_r_theta[vessel][pos].copy()
        self.modified_mask[vessel][pos] = False
        self._recalculate_r_eq(vessel, pos)
        self._update_display()
        self._update_pyvista_meshes()
        print(f"[contour_editor] Reset contour at position {pos}")

    def _recalculate_r_eq(self, vessel: str, pos: int) -> None:
        """Recalculate equivalent radius for a position."""
        data = self.vessel_data[vessel]
        r_theta = data["r_theta"][pos]
        n_angles = len(r_theta)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Compute area using shoelace formula
        x = r_theta * np.cos(angles)
        y = r_theta * np.sin(angles)
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

        # r_eq = sqrt(area / pi)
        data["r_eq"][pos] = np.sqrt(area / np.pi)

    def _smooth_contour_inplace(self, sigma_deg: float = 5.0) -> None:
        """Apply Gaussian smoothing with wrap-around to current contour."""
        vessel = self.current_vessel
        pos = self.current_position
        r_theta = self.vessel_data[vessel]["r_theta"][pos]
        n_angles = len(r_theta)
        sigma_samples = sigma_deg / (360.0 / n_angles)
        self.vessel_data[vessel]["r_theta"][pos] = gaussian_filter1d(
            r_theta, sigma=sigma_samples, mode='wrap'
        )

    def _auto_snap_boundary(self) -> None:
        """Re-detect vessel boundary at current position using gradient analysis."""
        vessel = self.current_vessel
        pos = self.current_position
        data = self.vessel_data[vessel]
        center_mm = data["positions_mm"][pos]
        N = data["N_frame"][pos]
        B = data["B_frame"][pos]

        # Sample polar cross-section
        polar_image, angles, radii = _polar_transform_cross_section(
            self.volume, self.vox_size, center_mm, N, B,
            n_angles=data["r_theta"].shape[1],
            max_radius_mm=8.0,
        )

        # Detect boundary
        r_theta_new = _detect_adventitial_boundary(polar_image, radii)

        # Smooth the detected boundary
        n_angles = len(r_theta_new)
        sigma_samples = 5.0 / (360.0 / n_angles)
        r_theta_new = gaussian_filter1d(r_theta_new, sigma=sigma_samples, mode='wrap')

        # Apply
        data["r_theta"][pos] = r_theta_new
        self.modified_mask[vessel][pos] = True
        self._recalculate_r_eq(vessel, pos)
        self._update_display()
        self._update_pyvista_meshes()
        print(f"[contour_editor] Auto-snapped boundary at {vessel} pos {pos}")

    def _save_and_close(self) -> None:
        """Save corrected contours and close the editor."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build save dictionary
        save_data = {}

        for vessel_name, data in self.vessel_data.items():
            # Save corrected r_theta
            save_data[f"{vessel_name}_r_theta_corrected"] = data["r_theta"]
            save_data[f"{vessel_name}_r_theta"] = data["r_theta"]  # Also save as original key

            # Copy all other data unchanged
            for key, value in data.items():
                if key != "r_theta" and value is not None:
                    np_key = f"{vessel_name}_{key}"
                    save_data[np_key] = value

        # Save to .npz
        output_file = self.output_dir / f"{self.prefix}_contour_data_corrected.npz"
        np.savez(str(output_file), **save_data)
        print(f"[contour_editor] Saved corrected contours to {output_file}")

        # Write signal file
        signal_file = self.output_dir / f"{self.prefix}_contour_editor.done"
        signal_file.write_text("")
        print(f"[contour_editor] Wrote signal file: {signal_file}")

        # Close windows
        if self.plotter is not None:
            self.plotter.close()
        plt.close(self.fig)

    def run(self) -> None:
        """Run the interactive editor (blocks until window is closed)."""
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def launch_contour_editor(
    volume: np.ndarray,
    spacing_mm: List[float],
    contour_data: Dict[str, np.ndarray],
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Launch the interactive contour editor.

    Parameters
    ----------
    volume : (Z, Y, X) float32
        CT HU volume
    spacing_mm : [sz, sy, sx]
        Voxel spacing in mm
    contour_data : dict
        Dictionary containing vessel contour data from .npz file
    output_dir : Path
        Directory for saving results
    prefix : str
        Filename prefix for output files
    """
    editor = ContourEditor(
        volume=volume,
        spacing_mm=spacing_mm,
        contour_data=contour_data,
        output_dir=output_dir,
        prefix=prefix,
    )
    editor.run()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Professional clinical GUI for manually correcting auto-extracted vessel wall "
            "contours before PCAT VOI computation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls:\n"
            "  Draw lasso to edit contour (left-drag or right-drag)\n"
            "  A             Auto-snap boundary (re-detect using gradient)\n"
            "  E             Toggle fill/erase mode\n"
            "  ←/→ arrows    Navigate ±1 position\n"
            "  ↑/↓ arrows    Navigate ±5 positions\n"
            "  1/2/3         Switch vessel (LAD/LCX/RCA)\n"
            "  R             Reset current contour to auto-detected values\n"
            "  I             Fill between slices (interpolate)\n"
            "  S             Save all corrections and close\n"
            "  Q             Quit without saving\n"
            "  Space         Toggle contour visibility\n"
            "  V             Toggle VOI visibility\n"
            "  Scroll        Zoom cross-section view\n"
        ),
    )

    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory",
    )
    parser.add_argument(
        "--contour-data", required=True,
        dest="contour_data",
        help="Path to contour extraction .npz file",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for saving results",
    )
    parser.add_argument(
        "--prefix", default="patient",
        help="Filename prefix for output files (default: patient)",
    )

    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.dicom_loader import load_dicom_series

    # ── Load DICOM ───────────────────────────────────────────────────────────
    print(f"[contour_editor] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[contour_editor] Volume shape: {volume.shape}  spacing: {spacing_mm}")

    # ── Load contour data ─────────────────────────────────────────────────────
    contour_path = Path(args.contour_data)
    if not contour_path.exists():
        print(f"[contour_editor] ERROR: Contour data file not found: {contour_path}")
        sys.exit(1)

    print(f"[contour_editor] Loading contour data from {contour_path} ...")
    contour_data = dict(np.load(str(contour_path), allow_pickle=True))

    # ── Launch editor ─────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_contour_editor(
        volume=volume,
        spacing_mm=spacing_mm,
        contour_data=contour_data,
        output_dir=output_dir,
        prefix=args.prefix,
    )

    print("[contour_editor] Editor closed")


if __name__ == "__main__":
    main()
