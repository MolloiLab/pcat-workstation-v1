#!/usr/bin/env python3
"""
contour_game_editor.py

Game-style GUI for manually correcting auto-extracted vessel wall contours
before PCAT VOI computation.

Features:
- Dark theme with neon color accents (game-style HUD)
- CPR cross-section viewer with polar contour overlay
- Draggable control points (every 15°) for contour correction
- Navigation through centerline positions (slider/arrow keys)
- Per-vessel switching (LAD/LCX/RCA)
- Save corrected contours to .npz
- PCAT VOI ring preview (3× r_eq)

Controls:
- Left/Right arrows or slider: navigate centerline positions
- 1/2/3: switch vessel (LAD/LCX/RCA)
- Mouse drag on control points: reshape contour
- R: reset current contour to auto-detected values
- S: save all corrections and close
- Q: quit without saving
- Scroll wheel: zoom cross-section view
- Space: toggle contour visibility
- C: copy current contour to all positions
- F: toggle fallback indicator

CLI Usage:
    python contour_game_editor.py \
        --dicom path/to/dicom \
        --contour-data path/to/contour_data.npz \
        --output path/to/output \
        --prefix patient
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

# Interactive backend — MUST be set before pyplot is imported
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────────────────────────────────────
# Color Theme (Game-style dark with neon accents)
# ─────────────────────────────────────────────────────────────────────────────

# Background colors
BG_DARK = "#1a1a2e"
BG_PANEL = "#0d0d1a"
BG_HUD = "#16213e"
BG_BORDER = "#3a3a5a"

# Neon accent colors
NEON_CYAN = "#00f0ff"
NEON_MAGENTA = "#ff00ff"
NEON_YELLOW = "#ffff00"
NEON_GREEN = "#00ff88"
NEON_ORANGE = "#ff6600"
NEON_RED = "#ff3366"
NEON_WHITE = "#ffffff"

# Vessel colors (matching existing pipeline)
VESSEL_COLORS = {
    "LAD": "#E8533A",
    "LCX": "#4A90D9",
    "RCA": "#2ECC71",
}

# Text colors
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#aaaacc"
TEXT_MUTED = "#666688"

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

class ContourGameEditor:
    """
    Game-style interactive contour editor for vessel wall correction.

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
        self.voi_visible = True

        # Control points: every 15° = 24 points around the contour
        self.n_control_points = 24
        self.control_angles = np.linspace(0, 2 * np.pi, self.n_control_points, endpoint=False)

        # Dragging state
        self.dragging_idx: Optional[int] = None
        self.dragging_vessel: Optional[str] = None

        # Store original r_theta for reset functionality
        self.original_r_theta: Dict[str, np.ndarray] = {}
        for vessel_name, data in self.vessel_data.items():
            self.original_r_theta[vessel_name] = data["r_theta"].copy()

        # Matplotlib handles
        self.fig = None
        self.ax_cross = None
        self.ax_long = None
        self.ax_hud = None
        self.ax_slider = None

        # Image handles
        self._cross_im = None
        self._contour_line = None
        self._contour_glow = None
        self._voi_ring = None
        self._voi_ring_glow = None
        self._control_points = []
        self._control_point_hovers = []

        # Longitudinal view
        self._long_im = None
        self._long_position_line = None

        # HUD elements
        self._hud_texts = {}

        # Slider
        self._slider = None

        # Build GUI
        self._build_gui()
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
                print(f"[contour_game] Loaded {vessel}: {len(data['positions_mm'])} positions")

    def _build_gui(self) -> None:
        """Build the matplotlib figure with game-style UI."""
        # Create figure with dark background
        self.fig = plt.figure(figsize=(18, 11), facecolor=BG_DARK)
        self.fig.canvas.manager.set_window_title(
            f"PCAT Contour Game Editor — {self.prefix}"
        )

        # Layout: Cross-section (left) | Longitudinal view (right) | HUD (bottom)
        gs = self.fig.add_gridspec(
            3, 2,
            height_ratios=[15, 1, 2],
            width_ratios=[1.2, 1],
            left=0.06, right=0.98,
            bottom=0.08, top=0.92,
            hspace=0.15, wspace=0.12,
        )

        # Main panels
        self.ax_cross = self.fig.add_subplot(gs[0, 0])
        self.ax_long = self.fig.add_subplot(gs[0, 1])
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        self.ax_hud = self.fig.add_subplot(gs[2, :])

        # Style all axes
        for ax in [self.ax_cross, self.ax_long, self.ax_slider]:
            ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BG_BORDER)
                spine.set_linewidth(1.5)

        self.ax_hud.set_facecolor(BG_HUD)
        self.ax_hud.axis('off')

        # ── Setup Cross-section Panel ───────────────────────────────────────
        self.ax_cross.set_title(
            "CROSS-SECTION VIEW",
            color=TEXT_PRIMARY, fontsize=11, fontweight='bold', pad=8,
        )

        # ── Setup Longitudinal Panel ────────────────────────────────────────
        self.ax_long.set_title(
            "LONGITUDINAL CPR",
            color=TEXT_PRIMARY, fontsize=11, fontweight='bold', pad=8,
        )

        # ── Setup Slider ────────────────────────────────────────────────────
        self.ax_slider.set_facecolor(BG_PANEL)
        n_positions = len(self.vessel_data[self.current_vessel]["positions_mm"])
        arclengths = self.vessel_data[self.current_vessel]["arclengths"]

        self._slider = Slider(
            ax=self.ax_slider,
            label="POSITION",
            valmin=0,
            valmax=n_positions - 1,
            valinit=0,
            valstep=1,
            color=NEON_CYAN,
        )
        self._slider.label.set_color(TEXT_PRIMARY)
        self._slider.label.set_fontweight('bold')
        self._slider.valtext.set_color(NEON_CYAN)
        self._slider.valtext.set_fontweight('bold')
        self._slider.on_changed(self._on_slider_change)

        # ── Setup HUD Panel ─────────────────────────────────────────────────
        self._build_hud()

        # ── Event Handlers ──────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # ── Title Bar ───────────────────────────────────────────────────────
        self._update_title()

    def _build_hud(self) -> None:
        """Build the HUD (Heads-Up Display) status bar."""
        # Create HUD background with rounded corners effect
        hud_bg = FancyBboxPatch(
            (0.01, 0.1), 0.98, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=BG_HUD,
            edgecolor=BG_BORDER,
            linewidth=2,
            transform=self.ax_hud.transAxes,
            zorder=1,
        )
        self.ax_hud.add_patch(hud_bg)

        # HUD text elements
        self._hud_texts = {
            "vessel": self.ax_hud.text(
                0.02, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=12, fontweight='bold', color=TEXT_PRIMARY,
                va='center', ha='left',
            ),
            "position": self.ax_hud.text(
                0.15, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=11, color=TEXT_SECONDARY,
                va='center', ha='left',
            ),
            "arclength": self.ax_hud.text(
                0.35, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=11, color=NEON_CYAN,
                va='center', ha='left',
            ),
            "radius": self.ax_hud.text(
                0.55, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=11, color=NEON_GREEN,
                va='center', ha='left',
            ),
            "fallback": self.ax_hud.text(
                0.75, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=11, color=NEON_ORANGE,
                va='center', ha='left',
            ),
            "controls": self.ax_hud.text(
                0.92, 0.5, "",
                transform=self.ax_hud.transAxes,
                fontsize=9, color=TEXT_MUTED,
                va='center', ha='right',
            ),
        }

    def _update_hud(self) -> None:
        """Update HUD text elements with current state."""
        vessel = self.current_vessel
        data = self.vessel_data[vessel]
        pos = self.current_position
        n_total = len(data["positions_mm"])

        # Vessel name with color
        vessel_color = VESSEL_COLORS.get(vessel, TEXT_PRIMARY)
        self._hud_texts["vessel"].set_text(f"■ {vessel}")
        self._hud_texts["vessel"].set_color(vessel_color)

        # Position
        self._hud_texts["position"].set_text(f"POS: {pos + 1}/{n_total}")

        # Arc-length
        arc_mm = data["arclengths"][pos]
        self._hud_texts["arclength"].set_text(f"ARC: {arc_mm:.1f} mm")

        # Radius
        r_eq = data["r_eq"][pos]
        self._hud_texts["radius"].set_text(f"R_eq: {r_eq:.2f} mm")

        # Fallback status
        fallback = data["fallback_mask"][pos] if data["fallback_mask"] is not None else False
        if fallback:
            self._hud_texts["fallback"].set_text("⚠ FALLBACK")
            self._hud_texts["fallback"].set_color(NEON_ORANGE)
        else:
            self._hud_texts["fallback"].set_text("✓ GRADIENT")
            self._hud_texts["fallback"].set_color(NEON_GREEN)

        # Controls hint
        self._hud_texts["controls"].set_text("←/→ Navigate | 1/2/3 Vessel | R Reset | S Save | Q Quit")

    def _update_title(self) -> None:
        """Update the figure title."""
        vessel_color = VESSEL_COLORS.get(self.current_vessel, TEXT_PRIMARY)
        self.fig.suptitle(
            f"◆ PCAT CONTOUR GAME EDITOR ◆  |  {self.prefix.upper()}  |  "
            f"<span style='color:{vessel_color}'>{self.current_vessel}</span>  |  "
            f"ZOOM: {self.zoom_level:.1f}×",
            color=TEXT_PRIMARY, fontsize=12, fontweight='bold',
        )

    def _update_display(self) -> None:
        """Update all display panels."""
        self._draw_crosssection()
        self._draw_longitudinal()
        self._update_hud()
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

        # Grayscale normalization
        gray = np.clip(cs_img, -200.0, 400.0)
        gray = (gray + 200.0) / 600.0
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
            # Add crosshairs
            self.ax_cross.axhline(0, color=TEXT_MUTED, linewidth=0.5, linestyle=':', alpha=0.5)
            self.ax_cross.axvline(0, color=TEXT_MUTED, linewidth=0.5, linestyle=':', alpha=0.5)
        else:
            self._cross_im.set_data(gray)
            self._cross_im.set_extent((-extent, extent, extent, -extent))

        # Clear old contour elements
        if self._contour_line is not None:
            self._contour_line.remove()
            self._contour_line = None
        if self._contour_glow is not None:
            self._contour_glow.remove()
            self._contour_glow = None
        if self._voi_ring is not None:
            self._voi_ring.remove()
            self._voi_ring = None
        if self._voi_ring_glow is not None:
            self._voi_ring_glow.remove()
            self._voi_ring_glow = None
        for cp in self._control_points:
            cp.remove()
        self._control_points.clear()
        for cp in self._control_point_hovers:
            cp.remove()
        self._control_point_hovers.clear()

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
            vessel_color = VESSEL_COLORS.get(vessel, NEON_CYAN)

            # Draw glow effect (thicker semi-transparent line behind)
            self._contour_glow, = self.ax_cross.plot(
                x_closed, y_closed,
                color=vessel_color,
                linewidth=6.0,
                alpha=0.3,
                zorder=3,
            )

            # Draw main contour line
            self._contour_line, = self.ax_cross.plot(
                x_closed, y_closed,
                color=NEON_CYAN,
                linewidth=2.0,
                alpha=0.95,
                zorder=4,
            )

            # Draw control points (every 15°)
            for i, angle in enumerate(self.control_angles):
                # Interpolate r at this angle
                idx_continuous = angle / (2 * np.pi) * n_angles
                idx_lo = int(idx_continuous) % n_angles
                idx_hi = (idx_lo + 1) % n_angles
                frac = idx_continuous - int(idx_continuous)
                r_interp = r_theta[idx_lo] * (1 - frac) + r_theta[idx_hi] * frac

                x_cp = r_interp * np.cos(angle)
                y_cp = r_interp * np.sin(angle)

                # Control point glow
                glow = self.ax_cross.plot(
                    x_cp, y_cp, 'o',
                    color=vessel_color,
                    markersize=12,
                    alpha=0.3,
                    zorder=5,
                )[0]
                self._control_point_hovers.append(glow)

                # Control point
                cp = self.ax_cross.plot(
                    x_cp, y_cp, 'o',
                    color=NEON_WHITE,
                    markersize=7,
                    markeredgecolor=vessel_color,
                    markeredgewidth=2,
                    alpha=0.9,
                    zorder=6,
                )[0]
                self._control_points.append(cp)

        # Draw PCAT VOI ring (3× r_eq)
        if self.voi_visible:
            r_voi = r_eq * 3.0

            # VOI ring glow
            self._voi_ring_glow = mpatches.Circle(
                (0, 0), radius=r_voi,
                fill=False,
                edgecolor=NEON_YELLOW,
                linewidth=5.0,
                alpha=0.25,
                zorder=2,
            )
            self.ax_cross.add_patch(self._voi_ring_glow)

            # VOI ring
            self._voi_ring = mpatches.Circle(
                (0, 0), radius=r_voi,
                fill=False,
                edgecolor=NEON_YELLOW,
                linewidth=1.5,
                linestyle='--',
                alpha=0.8,
                zorder=2,
            )
            self.ax_cross.add_patch(self._voi_ring)

        # Set axis limits
        self.ax_cross.set_xlim(-extent, extent)
        self.ax_cross.set_ylim(extent, -extent)

        # Update title
        self.ax_cross.set_title(
            f"CROSS-SECTION  |  {vessel}  |  Pos {pos + 1}",
            color=TEXT_PRIMARY, fontsize=11, fontweight='bold', pad=8,
        )

    def _draw_longitudinal(self) -> None:
        """Draw the longitudinal CPR view showing all contours as a heatmap."""
        vessel = self.current_vessel
        data = self.vessel_data[vessel]

        r_theta_all = data["r_theta"]  # (n_positions, n_angles)
        arclengths = data["arclengths"]
        n_positions = len(arclengths)

        # Create a heatmap: rows = positions, cols = angles
        # Use symmetric colormap centered on mean radius
        r_mean = np.mean(data["r_eq"])
        r_min = r_mean * 0.5
        r_max = r_mean * 2.0

        # Draw or update heatmap
        if self._long_im is None:
            self._long_im = self.ax_long.imshow(
                r_theta_all,
                aspect='auto',
                origin='upper',
                cmap='plasma',
                vmin=r_min,
                vmax=r_max,
                interpolation='bilinear',
            )
            # Add colorbar
            cbar = self.fig.colorbar(self._long_im, ax=self.ax_long, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            cbar.set_label('Radius (mm)', color=TEXT_SECONDARY, fontsize=9)
            cbar.outline.set_edgecolor(BG_BORDER)
        else:
            self._long_im.set_data(r_theta_all)
            self._long_im.set_clim(r_min, r_max)

        # Clear old position line
        if self._long_position_line is not None:
            self._long_position_line.remove()

        # Draw current position indicator
        self._long_position_line = self.ax_long.axhline(
            self.current_position,
            color=NEON_CYAN,
            linewidth=2.0,
            linestyle='-',
            alpha=0.9,
        )

        # Set axis labels
        n_angles = r_theta_all.shape[1]
        self.ax_long.set_xticks([0, n_angles // 4, n_angles // 2, 3 * n_angles // 4, n_angles - 1])
        self.ax_long.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=8, color=TEXT_SECONDARY)
        self.ax_long.set_xlabel('Angle', color=TEXT_SECONDARY, fontsize=9)

        # Y-axis: position indices
        tick_positions = np.linspace(0, n_positions - 1, min(6, n_positions), dtype=int)
        self.ax_long.set_yticks(tick_positions)
        self.ax_long.set_yticklabels(
            [f"{arclengths[i]:.0f}" for i in tick_positions],
            fontsize=8, color=TEXT_SECONDARY
        )
        self.ax_long.set_ylabel('Arc-length (mm)', color=TEXT_SECONDARY, fontsize=9)

        # Update title
        self.ax_long.set_title(
            f"CONTOUR MAP  |  {vessel}",
            color=TEXT_PRIMARY, fontsize=11, fontweight='bold', pad=8,
        )

    def _update_slider(self) -> None:
        """Update slider to match current vessel."""
        data = self.vessel_data[self.current_vessel]
        n_positions = len(data["positions_mm"])

        # Update slider range
        self._slider.valmin = 0
        self._slider.valmax = n_positions - 1
        self._slider.ax.set_xlim(0, n_positions - 1)

        # Set current value
        self._slider.set_val(self.current_position)

        # Update color based on vessel
        vessel_color = VESSEL_COLORS.get(self.current_vessel, NEON_CYAN)
        self._slider.poly.set_facecolor(vessel_color)

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
        """Handle mouse press for control point dragging."""
        if event.inaxes != self.ax_cross:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Find nearest control point
        data = self.vessel_data[self.current_vessel]
        r_theta = data["r_theta"][self.current_position]

        min_dist = float('inf')
        nearest_idx = None

        for i, angle in enumerate(self.control_angles):
            # Interpolate r at this angle
            n_angles = len(r_theta)
            idx_continuous = angle / (2 * np.pi) * n_angles
            idx_lo = int(idx_continuous) % n_angles
            idx_hi = (idx_lo + 1) % n_angles
            frac = idx_continuous - int(idx_continuous)
            r_interp = r_theta[idx_lo] * (1 - frac) + r_theta[idx_hi] * frac

            x_cp = r_interp * np.cos(angle)
            y_cp = r_interp * np.sin(angle)

            dist = np.sqrt((x_cp - event.xdata) ** 2 + (y_cp - event.ydata) ** 2)
            if dist < min_dist and dist < 2.0 / self.zoom_level:
                min_dist = dist
                nearest_idx = i

        if nearest_idx is not None:
            self.dragging_idx = nearest_idx
            self.dragging_vessel = self.current_vessel

    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dragging control points."""
        if self.dragging_idx is None or event.inaxes != self.ax_cross:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Calculate new radius from mouse position
        r_new = np.sqrt(event.xdata ** 2 + event.ydata ** 2)

        # Calculate angle from mouse position
        angle_mouse = np.arctan2(event.ydata, event.xdata)
        if angle_mouse < 0:
            angle_mouse += 2 * np.pi

        # Update r_theta at nearby angles
        data = self.vessel_data[self.current_vessel]
        n_angles = data["r_theta"].shape[1]
        r_theta = data["r_theta"][self.current_position]

        # Find the angle indices to update (smooth update around the dragged point)
        center_idx = int(angle_mouse / (2 * np.pi) * n_angles) % n_angles

        # Update a small window of angles for smooth deformation
        window = 5  # Update 5 angles on each side
        for offset in range(-window, window + 1):
            idx = (center_idx + offset) % n_angles

            # Calculate angle for this index
            angle_idx = idx / n_angles * 2 * np.pi
            angle_diff = abs(angle_idx - angle_mouse)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # Weight by distance from mouse angle
            weight = np.exp(-angle_diff ** 2 / 0.1)  # Gaussian falloff
            r_theta[idx] = r_theta[idx] * (1 - weight) + r_new * weight

        # Ensure minimum radius
        r_theta[:] = np.maximum(r_theta, 0.3)

        # Update display
        self._draw_crosssection()
        self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event) -> None:
        """Handle mouse release to finish dragging."""
        if self.dragging_idx is not None:
            # Recalculate r_eq based on updated contour
            self._recalculate_r_eq(self.current_vessel, self.current_position)
            self._update_display()
        self.dragging_idx = None
        self.dragging_vessel = None

    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        if event.key == 'q':
            # Quit without saving
            plt.close(self.fig)
        elif event.key == 's':
            # Save and close
            self._save_and_close()
        elif event.key == 'r':
            # Reset current contour
            self._reset_current_contour()
        elif event.key == 'c':
            # Copy current contour to all positions
            self._copy_contour_to_all()
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
            # Navigate by 5 positions
            self._navigate_position(-5)
        elif event.key == 'down':
            # Navigate by 5 positions
            self._navigate_position(5)
        elif event.key == 'f':
            # Toggle fallback for current position
            self._toggle_fallback()

    def _on_scroll(self, event) -> None:
        """Handle scroll wheel for zoom."""
        if event.inaxes != self.ax_cross:
            return

        if event.button == 'up':
            self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        elif event.button == 'down':
            self.zoom_level = max(self.zoom_level / 1.2, 0.5)

        self._update_title()
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
        self._recalculate_r_eq(vessel, pos)
        self._update_display()
        print(f"[contour_game] Reset contour at position {pos}")

    def _copy_contour_to_all(self) -> None:
        """Copy current contour to all positions in current vessel."""
        vessel = self.current_vessel
        pos = self.current_position
        current_r_theta = self.vessel_data[vessel]["r_theta"][pos].copy()

        n_positions = len(self.vessel_data[vessel]["positions_mm"])
        for i in range(n_positions):
            self.vessel_data[vessel]["r_theta"][i] = current_r_theta.copy()
            self._recalculate_r_eq(vessel, i)

        self._update_display()
        print(f"[contour_game] Copied contour to all {n_positions} positions")

    def _toggle_fallback(self) -> None:
        """Toggle fallback status for current position."""
        vessel = self.current_vessel
        pos = self.current_position
        data = self.vessel_data[vessel]

        if data["fallback_mask"] is not None:
            data["fallback_mask"][pos] = not data["fallback_mask"][pos]
            self._update_display()

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
        print(f"[contour_game] Saved corrected contours to {output_file}")

        # Write signal file
        signal_file = self.output_dir / f"{self.prefix}_contour_game_editor.done"
        signal_file.write_text("")
        print(f"[contour_game] Wrote signal file: {signal_file}")

        # Close window
        plt.close(self.fig)

    def run(self) -> None:
        """Run the interactive editor (blocks until window is closed)."""
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def launch_contour_game_editor(
    volume: np.ndarray,
    spacing_mm: List[float],
    contour_data: Dict[str, np.ndarray],
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Launch the interactive contour game editor.

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
    editor = ContourGameEditor(
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
            "Game-style GUI for manually correcting auto-extracted vessel wall "
            "contours before PCAT VOI computation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls:\n"
            "  ←/→ arrows    Navigate centerline positions\n"
            "  1/2/3         Switch vessel (LAD/LCX/RCA)\n"
            "  Mouse drag    Reshape contour at current position\n"
            "  R             Reset current contour to auto-detected values\n"
            "  C             Copy current contour to all positions\n"
            "  S             Save all corrections and close\n"
            "  Q             Quit without saving\n"
            "  Scroll        Zoom cross-section view\n"
            "  Space         Toggle contour visibility\n"
            "  V             Toggle VOI visibility\n"
            "  F             Toggle fallback status\n"
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
    print(f"[contour_game] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[contour_game] Volume shape: {volume.shape}  spacing: {spacing_mm}")

    # ── Load contour data ─────────────────────────────────────────────────────
    contour_path = Path(args.contour_data)
    if not contour_path.exists():
        print(f"[contour_game] ERROR: Contour data file not found: {contour_path}")
        sys.exit(1)

    print(f"[contour_game] Loading contour data from {contour_path} ...")
    contour_data = dict(np.load(str(contour_path), allow_pickle=True))

    # ── Launch editor ─────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_contour_game_editor(
        volume=volume,
        spacing_mm=spacing_mm,
        contour_data=contour_data,
        output_dir=output_dir,
        prefix=args.prefix,
    )

    print("[contour_game] Editor closed")


if __name__ == "__main__":
    main()
