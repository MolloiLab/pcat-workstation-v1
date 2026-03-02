"""
cpr_browser.py
Interactive CPR (Curved Planar Reformation) browser with live cross-section panel.

Shows two linked panels:
  Left  — Straightened CPR (vessel top→bottom, ostium at top).
           A movable horizontal needle line indicates the current position.
  Right — Orthogonal cross-section at the needle position:
           True N-B plane sampling with vessel lumen circle (white) and PCAT VOI
           ring (semi-transparent yellow) overlaid.

Rotation slider rotates the cutting plane around the vessel axis (rotational CPR).

Usage (standalone):
    python pipeline/cpr_browser.py \
        --dicom  Rahaf_Patients/1200.2 \
        --seeds  seeds/patient_1200.json \
        --vessel LAD

    # Or after pipeline has already saved a reviewed VOI:
    python pipeline/cpr_browser.py \
        --dicom  Rahaf_Patients/1200.2 \
        --seeds  seeds/patient_1200.json \
        --vessel LAD \
        --voi    output/patient_1200/patient1200_LAD_voi_reviewed.npy

Controls:
    Slider (Arc-length) — Drag to move the needle along the vessel
    Slider (Rotation)   — Drag to rotate the cutting plane (0-360)
    Click CPR           — Click directly on the CPR panel to jump the needle
    ← / →               — Arrow keys: step needle by one centerline point
    s                   — Save a PNG snapshot of the current view to the output dir
    a                   — Toggle anchor point mode (click to place, right-click to delete)
    p                   — Apply anchor points (print data, UI scaffolding for future)
    r                   — Reset rotation to 0
    q                   — Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Interactive backend — MUST be set before pyplot is imported.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.dicom_loader import load_dicom_series
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
    _sample_volume_trilinear,
    _fai_colormap,
    FAI_HU_MIN,
    FAI_HU_MAX,
)


# ---------------------------------------------------------------------------
# Core interactive class
# ---------------------------------------------------------------------------

class CPRBrowser:
    """
    Interactive CPR browser with rotational CPR and anchor points.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) centerline voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii in mm
    spacing_mm      : [sz, sy, sx]
    vessel_name     : display label (e.g. "LAD")
    voi_mask        : (Z, Y, X) bool optional — PCAT VOI for ring overlay
    width_mm        : half-width of CPR / cross-section plane in mm (default 25)
    slab_thickness_mm : MIP slab thickness for CPR (default 3 mm)
    output_dir      : where to save snapshots (default: current dir)
    rotation_deg    : initial rotation angle in degrees (default 0)
    """

    def __init__(
        self,
        volume: np.ndarray,
        centerline_ijk: np.ndarray,
        radii_mm: np.ndarray,
        spacing_mm: List[float],
        vessel_name: str = "vessel",
        voi_mask: Optional[np.ndarray] = None,
        width_mm: float = 25.0,
        slab_thickness_mm: float = 3.0,
        output_dir: str | Path = ".",
        rotation_deg: float = 0.0,
    ):
        self.volume = volume
        self.centerline_ijk = centerline_ijk
        self.radii_mm = radii_mm
        self.spacing_mm = spacing_mm
        self.vessel_name = vessel_name
        self.voi_mask = voi_mask
        self.width_mm = width_mm
        self.slab_thickness_mm = slab_thickness_mm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.N_pts = len(centerline_ijk)
        self._needle_idx = 0  # current centerline index shown in cross-section
        self._rotation_deg = rotation_deg  # current rotation angle

        # Anchor point system
        self._anchor_mode = False  # whether we're in anchor placement mode
        self._anchor_points: List[Tuple[float, float]] = []  # (arc_length_mm, lateral_offset_mm)
        self._anchor_markers: List[Line2D] = []  # matplotlib marker objects

        # vox_size for _sample_volume_trilinear
        self._vox_size = np.array(spacing_mm, dtype=np.float64)

        # Compute initial CPR data
        self._recompute_cpr_data(rotation_deg)

        self._build_gui()

    def _recompute_cpr_data(self, rotation_deg: float) -> None:
        """Compute CPR data with given rotation angle (expensive, cache result)."""
        print(f"[cpr_browser] Computing CPR data for {self.vessel_name} "
              f"({self.N_pts} centerline points, rotation={rotation_deg:.1f})...")
        (
            self.cpr_volume,    # (pixels_wide, pixels_high)
            self.N_frame,       # (pixels_wide, 3) Bishop normal
            self.B_frame,       # (pixels_wide, 3) Bishop binormal
            self.cl_mm,         # (pixels_wide, 3) centerline in mm
            self.arclengths,    # (pixels_wide,) cumulative arc-length in mm
            self.n_height,
            self.n_width,
        ) = _compute_cpr_data(
            self.volume, self.centerline_ijk, self.spacing_mm,
            slab_thickness_mm=self.slab_thickness_mm,
            width_mm=self.width_mm,
            rotation_deg=rotation_deg,
        )
        self._rotation_deg = rotation_deg
        print(f"[cpr_browser] CPR volume shape: {self.cpr_volume.shape}")

    # -----------------------------------------------------------------------
    def _build_gui(self):
        """Build the matplotlib figure with CPR + cross-section panels."""
        fig = plt.figure(figsize=(16, 12), facecolor="#1a1a2e")
        fig.canvas.manager.set_window_title(
            f"PCAT CPR Browser — {self.vessel_name}"
        )
        self.fig = fig

        # Layout: CPR left | cross-section right
        # Reserve bottom strip for two sliders
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[20, 1, 1],
            width_ratios=[5, 5],
            left=0.07, right=0.97,
            bottom=0.10, top=0.93,
            hspace=0.12, wspace=0.22,
        )
        self.ax_cpr   = fig.add_subplot(gs[0, 0])
        self.ax_cross = fig.add_subplot(gs[0, 1])
        ax_slider_arclen = fig.add_subplot(gs[1, :])
        ax_slider_rot    = fig.add_subplot(gs[2, :])

        # ── Style axes ───────────────────────────────────────────────────
        for ax in (self.ax_cpr, self.ax_cross):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5a")

        # ── Draw CPR (will be updated when rotation changes) ──────────────
        self._cpr_gray_im = None
        self._cpr_fai_im = None
        self._draw_cpr_image()

        # Needle line (horizontal — moves with slider)
        self._needle_line = self.ax_cpr.axhline(
            0, color="#00ffcc", linewidth=1.6, linestyle="-", alpha=0.9,
        )

        # Y-axis: arc-length ticks
        self._setup_cpr_ticks()

        # ── Arc-length Slider ─────────────────────────────────────────────
        ax_slider_arclen.set_facecolor("#0d0d1a")
        self._slider_arclen = Slider(
            ax=ax_slider_arclen,
            label="Arc-length (mm)",
            valmin=0.0,
            valmax=float(self.arclengths[-1]),
            valinit=0.0,
            color="#00ffcc",
        )
        self._slider_arclen.label.set_color("#aaaacc")
        self._slider_arclen.valtext.set_color("#aaaacc")
        self._slider_arclen.on_changed(self._on_arclen_slider)

        # ── Rotation Slider ───────────────────────────────────────────────
        ax_slider_rot.set_facecolor("#0d0d1a")
        self._slider_rot = Slider(
            ax=ax_slider_rot,
            label="Rotation (°)",
            valmin=0.0,
            valmax=360.0,
            valinit=self._rotation_deg,
            color="#ff6600",
            valfmt="%1.0f°",
        )
        self._slider_rot.label.set_color("#aaaacc")
        self._slider_rot.valtext.set_color("#aaaacc")
        self._slider_rot.on_changed(self._on_rot_slider)
        # Debounce handled via button_release_event

        # ── Initial cross-section ─────────────────────────────────────────
        self._xs_im       = None   # imshow handle for grayscale cross-section
        self._xs_fai_im   = None   # imshow handle for FAI overlay
        self._xs_lumen    = None   # Circle patch for lumen
        self._xs_voi_ring = None   # Circle patch for VOI outer boundary
        self._xs_needle_txt = None # Text annotation in cross-section panel

        self._draw_crosssection(0)

        # ── Mode indicator text ────────────────────────────────────────────
        self._mode_txt = self.ax_cpr.text(
            0.02, 0.98, "",
            transform=self.ax_cpr.transAxes,
            va="top", ha="left",
            fontsize=10, color="#ff00ff",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1a", alpha=0.8),
        )
        self._update_mode_indicator()

        # ── Event handlers ────────────────────────────────────────────────
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        # Title
        self._update_title()

    def _setup_cpr_ticks(self) -> None:
        """Setup CPR axis ticks and labels."""
        # Y-axis: arc-length ticks
        x_ticks_mm  = np.arange(0, self.arclengths[-1] + 1, 10.0)
        x_tick_idxs = [int(np.argmin(np.abs(self.arclengths - t))) for t in x_ticks_mm]
        self.ax_cpr.set_yticks(x_tick_idxs)
        self.ax_cpr.set_yticklabels(
            [f"{t:.0f}" for t in x_ticks_mm], fontsize=8, color="#aaaacc"
        )
        self.ax_cpr.set_ylabel("Distance along vessel (mm)", color="#aaaacc", fontsize=10)

        # X-axis: lateral distance ticks
        sz, sy, sx = self.spacing_mm
        n_width = self.n_width
        y_ticks_mm  = np.arange(-self.width_mm, self.width_mm + 1, 5.0)
        y_tick_idxs = (
            (y_ticks_mm + self.width_mm) / (2 * self.width_mm) * (n_width - 1)
        ).astype(int)
        self.ax_cpr.set_xticks(np.clip(y_tick_idxs, 0, n_width - 1))
        self.ax_cpr.set_xticklabels(
            [f"{t:.0f}" for t in y_ticks_mm], fontsize=7, color="#aaaacc"
        )
        self.ax_cpr.set_xlabel(
            "Lateral distance from centreline (mm)", color="#aaaacc", fontsize=9
        )

    def _draw_cpr_image(self) -> None:
        """Draw or update the CPR image (called on rotation change)."""
        cpr_image = self.cpr_volume.T  # (n_height, n_width)
        gray_img  = np.clip(cpr_image, -200.0, 400.0)
        gray_norm = (gray_img + 200.0) / 600.0
        gray_norm = np.nan_to_num(gray_norm, nan=0.0)

        fai_img = np.where(
            (cpr_image >= FAI_HU_MIN) & (cpr_image <= FAI_HU_MAX),
            cpr_image, np.nan,
        )

        if self._cpr_gray_im is None:
            # First draw
            self._cpr_gray_im = self.ax_cpr.imshow(
                gray_norm,
                aspect="auto",
                origin="upper",
                cmap="gray",
                vmin=0.0, vmax=1.0,
                interpolation="bilinear",
            )
            self._cpr_fai_im = self.ax_cpr.imshow(
                fai_img,
                aspect="auto",
                origin="upper",
                cmap=_fai_colormap(),
                vmin=FAI_HU_MIN, vmax=FAI_HU_MAX,
                alpha=0.85,
                interpolation="bilinear",
            )
            # Centreline axis marker
            self.ax_cpr.axvline(
                self.n_width // 2, color="white",
                linewidth=0.8, linestyle="--", alpha=0.5,
            )
        else:
            # Update existing images
            self._cpr_gray_im.set_data(gray_norm)
            self._cpr_fai_im.set_data(fai_img)

        self.ax_cpr.set_title(
            f"CPR — {self.vessel_name}  (rotation: {self._rotation_deg:.0f}°)",
            color="#ddddff", fontsize=10, fontweight="bold", pad=6,
        )

    def _update_mode_indicator(self) -> None:
        """Update the anchor mode indicator text."""
        if self._anchor_mode:
            n_anchors = len(self._anchor_points)
            self._mode_txt.set_text(f"ANCHOR MODE ({n_anchors} points) | Click to place, Right-click to delete")
            self._mode_txt.set_visible(True)
        else:
            self._mode_txt.set_visible(False)

    def _update_title(self) -> None:
        """Update the figure title."""
        self.fig.suptitle(
            f"PCAT CPR Browser  |  {self.vessel_name}  |  "
            f"{self.N_pts} pts  |  Rotation: {self._rotation_deg:.0f}°  |  "
            f"'s' save  'a' anchors  'r' reset  'q' quit",
            color="#ffffff", fontsize=11, fontweight="bold",
        )

    # -----------------------------------------------------------------------
    def _draw_crosssection(self, idx: int):
        """
        Render the orthogonal cross-section at centerline index `idx`.

        Samples a 2D grid on the (N, B) plane perpendicular to the tangent T
        at the centerline point, using trilinear interpolation.
        """
        idx = int(np.clip(idx, 0, len(self.cl_mm) - 1))
        self._needle_idx = idx

        # ── Compute true orthogonal cross-section (N-B plane) ─────────────
        n_cs = 256  # cross-section resolution
        offsets = np.linspace(-self.width_mm, self.width_mm, n_cs)

        # Grid in N-B plane
        nn, bb = np.meshgrid(offsets, offsets)  # (n_cs, n_cs) each

        # Get frame vectors at this index
        center = self.cl_mm[idx]       # (3,) center point in mm
        N_vec = self.N_frame[idx]      # (3,) normal direction
        B_vec = self.B_frame[idx]      # (3,) binormal direction

        # Sample positions: center + n*N + b*B
        # Shape: (n_cs, n_cs, 3)
        pts = (
            center[np.newaxis, np.newaxis, :]
            + nn[:, :, np.newaxis] * N_vec[np.newaxis, np.newaxis, :]
            + bb[:, :, np.newaxis] * B_vec[np.newaxis, np.newaxis, :]
        )

        # Sample volume using trilinear interpolation
        cs_img = _sample_volume_trilinear(self.volume, self._vox_size, pts)

        # Grayscale anatomy
        gray = np.clip(cs_img, -200.0, 400.0)
        gray = (gray + 200.0) / 600.0
        gray = np.nan_to_num(gray, nan=0.0)

        # FAI overlay
        fai = np.where(
            (cs_img >= FAI_HU_MIN) & (cs_img <= FAI_HU_MAX),
            cs_img, np.nan,
        )

        if self._xs_im is None:
            # First draw: create imshow handles
            self._xs_im = self.ax_cross.imshow(
                gray,
                aspect="equal",
                origin="upper",
                cmap="gray",
                vmin=0.0, vmax=1.0,
                interpolation="bilinear",
                extent=(
                    -self.width_mm, self.width_mm,
                    self.width_mm, -self.width_mm,
                ),
            )
            self._xs_fai_im = self.ax_cross.imshow(
                fai,
                aspect="equal",
                origin="upper",
                cmap=_fai_colormap(),
                vmin=FAI_HU_MIN, vmax=FAI_HU_MAX,
                alpha=0.80,
                interpolation="bilinear",
                extent=(
                    -self.width_mm, self.width_mm,
                    self.width_mm, -self.width_mm,
                ),
            )
            # Axis labels
            self.ax_cross.set_xlabel("N direction (mm)", color="#aaaacc", fontsize=9)
            self.ax_cross.set_ylabel("B direction (mm)", color="#aaaacc", fontsize=9)
            self.ax_cross.axhline(0, color="white", linewidth=0.6, linestyle=":", alpha=0.4)
            self.ax_cross.axvline(0, color="white", linewidth=0.6, linestyle=":", alpha=0.4)
        else:
            # Update existing imshow data
            self._xs_im.set_data(gray)
            self._xs_fai_im.set_data(fai)

        # ── Draw / update vessel lumen and VOI ring circles ──────────────
        # Map needle index back to original centerline for radius lookup
        # The CPR has n_width columns, corresponding to arc-length positions
        r_lumen_mm = float(self.radii_mm[min(idx, len(self.radii_mm) - 1)])
        r_voi_mm   = r_lumen_mm * 2.0  # VOI outer boundary = 2× vessel radius

        # Remove old patches
        if self._xs_lumen is not None:
            self._xs_lumen.remove()
        if self._xs_voi_ring is not None:
            self._xs_voi_ring.remove()

        self._xs_lumen = mpatches.Circle(
            (0, 0), radius=r_lumen_mm,
            fill=False, edgecolor="#ffffff", linewidth=2.0, linestyle="-",
            label=f"Lumen wall (r={r_lumen_mm:.1f} mm)",
            zorder=5,
        )
        self.ax_cross.add_patch(self._xs_lumen)

        self._xs_voi_ring = mpatches.Circle(
            (0, 0), radius=r_voi_mm,
            fill=False, edgecolor="#ffee00", linewidth=1.5, linestyle="--",
            label=f"VOI boundary (r={r_voi_mm:.1f} mm)",
            zorder=5,
        )
        self.ax_cross.add_patch(self._xs_voi_ring)

        # ── Arc-length annotation ─────────────────────────────────────────
        arc_mm = float(self.arclengths[idx])
        if self._xs_needle_txt is not None:
            self._xs_needle_txt.remove()
        self._xs_needle_txt = self.ax_cross.text(
            0.03, 0.97,
            f"Arc: {arc_mm:.1f} mm  |  pt {idx}/{len(self.cl_mm) - 1}  |  rot {self._rotation_deg:.0f}°",
            transform=self.ax_cross.transAxes,
            va="top", ha="left",
            fontsize=9, color="#00ffcc",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1a", alpha=0.7),
        )

        self.ax_cross.set_title(
            f"Cross-section at {arc_mm:.1f} mm  —  {self.vessel_name}",
            color="#ddddff", fontsize=10, fontweight="bold", pad=6,
        )
        self.ax_cross.set_xlim(-self.width_mm, self.width_mm)
        self.ax_cross.set_ylim(self.width_mm, -self.width_mm)

        # Update needle on CPR
        self._needle_line.set_ydata([idx, idx])

        self.fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    def _on_arclen_slider(self, val: float):
        """Called when arc-length slider value changes."""
        idx = int(np.argmin(np.abs(self.arclengths - val)))
        if idx != self._needle_idx:
            self._draw_crosssection(idx)

    # -----------------------------------------------------------------------
    def _on_rot_slider(self, val: float):
        """Called when rotation slider changes (debounced via release event)."""
        # Store the pending rotation value
        self._pending_rotation = val

    def _on_rot_slider_release(self) -> None:
        """Actually apply rotation change on mouse release."""
        if not hasattr(self, '_pending_rotation'):
            return

        new_rot = float(self._pending_rotation)
        if abs(new_rot - self._rotation_deg) < 0.5:
            return  # No significant change

        # Recompute CPR data with new rotation
        self._recompute_cpr_data(new_rot)

        # Update CPR image
        self._draw_cpr_image()

        # Update cross-section (uses new N_frame, B_frame)
        self._draw_crosssection(self._needle_idx)

        # Update title
        self._update_title()

        # Redraw anchor points
        self._redraw_anchor_markers()

    # -----------------------------------------------------------------------
    def _on_click(self, event):
        """Handle mouse click events."""
        # Check for rotation slider release
        if event.inaxes is self._slider_rot.ax:
            return  # Will be handled by release event

        # Click on CPR panel
        if event.inaxes is self.ax_cpr:
            if event.ydata is None:
                return

            # In anchor mode: place or delete anchor points
            if self._anchor_mode:
                if event.button == 1:  # Left click: place anchor
                    arc_mm = float(self.arclengths[int(np.clip(round(event.ydata), 0, len(self.arclengths) - 1))])
                    lat_mm = (event.xdata / (self.n_width - 1) - 0.5) * 2 * self.width_mm
                    self._anchor_points.append((arc_mm, lat_mm))
                    self._add_anchor_marker(event.ydata, event.xdata)
                    self._update_mode_indicator()
                    print(f"[cpr_browser] Anchor placed: arc={arc_mm:.1f}mm, lateral={lat_mm:.1f}mm")
                elif event.button == 3:  # Right click: delete nearest anchor
                    if self._anchor_points:
                        # Find nearest anchor to click position
                        click_arc = float(self.arclengths[int(np.clip(round(event.ydata), 0, len(self.arclengths) - 1))])
                        click_lat = (event.xdata / (self.n_width - 1) - 0.5) * 2 * self.width_mm

                        min_dist = float('inf')
                        min_idx = -1
                        for i, (arc, lat) in enumerate(self._anchor_points):
                            dist = np.sqrt((arc - click_arc)**2 + (lat - click_lat)**2)
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = i

                        if min_idx >= 0 and min_dist < 10.0:  # 10mm threshold
                            del self._anchor_points[min_idx]
                            self._remove_anchor_marker(min_idx)
                            self._update_mode_indicator()
                            print(f"[cpr_browser] Anchor removed at index {min_idx}")
            else:
                # Normal mode: jump needle to clicked position
                idx = int(np.clip(round(event.ydata), 0, len(self.arclengths) - 1))
                self._slider_arclen.set_val(float(self.arclengths[idx]))

    def _on_release(self, event):
        """Handle mouse release events (for rotation slider debounce)."""
        # Check if rotation slider was released
        if event.inaxes is self._slider_rot.ax:
            self._on_rot_slider_release()

    def _on_scroll(self, event):
        """Handle scroll events for rotation."""
        if event.inaxes is self.ax_cpr or event.inaxes is self.ax_cross:
            # Scroll changes rotation
            delta = 5.0 if event.button == "up" else -5.0
            new_rot = (self._rotation_deg + delta) % 360.0
            self._slider_rot.set_val(new_rot)

    # -----------------------------------------------------------------------
    def _add_anchor_marker(self, row: float, col: float) -> None:
        """Add a visual marker for an anchor point on the CPR."""
        marker, = self.ax_cpr.plot(
            col, row, 'o',
            color="#ff00ff",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            alpha=0.9,
            zorder=10,
        )
        self._anchor_markers.append(marker)

    def _remove_anchor_marker(self, idx: int) -> None:
        """Remove an anchor marker by index."""
        if 0 <= idx < len(self._anchor_markers):
            self._anchor_markers[idx].remove()
            del self._anchor_markers[idx]

    def _redraw_anchor_markers(self) -> None:
        """Redraw all anchor markers after rotation change."""
        # Remove old markers
        for marker in self._anchor_markers:
            marker.remove()
        self._anchor_markers.clear()

        # Recalculate positions and redraw
        for arc_mm, lat_mm in self._anchor_points:
            # Find row index from arc-length
            row_idx = int(np.argmin(np.abs(self.arclengths - arc_mm)))
            # Find column index from lateral offset
            col_idx = int((lat_mm / (2 * self.width_mm) + 0.5) * (self.n_width - 1))
            col_idx = int(np.clip(col_idx, 0, self.n_width - 1))

            marker, = self.ax_cpr.plot(
                col_idx, row_idx, 'o',
                color="#ff00ff",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=2,
                alpha=0.9,
                zorder=10,
            )
            self._anchor_markers.append(marker)

    # -----------------------------------------------------------------------
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key in ("right", "up"):
            new_idx = min(self._needle_idx + 1, len(self.cl_mm) - 1)
            self._slider_arclen.set_val(float(self.arclengths[new_idx]))
        elif event.key in ("left", "down"):
            new_idx = max(self._needle_idx - 1, 0)
            self._slider_arclen.set_val(float(self.arclengths[new_idx]))
        elif event.key == "s":
            self._save_snapshot()
        elif event.key == "a":
            # Toggle anchor mode
            self._anchor_mode = not self._anchor_mode
            self._update_mode_indicator()
            self.fig.canvas.draw_idle()
            mode_str = "ON" if self._anchor_mode else "OFF"
            print(f"[cpr_browser] Anchor mode: {mode_str}")
        elif event.key == "p":
            # Apply anchors (print data for now)
            self._apply_anchors()
        elif event.key == "r":
            # Reset rotation to 0
            self._slider_rot.set_val(0.0)
            self._on_rot_slider_release()
        elif event.key == "q":
            plt.close(self.fig)

    # -----------------------------------------------------------------------
    def _apply_anchors(self) -> None:
        """Apply anchor points (UI scaffolding - prints data for now)."""
        if not self._anchor_points:
            print("[cpr_browser] No anchor points to apply.")
            return

        print(f"[cpr_browser] === ANCHOR POINT DATA ({len(self._anchor_points)} points) ===")
        for i, (arc_mm, lat_mm) in enumerate(self._anchor_points):
            print(f"  Anchor {i+1}: arc-length={arc_mm:.2f}mm, lateral_offset={lat_mm:.2f}mm")

        # Future: adjust centerline and recompute CPR
        print("[cpr_browser] (Centerline adjustment not yet implemented)")
        print("[cpr_browser] ================================================")

    # -----------------------------------------------------------------------
    def _save_snapshot(self):
        """Save current view as PNG."""
        arc_mm = float(self.arclengths[self._needle_idx])
        fname  = self.output_dir / f"cpr_browser_{self.vessel_name}_{arc_mm:.0f}mm_rot{self._rotation_deg:.0f}.png"
        self.fig.savefig(str(fname), dpi=150, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        print(f"[cpr_browser] Snapshot saved: {fname.name}")

    # -----------------------------------------------------------------------
    def show(self):
        """Display the interactive browser window (blocks until closed)."""
        # Rebuild legend for cross-section
        self.ax_cross.legend(
            handles=[self._xs_lumen, self._xs_voi_ring],
            loc="lower right",
            fontsize=8,
            facecolor="#0d0d1a",
            edgecolor="#3a3a5a",
            labelcolor="#aaaacc",
        )
        plt.show()


# ---------------------------------------------------------------------------
# Public launcher (called from run_pipeline.py)
# ---------------------------------------------------------------------------

def launch_cpr_browser(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str = "vessel",
    voi_mask: Optional[np.ndarray] = None,
    width_mm: float = 25.0,
    slab_thickness_mm: float = 3.0,
    output_dir: str | Path = ".",
    rotation_deg: float = 0.0,
) -> None:
    """
    Launch the interactive CPR browser and block until the window is closed.

    This function switches matplotlib to an interactive backend. Call ONLY
    from an interactive (non-headless) context.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) centerline voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii in mm
    spacing_mm      : [sz, sy, sx]
    vessel_name     : display label
    voi_mask        : (Z, Y, X) bool optional — PCAT VOI mask
    width_mm        : half-width of CPR in mm
    slab_thickness_mm : MIP slab thickness in mm
    output_dir      : directory for snapshots
    rotation_deg    : initial rotation angle in degrees
    """
    browser = CPRBrowser(
        volume=volume,
        centerline_ijk=centerline_ijk,
        radii_mm=radii_mm,
        spacing_mm=spacing_mm,
        vessel_name=vessel_name,
        voi_mask=voi_mask,
        width_mm=width_mm,
        slab_thickness_mm=slab_thickness_mm,
        output_dir=output_dir,
        rotation_deg=rotation_deg,
    )
    browser.show()


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive CPR browser with live cross-section panel.\n\n"
            "Displays the straightened CPR for a vessel with a movable needle that "
            "shows the orthogonal cross-section (with lumen and VOI ring) at any "
            "arc-length position along the vessel. Supports rotational CPR via "
            "the rotation slider."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline/cpr_browser.py \\\n"
            "      --dicom Rahaf_Patients/1200.2 \\\n"
            "      --seeds seeds/patient_1200.json \\\n"
            "      --vessel LAD\n\n"
            "  python pipeline/cpr_browser.py \\\n"
            "      --dicom Rahaf_Patients/1200.2 \\\n"
            "      --seeds seeds/patient_1200.json \\\n"
            "      --vessel RCA \\\n"
            "      --voi output/patient_1200/patient1200_RCA_voi_reviewed.npy\n\n"
            "Controls:\n"
            "  Slider (Arc-length) — Drag to move needle\n"
            "  Slider (Rotation)   — Drag to rotate cutting plane (0-360)\n"
            "  Click CPR           — Jump to position (or place anchor in anchor mode)\n"
            "  ←/→ arrows          — Step needle by one point\n"
            "  s                   — Save snapshot PNG\n"
            "  a                   — Toggle anchor point mode\n"
            "  p                   — Apply anchor points (print data)\n"
            "  r                   — Reset rotation to 0\n"
            "  q                   — Quit\n"
        ),
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory",
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Path to seed JSON file (from seed_picker or auto_seeds)",
    )
    parser.add_argument(
        "--vessel", required=True, choices=["LAD", "LCX", "RCA"],
        help="Vessel to display",
    )
    parser.add_argument(
        "--voi", type=str, default=None,
        help=(
            "Path to .npy VOI mask (bool, Z×Y×X). Optional — if provided, "
            "the VOI ring is drawn from the actual reviewed mask."
        ),
    )
    parser.add_argument(
        "--width", type=float, default=25.0,
        dest="width_mm",
        help="Half-width of CPR / cross-section plane in mm (default: 25)",
    )
    parser.add_argument(
        "--slab", type=float, default=3.0,
        dest="slab_mm",
        help="MIP slab thickness for CPR in mm (default: 3)",
    )
    parser.add_argument(
        "--rotation", type=float, default=0.0,
        dest="rotation_deg",
        help="Initial rotation angle in degrees (default: 0)",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Directory for saving snapshot PNGs (default: current dir)",
    )
    args = parser.parse_args()

    # ── Load DICOM ───────────────────────────────────────────────────────────
    print(f"[cpr_browser] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[cpr_browser] Volume shape: {volume.shape}  spacing: {spacing_mm}")

    # ── Load seeds ───────────────────────────────────────────────────────────
    print(f"[cpr_browser] Loading seeds from {args.seeds} ...")
    seeds_data = load_seeds(args.seeds)

    vessel_name = args.vessel
    if vessel_name not in seeds_data:
        print(f"[cpr_browser] ERROR: vessel '{vessel_name}' not found in seeds file.")
        print(f"[cpr_browser] Available: {list(seeds_data.keys())}")
        sys.exit(1)

    vsd         = seeds_data[vessel_name]
    ostium_ijk  = vsd["ostium_ijk"]
    waypoints   = vsd.get("waypoints_ijk", [])

    if not ostium_ijk or any(v is None for v in ostium_ijk):
        print(f"[cpr_browser] ERROR: '{vessel_name}' has null/missing ostium seed.")
        sys.exit(1)

    seg_start  = float(VESSEL_CONFIGS[vessel_name].get("start_mm",  0.0))
    seg_length = float(VESSEL_CONFIGS[vessel_name].get("length_mm", 40.0))

    # ── Vesselness + centerline ──────────────────────────────────────────────
    all_pts = [ostium_ijk] + [wp for wp in waypoints if wp and all(v is not None for v in wp)]
    print("[cpr_browser] Computing Frangi vesselness (ROI-cropped) ...")
    vesselness = compute_vesselness(
        volume, spacing_mm,
        seed_points=all_pts,
        roi_margin_mm=20.0,
    )
    print("[cpr_browser] Extracting centerline ...")
    cl_full = extract_centerline_seeds(
        volume=volume,
        vesselness=vesselness,
        spacing_mm=spacing_mm,
        ostium_ijk=ostium_ijk,
        waypoints_ijk=waypoints,
        roi_radius_mm=35.0,
    )
    centerline = clip_centerline_by_arclength(
        cl_full, spacing_mm,
        start_mm=seg_start,
        length_mm=seg_length,
    )
    print(f"[cpr_browser] Centerline: {len(centerline)} points ({seg_start}-{seg_start+seg_length} mm)")

    radii_mm = estimate_vessel_radii(volume, centerline, spacing_mm)

    # ── Optional VOI mask ────────────────────────────────────────────────────
    voi_mask = None
    if args.voi:
        voi_path = Path(args.voi)
        if voi_path.exists():
            voi_mask = np.load(str(voi_path))
            print(f"[cpr_browser] VOI mask loaded: {voi_path.name}  ({voi_mask.sum():,} voxels)")
        else:
            print(f"[cpr_browser] WARNING: --voi path not found: {voi_path}  (continuing without VOI)")

    # ── Launch browser ───────────────────────────────────────────────────────
    launch_cpr_browser(
        volume=volume,
        centerline_ijk=centerline,
        radii_mm=radii_mm,
        spacing_mm=spacing_mm,
        vessel_name=vessel_name,
        voi_mask=voi_mask,
        width_mm=args.width_mm,
        slab_thickness_mm=args.slab_mm,
        output_dir=args.output,
        rotation_deg=args.rotation_deg,
    )


if __name__ == "__main__":
    main()
