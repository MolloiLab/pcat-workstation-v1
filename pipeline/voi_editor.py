"""
voi_editor.py
Interactive matplotlib GUI for post-segmentation manual review and editing of PCAT VOI masks.

Usage:
    python voi_editor.py --dicom /path/to/dicom/dir --voi voi_mask.npy --vessel LAD --output edited_voi.npy

Interface:
  - Shows three orthogonal slices (axial, coronal, sagittal) with VOI mask overlay
  - Left-click drag: paint voxels as VOI=True (add mode)
  - Right-click drag: paint voxels as VOI=False (remove mode)
  - Scroll wheel changes the displayed slice
  - Keys: +/- (brush size), u/ctrl+z (undo), s (save), q (quit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dicom_loader import load_dicom_series


class VOIEditor:
    """
    Interactive 3-plane VOI mask editor for PCAT pipeline.

    Displays axial/coronal/sagittal MPR views with VOI mask overlay and allows
    interactive painting to add/remove voxels from the VOI mask.
    """

    def __init__(self, volume: np.ndarray, voi_mask: np.ndarray, spacing_mm: List[float], 
                 vessel_name: str, output_path: str | Path):
        self.volume = volume
        self.voi_mask = voi_mask.astype(bool)
        self.original_voi_mask = voi_mask.astype(bool).copy()  # Keep original for reference
        self.spacing_mm = spacing_mm  # [sz, sy, sx]
        self.vessel_name = vessel_name
        self.output_path = Path(output_path)
        self.shape = volume.shape  # (Z, Y, X)

        # Current view state
        self.z_slice = self.shape[0] // 2
        self.y_slice = self.shape[1] // 2
        self.x_slice = self.shape[2] // 2

        # Brush settings
        self.brush_radius = 2
        self.painting = False
        self.paint_mode = "add"  # "add" or "remove"
        
        # Undo stack - store up to 20 previous mask states
        self.undo_stack = deque(maxlen=20)

        # Window/level
        self.ww = 600   # window width
        self.wl = 50    # window level (center)

        # First launch flag for sanity check warning
        self.first_launch = True

        self._build_figure()
        self._connect_events()

    # ─────────────────────────────────────────
    # Figure construction
    # ─────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(
            "PCAT VOI Editor — Left-click to ADD, Right-click to REMOVE | Scroll to change slice\n"
            "Keys: +/- (brush size)  u/ctrl+z (undo)  s (save)  q (quit)",
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
        self.ax_info.set_title("VOI Info", fontsize=9)
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

        # Initialize VOI mask overlays
        self.overlay_axial = self.ax_axial.imshow(
            self._axial_voi_overlay(), aspect="equal", origin="upper"
        )
        self.overlay_coronal = self.ax_coronal.imshow(
            self._coronal_voi_overlay(), aspect="auto", origin="upper"
        )
        self.overlay_sagittal = self.ax_sagittal.imshow(
            self._sagittal_voi_overlay(), aspect="auto", origin="upper"
        )

        # Crosshair lines
        self._draw_crosshairs()

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
    # VOI overlay creation
    # ─────────────────────────────────────────

    def _axial_voi_overlay(self):
        """Create RGBA overlay for axial slice showing VOI mask in yellow."""
        overlay = np.zeros((self.shape[1], self.shape[2], 4), dtype=np.float32)
        voi_slice = self.voi_mask[self.z_slice, :, :]
        # Yellow color with alpha=0.35
        overlay[voi_slice] = [1.0, 0.843, 0.0, 0.35]  # #FFD700 with alpha
        return overlay

    def _coronal_voi_overlay(self):
        """Create RGBA overlay for coronal slice showing VOI mask in yellow."""
        overlay = np.zeros((self.shape[0], self.shape[2], 4), dtype=np.float32)
        voi_slice = self.voi_mask[:, self.y_slice, :]
        # Yellow color with alpha=0.35
        overlay[voi_slice] = [1.0, 0.843, 0.0, 0.35]  # #FFD700 with alpha
        return overlay

    def _sagittal_voi_overlay(self):
        """Create RGBA overlay for sagittal slice showing VOI mask in yellow."""
        overlay = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.float32)
        voi_slice = self.voi_mask[:, :, self.x_slice]
        # Yellow color with alpha=0.35
        overlay[voi_slice] = [1.0, 0.843, 0.0, 0.35]  # #FFD700 with alpha
        return overlay

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
        self._ch_co = [
            self.ax_coronal.axhline(self.z_slice, **kw),
            self.ax_coronal.axvline(self.x_slice, **kw),
        ]
        self._ch_sa = [
            self.ax_sagittal.axhline(self.z_slice, **kw),
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
        
        # Update VOI overlays
        self.overlay_axial.set_data(self._axial_voi_overlay())
        self.overlay_coronal.set_data(self._coronal_voi_overlay())
        self.overlay_sagittal.set_data(self._sagittal_voi_overlay())
        
        self._draw_crosshairs()
        self._update_status_bar()
        self.fig.canvas.draw_idle()

    def _update_status_bar(self):
        self.ax_status.cla()
        self.ax_status.axis("off")
        mode = "ADD" if self.paint_mode == "add" else "REMOVE"
        msg = (
            f"  Slice Z={self.z_slice} Y={self.y_slice} X={self.x_slice}  |  "
            f"Brush radius: {self.brush_radius}  |  "
            f"Mode: {mode}  |  "
            f"VOI voxels: {self.voi_mask.sum()}  |  "
            f"W/L: {self.ww}/{self.wl}"
        )
        color = "green" if self.paint_mode == "add" else "red"
        self.ax_status.text(
            0.01, 0.5, msg,
            ha="left", va="center",
            transform=self.ax_status.transAxes,
            fontsize=9,
            color=color,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=3),
        )

    def _update_info_panel(self):
        self.ax_info.cla()
        self.ax_info.axis("off")
        
        # Count voxels
        total_voi_voxels = int(self.voi_mask.sum())
        original_voi_voxels = int(self.original_voi_mask.sum())
        diff = total_voi_voxels - original_voi_voxels
        
        # Count fat voxels in VOI (if in FAI range)
        fat_mask = (self.volume >= -190) & (self.volume <= -30)
        fat_voi_voxels = int(np.logical_and(self.voi_mask, fat_mask).sum())
        
        lines = [
            f"Vessel: {self.vessel_name}",
            "",
            f"VOI voxels: {total_voi_voxels}",
            f"  Original: {original_voi_voxels}",
            f"  Change: {'+' if diff >= 0 else ''}{diff}",
            "",
            f"Fat voxels: {fat_voi_voxels}",
            "",
            f"Brush radius: {self.brush_radius}",
        ]
        
        # Add warning on first launch
        if self.first_launch:
            lines.extend([
                "",
                "",
                "⚠️  MANDATORY SANITY CHECK ⚠️",
                "",
                "Review VOI boundaries",
                "before saving.",
                "Add/remove voxels as needed.",
            ])
            self.first_launch = False

        self.ax_info.text(
            0.05, 0.95, "\n".join(lines),
            ha="left", va="top",
            transform=self.ax_info.transAxes,
            fontsize=8,
            family="monospace",
        )

    # ─────────────────────────────────────────
    # Painting operations
    # ─────────────────────────────────────────

    def _paint_voxels(self, z, y, x):
        """Paint voxels within brush radius at given coordinate."""
        # Create spherical brush
        Z, Y, X = self.shape
        
        # Generate grid of coordinates within brush radius
        zz, yy, xx = np.meshgrid(
            np.arange(max(0, z - self.brush_radius), min(Z, z + self.brush_radius + 1)),
            np.arange(max(0, y - self.brush_radius), min(Y, y + self.brush_radius + 1)),
            np.arange(max(0, x - self.brush_radius), min(X, x + self.brush_radius + 1)),
            indexing='ij'
        )
        
        # Calculate distances
        dist_sq = (zz - z)**2 + (yy - y)**2 + (xx - x)**2
        mask = dist_sq <= self.brush_radius**2
        
        # Apply paint to mask
        if self.paint_mode == "add":
            self.voi_mask[zz[mask], yy[mask], xx[mask]] = True
        else:  # remove mode
            self.voi_mask[zz[mask], yy[mask], xx[mask]] = False

    def _save_to_undo_stack(self):
        """Save current VOI mask state to undo stack."""
        self.undo_stack.append(self.voi_mask.copy())

    def _undo(self):
        """Restore previous VOI mask state from undo stack."""
        if self.undo_stack:
            self.voi_mask = self.undo_stack.pop()
            self._refresh_images()
            self._update_info_panel()
            print("[voi_editor] Undo: restored previous VOI mask state")

    # ─────────────────────────────────────────
    # Events
    # ─────────────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_button_press(self, event):
        if event.inaxes is None:
            return

        if event.button == 1:  # Left click
            self.paint_mode = "add"
            self.painting = True
        elif event.button == 3:  # Right click
            self.paint_mode = "remove"
            self.painting = True
        else:
            return

        # Save state for undo
        self._save_to_undo_stack()
        
        # Apply paint at initial position
        self._apply_paint_at_position(event)

    def _on_button_release(self, event):
        self.painting = False

    def _on_motion(self, event):
        if not self.painting or event.inaxes is None:
            return
        
        self._apply_paint_at_position(event)

    def _apply_paint_at_position(self, event):
        """Apply paint at the current mouse position."""
        ax = event.inaxes
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        # Determine which view was clicked and derive the (z, y, x) coordinate
        if ax == self.ax_axial:
            z, y, x = self.z_slice, iy, ix
        elif ax == self.ax_coronal:
            # FIX: Un-flip Z coordinate for actual position
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, self.y_slice, ix
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        elif ax == self.ax_sagittal:
            # FIX: Un-flip Z coordinate for actual position
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, iy, self.x_slice
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        else:
            return

        # Clamp to volume
        z = int(np.clip(z, 0, self.shape[0] - 1))
        y = int(np.clip(y, 0, self.shape[1] - 1))
        x = int(np.clip(x, 0, self.shape[2] - 1))

        # Update crosshair position
        self.y_slice = y
        self.x_slice = x

        # Apply paint
        self._paint_voxels(z, y, x)
        
        # Refresh display
        self._refresh_images()
        self._update_info_panel()

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

        if key == "u" or (key == "ctrl+z"):
            self._undo()
        elif key == "+":
            self.brush_radius = min(self.brush_radius + 1, 10)
            print(f"[voi_editor] Brush radius increased to {self.brush_radius}")
            self._refresh_images()
            self._update_info_panel()
        elif key == "-":
            self.brush_radius = max(self.brush_radius - 1, 1)
            print(f"[voi_editor] Brush radius decreased to {self.brush_radius}")
            self._refresh_images()
            self._update_info_panel()
        elif key == "s":
            self._save()
            return
        elif key == "q":
            print("[voi_editor] Quitting without saving.")
            plt.close(self.fig)
            return
        elif key == "w":
            self.ww = min(self.ww + 50, 3000)
            self._refresh_images()
        elif key == "W":
            self.ww = max(self.ww - 50, 50)
            self._refresh_images()
        elif key == "l":
            self.wl += 20
            self._refresh_images()
        elif key == "L":
            self.wl -= 20
            self._refresh_images()

        self._update_status_bar()
        self.fig.canvas.draw_idle()

    def _save(self):
        """Save the edited VOI mask to the specified output path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, self.voi_mask)
        
        # Also try to save as NIfTI if nibabel is available
        try:
            import nibabel as nib
            
            # Create NIfTI image with proper orientation
            affine = np.diag([-self.spacing_mm[2], -self.spacing_mm[1], self.spacing_mm[0], 1])
            nii = nib.Nifti1Image(self.voi_mask.astype(np.uint8), affine)
            
            nii_path = self.output_path.with_suffix('.nii.gz')
            nib.save(nii, nii_path)
            print(f"[voi_editor] VOI mask saved to {self.output_path} and {nii_path}")
        except ImportError:
            print(f"[voi_editor] VOI mask saved to {self.output_path} (nibabel not available for NIfTI)")
        
        self.fig.suptitle(f"SAVED to {self.output_path.name}", fontsize=10, color="green")
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n=== PCAT VOI Editor ===")
        print("Mouse: Left-click drag to ADD voxels, Right-click drag to REMOVE voxels")
        print("Keys:  +/- (brush size)  u/ctrl+z (undo)  s (save)  q (quit)")
        print("      w/W (window width)  l/L (window level)")
        print("\n⚠️  MANDATORY SANITY CHECK: Review VOI boundaries before saving\n")
        plt.show()


# ─────────────────────────────────────────────
# Pipeline integration helper
# ─────────────────────────────────────────────


def launch_voi_editor(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    vessel_name: str,
    output_path,
    spacing_mm,
) -> np.ndarray:
    """
    Launch the interactive VOI editor for mandatory clinical review.

    Called from run_pipeline.py after build_tubular_voi().
    Blocks until the user closes the editor window.
    Returns the (possibly edited) VOI mask as a (Z, Y, X) bool array.

    Parameters
    ----------
    volume      : (Z, Y, X) float32 HU volume
    voi_mask    : (Z, Y, X) bool  initial VOI from build_tubular_voi()
    vessel_name : label shown in GUI title/info panel
    output_path : path where the reviewed mask will be auto-saved as .npy
    spacing_mm  : [sz, sy, sx] voxel spacing in mm
    """
    editor = VOIEditor(
        volume=volume,
        voi_mask=voi_mask,
        spacing_mm=spacing_mm,
        vessel_name=vessel_name,
        output_path=output_path,
    )
    editor.run()  # blocks until window is closed
    return editor.voi_mask


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive VOI mask editor for PCAT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory"
    )
    parser.add_argument(
        "--voi", required=True,
        help="Path to .npy VOI mask file"
    )
    parser.add_argument(
        "--vessel", default="VOI",
        help="Vessel name label"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for edited .npy mask"
    )
    args = parser.parse_args()

    print(f"[voi_editor] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[voi_editor] Volume shape: {volume.shape}, spacing: {spacing_mm}")
    print(f"[voi_editor] HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    print(f"[voi_editor] Loading VOI mask from {args.voi} ...")
    voi_mask = np.load(args.voi)
    print(f"[voi_editor] VOI mask shape: {voi_mask.shape}, voxels: {voi_mask.sum()}")

    editor = VOIEditor(
        volume=volume,
        voi_mask=voi_mask,
        spacing_mm=spacing_mm,
        vessel_name=args.vessel,
        output_path=args.output
    )
    editor.run()


if __name__ == "__main__":
    main()