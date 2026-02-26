"""
seed_picker.py
Interactive matplotlib tool for manually selecting coronary artery seed points.

Usage:
    python seed_picker.py --dicom /path/to/patient/dir --output seeds/patient_X.json

Interface:
  - Shows three orthogonal slices (axial, coronal, sagittal) side by side
  - Scroll wheel changes the displayed slice
  - Click on the vessel lumen → adds a seed point
  - Seeds are categorized as: ostium (first click per vessel) or waypoints (subsequent)
  - Supports LAD, LCX, RCA — switch vessels with number keys
  - Press 's' to save, 'u' to undo last point, 'r' to reset current vessel

Output JSON format:
{
  "LAD": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z, y, x], ...],
    "segment_length_mm": 40.0
  },
  "LCX": { ... },
  "RCA": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [[z, y, x], ...],
    "segment_start_mm": 10.0,
    "segment_length_mm": 40.0
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, RadioButtons

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


class SeedPicker:
    """
    Interactive 3-plane seed picker for coronary artery seeds.

    Displays axial/coronal/sagittal MPR views and lets the user click to add seeds.
    """

    def __init__(self, volume: np.ndarray, spacing_mm: List[float], output_path: str | Path):
        self.volume = volume
        self.spacing_mm = spacing_mm  # [sz, sy, sx]
        self.output_path = Path(output_path)
        self.shape = volume.shape  # (Z, Y, X)

        # Current view state
        self.z_slice = self.shape[0] // 2
        self.y_slice = self.shape[1] // 2
        self.x_slice = self.shape[2] // 2

        # Seeds: {vessel: {"ostium": [z,y,x] or None, "waypoints": [[z,y,x], ...]}}
        self.seeds: Dict[str, Dict] = {
            v: {"ostium": None, "waypoints": []}
            for v in VESSEL_KEYS
        }
        self.current_vessel = "LAD"
        self.active_view = "axial"  # which view was last clicked

        # Window/level
        self.ww = 600   # window width
        self.wl = 50    # window level (center)

        self._build_figure()
        self._connect_events()

    # ─────────────────────────────────────────
    # Figure construction
    # ─────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(
            "PCAT Seed Picker — Click to add seeds | Scroll to change slice\n"
            "Keys: 1=LAD  2=LCX  3=RCA  |  u=Undo  r=Reset vessel  s=Save  q=Quit",
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
        self.ax_info.set_title("Seeds", fontsize=9)
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

    def _update_status_bar(self):
        self.ax_status.cla()
        self.ax_status.axis("off")
        v = self.current_vessel
        sd = self.seeds[v]
        n_wp = len(sd["waypoints"])
        has_ostium = sd["ostium"] is not None
        msg = (
            f"  Active vessel: {v} (color: {VESSEL_COLORS[v]})  |  "
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
        lines = ["Seeds summary:", ""]
        for v in VESSEL_KEYS:
            sd = self.seeds[v]
            c = VESSEL_COLORS[v]
            has_o = sd["ostium"] is not None
            n_w = len(sd["waypoints"])
            status = "✓" if has_o else "✗"
            lines.append(f"[{status}] {v}")
            if has_o:
                z, y, x = sd["ostium"]
                lines.append(f"   ostium: ({z},{y},{x})")
            else:
                lines.append(f"   ostium: —")
            lines.append(f"   waypoints: {n_w}")
            lines.append("")

        self.ax_info.text(
            0.05, 0.95, "\n".join(lines),
            ha="left", va="top",
            transform=self.ax_info.transAxes,
            fontsize=8,
            family="monospace",
        )

    # ─────────────────────────────────────────
    # Events
    # ─────────────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
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
            # FIX: Un-flip Z coordinate
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, self.y_slice, ix
            self.active_view = "coronal"
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        elif ax == self.ax_sagittal:
            # FIX: Un-flip Z coordinate
            actual_z = self.shape[0] - 1 - iy
            z, y, x = actual_z, ix, self.x_slice
            self.active_view = "sagittal"
            self.z_slice = max(0, min(z, self.shape[0] - 1))
        else:
            return

        # Clamp to volume
        z = int(np.clip(z, 0, self.shape[0] - 1))
        y = int(np.clip(y, 0, self.shape[1] - 1))
        x = int(np.clip(x, 0, self.shape[2] - 1))

        # Update crosshair cursor
        self.y_slice = y
        self.x_slice = x

        # Add seed point
        sd = self.seeds[self.current_vessel]
        if sd["ostium"] is None:
            sd["ostium"] = [z, y, x]
            print(f"[seed_picker] {self.current_vessel} ostium set: ({z}, {y}, {x})")
        else:
            sd["waypoints"].append([z, y, x])
            print(f"[seed_picker] {self.current_vessel} waypoint {len(sd['waypoints'])}: ({z}, {y}, {x})")

        self._refresh_images()
        self._update_info_panel()
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
            print("[seed_picker] Switched to LAD")
        elif key == "2":
            self.current_vessel = "LCX"
            print("[seed_picker] Switched to LCX")
        elif key == "3":
            self.current_vessel = "RCA"
            print("[seed_picker] Switched to RCA")
        elif key == "u":
            # Undo last point
            sd = self.seeds[self.current_vessel]
            if sd["waypoints"]:
                removed = sd["waypoints"].pop()
                print(f"[seed_picker] Undid waypoint: {removed}")
            elif sd["ostium"] is not None:
                removed = sd["ostium"]
                sd["ostium"] = None
                print(f"[seed_picker] Undid ostium: {removed}")
        elif key == "r":
            # Reset current vessel
            self.seeds[self.current_vessel] = {"ostium": None, "waypoints": []}
            print(f"[seed_picker] Reset {self.current_vessel}")
        elif key == "s":
            self._save()
            return
        elif key == "q":
            print("[seed_picker] Quitting without saving.")
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
                print(f"[seed_picker] WARNING: {v} has no ostium — skipping in output")
                continue
            entry = {
                "ostium_ijk": sd["ostium"],
                "waypoints_ijk": sd["waypoints"],
            }
            entry.update(VESSEL_CONFIGS[v])
            output[v] = entry

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[seed_picker] Seeds saved to {self.output_path}")
        plt.title(f"Saved to {self.output_path.name}", fontsize=9)
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n=== PCAT Seed Picker ===")
        print("Keys:  1=LAD  2=LCX  3=RCA  |  u=Undo  r=Reset  s=Save  q=Quit")
        print("       w/W=wider/narrower window  l/L=brighter/darker level")
        print("Click on the vessel ostium first, then add waypoints along the proximal segment.\n")
        plt.show()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    epilog = """
HOW TO USE
----------
This tool opens a 3-panel MPR viewer (Axial / Coronal / Sagittal) for each patient.
You identify the start and course of each coronary artery by clicking directly on
the vessel lumen.

STEP-BY-STEP
  1. The window opens showing a mid-volume axial slice.
     Scroll the mouse wheel over any panel to navigate through slices.

  2. Locate the LEFT MAIN / LAD ostium (origin of the LAD off the left coronary sinus).
     Press 1 to select LAD, then CLICK on the ostium in the axial or coronal view.
     This sets the orange square marker -- the pipeline starts tracking here.

  3. Continue clicking 3-5 waypoints along the proximal 40 mm of the LAD
     (scroll down through slices, click where you see the vessel centre).
     Waypoints guide the centerline extraction -- more = more accurate.

  4. Press 2 to switch to LCX. Repeat the same process:
     click the LCX ostium (off the left main, turning posteriorly), then add waypoints
     along the proximal 40 mm.

  5. Press 3 to switch to RCA. Click the RCA ostium (right coronary sinus),
     then add waypoints covering at least the 10-50 mm proximal segment.

  6. Press S to save the seed file.  All three vessels are written to the JSON.
     The pipeline uses these coordinates to extract centerlines and build the PCAT VOI.

KEYBOARD SHORTCUTS
  1 / 2 / 3   Switch active vessel (LAD / LCX / RCA)
  u           Undo last point (waypoints first, then ostium)
  r           Reset all points for the current vessel
  s           Save seeds to JSON and continue
  q           Quit without saving
  w / W       Window width wider / narrower  (adjust contrast)
  l / L       Window level brighter / darker (adjust brightness)

TIPS
  - Seed in soft-tissue window (W=600 L=50 -- the default).  The vessel lumen
    appears dark (near 0 HU); aim for the centre of the dark tube.
  - Use the coronal view to quickly find the ostia -- the aortic root is visible
    as a bright ring at the top of the heart.
  - You only need the OSTIUM and a few WAYPOINTS per vessel.
    The centerline algorithm fills in the rest automatically.
  - If you make a mistake, press u to undo or r to start the vessel over.
  - Markers are colour-coded: LAD=orange, LCX=blue, RCA=green.
    Square = ostium, circle = waypoint.
"""
    parser = argparse.ArgumentParser(
        description="Interactive seed point picker for PCAT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory for one patient"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for seed JSON file (e.g. seeds/patient1200.json)"
    )
    args = parser.parse_args()

    print(f"[seed_picker] Loading DICOM from {args.dicom} ...")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[seed_picker] Volume shape: {volume.shape}, spacing: {spacing_mm}")
    print(f"[seed_picker] HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    picker = SeedPicker(volume, spacing_mm, output_path=args.output)
    picker.run()


if __name__ == "__main__":
    main()
