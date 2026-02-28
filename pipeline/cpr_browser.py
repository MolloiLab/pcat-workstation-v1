"""
cpr_browser.py
Interactive CPR (Curved Planar Reformation) browser with live cross-section panel.

Shows two linked panels:
  Left  — Straightened CPR (vessel top→bottom, ostium at top).
           A movable horizontal needle line indicates the current position.
  Right — Orthogonal cross-section at the needle position:
           grayscale anatomy with vessel lumen circle (white) and PCAT VOI
           ring (semi-transparent yellow) overlaid.

Usage (standalone):
    python pipeline/cpr_browser.py \\
        --dicom  Rahaf_Patients/1200.2 \\
        --seeds  seeds/patient_1200.json \\
        --vessel LAD

    # Or after pipeline has already saved a reviewed VOI:
    python pipeline/cpr_browser.py \\
        --dicom  Rahaf_Patients/1200.2 \\
        --seeds  seeds/patient_1200.json \\
        --vessel LAD \\
        --voi    output/patient_1200/patient1200_LAD_voi_reviewed.npy

Controls:
    Slider      — Drag the arc-length slider to move the needle
    Click CPR   — Click directly on the CPR panel to jump the needle
    ← / →       — Arrow keys: step needle by one centerline point
    s           — Save a PNG snapshot of the current view to the output dir
    q           — Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Interactive backend — MUST be set before pyplot is imported.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from scipy.ndimage import map_coordinates

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
from pipeline.visualize import _compute_cpr_data, _fai_colormap, FAI_HU_MIN, FAI_HU_MAX


# ---------------------------------------------------------------------------
# Core interactive class
# ---------------------------------------------------------------------------

class CPRBrowser:
    """
    Interactive CPR browser.

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

        print(f"[cpr_browser] Computing CPR data for {vessel_name}  "
              f"({self.N_pts} centerline points)…")
        (
            self.cpr_volume,    # (N_pts, n_height, n_width)
            self.N_frame,       # (N_pts, 3) Bishop normal
            self.B_frame,       # (N_pts, 3) Bishop binormal
            self.cl_mm,         # (N_pts, 3) centerline in mm
            self.arclengths,    # (N_pts,) cumulative arc-length in mm
            self.n_height,
            self.n_width,
        ) = _compute_cpr_data(
            volume, centerline_ijk, spacing_mm,
            slab_thickness_mm=slab_thickness_mm, width_mm=width_mm,
        )
        print(f"[cpr_browser] CPR volume shape: {self.cpr_volume.shape}")

        self._build_gui()

    # -----------------------------------------------------------------------
    def _build_gui(self):
        """Build the matplotlib figure with CPR + cross-section panels."""
        fig = plt.figure(figsize=(16, 12), facecolor="#1a1a2e")
        fig.canvas.manager.set_window_title(
            f"PCAT CPR Browser — {self.vessel_name}"
        )
        self.fig = fig

        # Layout: CPR left (narrower) | cross-section right (square)
        # Reserve bottom strip for slider
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[20, 1],
            width_ratios=[5, 5],
            left=0.07, right=0.97,
            bottom=0.08, top=0.93,
            hspace=0.12, wspace=0.22,
        )
        self.ax_cpr   = fig.add_subplot(gs[0, 0])
        self.ax_cross = fig.add_subplot(gs[0, 1])
        ax_slider     = fig.add_subplot(gs[1, :])

        # ── Style axes ───────────────────────────────────────────────────
        for ax in (self.ax_cpr, self.ax_cross):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5a")

        # ── Draw CPR (static — does not change with needle) ──────────────
        cpr_image = self.cpr_volume.T  # (n_height, n_width) — transpose of (pixels_wide, pixels_high)
        gray_img  = np.clip(cpr_image, -200.0, 400.0)
        gray_norm = (gray_img + 200.0) / 600.0
        gray_norm = np.nan_to_num(gray_norm, nan=0.0)

        self.ax_cpr.imshow(
            gray_norm,
            aspect="auto",
            origin="upper",
            cmap="gray",
            vmin=0.0, vmax=1.0,
            interpolation="bilinear",
        )
        fai_img = np.where(
            (cpr_image >= FAI_HU_MIN) & (cpr_image <= FAI_HU_MAX),
            cpr_image, np.nan,
        )
        self.ax_cpr.imshow(
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

        # Needle line (horizontal — moves with slider)
        self._needle_line = self.ax_cpr.axhline(
            0, color="#00ffcc", linewidth=1.6, linestyle="-", alpha=0.9,
        )

        # Y-axis: arc-length ticks
        x_ticks_mm  = np.arange(0, self.arclengths[-1] + 1, 10.0)
        x_tick_idxs = [int(np.argmin(np.abs(self.arclengths - t))) for t in x_ticks_mm]
        self.ax_cpr.set_yticks(x_tick_idxs)
        self.ax_cpr.set_yticklabels(
            [f"{t:.0f}" for t in x_ticks_mm], fontsize=8, color="#aaaacc"
        )
        self.ax_cpr.set_ylabel("Distance along vessel (mm)", color="#aaaacc", fontsize=10)

        sz, sy, sx = self.spacing_mm
        mean_sp_xy = float(np.mean([sy, sx]))
        n_width    = self.n_width
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
        self.ax_cpr.set_title(
            f"CPR — {self.vessel_name}  (click or use slider to move needle)",
            color="#ddddff", fontsize=10, fontweight="bold", pad=6,
        )

        # ── Slider ───────────────────────────────────────────────────────
        ax_slider.set_facecolor("#0d0d1a")
        self._slider = Slider(
            ax=ax_slider,
            label="Arc-length (mm)",
            valmin=0.0,
            valmax=float(self.arclengths[-1]),
            valinit=0.0,
            color="#00ffcc",
        )
        self._slider.label.set_color("#aaaacc")
        self._slider.valtext.set_color("#aaaacc")
        self._slider.on_changed(self._on_slider)

        # ── Initial cross-section ─────────────────────────────────────────
        self._xs_im       = None   # imshow handle for grayscale cross-section
        self._xs_fai_im   = None   # imshow handle for FAI overlay
        self._xs_lumen    = None   # Circle patch for lumen
        self._xs_voi_ring = None   # Circle patch for VOI outer boundary
        self._xs_needle_txt = None # Text annotation in cross-section panel

        self._draw_crosssection(0)

        # ── Event handlers ────────────────────────────────────────────────
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Title
        fig.suptitle(
            f"PCAT CPR Browser  |  {self.vessel_name}  |  "
            f"{self.N_pts} pts  |  Press 's' to save snapshot, 'q' to quit",
            color="#ffffff", fontsize=11, fontweight="bold",
        )

    # -----------------------------------------------------------------------
    def _draw_crosssection(self, idx: int):
        """
        Render the orthogonal cross-section at centerline index `idx`.

        The cross-section is the full 2D (n_height × n_width) plane from
        cpr_volume[idx, :, :] — it spans ±width_mm in both N and B directions,
        centred on the vessel centreline at that point.
        """
        idx = int(np.clip(idx, 0, self.N_pts - 1))
        self._needle_idx = idx

        # cpr_volume is now (pixels_wide, pixels_high) — 2D straightened CPR.
        # Extract column at arc-length index `idx` as cross-section proxy.
        col_idx = int(np.clip(idx, 0, self.cpr_volume.shape[0] - 1))
        cs_plane = self.cpr_volume[col_idx : col_idx + 1, :].T  # (n_height, 1)

        # Grayscale anatomy
        gray = np.clip(cs_plane, -200.0, 400.0)
        gray = (gray + 200.0) / 600.0
        gray = np.nan_to_num(gray, nan=0.0)

        # FAI overlay
        fai = np.where(
            (cs_plane >= FAI_HU_MIN) & (cs_plane <= FAI_HU_MAX),
            cs_plane, np.nan,
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
                extent=[
                    -self.width_mm, self.width_mm,
                    self.width_mm, -self.width_mm,
                ],
            )
            self._xs_fai_im = self.ax_cross.imshow(
                fai,
                aspect="equal",
                origin="upper",
                cmap=_fai_colormap(),
                vmin=FAI_HU_MIN, vmax=FAI_HU_MAX,
                alpha=0.80,
                interpolation="bilinear",
                extent=[
                    -self.width_mm, self.width_mm,
                    self.width_mm, -self.width_mm,
                ],
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
        r_lumen_mm = float(self.radii_mm[idx])
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
            f"Arc: {arc_mm:.1f} mm  |  pt {idx}/{self.N_pts - 1}",
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
    def _on_slider(self, val: float):
        """Called when slider value changes."""
        idx = int(np.argmin(np.abs(self.arclengths - val)))
        if idx != self._needle_idx:
            self._draw_crosssection(idx)

    # -----------------------------------------------------------------------
    def _on_click(self, event):
        """Click on CPR panel jumps needle to that arc-length position."""
        if event.inaxes is not self.ax_cpr:
            return
        if event.ydata is None:
            return
        idx = int(np.clip(round(event.ydata), 0, self.N_pts - 1))
        # Update slider (which triggers _on_slider)
        self._slider.set_val(float(self.arclengths[idx]))

    # -----------------------------------------------------------------------
    def _on_key(self, event):
        """Keyboard: left/right arrows step needle; 's' saves snapshot; 'q' quits."""
        if event.key in ("right", "up"):
            new_idx = min(self._needle_idx + 1, self.N_pts - 1)
            self._slider.set_val(float(self.arclengths[new_idx]))
        elif event.key in ("left", "down"):
            new_idx = max(self._needle_idx - 1, 0)
            self._slider.set_val(float(self.arclengths[new_idx]))
        elif event.key == "s":
            self._save_snapshot()
        elif event.key == "q":
            plt.close(self.fig)

    # -----------------------------------------------------------------------
    def _save_snapshot(self):
        """Save current view as PNG."""
        arc_mm = float(self.arclengths[self._needle_idx])
        fname  = self.output_dir / f"cpr_browser_{self.vessel_name}_{arc_mm:.0f}mm.png"
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
) -> None:
    """
    Launch the interactive CPR browser and block until the window is closed.

    This function switches matplotlib to an interactive backend. Call ONLY
    from an interactive (non-headless) context.
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
            "arc-length position along the vessel."
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
            "      --voi output/patient_1200/patient1200_RCA_voi_reviewed.npy\n"
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
        "--output", type=str, default=".",
        help="Directory for saving snapshot PNGs (default: current dir)",
    )
    args = parser.parse_args()

    # ── Load DICOM ───────────────────────────────────────────────────────────
    print(f"[cpr_browser] Loading DICOM from {args.dicom} …")
    volume, meta = load_dicom_series(args.dicom)
    spacing_mm = meta["spacing_mm"]
    print(f"[cpr_browser] Volume shape: {volume.shape}  spacing: {spacing_mm}")

    # ── Load seeds ───────────────────────────────────────────────────────────
    print(f"[cpr_browser] Loading seeds from {args.seeds} …")
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
    print("[cpr_browser] Computing Frangi vesselness (ROI-cropped) …")
    vesselness = compute_vesselness(
        volume, spacing_mm,
        seed_points=all_pts,
        roi_margin_mm=20.0,
    )
    print("[cpr_browser] Extracting centerline …")
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
    print(f"[cpr_browser] Centerline: {len(centerline)} points ({seg_start}–{seg_start+seg_length} mm)")

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
    )


if __name__ == "__main__":
    main()
