#!/usr/bin/env python3
"""
validate_cpr.py
Non-interactive CPR validation script.

Loads patient 1200 DICOM, reads saved centerlines from contour_data.npz,
runs _compute_cpr_data (with the current cubic interpolation fix), and
compares the output arrays against existing CPR DICOM outputs.

Saves results to output/patient_1200/cpr_validation/

Usage:
    python validate_cpr.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

# Force non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.dicom_loader import load_dicom_series
from pipeline.visualize import _compute_cpr_data

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DICOM_DIR = Path("Rahaf_Patients/1200.2")
CONTOUR_DATA = Path("output/patient_1200/raw/patient1200_contour_data.npz")
EXISTING_CPR_DIR = Path("output/patient_1200/cpr")
VALIDATION_DIR = Path("output/patient_1200/cpr_validation")
VESSELS = ["LAD", "LCX", "RCA"]


def load_existing_cpr_dcm(path: Path) -> np.ndarray | None:
    """Load a CPR DICOM Secondary Capture as a float32 array."""
    try:
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        return arr
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None


def compute_column_variance(img: np.ndarray) -> np.ndarray:
    """
    Compute per-column variance (across rows) as a stripe artifact metric.
    High column-to-column variance jumps = stripe artifacts.
    """
    col_var = np.nanvar(img, axis=0)
    return col_var


def compute_stripe_metric(img: np.ndarray) -> float:
    """
    Stripe artifact metric: ratio of column variance standard deviation
    to its mean. High ratio = visible column-to-column banding.
    """
    col_var = compute_column_variance(img)
    mean_v = np.nanmean(col_var)
    std_v = np.nanstd(col_var)
    return float(std_v / mean_v) if mean_v > 0 else 0.0


def main():
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load DICOM ────────────────────────────────────────────────────────
    print("=" * 60)
    print("CPR Validation Script — Patient 1200")
    print("=" * 60)
    t0 = time.time()
    print("\n[1/4] Loading DICOM volume...")
    volume, meta = load_dicom_series(DICOM_DIR)
    spacing_mm = meta["spacing_mm"]
    print(f"  Volume: {volume.shape}, spacing: {[f'{s:.4f}' for s in spacing_mm]}")
    print(f"  HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # ── Load centerlines ──────────────────────────────────────────────────
    print(f"\n[2/4] Loading centerlines from {CONTOUR_DATA}...")
    data = np.load(str(CONTOUR_DATA), allow_pickle=True)
    centerlines = {}
    for v in VESSELS:
        key = f"{v}_centerline"
        if key in data.files:
            centerlines[v] = data[key]
            print(f"  {v}: {data[key].shape[0]} centerline points")
        else:
            print(f"  {v}: NOT FOUND in contour data")

    # ── Generate new CPR arrays ───────────────────────────────────────────
    print(f"\n[3/4] Computing CPR with cubic interpolation...")
    new_cprs = {}
    for vessel_name, cl_ijk in centerlines.items():
        t1 = time.time()
        print(f"\n  ── {vessel_name} ──")
        cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_h, n_w = _compute_cpr_data(
            volume=volume,
            centerline_ijk=cl_ijk,
            spacing_mm=spacing_mm,
            slab_thickness_mm=3.0,
            width_mm=40.0,
            pixels_wide=512,
            pixels_high=512,
        )
        # cpr_volume is (pixels_wide, pixels_high) = (512, 512)
        # Transpose for display: (pixels_high, pixels_wide) = (rows=lateral, cols=arclength)
        cpr_img = cpr_volume.T  # (512, 512)
        new_cprs[vessel_name] = cpr_img

        nan_count = int(np.isnan(cpr_img).sum())
        nan_pct = 100.0 * nan_count / cpr_img.size
        valid = cpr_img[~np.isnan(cpr_img)]

        print(f"  Shape: {cpr_img.shape}")
        print(f"  NaN:   {nan_count:,} ({nan_pct:.2f}%)")
        if len(valid) > 0:
            print(f"  HU range: [{valid.min():.1f}, {valid.max():.1f}]")
            print(f"  HU mean:  {valid.mean():.1f} ± {valid.std():.1f}")
        stripe = compute_stripe_metric(cpr_img)
        print(f"  Stripe metric (col-var CV): {stripe:.4f}")
        print(f"  Time: {time.time() - t1:.1f}s")

        # Save .npy
        npy_path = VALIDATION_DIR / f"{vessel_name}_cpr_cubic.npy"
        np.save(str(npy_path), cpr_img)
        print(f"  Saved: {npy_path}")

    # ── Compare with existing CPR ─────────────────────────────────────────
    print(f"\n[4/4] Comparing with existing CPR outputs...")
    for vessel_name, new_img in new_cprs.items():
        dcm_path = EXISTING_CPR_DIR / f"patient1200_{vessel_name}_cpr_hu.dcm"
        print(f"\n  ── {vessel_name} ──")
        if not dcm_path.exists():
            print(f"  No existing DICOM at {dcm_path}")
            continue

        old_img = load_existing_cpr_dcm(dcm_path)
        if old_img is None:
            continue

        print(f"  Old shape: {old_img.shape}, New shape: {new_img.shape}")

        # Stripe metric comparison
        old_stripe = compute_stripe_metric(old_img)
        new_stripe = compute_stripe_metric(new_img)
        improvement = (old_stripe - new_stripe) / old_stripe * 100 if old_stripe > 0 else 0
        print(f"  Old stripe metric: {old_stripe:.4f}")
        print(f"  New stripe metric: {new_stripe:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

        # NaN comparison
        old_nan = int(np.isnan(old_img).sum()) if np.isnan(old_img).any() else 0
        new_nan = int(np.isnan(new_img).sum()) if np.isnan(new_img).any() else 0
        print(f"  Old NaN count: {old_nan:,}")
        print(f"  New NaN count: {new_nan:,}")

        # Mean HU comparison (where both are valid)
        if old_img.shape == new_img.shape:
            both_valid = ~np.isnan(new_img) & (old_img != 0)
            if both_valid.any():
                diff = new_img[both_valid] - old_img[both_valid]
                print(f"  Mean HU diff (new - old): {diff.mean():.2f} ± {diff.std():.2f}")
                print(f"  Max |diff|: {np.abs(diff).max():.1f}")
        else:
            print(f"  Shapes differ — cannot compute pixel-wise difference")

        # Save comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

        # Old CPR
        ax = axes[0]
        old_disp = np.clip(old_img, -200, 400)
        ax.imshow(old_disp, cmap="gray", aspect="auto", vmin=-200, vmax=400)
        ax.set_title(f"{vessel_name} — Old (trilinear)\nStripe={old_stripe:.4f}")
        ax.axis("off")

        # New CPR
        ax = axes[1]
        new_disp = np.nan_to_num(np.clip(new_img, -200, 400), nan=0)
        ax.imshow(new_disp, cmap="gray", aspect="auto", vmin=-200, vmax=400)
        ax.set_title(f"{vessel_name} — New (cubic)\nStripe={new_stripe:.4f}")
        ax.axis("off")

        # Column variance comparison
        ax = axes[2]
        old_cv = compute_column_variance(old_img)
        new_cv = compute_column_variance(new_img)
        ax.plot(old_cv, alpha=0.7, label=f"Old (mean={old_cv.mean():.0f})", linewidth=0.5)
        ax.plot(new_cv, alpha=0.7, label=f"New (mean={new_cv.mean():.0f})", linewidth=0.5)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Variance across rows")
        ax.set_title("Per-column variance (stripe indicator)")
        ax.legend()

        fig.tight_layout()
        fig_path = VALIDATION_DIR / f"{vessel_name}_comparison.png"
        fig.savefig(str(fig_path))
        plt.close(fig)
        print(f"  Saved comparison: {fig_path}")

    print(f"\n{'=' * 60}")
    print(f"Validation complete. Total time: {time.time() - t0:.1f}s")
    print(f"Results in: {VALIDATION_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
