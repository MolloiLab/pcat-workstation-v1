"""
run_pipeline.py
Main CLI entry point for the PCAT segmentation pipeline.
    # Option A: Fully automatic (no manual seeds needed)
    python pipeline/run_pipeline.py \
        --dicom Rahaf_Patients/1200.2 \
        --output output/patient_1200 \
        --prefix patient1200 \
        --auto-seeds

    # Option B: Manual seeds (traditional workflow)
    python pipeline/seed_picker.py \
        --dicom Rahaf_Patients/1200.2 \
        --output seeds/patient_1200.json
        --dicom   Rahaf_Patients/1200.2 \
        --seeds   seeds/patient_1200.json \
        --output  output/patient_1200 \
        --prefix  patient1200
    python pipeline/run_pipeline.py --batch
    python pipeline/run_pipeline.py --batch --auto-seeds
Full pipeline per patient:
  1. Load DICOM → float32 HU volume
  2. (Optional) Auto-generate seeds via TotalSegmentator if --auto-seeds set
  3. Compute Frangi vesselness (multi-scale)
  4. For each vessel (LAD, LCX, RCA):
     a. Extract centerline via FMM/Dijkstra shortest path from seeds
     b. Clip to proximal segment (40mm for LAD/LCX; 10–50mm for RCA)
     c. Estimate vessel radii via EDT
     d. Build tubular VOI mask (outer shell = mean_diameter thick)
     e. Export per-vessel .raw + metadata JSON
     f. Plot: CPR FAI, HU histogram, radial HU profile
  5. Compute combined (all-vessel) VOI and export as single .raw
  6. Render combined 3D VOI visualization
  7. Write per-patient stats JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
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
from pipeline.pcat_segment import (
    build_tubular_voi,
    build_vessel_mask,
    apply_fai_filter,
    compute_pcat_stats,
)
from pipeline.export_raw import export_voi_raw, export_combined_voi_raw, export_voi_nifti, export_combined_voi_nifti
from pipeline.visualize import (
    render_3d_voi,
    render_cpr_fai,
    plot_hu_histogram,
    plot_radial_hu_profile,
    plot_summary,
)

from pipeline.voi_editor import launch_voi_editor

from pipeline.auto_seeds import _detect_best_device as _auto_detect_device

_DEFAULT_DEVICE = _auto_detect_device()


# ─────────────────────────────────────────────────────────────────────────────
# Known patient configurations (for batch mode)
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_CONFIGS = [
    {
        "patient_id": "1200",
        "dicom": "Rahaf_Patients/1200.2",
        "seeds": "seeds/patient_1200.json",
        "output": "output/patient_1200",
        "prefix": "patient1200",
    },
    {
        "patient_id": "2",
        "dicom": "Rahaf_Patients/2.1",
        "seeds": "seeds/patient_2.json",
        "output": "output/patient_2",
        "prefix": "patient2",
    },
    {
        "patient_id": "317",
        "dicom": "Rahaf_Patients/317.6",
        "seeds": "seeds/patient_317.json",
        "output": "output/patient_317",
        "prefix": "patient317",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Single patient pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_seeds(
    seeds_path: Path,
    dicom_dir: Path,
    auto_seeds: bool,
    auto_seeds_device: str = _DEFAULT_DEVICE,
    auto_seeds_license: Optional[str] = None,
) -> None:
    """
    Ensure seeds_path exists. If it does not exist and auto_seeds is True,
    call generate_seeds() automatically via TotalSegmentator.
    Raises FileNotFoundError / RuntimeError on failure.
    """
    if seeds_path.exists():
        return
    if auto_seeds:
        print(
            f"[pipeline] Seeds file not found: {seeds_path}\n"
            f"[pipeline] --auto-seeds enabled -> running TotalSegmentator..."
        )
        try:
            from pipeline.auto_seeds import generate_seeds
        except ImportError as exc:
            raise RuntimeError(
                "auto_seeds module could not be imported. "
                "Make sure TotalSegmentator is installed: pip install TotalSegmentator"
            ) from exc
        generate_seeds(
            dicom_dir=dicom_dir,
            output_json=seeds_path,
            device=auto_seeds_device,
            license_number=auto_seeds_license,
        )
        if not seeds_path.exists():
            raise RuntimeError(
                f"generate_seeds() completed but {seeds_path} was not written."
            )
        print(f"[pipeline] Auto-seeds written: {seeds_path}")
    else:
        raise FileNotFoundError(
            f"Seeds file not found: {seeds_path}\n"
            f"Run seed_picker.py first:\n"
            f"  python pipeline/seed_picker.py --dicom {dicom_dir} --output {seeds_path}\n"
            f"Or use --auto-seeds to generate seeds automatically via TotalSegmentator."
        )


def run_patient(
    dicom_dir: str | Path,
    seeds_path: str | Path,
    output_dir: str | Path,
    prefix: str = "pcat",
    vessels: Optional[List[str]] = None,
    skip_3d: bool = False,
    vesselness_sigmas: Optional[List[float]] = None,
    auto_seeds: bool = False,
    auto_seeds_device: str = _DEFAULT_DEVICE,
    auto_seeds_license: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full PCAT pipeline for one patient.
    ----------
    dicom_dir   : path to DICOM series directory
    seeds_path  : path to seed JSON file
    output_dir  : directory to write all outputs
    prefix      : filename prefix for all outputs
    vessels     : list of vessels to process (default: all in seeds file)
    skip_3d     : skip 3D pyvista render (use in headless/CI environments)
    vesselness_sigmas : Frangi scale sigmas in mm (default: [0.5, 1.0, 1.5, 2.0, 2.5])
    auto_seeds        : if True and seeds_path missing, call TotalSegmentator auto-seed
    auto_seeds_device : device for TotalSegmentator ("cpu"|"gpu"|"mps")
    auto_seeds_license: TotalSegmentator licence key (or set TOTALSEG_LICENSE env var)
    Returns
    -------
    results dict with per-vessel stats and output file paths
    """
    dicom_dir = Path(dicom_dir)
    seeds_path = Path(seeds_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "patient_prefix": prefix,
        "dicom_dir": str(dicom_dir),
        "seeds_path": str(seeds_path),
        "vessels": {},
        "outputs": [],
        "errors": [],
    }

    # -- Step 1: Load DICOM ----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[pipeline] Patient: {prefix}")
    print(f"[pipeline] DICOM: {dicom_dir}")
    print(f"{'='*60}")
    t0 = time.time()
    print("[pipeline] Loading DICOM series...")
    volume, meta = load_dicom_series(dicom_dir)
    spacing_mm = meta["spacing_mm"]
    print(f"[pipeline] Volume shape: {volume.shape}, spacing_mm: {[f'{s:.4f}' for s in spacing_mm]}")
    print(f"[pipeline] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    results["meta"] = meta
    # -- Step 2: Ensure seeds exist (auto-generate if needed) ------------------
    _ensure_seeds(
        seeds_path=seeds_path,
        dicom_dir=dicom_dir,
        auto_seeds=auto_seeds,
        auto_seeds_device=auto_seeds_device,
        auto_seeds_license=auto_seeds_license,
    )

    seeds_data = load_seeds(seeds_path)
    if vessels is None:
        vessels = list(seeds_data.keys())
    print(f"[pipeline] Vessels to process: {vessels}")

    # ── Step 3: Compute vesselness (ROI-cropped around seed points) ────────
    # Collect ALL seed points across all vessels so the ROI covers the whole
    # coronary tree — Frangi runs on a small sub-volume instead of the full
    # 512×512×405, giving a 50-100× speedup.
    all_seed_pts: List[List[int]] = []
    for vname in vessels:
        if vname not in seeds_data:
            continue
        vsd = seeds_data[vname]
        ostium = vsd["ostium_ijk"]
        # Skip null/placeholder seeds (auto_seeds inserts [None,None,None] for missing vessels)
        if ostium and all(v is not None for v in ostium):
            all_seed_pts.append(ostium)
        for wp in vsd.get("waypoints_ijk", []):
            if wp and all(v is not None for v in wp):
                all_seed_pts.append(wp)

    print("\n[pipeline] Computing Frangi vesselness filter (ROI-cropped — should take ~10s)...")
    t_v = time.time()
    vesselness = compute_vesselness(
        volume, spacing_mm,
        sigmas=vesselness_sigmas,
        seed_points=all_seed_pts if all_seed_pts else None,
        roi_margin_mm=20.0,
    )
    print(f"[pipeline] Vesselness computed in {time.time() - t_v:.1f}s")

    # ── Per-vessel processing ─────────────────────────────────────────────
    vessel_voi_masks: Dict[str, np.ndarray] = {}
    vessel_centerlines: Dict[str, np.ndarray] = {}
    vessel_radii_dict: Dict[str, np.ndarray] = {}
    vessel_stats: Dict[str, Any] = {}

    for vessel_name in vessels:
        print(f"\n[pipeline] ── Processing {vessel_name} ──────────────────────")
        t_vsl = time.time()

        if vessel_name not in seeds_data:
            msg = f"Vessel {vessel_name} not found in seeds file"
            print(f"[pipeline] WARNING: {msg}")
            results["errors"].append(msg)
            continue
        vsd = seeds_data[vessel_name]
        ostium_ijk = vsd["ostium_ijk"]
        # Skip vessels with null/placeholder seeds (auto_seeds inserts [None,None,None] for missing vessels)
        if not ostium_ijk or any(v is None for v in ostium_ijk):
            msg = f"Vessel {vessel_name} has null seeds -- skipping (run seed_picker.py to add manually)"
            print(f"[pipeline] WARNING: {msg}")
            results["errors"].append(msg)
            continue
        waypoints_ijk = vsd.get("waypoints_ijk", [])
        seg_start = float(VESSEL_CONFIGS[vessel_name].get("start_mm", 0.0))
        seg_length = float(VESSEL_CONFIGS[vessel_name].get("length_mm", 40.0))

        # ── Centerline extraction ──────────────────────────────────────
        print(f"[pipeline] Extracting {vessel_name} centerline...")
        try:
            centerline_full = extract_centerline_seeds(
                volume=volume,
                vesselness=vesselness,
                spacing_mm=spacing_mm,
                ostium_ijk=ostium_ijk,
                waypoints_ijk=waypoints_ijk,
                roi_radius_mm=35.0,
            )
        except Exception as e:
            msg = f"{vessel_name} centerline extraction failed: {e}"
            print(f"[pipeline] ERROR: {msg}")
            results["errors"].append(msg)
            continue

        print(f"[pipeline] Full centerline: {len(centerline_full)} points")

        # ── Clip to proximal segment ───────────────────────────────────
        centerline = clip_centerline_by_arclength(
            centerline_full, spacing_mm,
            start_mm=seg_start,
            length_mm=seg_length,
        )
        if len(centerline) < 5:
            msg = f"{vessel_name} clipped centerline too short ({len(centerline)} pts)"
            print(f"[pipeline] WARNING: {msg}")
            results["errors"].append(msg)
            continue

        print(f"[pipeline] Proximal segment [{seg_start}–{seg_start+seg_length}mm]: {len(centerline)} points")
        vessel_centerlines[vessel_name] = centerline

        # ── Radius estimation ──────────────────────────────────────────
        print(f"[pipeline] Estimating {vessel_name} vessel radii...")
        radii_mm = estimate_vessel_radii(volume, centerline, spacing_mm)
        mean_r = float(np.mean(radii_mm))
        print(f"[pipeline] Mean radius: {mean_r:.2f} mm  (range: {radii_mm.min():.2f}–{radii_mm.max():.2f} mm)")
        vessel_radii_dict[vessel_name] = radii_mm

        # ── Build VOI ──────────────────────────────────────────────────
        print(f"[pipeline] Building {vessel_name} tubular VOI...")
        voi_mask = build_tubular_voi(volume.shape, centerline, spacing_mm, radii_mm)
        vessel_voi_masks[vessel_name] = voi_mask
        print(f"[pipeline] VOI voxels: {voi_mask.sum():,}")

        # ── Mandatory sanity check: let the clinician review/edit VOI ────
        print(f"[pipeline] MANDATORY SANITY CHECK: launching VOI editor for {vessel_name}...")
        voi_npy_path = output_dir / f"{prefix}_{vessel_name}_voi_reviewed.npy"
        voi_mask = launch_voi_editor(
            volume=volume,
            voi_mask=voi_mask,
            vessel_name=vessel_name,
            output_path=voi_npy_path,
            spacing_mm=spacing_mm,
        )
        print(f"[pipeline] VOI review complete. Voxels: {voi_mask.sum():,}")

        # ── Compute stats ──────────────────────────────────────────────
        stats = compute_pcat_stats(volume, voi_mask, vessel_name)
        vessel_stats[vessel_name] = stats
        print(
            f"[pipeline] {vessel_name} stats: "
            f"mean_HU={stats['hu_mean']:.1f}, "
            f"fat_fraction={100*stats['fat_fraction']:.1f}%, "
            f"n_fat={stats['n_fat_voxels']:,}"
        )
        risk = stats.get("fai_risk", "UNKNOWN")
        threshold = stats.get("fai_risk_threshold_hu", -70.1)
        risk_icon = "\u26a0\ufe0f  HIGH RISK" if risk == "HIGH" else ("\u2713 LOW RISK" if risk == "LOW" else "? UNKNOWN")
        print(
            f"[pipeline] {vessel_name} FAI RISK: {risk_icon} "
            f"(mean HU {stats['hu_mean']:.1f} vs threshold {threshold} HU)"
        )

        # ── Export per-vessel NIfTI ─────────────────────────────────────
        nii_path = export_voi_nifti(
            voi_mask=voi_mask,
            spacing_mm=spacing_mm,
            output_dir=output_dir,
            prefix=f"{prefix}_{vessel_name}",
        )
        results["outputs"].append(str(nii_path))

        # ── Visualizations ─────────────────────────────────────────────
        print(f"[pipeline] Generating {vessel_name} visualizations...")

        # Output 3: CPR FAI
        cpr_path = render_cpr_fai(
            volume=volume,
            centerline_ijk=centerline,
            radii_mm=radii_mm,
            spacing_mm=spacing_mm,
            vessel_name=vessel_name,
            output_dir=output_dir,
            prefix=prefix,
        )
        if cpr_path:
            results["outputs"].append(str(cpr_path))

        # Output 4: HU histogram
        hist_path = plot_hu_histogram(
            volume=volume,
            voi_mask=voi_mask,
            vessel_name=vessel_name,
            output_dir=output_dir,
            prefix=prefix,
        )
        results["outputs"].append(str(hist_path))

        # Output 5: Radial HU profile
        profile_path = plot_radial_hu_profile(
            volume=volume,
            centerline_ijk=centerline,
            radii_mm=radii_mm,
            spacing_mm=spacing_mm,
            vessel_name=vessel_name,
            output_dir=output_dir,
            prefix=prefix,
        )
        results["outputs"].append(str(profile_path))

        results["vessels"][vessel_name] = stats
        print(f"[pipeline] {vessel_name} done in {time.time() - t_vsl:.1f}s")

    # ── Step 4: Combined VOI export ───────────────────────────────────────
    if vessel_voi_masks:
        print("\n[pipeline] Exporting combined all-vessel VOI NIfTI...")
        combined_nii = export_combined_voi_nifti(
            vessel_masks=vessel_voi_masks,
            spacing_mm=spacing_mm,
            output_dir=output_dir,
            prefix=f"{prefix}_combined",
        )
        results["outputs"].append(str(combined_nii))

    # ── Step 5: 3D visualization ──────────────────────────────────────────
    if vessel_voi_masks and not skip_3d:
        print("\n[pipeline] Rendering 3D VOI visualization...")
        combined_mask = np.zeros(volume.shape, dtype=bool)
        for m in vessel_voi_masks.values():
            combined_mask |= m

        render_3d_voi(
            volume=volume,
            voi_mask=combined_mask,
            vessel_centerlines=vessel_centerlines,
            vessel_radii=vessel_radii_dict,
            spacing_mm=spacing_mm,
            output_dir=output_dir,
            prefix=prefix,
            screenshot=True,
            interactive=False,
        )

    # ── Step 6: Summary chart ─────────────────────────────────────────────
    if vessel_stats:
        summary_path = plot_summary(vessel_stats, output_dir=output_dir, prefix=prefix)
        results["outputs"].append(str(summary_path))

    # ── Save results JSON ─────────────────────────────────────────────────
    results_path = output_dir / f"{prefix}_results.json"
    _save_results(results, results_path)

    elapsed = time.time() - t0
    print(f"\n[pipeline] {'='*50}")
    print(f"[pipeline] Patient {prefix} complete in {elapsed:.1f}s")
    print(f"[pipeline] Outputs in: {output_dir}")
    print(f"[pipeline] Summary: {results_path}")
    if results["errors"]:
        print(f"[pipeline] Warnings/errors: {results['errors']}")
    print(f"[pipeline] {'='*50}\n")

    return results


def _save_results(results: Dict, path: Path):
    """Save results dict to JSON, handling numpy types."""
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"[pipeline] Results JSON saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCAT automated segmentation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--batch", action="store_true",
        help="Run all 3 patients using configs in PATIENT_CONFIGS"
    )
    mode.add_argument(
        "--dicom", type=str,
        help="Path to DICOM series directory (single patient mode)"
    )

    parser.add_argument(
        "--seeds", type=str,
        help="Path to seed JSON file (required in single patient mode)"
    )
    parser.add_argument(
        "--output", type=str, default="output/patient",
        help="Output directory"
    )
    parser.add_argument(
        "--prefix", type=str, default="pcat",
        help="Filename prefix for outputs"
    )
    parser.add_argument(
        "--vessels", type=str, nargs="+",
        choices=["LAD", "LCX", "RCA"],
        default=None,
        help="Vessels to process (default: all vessels in seeds file)"
    )
    parser.add_argument(
        "--skip-3d", action="store_true",
        help="Skip 3D pyvista rendering (useful on headless servers)"
    )
    parser.add_argument(
        "--project-root", type=str, default=".",
        help="Project root directory (for resolving relative paths in batch mode)"
    )
    parser.add_argument(
        "--auto-seeds", action="store_true",
        help=(
            "Automatically generate seeds via TotalSegmentator if seeds JSON is missing. "
            "Requires a free academic licence from "
            "https://backend.totalsegmentator.com/license-academic/"
        ),
    )
    parser.add_argument(
        "--auto-seeds-device", type=str, default=_DEFAULT_DEVICE,
        choices=["cpu", "gpu", "mps"],
        dest="auto_seeds_device",
        help=f"Device for TotalSegmentator inference (auto-detected default: '{_DEFAULT_DEVICE}'). 'mps' on Apple Silicon is 5-10x faster than cpu."
    )
    parser.add_argument(
        "--auto-seeds-license", type=str, default=None,
        dest="auto_seeds_license",
        metavar="KEY",
        help=(
            "TotalSegmentator academic licence key. "
            "Alternatively set the TOTALSEG_LICENSE environment variable."
        ),
    )

    args = parser.parse_args()

    if args.batch:
        # Batch mode: run all patients
        root = Path(args.project_root)
        all_results = []

        for cfg in PATIENT_CONFIGS:
            dicom_path = root / cfg["dicom"]
            seeds_path = root / cfg["seeds"]
            output_path = root / cfg["output"]

            if not dicom_path.exists():
                print(f"[pipeline] Skipping {cfg['patient_id']}: DICOM not found at {dicom_path}")
                continue
            if not seeds_path.exists() and not args.auto_seeds:
                print(
                    f"[pipeline] Skipping {cfg['patient_id']}: seeds file not found at {seeds_path}\n"
                    f"           Run seed_picker.py first OR use --auto-seeds:\n"
                    f"           python pipeline/run_pipeline.py --batch --auto-seeds"
                )
                continue

            try:
                r = run_patient(
                    dicom_dir=dicom_path,
                    seeds_path=seeds_path,
                    output_dir=output_path,
                    prefix=cfg["prefix"],
                    vessels=args.vessels,
                    skip_3d=args.skip_3d,
                    auto_seeds=args.auto_seeds,
                    auto_seeds_device=args.auto_seeds_device,
                    auto_seeds_license=args.auto_seeds_license,
                )
                all_results.append(r)
            except Exception as e:
                print(f"[pipeline] ERROR for patient {cfg['patient_id']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[pipeline] Batch complete: {len(all_results)}/{len(PATIENT_CONFIGS)} patients processed")

    else:
        # Single patient mode
        if not args.seeds and not args.auto_seeds:
            parser.error("--seeds is required unless --auto-seeds is set")

        # Derive a default seeds path when --auto-seeds is used without --seeds
        seeds_arg = args.seeds
        if seeds_arg is None and args.auto_seeds:
            patient_name = Path(args.dicom).name.replace(".", "_")
            seeds_arg = f"seeds/{patient_name}_auto.json"
            print(f"[pipeline] --auto-seeds: will write seeds to {seeds_arg}")
        run_patient(
            dicom_dir=args.dicom,
            seeds_path=seeds_arg,
            output_dir=args.output,
            prefix=args.prefix,
            vessels=args.vessels,
            skip_3d=args.skip_3d,
            auto_seeds=args.auto_seeds,
            auto_seeds_device=args.auto_seeds_device,
            auto_seeds_license=args.auto_seeds_license,
        )


if __name__ == "__main__":
    main()
