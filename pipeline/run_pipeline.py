"""
run_pipeline.py
Main CLI entry point for the PCAT segmentation pipeline.

Usage:
    # Step 1: Pick seeds interactively (one-time per patient)
    python pipeline/seed_picker.py \
        --dicom Rahaf_Patients/1200.2 \
        --output seeds/patient_1200.json

    # Step 2: Run full pipeline
    python pipeline/run_pipeline.py \
        --dicom   Rahaf_Patients/1200.2 \
        --seeds   seeds/patient_1200.json \
        --output  output/patient_1200 \
        --prefix  patient1200

    # Or run all patients in batch:
    python pipeline/run_pipeline.py --batch

Full pipeline per patient:
  1. Load DICOM → float32 HU volume
  2. Compute Frangi vesselness (multi-scale)
  3. For each vessel (LAD, LCX, RCA):
     a. Extract centerline via Dijkstra shortest path from seeds
     b. Clip to proximal segment (40mm for LAD/LCX; 10–50mm for RCA)
     c. Estimate vessel radii via EDT
     d. Build tubular VOI mask (outer shell = mean_diameter thick)
     e. Export per-vessel .raw + metadata JSON
     f. Plot: CPR FAI, HU histogram, radial HU profile
  4. Compute combined (all-vessel) VOI and export as single .raw
  5. Render combined 3D VOI visualization
  6. Write per-patient stats JSON
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
from pipeline.export_raw import export_voi_raw, export_combined_voi_raw
from pipeline.visualize import (
    render_3d_voi,
    render_cpr_fai,
    plot_hu_histogram,
    plot_radial_hu_profile,
    plot_summary,
)


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

def run_patient(
    dicom_dir: str | Path,
    seeds_path: str | Path,
    output_dir: str | Path,
    prefix: str = "pcat",
    vessels: Optional[List[str]] = None,
    skip_3d: bool = False,
    vesselness_sigmas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Run the full PCAT pipeline for one patient.

    Parameters
    ----------
    dicom_dir   : path to DICOM series directory
    seeds_path  : path to seed JSON file
    output_dir  : directory to write all outputs
    prefix      : filename prefix for all outputs
    vessels     : list of vessels to process (default: all in seeds file)
    skip_3d     : skip 3D pyvista render (use in headless/CI environments)
    vesselness_sigmas : Frangi scale sigmas in mm (default: [0.5, 1.0, 1.5, 2.0, 2.5])

    Returns
    -------
    results dict with per-vessel stats and output file paths
    """
    dicom_dir = Path(dicom_dir)
    seeds_path = Path(seeds_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: Dict[str, Any] = {
        "patient_prefix": prefix,
        "dicom_dir": str(dicom_dir),
        "seeds_path": str(seeds_path),
        "vessels": {},
        "outputs": [],
        "errors": [],
    }

    # ── Step 1: Load DICOM ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] Patient: {prefix}")
    print(f"[pipeline] DICOM: {dicom_dir}")
    print(f"{'='*60}")
    print("[pipeline] Loading DICOM series...")
    volume, meta = load_dicom_series(dicom_dir)
    spacing_mm = meta["spacing_mm"]
    print(f"[pipeline] Volume shape: {volume.shape}, spacing_mm: {[f'{s:.4f}' for s in spacing_mm]}")
    print(f"[pipeline] HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    results["meta"] = meta

    # ── Step 2: Load seeds ───────────────────────────────────────────────
    if not seeds_path.exists():
        raise FileNotFoundError(
            f"Seeds file not found: {seeds_path}\n"
            f"Run seed_picker.py first:\n"
            f"  python pipeline/seed_picker.py --dicom {dicom_dir} --output {seeds_path}"
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
        all_seed_pts.append(vsd["ostium_ijk"])
        all_seed_pts.extend(vsd.get("waypoints_ijk", []))

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

        # ── Compute stats ──────────────────────────────────────────────
        stats = compute_pcat_stats(volume, voi_mask, vessel_name)
        vessel_stats[vessel_name] = stats
        print(
            f"[pipeline] {vessel_name} stats: "
            f"mean_HU={stats['hu_mean']:.1f}, "
            f"fat_fraction={100*stats['fat_fraction']:.1f}%, "
            f"n_fat={stats['n_fat_voxels']:,}"
        )

        # ── Export per-vessel .raw ─────────────────────────────────────
        raw_path, json_path = export_voi_raw(
            volume=volume,
            voi_mask=voi_mask,
            meta=meta,
            output_dir=output_dir,
            prefix=f"{prefix}_{vessel_name}",
        )
        results["outputs"].extend([str(raw_path), str(json_path)])

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
        print("\n[pipeline] Exporting combined all-vessel VOI .raw...")
        combined_raw, combined_json = export_combined_voi_raw(
            volume=volume,
            vessel_masks=vessel_voi_masks,
            meta=meta,
            output_dir=output_dir,
            prefix=f"{prefix}_combined",
        )
        results["outputs"].extend([str(combined_raw), str(combined_json)])

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
            if not seeds_path.exists():
                print(
                    f"[pipeline] Skipping {cfg['patient_id']}: seeds file not found at {seeds_path}\n"
                    f"           Run seed_picker.py first:\n"
                    f"           python pipeline/seed_picker.py "
                    f"--dicom {dicom_path} --output {seeds_path}"
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
                )
                all_results.append(r)
            except Exception as e:
                print(f"[pipeline] ERROR for patient {cfg['patient_id']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[pipeline] Batch complete: {len(all_results)}/{len(PATIENT_CONFIGS)} patients processed")

    else:
        # Single patient mode
        if not args.seeds:
            parser.error("--seeds is required in single patient mode")

        run_patient(
            dicom_dir=args.dicom,
            seeds_path=args.seeds,
            output_dir=args.output,
            prefix=args.prefix,
            vessels=args.vessels,
            skip_3d=args.skip_3d,
        )


if __name__ == "__main__":
    main()
