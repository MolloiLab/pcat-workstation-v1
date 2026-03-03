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
     c. Extract vessel wall contours via polar-transform boundary detection
     d. Estimate vessel radii via EDT (for CPR rendering)
  5. Centerline verification visualization (with TotalSeg overlay if available)
  6. Interactive contour correction (game-style GUI editor)
  7. Build contour-based PCAT VOI masks from corrected contours
  8. Generate outputs: per-vessel stats, .raw exports, CPR visualizations
  9. Interactive CPR browser (one per vessel)
  10. Combined VOI export + 3D visualization
  11. Write per-patient stats JSON + summary chart
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Non-interactive backend for batch/headless rendering — MUST be set before
# importing pipeline.visualize which uses matplotlib.pyplot at module level.
import matplotlib
matplotlib.use("Agg")

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
    apply_fai_filter,
    compute_pcat_stats,
)
from pipeline.contour_extraction import (
    extract_vessel_contours,
    build_contour_based_voi,
    _contour_to_cartesian,
)
from pipeline.export_raw import export_voi_raw, export_combined_voi_raw
from pipeline.visualize import (
    render_3d_voi_dicom,
    render_cpr_fai,
    render_cpr_dicom,
    render_cpr_png,
    plot_hu_histogram,
    plot_radial_hu_profile,
    render_centerline_verification,
    plot_summary,
)


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



def _find_reviewed_seeds(seeds_path: Path) -> Optional[Path]:
    """
    Check if a reviewed version of the seeds file exists.
    Convention: replace '_auto.json' with '_reviewed.json' in filename.
    Also checks for a .done signal file (written by seed_reviewer after save).
    """
    name = seeds_path.name
    if "_auto" in name:
        reviewed_path = seeds_path.parent / name.replace("_auto", "_reviewed")
        if reviewed_path.exists():
            print(f"[pipeline] Found reviewed seeds: {reviewed_path}")
            return reviewed_path
    return None

def run_patient(
    dicom_dir: str | Path,
    seeds_path: str | Path,
    output_dir: str | Path,
    prefix: str = "pcat",
    vessels: Optional[List[str]] = None,
    skip_3d: bool = False,
    skip_editor: bool = False,
    skip_cpr_browser: bool = False,
    legacy_voi: bool = False,
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
    skip_editor  : skip interactive VOI editor (use in headless/CI environments)
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

    # Create subdirectories for organized output
    raw_dir   = output_dir / "raw"
    cpr_dir   = output_dir / "cpr"
    plots_dir = output_dir / "plots"
    d3_dir    = output_dir / "3d"
    for d in [raw_dir, cpr_dir, plots_dir, d3_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
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
    # -- Step 2b: Seed review (auto-skip if reviewed seeds already exist) -----
    reviewed = _find_reviewed_seeds(seeds_path)
    if reviewed is not None:
        seeds_path = reviewed
        print(f"[pipeline] Using reviewed seeds: {seeds_path}")
    else:
        # Derive reviewed output path
        name = seeds_path.name
        if "_auto" in name:
            reviewed_path = seeds_path.parent / name.replace("_auto", "_reviewed")
        else:
            reviewed_path = seeds_path.parent / (seeds_path.stem + "_reviewed.json")
        print(f"[pipeline] No reviewed seeds found. Opening seed reviewer...")
        print(f"[pipeline] Review seeds then press 's' to save and continue the pipeline.")
        import subprocess as _sp, sys as _sys
        _sp.run(
            [
                _sys.executable,
                str(Path(__file__).parent / "seed_reviewer.py"),
                "--dicom", str(dicom_dir),
                "--seeds", str(seeds_path),
                "--output", str(reviewed_path),
            ],
            check=False,
        )
        # After seed_reviewer closes, use reviewed seeds if written
        if reviewed_path.exists():
            seeds_path = reviewed_path
            print(f"[pipeline] Seed review complete. Using: {seeds_path}")
        else:
            print(f"[pipeline] WARNING: Seed reviewer closed without saving. Using original seeds.")
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

    print("\n[pipeline] Computing Frangi vesselness filter (ROI-cropped — ~120s)...")
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
    vessel_radii_full_dict: Dict[str, np.ndarray] = {}
    vessel_centerlines_proximal: Dict[str, np.ndarray] = {}
    vessel_stats: Dict[str, Any] = {}
    vessel_contour_results: Dict[str, Any] = {}

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
        vessel_centerlines[vessel_name] = centerline_full
        vessel_centerlines_proximal[vessel_name] = centerline

        # ── Radius estimation (clipped segment — for VOI/stats) ──────────────────────────────────────────
        print(f"[pipeline] Estimating {vessel_name} vessel radii (proximal segment)...")
        radii_mm = estimate_vessel_radii(volume, centerline, spacing_mm)
        mean_r = float(np.mean(radii_mm))
        print(f"[pipeline] Mean radius: {mean_r:.2f} mm  (range: {radii_mm.min():.2f}–{radii_mm.max():.2f} mm)")
        vessel_radii_dict[vessel_name] = radii_mm

        # ── Radius estimation (full centerline — for CPR rendering) ──────────────────────────────────────────
        print(f"[pipeline] Estimating {vessel_name} vessel radii (full centerline)...")
        radii_mm_full = estimate_vessel_radii(volume, centerline_full, spacing_mm)
        print(f"[pipeline] Full radii: mean={radii_mm_full.mean():.2f} mm, range={radii_mm_full.min():.2f}–{radii_mm_full.max():.2f} mm")
        vessel_radii_full_dict[vessel_name] = radii_mm_full

        # ── Build VOI (legacy mode) or Extract contours (new default) ────────────────────
        if legacy_voi:
            # Legacy: Build tubular VOI via EDT
            print(f"[pipeline] Building {vessel_name} tubular VOI (legacy mode)...")
            voi_mask = build_tubular_voi(volume.shape, centerline, spacing_mm, radii_mm)
            print(f"[pipeline] VOI voxels (auto): {voi_mask.sum():,}")
            vessel_voi_masks[vessel_name] = voi_mask
        else:
            # New default: Extract vessel wall contours via polar transform
            print(f"[pipeline] Extracting {vessel_name} vessel wall contours...")
            contour_result = extract_vessel_contours(
                volume, centerline, spacing_mm, vessel_name=vessel_name
            )
            vessel_contour_results[vessel_name] = contour_result
            # Print contour extraction stats
            n_fallback = int(contour_result.fallback_mask.sum())
            print(
                f"[pipeline] {vessel_name} contours: r_eq mean={np.mean(contour_result.r_eq):.2f} mm, "
                f"fallback={n_fallback}/{len(contour_result.r_eq)} positions"
            )
            # VOI will be built later after contour editor corrections

        print(f"[pipeline] {vessel_name} data ready in {time.time() - t_vsl:.1f}s")

    # ── Centerline verification visualization ────────────────────────────
    print("\n[pipeline] Generating centerline verification visualization...")
    # Try to load TotalSeg mask if auto-seeds was used
    totalseg_mask = None
    try:
        ts_mask_path = Path(seeds_path).parent / f"patient_{prefix.replace('patient', '')}" / "coronary_arteries.nii.gz"
        # Also check common locations
        for candidate in [
            ts_mask_path,
            output_dir / "totalseg" / "coronary_arteries.nii.gz",
            Path(f"output/discarded/totalseg_{prefix.replace('patient', '')}") / "coronary_arteries.nii.gz",
        ]:
            if candidate.exists():
                import nibabel as nib
                ts_img = nib.load(str(candidate))
                ts_data = ts_img.get_fdata()
                # TotalSeg outputs in XYZ, pipeline uses ZYX
                totalseg_mask = ts_data.transpose(2, 1, 0).astype(bool)
                print(f"[pipeline] Loaded TotalSeg mask from {candidate} ({totalseg_mask.sum():,} voxels)")
                break
    except Exception as e:
        print(f"[pipeline] TotalSeg mask not available: {e}")
    
    verify_path = render_centerline_verification(
        volume=volume,
        vessel_centerlines=vessel_centerlines,
        spacing_mm=spacing_mm,
        output_dir=plots_dir,
        prefix=prefix,
        totalseg_mask=totalseg_mask,
    )
    results["outputs"].append(str(verify_path))

    # ── Save contour data for game editor (new mode only) ────────────────────
    if not legacy_voi and vessel_contour_results:
        contour_data_path = raw_dir / f"{prefix}_contour_data.npz"
        save_data = {}
        for vessel_name, cr in vessel_contour_results.items():
            save_data[f"{vessel_name}_r_theta"] = cr.r_theta
            save_data[f"{vessel_name}_positions_mm"] = cr.positions_mm
            save_data[f"{vessel_name}_N_frame"] = cr.N_frame
            save_data[f"{vessel_name}_B_frame"] = cr.B_frame
            save_data[f"{vessel_name}_r_eq"] = cr.r_eq
            save_data[f"{vessel_name}_arclengths"] = cr.arclengths
            save_data[f"{vessel_name}_fallback_mask"] = cr.fallback_mask
            # Also include centerline for reference
            save_data[f"{vessel_name}_centerline"] = vessel_centerlines_proximal.get(
                vessel_name, cr.positions_mm / np.array(spacing_mm)
            )
        np.savez(str(contour_data_path), **save_data)
        print(f"[pipeline] Saved contour data to {contour_data_path}")


    # ── Step 3b: Contour Editor ─────────────────────────────────────────────
    # Launch interactive editor to review/adjust centerlines and vessel wall contours.
    # This updates vessel_voi_masks and vessel_centerlines in place.
    # Launched as subprocess to avoid matplotlib backend conflict.
    if legacy_voi:
        # Legacy mode: use old coronary_contour_editor
        if not skip_editor and vessel_voi_masks:
            print("\n[pipeline] Launching Coronary Artery Contour Editor (legacy mode)...")
            print("[pipeline] Review centerlines and vessel walls, add PCAT volume, then close.")
            
            # Save vessel data to .npz file for subprocess
            contour_data_path = raw_dir / f"{prefix}_contour_data.npz"
            save_data = {}
            for vessel_name in vessel_centerlines:
                save_data[f"{vessel_name}_centerline"] = vessel_centerlines[vessel_name]
            for vessel_name in vessel_radii_dict:
                save_data[f"{vessel_name}_radii"] = vessel_radii_full_dict.get(vessel_name, vessel_radii_dict[vessel_name])
            for vessel_name in vessel_voi_masks:
                save_data[f"{vessel_name}_voi_mask"] = vessel_voi_masks[vessel_name]
            np.savez(str(contour_data_path), **save_data)
            print(f"[pipeline] Saved vessel data to {contour_data_path}")
            
            # Launch contour editor as subprocess
            try:
                import subprocess as _subprocess
                import sys as _sys
                _subprocess.run(
                    [
                        _sys.executable,
                        str(Path(__file__).parent / "coronary_contour_editor.py"),
                        "--dicom",  str(dicom_dir),
                        "--data",   str(contour_data_path),
                        "--output", str(raw_dir),
                        "--prefix", prefix,
                    ],
                    check=False,
                )
                
                # Load updated data from .npz file
                updated_data_path = raw_dir / f"{prefix}_contour_data_updated.npz"
                if updated_data_path.exists():
                    updated_data = np.load(str(updated_data_path), allow_pickle=True)
                    for key in updated_data.files:
                        if key.endswith("_centerline"):
                            vessel_name = key.replace("_centerline", "")
                            vessel_centerlines[vessel_name] = updated_data[key]
                        elif key.endswith("_radii"):
                            vessel_name = key.replace("_radii", "")
                            vessel_radii_dict[vessel_name] = updated_data[key]
                        elif key.endswith("_voi_mask"):
                            vessel_name = key.replace("_voi_mask", "")
                            vessel_voi_masks[vessel_name] = updated_data[key]
                    print("[pipeline] Contour editor: masks and centerlines updated.")
                    # Clean up temp files
                    contour_data_path.unlink(missing_ok=True)
                    updated_data_path.unlink(missing_ok=True)
                else:
                    print("[pipeline] Contour editor closed without saving changes.")
                    contour_data_path.unlink(missing_ok=True)
                    
            except Exception as _e:
                print(f"[pipeline] WARNING: contour editor error: {_e}")
    else:
        # New mode: use contour_game_editor
        if not skip_editor and vessel_contour_results:
            print("\n[pipeline] Launching Contour Game Editor...")
            print("[pipeline] Adjust vessel wall contours, then press 'S' to save and continue.")
            contour_data_path = raw_dir / f"{prefix}_contour_data.npz"
            try:
                import subprocess as _subprocess
                import sys as _sys
                _subprocess.run(
                    [
                        _sys.executable,
                        str(Path(__file__).parent / "contour_game_editor.py"),
                        "--dicom",        str(dicom_dir),
                        "--contour-data", str(contour_data_path),
                        "--output",       str(raw_dir),
                        "--prefix",       prefix,
                    ],
                    check=False,
                )
            except Exception as _e:
                print(f"[pipeline] WARNING: contour game editor error: {_e}")

            # Load corrected contours if available
            corrected_path = raw_dir / f"{prefix}_contour_data_corrected.npz"
            signal_path = raw_dir / f"{prefix}_contour_game_editor.done"
            if corrected_path.exists() and signal_path.exists():
                corrected_data = np.load(str(corrected_path), allow_pickle=True)
                n_angles = vessel_contour_results[list(vessel_contour_results.keys())[0]].r_theta.shape[1]
                angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
                for vessel_name in list(vessel_contour_results.keys()):
                    cr = vessel_contour_results[vessel_name]
                    key = f"{vessel_name}_r_theta_corrected"
                    if key in corrected_data:
                        cr.r_theta[:] = corrected_data[key]
                        # Also update r_eq from corrected contours
                        key_req = f"{vessel_name}_r_eq"
                        if key_req in corrected_data:
                            cr.r_eq[:] = corrected_data[key_req]
                        # Recompute contours from corrected r_theta
                        cr.contours = [
                            _contour_to_cartesian(cr.r_theta[i], angles, cr.positions_mm[i], cr.N_frame[i], cr.B_frame[i])
                            for i in range(len(cr.positions_mm))
                        ]
                print("[pipeline] Contour game editor: corrections loaded.")
                signal_path.unlink(missing_ok=True)
            else:
                print("[pipeline] Contour game editor closed without saving. Using auto-detected contours.")

        # Build contour-based VOI for each vessel
        for vessel_name, cr in vessel_contour_results.items():
            print(f"[pipeline] Building contour-based VOI for {vessel_name}...")
            voi_mask = build_contour_based_voi(
                volume_shape=volume.shape,
                contours=cr.contours,
                centerline_mm=cr.positions_mm,
                N_frame=cr.N_frame,
                B_frame=cr.B_frame,
                r_eq=cr.r_eq,
                spacing_mm=spacing_mm,
                pcat_scale=3.0,
            )
            vessel_voi_masks[vessel_name] = voi_mask
            print(f"[pipeline] {vessel_name} contour-based VOI: {voi_mask.sum():,} voxels")

    # ── Step 3b½: Generate outputs (AFTER contour editor adjustments) ────────
    # Stats, exports, and visualizations use potentially updated centerlines,
    # radii, and VOI masks from the contour editor.
    for vessel_name in vessels:
        if vessel_name not in vessel_centerlines:
            continue
        print(f"\n[pipeline] ── Generating {vessel_name} outputs ──────────────────────")

        centerline_full = vessel_centerlines[vessel_name]
        centerline = vessel_centerlines_proximal.get(vessel_name)
        if centerline is None:
            continue
        radii_mm = vessel_radii_dict.get(vessel_name)
        radii_mm_full = vessel_radii_full_dict.get(vessel_name, radii_mm)
        voi_mask = vessel_voi_masks.get(vessel_name)
        if voi_mask is None:
            continue

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

        # ── Export per-vessel .raw + metadata JSON ──────────────────────────
        raw_path, json_path = export_voi_raw(
            volume=volume,
            voi_mask=voi_mask,
            meta=meta,
            output_dir=raw_dir,
            prefix=f"{prefix}_{vessel_name}",
        )
        results["outputs"].append(str(raw_path))
        results["outputs"].append(str(json_path))

        # ── Visualizations ─────────────────────────────────────────────
        print(f"[pipeline] Generating {vessel_name} visualizations...")

        # Output 3: CPR FAI
        cpr_path = render_cpr_fai(
            volume=volume,
            centerline_ijk=centerline_full,
            radii_mm=radii_mm_full,
            spacing_mm=spacing_mm,
            vessel_name=vessel_name,
            output_dir=cpr_dir,
            prefix=prefix,
            width_mm=40.0,
        )
        if cpr_path:
            results["outputs"].append(str(cpr_path))

        # Output 3d: CPR DICOM Secondary Capture
        cpr_dcm_path = render_cpr_dicom(
            volume=volume,
            centerline_ijk=centerline_full,
            radii_mm=radii_mm_full,
            spacing_mm=spacing_mm,
            vessel_name=vessel_name,
            output_dir=cpr_dir,
            prefix=prefix,
            patient_meta=meta,
        )
        if cpr_dcm_path:
            results["outputs"].append(str(cpr_dcm_path))

        # Output 3e: CPR PNG with vessel wall + 4 green lines + FAI overlay
        cpr_wall_path = render_cpr_png(
            volume=volume,
            centerline_ijk=centerline_full,
            radii_mm=radii_mm_full,
            spacing_mm=spacing_mm,
            vessel_name=vessel_name,
            output_dir=cpr_dir,
            prefix=prefix,
            width_mm=40.0,
        )
        if cpr_wall_path:
            results["outputs"].append(str(cpr_wall_path))

        # Output 4: HU histogram
        hist_path = plot_hu_histogram(
            volume=volume,
            voi_mask=voi_mask,
            vessel_name=vessel_name,
            output_dir=plots_dir,
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
            output_dir=plots_dir,
            prefix=prefix,
        )
        results["outputs"].append(str(profile_path))

        results["vessels"][vessel_name] = stats
        print(f"[pipeline] {vessel_name} outputs complete.")

    # ── Step 3c: Interactive CPR browsers (one per vessel) ────────────────
    # Launched after contour editor so the user reviews contours first,
    # then browses CPR images with the final centerlines.
    if not skip_cpr_browser and vessel_centerlines:
        for vessel_name in vessels:
            if vessel_name not in vessel_centerlines:
                continue
            print(f"\n[pipeline] Launching interactive CPR browser for {vessel_name}...")
            print(f"[pipeline] (close the window or press 'q' to continue the pipeline)")
            try:
                import subprocess as _subprocess
                import sys as _sys
                _subprocess.run(
                    [
                        _sys.executable,
                        str(Path(__file__).parent / "cpr_browser.py"),
                        "--dicom",  str(dicom_dir),
                        "--seeds",  str(seeds_path),
                        "--vessel", vessel_name,
                        "--output", str(output_dir),
                    ],
                    check=False,
                )
            except Exception as _e:
                print(f"[pipeline] WARNING: CPR browser failed to launch: {_e}")

    # ── Step 4: Combined VOI export ───────────────────────────────────────
    if vessel_voi_masks:
        print("\n[pipeline] Exporting combined all-vessel VOI .raw...")
        raw_path, json_path = export_combined_voi_raw(
            volume=volume,
            vessel_masks=vessel_voi_masks,
            meta=meta,
            output_dir=raw_dir,
            prefix=f"{prefix}_combined",
        )
        results["outputs"].append(str(raw_path))
        results["outputs"].append(str(json_path))

    # ── Step 5: 3D visualization ──────────────────────────────────────────
    if vessel_voi_masks and not skip_3d:
        print("\n[pipeline] Rendering 3D VOI visualization...")
        combined_mask = np.zeros(volume.shape, dtype=bool)
        for m in vessel_voi_masks.values():
            combined_mask |= m

        render_3d_voi_dicom(
            volume=volume,
            voi_mask=combined_mask,
            vessel_centerlines=vessel_centerlines,
            vessel_radii=vessel_radii_dict,
            spacing_mm=spacing_mm,
            output_dir=d3_dir,
            prefix=prefix,
        )

    # ── Step 6: Summary chart ─────────────────────────────────────────────
    if vessel_stats:
        summary_path = plot_summary(vessel_stats, output_dir=plots_dir, prefix=prefix)
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
        "--skip-editor", action="store_true",
        dest="skip_editor",
        help=(
            "Skip the interactive VOI editor sanity check. Use in headless/CI/batch "
            "environments where no display is available. WARNING: bypasses mandatory "
            "clinical review step — only use when a separate manual review will be done."
        ),
    )
    parser.add_argument(
        "--skip-cpr-browser", action="store_true",
        dest="skip_cpr_browser",
        help=(
            "Skip the interactive CPR browser. Use in headless/CI/batch "
            "environments where no display is available."
        ),
    )
    parser.add_argument(
        "--legacy-voi", action="store_true",
        dest="legacy_voi",
        help=(
            "Use legacy circle-based VOI construction (EDT + tubular mask) "
            "instead of polar-transform contour extraction. Faster but less accurate."
        ),
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
                    skip_editor=args.skip_editor,
                    skip_cpr_browser=args.skip_cpr_browser,
                    legacy_voi=args.legacy_voi,
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
            skip_editor=args.skip_editor,
            skip_cpr_browser=args.skip_cpr_browser,
            legacy_voi=args.legacy_voi,
            auto_seeds=args.auto_seeds,
            auto_seeds_device=args.auto_seeds_device,
            auto_seeds_license=args.auto_seeds_license,
        )


if __name__ == "__main__":
    main()
