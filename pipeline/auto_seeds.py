"""
auto_seeds.py
Automatic coronary artery seed extraction using TotalSegmentator.
  1. Convert DICOM volume -> NIfTI (in a temp dir) for TotalSegmentator input
  2. Run TotalSegmentator coronary_arteries task -> binary mask NIfTI
  3. Separate the mask into LAD / LCX / RCA components via connected-component
     labelling + anatomical heuristics (relative position in the volume)
  4. 3-D skeletonize each component to get the vessel centreline voxels
  5. Order skeleton points from ostium -> distal using BFS/DFS from the point
     nearest the aorta
  6. Extract ostium (first ordered point) and 1-3 evenly-spaced waypoints
     along the proximal segment
  7. Write the standard seeds JSON used by run_pipeline.py
Device selection (auto-detected):
  - Apple Silicon (M1/M2/M3/M4): defaults to 'mps' (Metal GPU) -- 5-10x faster than CPU
  - NVIDIA GPU:                   pass device='gpu'
  - Fallback:                     'cpu'
Usage (standalone):
    python pipeline/auto_seeds.py \
        --dicom  Rahaf_Patients/1200.2 \
        --output seeds/patient_1200.json
    # Device is auto-detected (mps on Apple Silicon)
Usage (Python API):
    from pipeline.auto_seeds import generate_seeds
    seeds = generate_seeds(dicom_dir, output_path)  # device auto-detected
Requirements:
    pip install TotalSegmentator  (installs torch, nibabel, nnunetv2, ...)
License note:
    TotalSegmentator coronary_arteries model requires a free academic licence.
    Obtain one at: https://backend.totalsegmentator.com/license-academic/
    and set the environment variable  TOTALSEG_LICENSE=<your-key>  before running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as nd_label
from skimage.morphology import skeletonize

# ── Optional TotalSegmentator import ──────────────────────────────────────────
try:
    from totalsegmentator.python_api import totalsegmentator as _ts_run
    HAS_TOTALSEG = True
except ImportError:
    HAS_TOTALSEG = False
    warnings.warn(
        "TotalSegmentator not installed — auto_seeds.py will not work.\n"
        "Install with: pip install TotalSegmentator",
        RuntimeWarning,
    )

# ── NIfTI I/O (nibabel comes with TotalSegmentator) ───────────────────────────
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def _detect_best_device() -> str:
    """
    Return the best available device string for TotalSegmentator:
      - 'mps'  on Apple Silicon (M1/M2/M3/M4) -- Metal GPU, 5-10x faster than CPU
      - 'gpu'  if CUDA is available (NVIDIA)
      - 'cpu'  fallback
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


_DEFAULT_DEVICE = _detect_best_device()


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dicom_loader import load_dicom_series


# ─────────────────────────────────────────────────────────────────────────────
# VESSEL_CONFIGS (output JSON shape expected by run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

VESSEL_CONFIGS = {
    "LAD": {"segment_length_mm": 40.0},
    "LCX": {"segment_length_mm": 40.0},
    "RCA": {"segment_start_mm": 10.0, "segment_length_mm": 40.0},
}

# Number of waypoints to place per vessel (evenly spaced between ostium and
# 40 mm along the skeleton — or the full skeleton if shorter)
N_WAYPOINTS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: DICOM → NIfTI
# ─────────────────────────────────────────────────────────────────────────────

def dicom_to_nifti(
    dicom_dir: str | Path,
    out_nifti: str | Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DICOM series with our loader (handles Siemens intercept quirks),
    then save the HU volume as a NIfTI file that TotalSegmentator can read.

    Returns (volume, meta) so the caller can reuse the already-loaded array.
    """
    volume, meta = load_dicom_series(dicom_dir)

    sz, sy, sx = meta["spacing_mm"]   # [z, y, x] mm
    ox, oy, oz = meta["origin_mm"]    # [x, y, z] mm (patient coords)

    # NIfTI affine: maps (i, j, k) → (x, y, z) in mm
    # Our volume axes are (Z, Y, X) → NIfTI axes are (X, Y, Z) so we
    # must transpose the volume before saving.
    vol_xyz = volume.transpose(2, 1, 0)  # (X, Y, Z)

    affine = np.diag([sx, sy, sz, 1.0])
    affine[0, 3] = ox   # x origin
    affine[1, 3] = oy   # y origin
    affine[2, 3] = oz   # z origin (first slice position)

    img = nib.Nifti1Image(vol_xyz.astype(np.float32), affine)
    nib.save(img, str(out_nifti))
    print(f"[auto_seeds] NIfTI saved: {out_nifti}  shape={vol_xyz.shape}")

    return volume, meta


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run TotalSegmentator coronary_arteries task
# ─────────────────────────────────────────────────────────────────────────────

def run_totalsegmentator(
    nifti_input: str | Path,
    output_dir: str | Path,
    device: str = _DEFAULT_DEVICE,
    license_number: Optional[str] = None,
) -> Path:
    """
    Run TotalSegmentator on the NIfTI volume and return path to the
    coronary_arteries segmentation mask NIfTI.

    Parameters
    ----------
    nifti_input    : path to input CT NIfTI
    output_dir     : directory where masks are written
    device         : "cpu" | "gpu" | "mps"  (auto-detected; 'mps' on Apple Silicon)
    license_number : TotalSegmentator academic/commercial licence key,
                     or None to use the TOTALSEG_LICENSE env var

    Returns
    -------
    Path to coronary_arteries.nii.gz (inside output_dir)
    """
    if not HAS_TOTALSEG:
        raise RuntimeError(
            "TotalSegmentator is not installed.\n"
            "Install with: pip install TotalSegmentator"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer env-var over explicit arg
    if license_number is None:
        license_number = os.environ.get("TOTALSEG_LICENSE", None)

    print(f"[auto_seeds] Running TotalSegmentator coronary_arteries (device={device}) …")
    print("[auto_seeds] This downloads ~500 MB of model weights on first run.")

    _ts_run(
        input=str(nifti_input),
        output=str(output_dir),
        task="coronary_arteries",
        device=device,
        quiet=False,
        license_number=license_number,
    )

    mask_path = output_dir / "coronary_arteries.nii.gz"
    if not mask_path.exists():
        # TotalSegmentator v2 sometimes puts masks in a subdirectory
        alt = list(output_dir.rglob("coronary_arteries.nii.gz"))
        if alt:
            mask_path = alt[0]
        else:
            raise FileNotFoundError(
                f"TotalSegmentator did not produce coronary_arteries.nii.gz "
                f"in {output_dir}. Check for errors above."
            )

    print(f"[auto_seeds] Mask saved: {mask_path}")
    return mask_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load mask NIfTI → numpy array in (Z, Y, X) layout
# ─────────────────────────────────────────────────────────────────────────────

def load_mask_as_zyx(mask_nifti: str | Path, meta: Dict[str, Any]) -> np.ndarray:
    """
    Load a NIfTI mask saved by TotalSegmentator and reorder axes to match
    our (Z, Y, X) convention.

    TotalSegmentator saves in (X, Y, Z) NIfTI convention; our pipeline
    uses (Z, Y, X) numpy arrays.
    """
    img = nib.load(str(mask_nifti))
    data_xyz = img.get_fdata(dtype=np.float32)   # (X, Y, Z)
    data_zyx = data_xyz.transpose(2, 1, 0)        # (Z, Y, X)

    vol_shape = tuple(meta["shape"])
    if data_zyx.shape != vol_shape:
        # TotalSegmentator may resample internally; resize back to original grid
        from scipy.ndimage import zoom
        zoom_factors = tuple(v / d for v, d in zip(vol_shape, data_zyx.shape))
        data_zyx = zoom(data_zyx, zoom_factors, order=0)  # nearest-neighbour for mask

    return (data_zyx > 0.5).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Separate LAD / LCX / RCA using connected components + anatomy
# ─────────────────────────────────────────────────────────────────────────────

def separate_vessels(
    mask_zyx: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Split the combined coronary mask into LAD, LCX, RCA sub-masks using:
      - Connected-component labelling (26-connected)
      - Anatomical heuristics based on position in the volume

    Heuristics (standard cardiac CT orientation, patient supine):
      - RCA: rightmost large component (most negative X / leftmost in image X)
      - LAD: anterior (smallest X index in image) of the left-side components
      - LCX: posterior/lateral of the left-side components

    Returns dict: {"LAD": bool_array, "LCX": bool_array, "RCA": bool_array}
    If fewer than 3 components are found, returns the available ones and
    warns about the missing vessel.
    """
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connected
    labeled, n_components = nd_label(mask_zyx, structure=structure)

    if n_components == 0:
        raise ValueError(
            "TotalSegmentator produced an empty coronary mask. "
            "Ensure the CCTA covers the proximal coronary arteries."
        )

    # Compute centroid of each component and keep the N_largest
    component_info = []
    for comp_id in range(1, n_components + 1):
        pts = np.argwhere(labeled == comp_id)  # (N, 3) in (Z, Y, X)
        n_vox = len(pts)
        centroid = pts.mean(axis=0)  # [z_mean, y_mean, x_mean]
        component_info.append({
            "id": comp_id,
            "n_vox": n_vox,
            "centroid": centroid,
            "pts": pts,
        })

    # Keep only components large enough to be a major coronary (>= 20 voxels)
    min_vox = max(20, mask_zyx.sum() // 20)
    major = [c for c in component_info if c["n_vox"] >= min_vox]
    major.sort(key=lambda c: -c["n_vox"])  # largest first

    if len(major) == 0:
        raise ValueError(
            "No large enough connected components found in coronary mask. "
            f"Total mask voxels: {mask_zyx.sum()}"
        )

    # Sort by X centroid (image X = patient left-right):
    # In standard axial CT (supine, head-first), the RCA originates from the
    # right coronary sinus → lower X index (patient right = smaller X in LPS).
    # LAD & LCX originate from the left coronary sinus → higher X index.
    major.sort(key=lambda c: c["centroid"][2])  # sort by x_centroid ascending

    vessel_masks: Dict[str, np.ndarray] = {}

    if len(major) == 1:
        warnings.warn(
            "Only 1 connected component found — assigning to LAD. "
            "Consider adding more waypoints manually.",
            RuntimeWarning,
        )
        vessel_masks["LAD"] = labeled == major[0]["id"]

    elif len(major) == 2:
        # Assume leftmost = RCA, rightmost split into LAD/LCX
        # (RCA is rightmost in patient X → lowest image X)
        rca_cand = major[0]
        left_cand = major[1]

        vessel_masks["RCA"] = labeled == rca_cand["id"]
        # Can't separate LAD/LCX — assign both left-side voxels to LAD
        vessel_masks["LAD"] = labeled == left_cand["id"]
        warnings.warn(
            "Only 2 major components found. LAD and LCX are merged into 'LAD'. "
            "Run seed_picker.py to set separate LCX seeds.",
            RuntimeWarning,
        )

    else:
        # 3+ components:
        # Leftmost (smallest X) = RCA
        # Of the remaining, the one most anterior (smallest Y centroid in axial CT,
        # which maps to anterior direction in patient coords) = LAD
        # The other = LCX
        rca_comp = major[0]
        left_comps = major[1:]

        # Among left_comps, LAD is more anterior (smaller Y centroid)
        left_comps.sort(key=lambda c: c["centroid"][1])  # sort by y ascending
        lad_comp = left_comps[0]   # most anterior
        lcx_comp = left_comps[1]   # more posterior/lateral

        vessel_masks["RCA"] = labeled == rca_comp["id"]
        vessel_masks["LAD"] = labeled == lad_comp["id"]
        vessel_masks["LCX"] = labeled == lcx_comp["id"]

    print(f"[auto_seeds] Found {len(major)} major coronary components → "
          f"{list(vessel_masks.keys())}")
    for vn, vm in vessel_masks.items():
        print(f"[auto_seeds]   {vn}: {vm.sum()} voxels")

    return vessel_masks


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 & 6: Skeletonize → order → extract ostium + waypoints
# ─────────────────────────────────────────────────────────────────────────────

def _skeleton_to_ordered_path(
    skel_mask: np.ndarray,
    aorta_center_zyx: np.ndarray,
) -> np.ndarray:
    """
    Convert a 3-D skeleton (bool array) to an ordered list of voxels from
    the point closest to the aorta (ostium) outward.

    Uses a simple greedy nearest-neighbour walk starting from the ostium
    candidate, visiting each unvisited skeleton voxel by proximity.

    Returns
    -------
    ordered : (N, 3) int array  [z, y, x]
    """
    pts = np.argwhere(skel_mask)  # (N, 3)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=int)

    # Ostium = skeleton point closest to the aorta center
    dists_to_aorta = np.linalg.norm(pts - aorta_center_zyx, axis=1)
    start_idx = int(np.argmin(dists_to_aorta))

    # Greedy walk: at each step go to the unvisited neighbour with the
    # smallest Euclidean distance to the current point.
    visited = np.zeros(len(pts), dtype=bool)
    ordered_indices = [start_idx]
    visited[start_idx] = True

    for _ in range(len(pts) - 1):
        current = pts[ordered_indices[-1]]
        unvisited_mask = ~visited
        if not unvisited_mask.any():
            break
        unvisited_pts = pts[unvisited_mask]
        unvisited_idxs = np.where(unvisited_mask)[0]
        dists = np.linalg.norm(unvisited_pts - current, axis=1)
        nearest = int(unvisited_idxs[np.argmin(dists)])
        # Only continue if we haven't jumped more than 10 voxels
        if dists.min() > 10:
            break
        ordered_indices.append(nearest)
        visited[nearest] = True

    return pts[ordered_indices]


def _estimate_aorta_center(
    mask_zyx: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """
    Estimate where the aorta is in voxel space.

    We look for the centroid of the coronary mask in the most-superior
    (highest Z) quartile of the mask, which is roughly where the coronary
    ostia exit the aorta.

    Returns [z, y, x] float.
    """
    pts = np.argwhere(mask_zyx)
    if len(pts) == 0:
        Z, Y, X = mask_zyx.shape
        return np.array([Z // 2, Y // 2, X // 2], dtype=float)

    z_vals = pts[:, 0]
    z_thresh = np.percentile(z_vals, 75)  # upper quartile in Z
    superior_pts = pts[z_vals >= z_thresh]
    if len(superior_pts) == 0:
        superior_pts = pts

    return superior_pts.mean(axis=0)  # [z_mean, y_mean, x_mean]


def extract_seeds_from_mask(
    vessel_mask_zyx: np.ndarray,
    meta: Dict[str, Any],
    spacing_mm: List[float],
    vessel_name: str,
    n_waypoints: int = N_WAYPOINTS,
    proximal_mm: float = 45.0,
) -> Dict[str, Any]:
    """
    Skeletonize a binary vessel mask and extract:
      - ostium_ijk: first point (closest to aorta)
      - waypoints_ijk: N evenly spaced points within the proximal segment

    Parameters
    ----------
    vessel_mask_zyx : (Z, Y, X) bool mask of one coronary artery
    meta            : DICOM metadata from load_dicom_series
    spacing_mm      : [sz, sy, sx]
    vessel_name     : "LAD" | "LCX" | "RCA"
    n_waypoints     : number of waypoints to extract
    proximal_mm     : only use this many mm from the ostium for waypoint selection

    Returns
    -------
    seed dict matching the JSON schema in seeds/patient_*.json
    """
    print(f"[auto_seeds] Skeletonizing {vessel_name} ({vessel_mask_zyx.sum()} voxels)…")

    skel = skeletonize(vessel_mask_zyx)
    skel_pts = np.argwhere(skel)

    if len(skel_pts) < 2:
        warnings.warn(
            f"{vessel_name}: skeleton has only {len(skel_pts)} points. "
            "Using mask centroid as ostium with no waypoints.",
            RuntimeWarning,
        )
        mask_pts = np.argwhere(vessel_mask_zyx)
        centroid = mask_pts.mean(axis=0).astype(int).tolist()
        return {
            "ostium_ijk": centroid,
            "waypoints_ijk": [],
            **VESSEL_CONFIGS[vessel_name],
        }

    aorta_center = _estimate_aorta_center(vessel_mask_zyx, meta)
    ordered = _skeleton_to_ordered_path(skel, aorta_center)

    # Compute cumulative arc-length along ordered skeleton
    sp = np.array(spacing_mm)  # [sz, sy, sx]
    diffs = np.diff(ordered.astype(float), axis=0) * sp
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])

    # Clip to proximal_mm
    max_mm = min(proximal_mm, cumlen[-1])
    proximal_mask = cumlen <= max_mm
    prox_pts = ordered[proximal_mask]
    prox_len = cumlen[proximal_mask]

    ostium_ijk = ordered[0].tolist()

    # Evenly spaced waypoints within the proximal segment (skip the ostium itself)
    if len(prox_pts) > 1 and n_waypoints > 0:
        wp_positions_mm = np.linspace(
            max_mm / (n_waypoints + 1),
            max_mm * n_waypoints / (n_waypoints + 1),
            n_waypoints,
        )
        waypoints_ijk = []
        for wp_mm in wp_positions_mm:
            idx = int(np.argmin(np.abs(prox_len - wp_mm)))
            waypoints_ijk.append(prox_pts[idx].tolist())
    else:
        waypoints_ijk = []

    print(f"[auto_seeds] {vessel_name}: ostium={ostium_ijk}, "
          f"{len(waypoints_ijk)} waypoints, arc-length={cumlen[-1]:.1f}mm")

    return {
        "ostium_ijk": ostium_ijk,
        "waypoints_ijk": waypoints_ijk,
        **VESSEL_CONFIGS[vessel_name],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: generate_seeds
# ─────────────────────────────────────────────────────────────────────────────

def generate_seeds(
    dicom_dir: str | Path,
    output_json: Optional[str | Path] = None,
    device: str = _DEFAULT_DEVICE,
    license_number: Optional[str] = None,
    ts_output_dir: Optional[str | Path] = None,
    n_waypoints: int = N_WAYPOINTS,
) -> Dict[str, Any]:
    """
    Full automatic seed generation pipeline:
      DICOM → NIfTI → TotalSegmentator → skeletonize → seeds JSON

    Parameters
    ----------
    dicom_dir      : path to DICOM series directory
    output_json    : where to save seeds JSON (optional — returns dict either way)
    device         : "cpu" | "gpu" | "mps"  (auto-detected: 'mps' on Apple Silicon, 'gpu' on NVIDIA, 'cpu' fallback)
    license_number : TotalSegmentator academic licence key (or set TOTALSEG_LICENSE env var)
    ts_output_dir  : where to store TotalSegmentator output (default: temp dir)
    n_waypoints    : number of waypoints per vessel (default 3)

    Returns
    -------
    seeds dict in the standard format consumed by run_pipeline.py
    """
    if not HAS_TOTALSEG or not HAS_NIBABEL:
        raise RuntimeError(
            "TotalSegmentator and/or nibabel are not installed.\n"
            "Install with: pip install TotalSegmentator"
        )

    dicom_dir = Path(dicom_dir)

    # Create temp workspace
    use_tmp = ts_output_dir is None
    if use_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="pcat_autoseeds_")
        ts_output_dir = Path(tmp_dir)
    else:
        ts_output_dir = Path(ts_output_dir)
        ts_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. DICOM → NIfTI ─────────────────────────────────────────────
        nifti_path = ts_output_dir / "ct_input.nii.gz"
        volume, meta = dicom_to_nifti(dicom_dir, nifti_path)
        spacing_mm = meta["spacing_mm"]

        # ── 2. TotalSegmentator ───────────────────────────────────────────
        mask_nifti = run_totalsegmentator(
            nifti_path,
            ts_output_dir,
            device=device,
            license_number=license_number,
        )

        # ── 3. Load mask (Z, Y, X) ────────────────────────────────────────
        mask_zyx = load_mask_as_zyx(mask_nifti, meta)
        n_mask_vox = int(mask_zyx.sum())
        print(f"[auto_seeds] Combined coronary mask: {n_mask_vox} voxels")

        if n_mask_vox == 0:
            raise ValueError(
                "TotalSegmentator produced an empty coronary mask. "
                "Possible causes:\n"
                "  1. Missing TotalSegmentator academic license "
                "(set TOTALSEG_LICENSE env var)\n"
                "  2. CCTA does not cover the proximal coronary arteries\n"
                "  3. Low contrast CT (non-contrast scans may not work well)"
            )

        # ── 4. Separate LAD / LCX / RCA ──────────────────────────────────
        vessel_masks = separate_vessels(mask_zyx, meta)

        # ── 5 & 6. Skeletonize each vessel → seeds ─────────────────────────
        seeds: Dict[str, Any] = {}
        for vessel_name in ["LAD", "LCX", "RCA"]:
            if vessel_name not in vessel_masks:
                warnings.warn(
                    f"{vessel_name} component not found — add seeds manually "
                    "with seed_picker.py.",
                    RuntimeWarning,
                )
                # Insert null placeholder so run_pipeline.py skips gracefully
                seeds[vessel_name] = {
                    "ostium_ijk": [None, None, None],
                    "waypoints_ijk": [],
                    **VESSEL_CONFIGS[vessel_name],
                }
                continue

            seeds[vessel_name] = extract_seeds_from_mask(
                vessel_masks[vessel_name],
                meta,
                spacing_mm,
                vessel_name,
                n_waypoints=n_waypoints,
            )

        # ── 7. Save JSON ──────────────────────────────────────────────────
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(seeds, f, indent=2)
            print(f"[auto_seeds] Seeds saved: {output_json}")

        return seeds

    finally:
        # Clean up temp dir if we created it
        if use_tmp:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automatic coronary seed extraction via TotalSegmentator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dicom", required=True,
        help="Path to DICOM series directory"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write seeds JSON (e.g. seeds/patient_1200.json)"
    )
    parser.add_argument(
        "--device", default=_DEFAULT_DEVICE, choices=["cpu", "gpu", "mps"],
        help=(
            f"Device for TotalSegmentator inference (auto-detected default: '{_DEFAULT_DEVICE}'). "
            "'mps' uses Apple Metal GPU on M1/M2/M3/M4 -- 5-10x faster than cpu. "
            "'gpu' uses CUDA on NVIDIA GPUs. "
            "'cpu' works everywhere but is slow."
        ),
    )
    parser.add_argument(
        "--license", default=None, dest="license_number",
        help=(
            "TotalSegmentator academic licence key. "
            "Alternatively set the TOTALSEG_LICENSE environment variable. "
            "Get a free key at: https://backend.totalsegmentator.com/license-academic/"
        ),
    )
    parser.add_argument(
        "--ts-output-dir", default=None,
        help="Keep TotalSegmentator outputs in this directory (default: temp dir that is auto-deleted)"
    )
    parser.add_argument(
        "--waypoints", type=int, default=N_WAYPOINTS,
        help="Number of waypoints to extract per vessel"
    )

    args = parser.parse_args()

    generate_seeds(
        dicom_dir=args.dicom,
        output_json=args.output,
        device=args.device,
        license_number=args.license_number,
        ts_output_dir=args.ts_output_dir,
        n_waypoints=args.waypoints,
    )


if __name__ == "__main__":
    main()
