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

# ─────────────────────────────────────────────────────────────────────────────
# Minimum voxel threshold for a component to be considered a major coronary
# AFTER aorta exclusion.  Using an absolute floor (200 vox) rather than a
# fraction of total mask size avoids incorrectly filtering the smallest
# coronary (e.g. RCA proximal stub, typically 400–1000 voxels in CCTA).
# ─────────────────────────────────────────────────────────────────────────────
_MAJOR_VESSEL_MIN_VOX = 200      # absolute lower bound
_AORTA_SIZE_RATIO = 3.0          # largest/2nd-largest ratio → aorta flag


def _try_watershed_split(
    mask_zyx: np.ndarray,
    meta: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to split a single merged coronary blob into sub-regions using a
    morphological erosion-based watershed approach.

    Strategy:
      1. Iteratively erode the mask until ≥ 2 components of ≥ 50 voxels each
         appear (or until erosion destroys the mask).
      2. Use the eroded components as Voronoi seeds to partition the ORIGINAL
         mask voxels by nearest-centroid assignment.
      3. Return list of sub-mask dicts: [{mask, centroid, n_vox}, ...]

    Returns None if the blob cannot be split into ≥ 2 usable parts.
    """
    from scipy.ndimage import binary_erosion
    from scipy.ndimage import label as nd_label_ws

    structure = np.ones((3, 3, 3), dtype=int)
    SEED_MIN_VOX = 50
    N_SEEDS_TARGET = 2

    eroded = mask_zyx.copy()
    best_seeds_labeled: Optional[np.ndarray] = None
    best_n_seeds = 0

    for _iter in range(1, 8):
        eroded = binary_erosion(eroded, structure=structure, iterations=1)
        if not eroded.any():
            break
        lbl, n = nd_label_ws(eroded, structure=structure)
        seed_sizes = sorted(
            [int((lbl == i).sum()) for i in range(1, n + 1)], reverse=True
        )
        n_good = sum(1 for s in seed_sizes if s >= SEED_MIN_VOX)
        if n_good >= N_SEEDS_TARGET:
            best_seeds_labeled = lbl
            best_n_seeds = n_good
            if n_good >= 3:
                break

    if best_seeds_labeled is None or best_n_seeds < N_SEEDS_TARGET:
        return None

    # Collect up to 3 largest seeds
    seed_comps: List[Dict[str, Any]] = []
    n_lbl = int(best_seeds_labeled.max())
    for i in range(1, n_lbl + 1):
        sz = int((best_seeds_labeled == i).sum())
        if sz >= SEED_MIN_VOX:
            pts = np.argwhere(best_seeds_labeled == i)
            centroid = pts.mean(axis=0)
            seed_comps.append({"id": i, "n_vox": sz, "centroid": centroid})
    seed_comps.sort(key=lambda c: -c["n_vox"])
    seed_comps = seed_comps[:3]

    # Voronoi partition: assign each original mask voxel to nearest seed centroid
    mask_pts = np.argwhere(mask_zyx)  # (M, 3)
    seed_centroids = np.array([c["centroid"] for c in seed_comps])  # (K, 3)
    diffs = mask_pts[:, None, :] - seed_centroids[None, :, :]  # (M, K, 3)
    dists = np.linalg.norm(diffs, axis=2)  # (M, K)
    assignments = np.argmin(dists, axis=1)  # (M,)

    result: List[Dict[str, Any]] = []
    for k, sc in enumerate(seed_comps):
        vm = np.zeros_like(mask_zyx)
        vm_pts = mask_pts[assignments == k]
        if len(vm_pts) > 0:
            vm[vm_pts[:, 0], vm_pts[:, 1], vm_pts[:, 2]] = True
        result.append({
            "n_vox": int(vm.sum()),
            "centroid": sc["centroid"],
            "mask": vm,
            "id": -(k + 1),      # negative id signals watershed partition
            "use_mask": True,
        })

    print(
        "[auto_seeds] Watershed split → "
        + ", ".join(f"{r['n_vox']} vox" for r in result)
    )
    return result


def separate_vessels(
    mask_zyx: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Split the combined coronary mask into LAD, LCX, RCA sub-masks.

    Pipeline:
      1. Connected-component labelling (26-connected).
      2. Aorta-exclusion: if the largest component is >3× the second-largest,
         it is treated as an aortic over-segmentation and removed.
      3. Size filtering: keep components ≥ 200 voxels (absolute floor).
      4. Single-blob recovery: if only 1 component survives, attempt a
         morphological-erosion watershed to split merged vessels.
      5. Anatomical vessel assignment:
         - RCA:  smallest X centroid (patient-right coronary sinus)
         - LAD:  smallest Y centroid among left-side components (anterior)
         - LCX:  largest  Y centroid among left-side components (posterior)

    Returns dict {"LAD": bool_array, "LCX": bool_array, "RCA": bool_array}.
    Missing vessels are omitted with a RuntimeWarning.
    """
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connected
    labeled, n_components = nd_label(mask_zyx, structure=structure)
    if n_components == 0:
        raise ValueError(
            "TotalSegmentator produced an empty coronary mask. "
            "Ensure the CCTA covers the proximal coronary arteries."
        )

    # ── Build component info ──────────────────────────────────────────────
    component_info: List[Dict[str, Any]] = []
    for comp_id in range(1, n_components + 1):
        pts = np.argwhere(labeled == comp_id)  # (N, 3) in (Z, Y, X)
        component_info.append({
            "id": comp_id,
            "n_vox": len(pts),
            "centroid": pts.mean(axis=0),
        })
    component_info.sort(key=lambda c: -c["n_vox"])  # largest first

    # ── Aorta-exclusion heuristic ─────────────────────────────────────────
    if (
        len(component_info) >= 2
        and component_info[0]["n_vox"] / max(1, component_info[1]["n_vox"])
        > _AORTA_SIZE_RATIO
    ):
        aorta_comp = component_info[0]
        ratio = aorta_comp["n_vox"] / max(1, component_info[1]["n_vox"])
        warnings.warn(
            f"Largest component ({aorta_comp['n_vox']} vox) is {ratio:.1f}× "
            "larger than the next — treating as aortic over-segmentation and ",
            RuntimeWarning,
        )
        print(
            f"[auto_seeds] Aorta excluded: comp #{aorta_comp['id']} "
            f"({aorta_comp['n_vox']} vox, centroid="
            f"({aorta_comp['centroid'][0]:.0f},{aorta_comp['centroid'][1]:.0f},"
            f"{aorta_comp['centroid'][2]:.0f}))"
        )
        mask_zyx = mask_zyx & ~(labeled == aorta_comp["id"])
        component_info = component_info[1:]
    # ── Size filtering ──────────────────────────────────────────────────
    # Keep ALL components >= 200 vox (don't cap at 3 here — we need to
    # search beyond the top-2 largest for the RCA, which is often smaller
    # than distal LAD/LCX fragments).
    major: List[Dict[str, Any]] = [
        c for c in component_info if c["n_vox"] >= _MAJOR_VESSEL_MIN_VOX
    ]
    major.sort(key=lambda c: -c["n_vox"])  # largest first
    if not major:
        raise ValueError(
            "No large enough connected components found in coronary mask. "
            f"Total mask voxels: {int(mask_zyx.sum())}"
        )
    # ── Single-blob watershed split ─────────────────────────────────────
    if len(major) == 1:
        print("[auto_seeds] Only 1 major component — attempting watershed split …")
        single_mask = (labeled == major[0]["id"])
        ws_result = _try_watershed_split(single_mask, meta)
        if ws_result is not None and len(ws_result) >= 2:
            major = ws_result
        else:
            warnings.warn(
                "Watershed split failed — assigning single component to LAD. "
                "Run seed_picker.py to add LCX/RCA seeds manually.",
                RuntimeWarning,
            )
    # ── Vessel assignment ────────────────────────────────────────────────
    #
    # Strategy (handles the case where RCA is NOT among the 2 largest comps):
    #   1. Top-2 by size → left coronaries (LAD + LCX).
    #      Sorted by Y: smallest Y = anterior = LAD, largest Y = posterior = LCX.
    #   2. RCA → largest component NOT in top-2 whose X centroid is clearly
    #      to the patient-right of both left coronaries
    #      (i.e. centroid[2] < min(LAD_x, LCX_x) - _RCA_X_MARGIN_VOX).
    #   3. Fallback A: if no component satisfies the X constraint, pick the
    #      one with the smallest X centroid among all remaining >= 200-vox comps.
    #   4. Fallback B: if only 1–2 comps exist, fall back to legacy logic.
    #
    _RCA_X_MARGIN_VOX = 5  # RCA centroid must be >= 5 vox right of LAD/LCX
    def _get_mask(c: dict) -> np.ndarray:
        if c.get("use_mask"):
            return c["mask"].astype(bool)
        return (labeled == c["id"]).astype(bool)
    vessel_masks: Dict[str, np.ndarray] = {}
    if len(major) == 1:
        warnings.warn(
            "Only 1 component — assigning to LAD. Add seeds manually if needed.",
            RuntimeWarning,
        )
        vessel_masks["LAD"] = _get_mask(major[0])
    elif len(major) == 2:
        # Only 2 comps: smallest X = RCA, largest X = LAD (best guess)
        sorted2 = sorted(major, key=lambda c: c["centroid"][2])
        vessel_masks["RCA"] = _get_mask(sorted2[0])
        vessel_masks["LAD"] = _get_mask(sorted2[1])
        warnings.warn(
            "Only 2 major components found; LCX unavailable. "
            "Run seed_picker.py to set separate LCX seeds.",
            RuntimeWarning,
        )
    else:
        # ── Step 1: top-2 by size are the left coronaries ─────────────────
        left_candidates = list(major[:2])
        remaining = list(major[2:])

        left_x_vals = [c["centroid"][2] for c in left_candidates]
        left_x_min = min(left_x_vals)

        # ── Step 2: find RCA among remaining comps ────────────────────────
        # Prefer a comp clearly to the patient-right of the left coronaries
        rca_candidates = [
            c for c in remaining
            if c["centroid"][2] < left_x_min - _RCA_X_MARGIN_VOX
        ]
        if rca_candidates:
            # Take the largest qualifying candidate
            rca_comp = max(rca_candidates, key=lambda c: c["n_vox"])
        elif remaining:
            # Fallback A: no clear X separation — pick smallest-X remaining comp
            rca_comp = min(remaining, key=lambda c: c["centroid"][2])
            warnings.warn(
                f"RCA X-centroid not clearly separated from left coronaries "
                f"(left_x_min={left_x_min:.0f}). Using smallest-X remaining comp.",
                RuntimeWarning,
            )
        else:
            # Fallback B: no remaining comps — pull RCA from top-3 by X
            top3 = sorted(major[:3], key=lambda c: c["centroid"][2])
            rca_comp = top3[0]
            left_candidates = list(top3[1:])
            warnings.warn(
                "RCA not found outside top-2 components; falling back to "
                "smallest-X within top-3. Verify seeds visually.",
                RuntimeWarning,
            )

        # ── Step 3: assign LAD / LCX from left candidates by Y ───────────
        left_sorted_y = sorted(left_candidates, key=lambda c: c["centroid"][1])
        lad_comp = left_sorted_y[0]   # smallest Y = anterior = LAD
        lcx_comp = left_sorted_y[-1]  # largest  Y = posterior = LCX
        vessel_masks["RCA"] = _get_mask(rca_comp)
        vessel_masks["LAD"] = _get_mask(lad_comp)
        vessel_masks["LCX"] = _get_mask(lcx_comp)
    print(f"[auto_seeds] Vessel assignment: {list(vessel_masks.keys())}")
    for vn, vm in vessel_masks.items():
        pts = np.argwhere(vm)
        if len(pts):
            ctr = pts.mean(axis=0)
            print(
                f"[auto_seeds]   {vn}: {int(vm.sum())} voxels, "
                f"centroid ZYX=({ctr[0]:.0f},{ctr[1]:.0f},{ctr[2]:.0f})"
            )
        else:
            print(f"[auto_seeds]   {vn}: 0 voxels")
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
    Algorithm:
    1. Build a lookup of all skeleton voxel coordinates.
    2. BFS from the ostium voxel (closest to aorta center) over 26-connected
       skeleton neighbours.
    3. Return points in BFS order (proximal -> distal).

    This is robust to branching skeletons and avoids the early-stop
    problem of a greedy nearest-neighbour walk.
    Returns
    -------
    ordered : (N, 3) int array  [z, y, x]
    """
    pts = np.argwhere(skel_mask)  # (N, 3)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=int)
    if len(pts) == 1:
        return pts

    # Build a voxel lookup: coord tuple -> index
    coord_to_idx = {(int(p[0]), int(p[1]), int(p[2])): i for i, p in enumerate(pts)}
    # Ostium = skeleton point closest to the aorta center
    dists_to_aorta = np.linalg.norm(pts - aorta_center_zyx, axis=1)
    start_idx = int(np.argmin(dists_to_aorta))
    # BFS over 26-connected skeleton neighbours
    from collections import deque
    visited = np.zeros(len(pts), dtype=bool)
    ordered_indices: list = []
    queue: deque = deque([start_idx])
    visited[start_idx] = True
    while queue:
        idx = queue.popleft()
        ordered_indices.append(idx)
        p = pts[idx]
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nb = (int(p[0]) + dz, int(p[1]) + dy, int(p[2]) + dx)
                    nb_idx = coord_to_idx.get(nb)
                    if nb_idx is not None and not visited[nb_idx]:
                        visited[nb_idx] = True
                        queue.append(nb_idx)

    # Append any disconnected fragments (rare, but include them)
    for i in range(len(pts)):
        if not visited[i]:
            ordered_indices.append(i)

    return pts[np.array(ordered_indices, dtype=int)]


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
    # ── Radius quality check ─────────────────────────────────────────────
    # Estimate mean vessel radius using a distance-transform approximation.
    # If sub-voxel (< 0.5 px) the PCAT shell will fall inside the vessel wall
    # and the analysis will return NaN mean HU.
    try:
        from scipy.ndimage import distance_transform_edt
        dt = distance_transform_edt(vessel_mask_zyx)
        if len(prox_pts) > 0:
            prox_radii_vox = dt[prox_pts[:, 0], prox_pts[:, 1], prox_pts[:, 2]]
            mean_radius_vox = float(prox_radii_vox.mean())
            mean_radius_mm = mean_radius_vox * float(np.min(spacing_mm))
            print(
                f"[auto_seeds] {vessel_name}: mean proximal radius = "
                f"{mean_radius_mm:.2f} mm ({mean_radius_vox:.2f} vox)"
            )
            if mean_radius_vox < 0.5:
                warnings.warn(
                    f"{vessel_name}: mean proximal radius is sub-voxel "
                    f"({mean_radius_mm:.2f} mm / {mean_radius_vox:.2f} vox). "
                    "PCAT shell may fall inside vessel wall — expect NaN mean HU. "
                    "Adjust seeds manually with seed_picker.py.",
                    RuntimeWarning,
                )
    except Exception:
        pass  # radius check is non-critical

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
