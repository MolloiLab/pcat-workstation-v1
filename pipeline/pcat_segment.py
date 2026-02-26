"""
pcat_segment.py
Build the pericoronary adipose tissue (PCAT) VOI and apply FAI filtering.

Steps:
  1. Build tubular VOI mask: all voxels within (mean_radius + mean_radius) from
     the vessel centerline — i.e. the outer perivascular shell is 1× vessel radius thick
  2. Subtract vessel lumen (inner boundary = vessel wall)
  3. Optionally apply HU filter for FAI: -190 to -30 HU
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.ndimage import distance_transform_edt


# ─────────────────────────────────────────────
# Tubular VOI construction
# ─────────────────────────────────────────────

def build_tubular_voi(
    volume_shape: Tuple[int, int, int],
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    radii_mm: np.ndarray,
    inner_margin_mm: float = 0.0,
) -> np.ndarray:
    """
    Build a binary tubular VOI mask around the centerline.

    At each centerline point p_i with radius r_i:
      - Outer boundary: r_i + r_i = 2×r_i from the centerline (vessel wall + PCAT shell)
      - Inner boundary: r_i (vessel lumen — excluded from PCAT)

    The VOI is the region between inner and outer boundaries.

    Parameters
    ----------
    volume_shape   : (Z, Y, X)
    centerline_ijk : (N, 3) centerline voxel indices [z, y, x]
    spacing_mm     : [z, y, x]
    radii_mm       : (N,) per-point vessel radius in mm
    inner_margin_mm: extra margin to add to the inner boundary (default 0)

    Returns
    -------
    voi_mask : (Z, Y, X) bool array — True inside the perivascular shell
    """
    sz, sy, sx = spacing_mm
    mean_radius_mm = float(np.mean(radii_mm))

    # Determine bounding box around the centerline + max outer radius
    max_outer_mm = mean_radius_mm * 2.0
    margin_vox = np.array([
        int(np.ceil(max_outer_mm / sz)) + 2,
        int(np.ceil(max_outer_mm / sy)) + 2,
        int(np.ceil(max_outer_mm / sx)) + 2,
    ])

    lo = np.maximum(centerline_ijk.min(axis=0) - margin_vox, 0).astype(int)
    hi = np.minimum(centerline_ijk.max(axis=0) + margin_vox,
                    np.array(volume_shape) - 1).astype(int)

    # Subvolume dimensions
    sub_shape = tuple((hi - lo + 1).tolist())

    # Build binary mask of centerline points in subvolume
    cl_local = centerline_ijk - lo  # (N, 3) local coords

    cl_mask = np.zeros(sub_shape, dtype=bool)
    for pt in cl_local:
        z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
        if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
            cl_mask[z, y, x] = True

    # Distance transform from centerline (in mm)
    # EDT of the inverted centerline mask → each voxel's distance to nearest centerline point
    dist_mm = distance_transform_edt(~cl_mask, sampling=spacing_mm)  # mm

    # Per-point outer/inner radii (use mean for simplicity — can be per-point)
    outer_mm = mean_radius_mm * 2.0  # outer boundary of PCAT shell
    inner_mm = mean_radius_mm + inner_margin_mm  # inner boundary (vessel wall)

    # VOI: voxels between inner and outer shell
    voi_sub = (dist_mm >= inner_mm) & (dist_mm <= outer_mm)

    # Map back to full volume
    voi_full = np.zeros(volume_shape, dtype=bool)
    voi_full[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = voi_sub

    return voi_full


def build_vessel_mask(
    volume_shape: Tuple[int, int, int],
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    radii_mm: np.ndarray,
) -> np.ndarray:
    """
    Build binary mask of the vessel lumen (inner tube, radius = mean_radius).
    Useful for visualization overlays.
    """
    sz, sy, sx = spacing_mm
    mean_radius_mm = float(np.mean(radii_mm))

    margin_vox = np.array([
        int(np.ceil(mean_radius_mm / sz)) + 2,
        int(np.ceil(mean_radius_mm / sy)) + 2,
        int(np.ceil(mean_radius_mm / sx)) + 2,
    ])
    lo = np.maximum(centerline_ijk.min(axis=0) - margin_vox, 0).astype(int)
    hi = np.minimum(centerline_ijk.max(axis=0) + margin_vox,
                    np.array(volume_shape) - 1).astype(int)
    sub_shape = tuple((hi - lo + 1).tolist())

    cl_local = centerline_ijk - lo
    cl_mask = np.zeros(sub_shape, dtype=bool)
    for pt in cl_local:
        z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
        if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
            cl_mask[z, y, x] = True

    dist_mm = distance_transform_edt(~cl_mask, sampling=spacing_mm)
    vessel_sub = dist_mm <= mean_radius_mm

    vessel_full = np.zeros(volume_shape, dtype=bool)
    vessel_full[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = vessel_sub
    return vessel_full


# ─────────────────────────────────────────────
# FAI filtering (HU masking for fat)
# ─────────────────────────────────────────────

FAI_HU_MIN = -190.0
FAI_HU_MAX = -30.0


def apply_fai_filter(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    hu_min: float = FAI_HU_MIN,
    hu_max: float = FAI_HU_MAX,
) -> np.ndarray:
    """
    Apply fat HU range filter within the VOI.

    Returns a float32 array where:
      - voxels in VOI AND in fat HU range → their HU value
      - all other voxels → NaN

    Parameters
    ----------
    volume   : (Z, Y, X) HU float32
    voi_mask : (Z, Y, X) bool — tubular VOI
    hu_min   : lower HU threshold for fat
    hu_max   : upper HU threshold for fat

    Returns
    -------
    fai_volume : (Z, Y, X) float32 with NaN outside fat region
    """
    fat_mask = (volume >= hu_min) & (volume <= hu_max) & voi_mask

    fai_volume = np.full(volume.shape, np.nan, dtype=np.float32)
    fai_volume[fat_mask] = volume[fat_mask]

    return fai_volume


# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────

def compute_pcat_stats(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    vessel_name: str,
    hu_min: float = FAI_HU_MIN,
    hu_max: float = FAI_HU_MAX,
) -> Dict[str, Any]:
    """
    Compute PCAT statistics for a vessel VOI.

    Returns dict with:
      vessel, n_voi_voxels, n_fat_voxels, fat_fraction,
      hu_mean, hu_std, hu_median, hu_min, hu_max, hu_percentiles
    """
    hu_in_voi = volume[voi_mask]
    fat_voxels = hu_in_voi[(hu_in_voi >= hu_min) & (hu_in_voi <= hu_max)]

    stats = {
        "vessel": vessel_name,
        "n_voi_voxels": int(voi_mask.sum()),
        "n_fat_voxels": int(len(fat_voxels)),
        "fat_fraction": float(len(fat_voxels) / max(voi_mask.sum(), 1)),
        "hu_mean": float(np.mean(fat_voxels)) if len(fat_voxels) > 0 else float("nan"),
        "hu_std": float(np.std(fat_voxels)) if len(fat_voxels) > 0 else float("nan"),
        "hu_median": float(np.median(fat_voxels)) if len(fat_voxels) > 0 else float("nan"),
        "hu_min_measured": float(np.min(fat_voxels)) if len(fat_voxels) > 0 else float("nan"),
        "hu_max_measured": float(np.max(fat_voxels)) if len(fat_voxels) > 0 else float("nan"),
        "hu_p25": float(np.percentile(fat_voxels, 25)) if len(fat_voxels) > 0 else float("nan"),
        "hu_p75": float(np.percentile(fat_voxels, 75)) if len(fat_voxels) > 0 else float("nan"),
        "FAI_HU_range": [hu_min, hu_max],
    }
    return stats
