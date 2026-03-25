"""Compute radial HU profile from vessel wall out to 20mm.

Samples ALL tissue (not just VOI) in concentric 1mm-step rings from
the vessel outer wall, filtering each ring for fat-range voxels
(-190 to -30 HU). This matches the Oxford/CRISP-CT radial profile
methodology (Oikonomou et al., Lancet 2018).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


FAI_HU_MIN = -190.0
FAI_HU_MAX = -30.0


def compute_radial_profile(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    spacing_mm: list,
    centerline_ijk: np.ndarray = None,
    radii_mm: np.ndarray = None,
    max_distance_mm: float = 20.0,
    ring_step_mm: float = 1.0,
) -> tuple:
    """Compute mean FAI HU in concentric rings from vessel outer wall.

    For each 1mm ring from 0 to max_distance_mm, collects all voxels
    in the fat HU range (-190 to -30) and computes mean ± std.

    Parameters
    ----------
    volume : (Z, Y, X) float32 HU
    voi_mask : (Z, Y, X) bool — PCAT VOI (used as fallback if no centerline)
    spacing_mm : [sz, sy, sx]
    centerline_ijk : (N, 3) centerline voxels — if provided, distance is
        measured from the centerline with vessel radius subtracted
    radii_mm : (N,) per-point vessel radii — used with centerline_ijk
    max_distance_mm : how far from vessel wall to profile (default 20mm)
    ring_step_mm : ring width (default 1mm)

    Returns
    -------
    distances_mm : (n_bins,) — ring center distances from vessel wall
    mean_hu : (n_bins,) — mean FAI HU per ring (NaN where no fat voxels)
    std_hu : (n_bins,) — std FAI HU per ring (NaN where no fat voxels)
    """
    ring_edges = np.arange(0.0, max_distance_mm + ring_step_mm, ring_step_mm)
    distances_mm = (ring_edges[:-1] + ring_edges[1:]) / 2.0
    n_bins = len(distances_mm)
    mean_hu = np.full(n_bins, np.nan)
    std_hu = np.full(n_bins, np.nan)

    if centerline_ijk is not None and radii_mm is not None and len(centerline_ijk) >= 2:
        # Method 1: Distance from centerline minus vessel radius = distance from wall
        mean_radius = float(np.mean(radii_mm))

        # Build centerline mask in a local ROI
        pts = centerline_ijk.astype(int)
        max_outer = mean_radius + max_distance_mm + ring_step_mm
        margin = np.array([int(np.ceil(max_outer / s)) + 3 for s in spacing_mm])

        lo = np.maximum(pts.min(axis=0) - margin, 0)
        hi = np.minimum(pts.max(axis=0) + margin, np.array(volume.shape) - 1)
        sub_shape = tuple((hi - lo + 1).tolist())

        cl_local = pts - lo
        cl_mask = np.zeros(sub_shape, dtype=bool)
        for pt in cl_local:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
                cl_mask[z, y, x] = True

        # Distance from centerline in mm
        dist_from_cl = distance_transform_edt(~cl_mask, sampling=spacing_mm)

        # Distance from vessel wall = distance from centerline - mean radius
        dist_from_wall = dist_from_cl - mean_radius

        vol_sub = volume[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]

        # Bin by distance from wall
        for i, (r_inner, r_outer) in enumerate(zip(ring_edges[:-1], ring_edges[1:])):
            ring_mask = (dist_from_wall >= r_inner) & (dist_from_wall < r_outer)
            hu_ring = vol_sub[ring_mask]
            # Filter to FAI range
            fat = hu_ring[(hu_ring >= FAI_HU_MIN) & (hu_ring <= FAI_HU_MAX)]
            if len(fat) > 0:
                mean_hu[i] = float(np.mean(fat))
                std_hu[i] = float(np.std(fat))

    elif voi_mask.any():
        # Fallback: distance from VOI inner boundary
        dist = distance_transform_edt(voi_mask, sampling=spacing_mm)
        d_vals = dist[voi_mask]
        hu_vals = volume[voi_mask].astype(np.float32)

        for i, (r_inner, r_outer) in enumerate(zip(ring_edges[:-1], ring_edges[1:])):
            mask = (d_vals >= r_inner) & (d_vals < r_outer)
            if mask.any():
                fat = hu_vals[mask]
                fat = fat[(fat >= FAI_HU_MIN) & (fat <= FAI_HU_MAX)]
                if len(fat) > 0:
                    mean_hu[i] = float(np.mean(fat))
                    std_hu[i] = float(np.std(fat))

    return distances_mm, mean_hu, std_hu
