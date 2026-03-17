"""Compute radial HU profile from vessel wall for PCAT analysis.

The radial profile bins VOI voxels by their distance from the nearest
VOI boundary (approximating distance from the vessel wall) and computes
mean HU at each distance bin.  This gives a 1-D view of how fat
attenuation changes as a function of distance from the coronary artery.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_radial_profile(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    spacing_mm: list,
    n_bins: int = 30,
    max_distance_mm: float = 10.0,
) -> tuple:
    """Compute mean HU as a function of distance from vessel wall.

    For each voxel inside the VOI we compute its Euclidean distance to the
    nearest non-VOI voxel (i.e. the inner boundary of the VOI shell, which
    approximates the vessel wall).  Voxels are then binned by distance and
    the mean HU is reported per bin.

    Parameters
    ----------
    volume : (Z, Y, X) float32 HU
    voi_mask : (Z, Y, X) bool -- PCAT VOI region
    spacing_mm : [sz, sy, sx]
    n_bins : number of radial distance bins
    max_distance_mm : maximum distance to profile

    Returns
    -------
    distances_mm : (n_bins,) -- bin centres in mm
    mean_hu : (n_bins,) -- mean HU at each distance (NaN where no data)
    """
    if not voi_mask.any():
        return (
            np.linspace(0, max_distance_mm, n_bins),
            np.full(n_bins, np.nan),
        )

    # Distance from every VOI voxel to the nearest boundary (non-VOI voxel).
    # For the thin-shell VOI this gives distance from either the inner
    # (vessel wall) or outer edge -- since the shell is narrow, most voxels
    # are closest to the inner edge.
    dist = distance_transform_edt(voi_mask, sampling=spacing_mm)

    d_vals = dist[voi_mask]
    hu_vals = volume[voi_mask].astype(np.float32)

    # Clamp max_distance to actual data range so bins are meaningful
    effective_max = min(float(d_vals.max()), max_distance_mm) if d_vals.max() > 0 else max_distance_mm

    bin_edges = np.linspace(0, effective_max, n_bins + 1)
    distances_mm = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_hu = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (d_vals >= bin_edges[i]) & (d_vals < bin_edges[i + 1])
        if mask.any():
            mean_hu[i] = float(np.mean(hu_vals[mask]))

    return distances_mm, mean_hu
