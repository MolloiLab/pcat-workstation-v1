"""
centerline.py
Coronary artery centerline extraction from seed points.

Strategy:
  1. Frangi vesselness filter → tubular structure probability map
  2. Cost-weighted shortest-path (Dijkstra via scipy/skimage) from ostium seed
     through waypoints, producing ordered centerline voxels
  3. Per-point radius estimation via distance transform from vessel wall

Seed JSON format (per patient, per vessel):
{
  "LAD": {
    "ostium_ijk": [z, y, x],          # voxel index of LAD ostium
    "waypoints_ijk": [[z,y,x], ...],  # 1-3 waypoints along proximal LAD
    "segment_length_mm": 40.0
  },
  "LCX": { ... },
  "RCA": {
    "ostium_ijk": [z, y, x],
    "waypoints_ijk": [...],
    "segment_start_mm": 10.0,         # for RCA: skip first 10mm
    "segment_length_mm": 50.0         # then take next 40mm (10-50mm)
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import frangi

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import frangi
from skimage.morphology import skeletonize_3d


# ─────────────────────────────────────────────
# Vessel enhancement
# ─────────────────────────────────────────────

def compute_vesselness(
    volume: np.ndarray,
    spacing_mm: List[float],
    sigmas: Optional[List[float]] = None,
    hu_clip: Tuple[float, float] = (-200, 1200),
) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter on the HU volume.

    Parameters
    ----------
    volume     : (Z, Y, X) float32 HU array
    spacing_mm : [z, y, x] voxel size in mm
    sigmas     : scale range in mm (default: 0.5–3.0 mm for coronaries)
    hu_clip    : clip HU before filtering (remove bone artifacts)

    Returns
    -------
    vesselness : (Z, Y, X) float32, 0–1
    """
    if sigmas is None:
        # Coronary diameters typically 2–5mm lumen; sigmas in voxels
        sz = spacing_mm[0]
        # Convert mm sigmas to voxels for z; use x/y spacing for xy
        sy = spacing_mm[1]
        # Use isotropic assumption (sub-mm near-isotropic for these scans)
        sigmas = [0.5, 1.0, 1.5, 2.0, 2.5]  # in mm; will be divided by spacing below

    # Clip HU to suppress bone and air extremes
    vol = np.clip(volume, hu_clip[0], hu_clip[1]).astype(np.float32)

    # Normalize to [0, 1]
    vmin, vmax = hu_clip
    vol_norm = (vol - vmin) / (vmax - vmin)

    # Convert mm sigmas → voxel sigmas (use mean spacing)
    mean_sp = np.mean(spacing_mm)
    sigmas_vox = [s / mean_sp for s in sigmas]

    # scikit-image frangi expects (Y, X) or (Z, Y, X) — 3D supported
    # black_ridges=False: vessels are bright (contrast-enhanced CCTA)
    vessel = frangi(
        vol_norm,
        sigmas=sigmas_vox,
        black_ridges=False,
        alpha=0.5,
        beta=0.5,
        gamma=15,
    ).astype(np.float32)

    return vessel


# ─────────────────────────────────────────────
# Shortest-path centerline extraction
# ─────────────────────────────────────────────

def _neighbors_26(idx: int, shape: tuple):
    """Yield 26-connected neighbor flat indices and distances."""
    z, y, x = np.unravel_index(idx, shape)
    nz, ny, nx = shape
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nz2, ny2, nx2 = z + dz, y + dy, x + dx
                if 0 <= nz2 < nz and 0 <= ny2 < ny and 0 <= nx2 < nx:
                    dist = np.sqrt(dz * dz + dy * dy + dx * dx)
                    yield np.ravel_multi_index((nz2, ny2, nx2), shape), dist


def extract_centerline_seeds(
    volume: np.ndarray,
    vesselness: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    waypoints_ijk: List[List[int]],
    roi_radius_mm: float = 30.0,
) -> np.ndarray:
    """
    Extract centerline from ostium through waypoints using cost-weighted Dijkstra.

    The cost of each voxel = 1 / (vesselness + epsilon), so the path prefers
    high-vesselness regions (the vessel interior).

    Works on a local ROI around the seed region for efficiency.

    Parameters
    ----------
    volume        : (Z, Y, X) float32 HU array (unused here, kept for API consistency)
    vesselness    : (Z, Y, X) float32 vesselness map
    spacing_mm    : [z, y, x]
    ostium_ijk    : [z, y, x] ostium voxel
    waypoints_ijk : list of [z, y, x] waypoints
    roi_radius_mm : half-size of ROI cube to limit Dijkstra memory

    Returns
    -------
    centerline_ijk : (N, 3) array of ordered centerline voxel indices (z, y, x)
    """
    shape = vesselness.shape
    sz, sy, sx = spacing_mm

    # Convert all seed points to numpy arrays
    all_points = [np.array(ostium_ijk)] + [np.array(p) for p in waypoints_ijk]

    # Compute bounding ROI around all seed points + margin
    margin_vox = [int(roi_radius_mm / s) for s in spacing_mm]
    pts_arr = np.array(all_points)
    lo = np.maximum(pts_arr.min(axis=0) - margin_vox, 0).astype(int)
    hi = np.minimum(pts_arr.max(axis=0) + margin_vox, np.array(shape) - 1).astype(int)

    # Crop vesselness to ROI
    roi = vesselness[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]
    roi_shape = roi.shape

    # Cost = 1 / (vesselness + eps); clip to avoid zero-cost highways
    eps = 1e-3
    cost = 1.0 / (roi.astype(np.float64) + eps)

    # Build sparse graph (26-connectivity, weighted by cost × distance)
    n = roi.size
    rows_list, cols_list, data_list = [], [], []

    for flat_idx in range(n):
        c_src = cost.flat[flat_idx]
        for nb_idx, dist in _neighbors_26(flat_idx, roi_shape):
            c_nb = cost.flat[nb_idx]
            edge_cost = 0.5 * (c_src + c_nb) * dist
            rows_list.append(flat_idx)
            cols_list.append(nb_idx)
            data_list.append(edge_cost)

    graph = csr_matrix(
        (data_list, (rows_list, cols_list)),
        shape=(n, n)
    )

    # Map seed points from global → ROI local coords
    def global_to_roi(pt):
        return tuple(np.array(pt) - lo)

    def roi_to_flat(pt_local):
        return int(np.ravel_multi_index(pt_local, roi_shape))

    # Run Dijkstra from ostium
    ostium_local = global_to_roi(all_points[0])
    src_flat = roi_to_flat(ostium_local)

    dist_matrix, predecessors = dijkstra(
        graph, directed=False,
        indices=src_flat,
        return_predecessors=True,
    )

    # Trace path through each waypoint in order
    full_path_flat = [src_flat]
    current_src = src_flat

    for wp in all_points[1:]:
        wp_local = global_to_roi(wp)
        wp_flat = roi_to_flat(wp_local)

        # Retrace from wp back to current_src
        path = []
        node = wp_flat
        while node != current_src and node >= 0:
            path.append(node)
            node = predecessors[node]
        path.append(current_src)
        path.reverse()

        if len(path) > 1:
            full_path_flat.extend(path[1:])  # avoid duplicate of start
        current_src = wp_flat

    # Convert flat ROI indices → global voxel indices
    centerline_ijk = []
    seen = set()
    for flat_idx in full_path_flat:
        if flat_idx in seen:
            continue
        seen.add(flat_idx)
        local_ijk = np.unravel_index(flat_idx, roi_shape)
        global_ijk = tuple(int(local_ijk[i] + lo[i]) for i in range(3))
        centerline_ijk.append(global_ijk)

    return np.array(centerline_ijk)  # shape (N, 3): [z, y, x]


# ─────────────────────────────────────────────
# Arc-length clipping (proximal segment)
# ─────────────────────────────────────────────

def clip_centerline_by_arclength(
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    start_mm: float = 0.0,
    length_mm: float = 40.0,
) -> np.ndarray:
    """
    Clip centerline to [start_mm, start_mm + length_mm] from the ostium (index 0).

    Parameters
    ----------
    centerline_ijk : (N, 3) array [z, y, x]
    spacing_mm     : [z, y, x]
    start_mm       : skip this many mm from the start (e.g. 10mm for RCA)
    length_mm      : keep this many mm after start_mm

    Returns
    -------
    clipped : (M, 3) array
    """
    sz, sy, sx = spacing_mm
    scale = np.array([sz, sy, sx])

    # Compute arc-length distances along centerline
    diffs = np.diff(centerline_ijk.astype(float), axis=0)  # (N-1, 3)
    diffs_mm = diffs * scale[np.newaxis, :]
    seg_lengths = np.linalg.norm(diffs_mm, axis=1)  # mm between consecutive points
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])  # shape (N,)

    end_mm = start_mm + length_mm
    mask = (cumlen >= start_mm) & (cumlen <= end_mm)

    clipped = centerline_ijk[mask]
    return clipped


# ─────────────────────────────────────────────
# Radius estimation along centerline
# ─────────────────────────────────────────────

def estimate_vessel_radii(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    lumen_hu_range: Tuple[float, float] = (150, 1200),
    radius_search_mm: float = 8.0,
) -> np.ndarray:
    """
    Estimate vessel radius at each centerline point using:
      1. Segment lumen voxels (high HU after contrast enhancement)
      2. Distance transform → radius = distance from centerline to lumen edge

    Parameters
    ----------
    volume         : (Z, Y, X) HU array
    centerline_ijk : (N, 3) centerline voxels
    spacing_mm     : [z, y, x]
    lumen_hu_range : HU range of contrast-enhanced lumen (bright)
    radius_search_mm: max radius to consider (caps search)

    Returns
    -------
    radii_mm : (N,) array of estimated radii in mm
    """
    sz, sy, sx = spacing_mm

    # Binary lumen mask: contrast-enhanced blood pool is ~150–500 HU
    lumen_mask = (volume >= lumen_hu_range[0]) & (volume <= lumen_hu_range[1])

    # Distance transform: distance from each non-lumen voxel to nearest lumen voxel
    # We want distance from centerline points to vessel wall — approximate as:
    # binary EDT of the lumen mask → value at centerline point = radius
    edt = distance_transform_edt(lumen_mask, sampling=spacing_mm)  # in mm

    radii_mm = np.array([
        float(edt[p[0], p[1], p[2]])  # type: ignore[index]
        for p in centerline_ijk.astype(int)
    ])

    # Clip unrealistic values
    radii_mm = np.clip(radii_mm, 0.5, radius_search_mm)

    return radii_mm


# ─────────────────────────────────────────────
# Load seeds from JSON
# ─────────────────────────────────────────────

def load_seeds(seeds_path: str | Path) -> Dict[str, Any]:
    """Load vessel seed JSON file."""
    with open(seeds_path) as f:
        return json.load(f)


VESSEL_CONFIGS = {
    "LAD": {"start_mm": 0.0, "length_mm": 40.0},
    "LCX": {"start_mm": 0.0, "length_mm": 40.0},
    "RCA": {"start_mm": 10.0, "length_mm": 40.0},  # 10–50mm
}
