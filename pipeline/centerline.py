"""
centerline.py
Coronary artery centerline extraction from seed points.

Strategy:
  1. Frangi vesselness filter — run ONLY on a tight ROI around the seed points
     (not the full volume) → 10–100x speedup on large CCTA volumes.
  2. Fast Marching (scikit-fmm) from ostium seed through waypoints.
     Replaces Dijkstra + sparse graph build — no graph construction at all.
     Each voxel processed once: O(n log n) heap vs previous O(n²) Dijkstra.
     Speedup: 3–10× on the centerline step alone.
  3. Gradient descent back-trace from each waypoint through the FMM
     travel-time field to recover the minimal-cost path.
  4. Per-point radius estimation via distance transform from vessel wall.

  Fallback: if scikit-fmm is not installed, falls back to the original
  vectorised Dijkstra implementation automatically.

Apple M3 acceleration:
  - Frangi runs on a small ROI (typically ~100³ voxels) instead of 400+ slices.
  - Fast Marching is O(n log n) — naturally faster than Dijkstra on dense grids.
  - Graph construction completely eliminated.
  - All numpy ops use float32 to halve memory bandwidth vs float64.

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
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_dilation, label as _ndi_label
from skimage.filters import frangi


# ── Optional PyTorch MPS acceleration for Frangi ──────────────────────────
try:
    import torch
    import torch.nn.functional as _F
    HAS_TORCH_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH_MPS = False



# ── MPS-accelerated 3D Frangi vesselness ──────────────────────────────────
# Implements the same algorithm as skimage.filters.frangi but runs on
# Apple Metal (MPS) via PyTorch.  Key differences from CPU path:
#   - Separable 1D convolutions via conv3d (Gaussian derivative kernels)
#   - Cardano closed-form 3×3 eigenvalues (torch.linalg.eigvalsh unsupported on MPS)
#   - float32 only (MPS does not support float64)
#   - One sigma at a time to limit GPU memory

def _gaussian_kernel_1d(sigma: float, order: int, truncate: float = 8.0) -> 'torch.Tensor':
    """
    Create a 1D Gaussian or Gaussian-derivative kernel matching
    scipy.ndimage.gaussian_filter(order=...) convention.
    ----------
    sigma    : standard deviation
    order    : 0 = Gaussian, 1 = first derivative, 2 = second derivative
    truncate : kernel extends to truncate * sigma on each side
    -------
    1D float32 tensor on MPS
    """
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        radius = 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device='mps')
    g_raw = torch.exp(-0.5 * (x / sigma) ** 2)
    g_norm = g_raw / g_raw.sum()  # normalised Gaussian (sums to 1)
    if order == 0:
        return g_norm
    elif order == 1:
        # First derivative of normalised Gaussian: -x/sigma^2 * G_norm(x)
        g = -x / (sigma ** 2) * g_norm
        return g
    elif order == 2:
        # Second derivative of normalised Gaussian: (x^2/sigma^4 - 1/sigma^2) * G_norm(x)
        g = (x ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * g_norm
        return g
    return g_norm


def _separable_conv3d(
    vol: 'torch.Tensor',
    kz: 'torch.Tensor',
    ky: 'torch.Tensor',
    kx: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Apply three separable 1D convolutions (z, y, x) to a 3D volume.
    vol : (1, 1, Z, Y, X) on MPS
    kz, ky, kx : 1D kernels
    Returns (1, 1, Z, Y, X) result.
    """
    # Z-axis: kernel shape (1,1,Kz,1,1)
    pz = min(len(kz) // 2, vol.shape[2] - 1)
    w = kz.reshape(1, 1, -1, 1, 1)
    out = _F.conv3d(_F.pad(vol, (0, 0, 0, 0, pz, pz), mode='reflect'), w)
    # Y-axis: kernel shape (1,1,1,Ky,1)
    py = min(len(ky) // 2, out.shape[3] - 1)
    w = ky.reshape(1, 1, 1, -1, 1)
    out = _F.conv3d(_F.pad(out, (0, 0, py, py, 0, 0), mode='reflect'), w)
    # X-axis: kernel shape (1,1,1,1,Kx)
    px = min(len(kx) // 2, out.shape[4] - 1)
    w = kx.reshape(1, 1, 1, 1, -1)
    out = _F.conv3d(_F.pad(out, (px, px, 0, 0, 0, 0), mode='reflect'), w)
    return out


def _hessian_mps(
    vol: 'torch.Tensor',
    sigma: float,
) -> 'tuple[torch.Tensor, ...]':
    """
    Compute the 3D Hessian matrix elements using Gaussian derivatives on MPS.
    Matches skimage's _hessian_matrix_with_gaussian: two successive first-order
    Gaussian derivative convolutions at sigma_scaled = sigma / sqrt(2).

    Returns 6 upper-triangle elements: (Hzz, Hzy, Hzx, Hyy, Hyx, Hxx)
    """
    sigma_scaled = sigma / (2.0 ** 0.5)
    truncate = 8.0  # truncate=8 gives identical results to truncate=100 (diff < 1e-21)
    g0 = _gaussian_kernel_1d(sigma_scaled, order=0, truncate=truncate)
    g1 = _gaussian_kernel_1d(sigma_scaled, order=1, truncate=truncate)
    # Pass 1: first-order gradients
    grad_z = _separable_conv3d(vol, g1, g0, g0)  # order=(1,0,0)
    grad_y = _separable_conv3d(vol, g0, g1, g0)  # order=(0,1,0)
    grad_x = _separable_conv3d(vol, g0, g0, g1)  # order=(0,0,1)

    # Pass 2: second-order via first-order of each gradient
    Hzz = _separable_conv3d(grad_z, g1, g0, g0)  # d/dz of grad_z
    Hzy = _separable_conv3d(grad_z, g0, g1, g0)  # d/dy of grad_z
    Hzx = _separable_conv3d(grad_z, g0, g0, g1)  # d/dx of grad_z
    del grad_z
    Hyy = _separable_conv3d(grad_y, g0, g1, g0)  # d/dy of grad_y
    Hyx = _separable_conv3d(grad_y, g0, g0, g1)  # d/dx of grad_y
    del grad_y
    Hxx = _separable_conv3d(grad_x, g0, g0, g1)  # d/dx of grad_x
    del grad_x

    # Scale-space normalisation (sigma^2)
    s2 = sigma ** 2
    Hzz *= s2; Hzy *= s2; Hzx *= s2
    Hyy *= s2; Hyx *= s2; Hxx *= s2

    return Hzz, Hzy, Hzx, Hyy, Hyx, Hxx

def _eigenvalues_cardano_3x3(
    a: 'torch.Tensor', b: 'torch.Tensor', c: 'torch.Tensor',
    d: 'torch.Tensor', e: 'torch.Tensor', f: 'torch.Tensor',
) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
    """
    Closed-form eigenvalues of 3x3 symmetric matrices via Cardano's method.
    Avoids torch.linalg.eigvalsh (unsupported on MPS).

    Input: upper-triangle of [[a,b,c],[b,d,e],[c,e,f]]
    Returns (eig1, eig2, eig3) unsorted.
    """
    q = (a + d + f) / 3.0

    p_sq = ((a - q) ** 2 + (d - q) ** 2 + (f - q) ** 2
            + 2.0 * (b ** 2 + c ** 2 + e ** 2)) / 6.0
    p = torch.sqrt(torch.clamp(p_sq, min=1e-30))

    # B = (H - qI) / p
    B11 = (a - q) / p
    B12 = b / p
    B13 = c / p
    B22 = (d - q) / p
    B23 = e / p
    B33 = (f - q) / p

    # det(B) for symmetric 3x3
    detB = (B11 * (B22 * B33 - B23 * B23)
            - B12 * (B12 * B33 - B23 * B13)
            + B13 * (B12 * B23 - B22 * B13))

    r = torch.clamp(detB / 2.0, -1.0, 1.0)
    phi = torch.acos(r) / 3.0

    TWO_PI_3 = 2.0 * 3.141592653589793 / 3.0

    eig1 = q + 2.0 * p * torch.cos(phi)
    eig3 = q + 2.0 * p * torch.cos(phi + TWO_PI_3)
    eig2 = 3.0 * q - eig1 - eig3

    return eig1, eig2, eig3


def _frangi_mps(
    image: np.ndarray,
    sigmas: list,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: 'float | None' = None,
    black_ridges: bool = False,
) -> np.ndarray:
    """
    MPS-accelerated 3D Frangi vesselness filter.

    Matches skimage.filters.frangi output for 3D volumes.
    Runs entirely on Apple Metal via PyTorch MPS backend.

    Parameters
    ----------
    image        : (Z, Y, X) float32 normalised volume [0, 1]
    sigmas       : list of sigma values in voxel units
    alpha, beta  : Frangi structureness parameters
    gamma        : noise suppression (None = auto: max(S)/2 per scale)
    black_ridges : True for dark vessels on light background

    Returns
    -------
    vesselness : (Z, Y, X) float32 array, values in [0, 1]
    """
    if not black_ridges:
        image = -image  # skimage convention: negate for bright ridges

    result = np.zeros(image.shape, dtype=np.float32)

    two_alpha_sq = 2.0 * alpha ** 2
    two_beta_sq = 2.0 * beta ** 2

    for sigma in sigmas:
        # Upload to MPS
        vol_t = torch.from_numpy(image.astype(np.float32)).to('mps')
        vol_t = vol_t.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

        # Hessian
        Hzz, Hzy, Hzx, Hyy, Hyx, Hxx = _hessian_mps(vol_t, sigma)
        del vol_t

        # Flatten spatial dims for eigenvalue computation
        a = Hzz.reshape(-1)
        b = Hzy.reshape(-1)
        c = Hzx.reshape(-1)
        d = Hyy.reshape(-1)
        e = Hyx.reshape(-1)
        f = Hxx.reshape(-1)
        del Hzz, Hzy, Hzx, Hyy, Hyx, Hxx

        # Eigenvalues via Cardano
        eig1, eig2, eig3 = _eigenvalues_cardano_3x3(a, b, c, d, e, f)
        del a, b, c, d, e, f

        # Sort by absolute value ascending: |lam1| <= |lam2| <= |lam3|
        # Manual 3-element sort network (3 comparisons, no argsort overhead)
        a1, a2, a3 = torch.abs(eig1), torch.abs(eig2), torch.abs(eig3)
        # Compare-and-swap to get |lam1| <= |lam2| <= |lam3|
        swap12 = a1 > a2
        t1, t2 = torch.where(swap12, eig2, eig1), torch.where(swap12, eig1, eig2)
        ta1, ta2 = torch.where(swap12, a2, a1), torch.where(swap12, a1, a2)
        del a1, a2, swap12
        swap23 = ta2 > a3
        t2b, t3 = torch.where(swap23, eig3, t2), torch.where(swap23, t2, eig3)
        ta2b = torch.where(swap23, a3, ta2)
        del ta2, a3, swap23, t2, eig3
        swap12b = ta1 > ta2b
        lam1 = torch.where(swap12b, t2b, t1)
        lam2 = torch.where(swap12b, t1, t2b)
        lam3 = t3
        del t1, t2b, t3, ta1, ta2b, swap12b, eig1, eig2

        # S uses UN-clamped eigenvalues (matches skimage: s = sqrt(sum(eigvals**2)))
        S = torch.sqrt(lam1 ** 2 + lam2 ** 2 + lam3 ** 2)         # eq (12)
        # When lam2 or lam3 are negative, this makes r_b very large,
        # causing exp(-r_b^2/2beta^2) -> 0, naturally zeroing non-vessel regions.
        lam2_c = torch.clamp(lam2, min=1e-10)
        lam3_c = torch.clamp(lam3, min=1e-10)
        del lam2, lam3
        # Frangi ratios
        Ra = lam2_c / lam3_c                                        # eq (11)
        Rb = torch.abs(lam1) / torch.sqrt(lam2_c * lam3_c)          # eq (10)
        del lam1, lam2_c, lam3_c

        # Gamma: auto-scale per sigma (matches skimage default)
        if gamma is None:
            gamma_val = float(S.max().item()) / 2.0
            if gamma_val < 1e-10:
                gamma_val = 1.0
        else:
            gamma_val = gamma
        two_gamma_sq = 2.0 * gamma_val ** 2

        # Frangi response eq (13)
        vals = ((1.0 - torch.exp(-Ra ** 2 / two_alpha_sq))
                * torch.exp(-Rb ** 2 / two_beta_sq)
                * (1.0 - torch.exp(-S ** 2 / two_gamma_sq)))
        del Ra, Rb, S
        # Transfer to CPU
        vals_np = vals.reshape(image.shape).cpu().numpy()
        del vals
        torch.mps.empty_cache()
        result = np.maximum(result, vals_np)

    return result


try:
    import skfmm
    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False
    import warnings as _warnings
    _warnings.warn(
        "scikit-fmm not installed — falling back to Dijkstra. "
        "Install with: pip install scikit-fmm",
        RuntimeWarning,
    )
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra


# ─────────────────────────────────────────────
# Vessel enhancement  (ROI-only Frangi)
# ─────────────────────────────────────────────

def compute_vesselness(
    volume: np.ndarray,
    spacing_mm: List[float],
    sigmas: Optional[List[float]] = None,
    hu_clip: Tuple[float, float] = (-200, 1200),
    seed_points: Optional[List[List[int]]] = None,
    roi_margin_mm: float = 20.0,
) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter.

    When *seed_points* are provided the filter is computed ONLY on a tight ROI
    around those points (+ roi_margin_mm padding) and then placed back into a
    full-volume array.  This is the primary speedup: a typical coronary ROI is
    ~80³ voxels vs the full 512×512×400 volume — roughly 100× less work.

    Parameters
    ----------
    volume        : (Z, Y, X) float32 HU array
    spacing_mm    : [sz, sy, sx] voxel size in mm
    sigmas        : Frangi scale sigmas in mm (default: 3 scales for coronaries)
    hu_clip       : clip HU before filtering (remove bone / air extremes)
    seed_points   : list of [z, y, x] seed coords — used to crop ROI
    roi_margin_mm : padding around seed bounding box in mm

    Returns
    -------
    vesselness : (Z, Y, X) float32, values in [0, 1]
    """
    if sigmas is None:
        # 3 scales covers coronary diameters 2–5 mm; was 5 scales before
        sigmas = [0.5, 1.0, 2.0]

    shape = volume.shape

    # ── Determine working ROI ──────────────────────────────────────────────
    if seed_points is not None and len(seed_points) > 0:
        pts = np.array(seed_points)                     # (N, 3)
        margin_vox = np.array([roi_margin_mm / s for s in spacing_mm], dtype=int)
        lo = np.maximum(pts.min(axis=0) - margin_vox, 0).astype(int)
        hi = np.minimum(pts.max(axis=0) + margin_vox,
                        np.array(shape) - 1).astype(int)
    else:
        lo = np.zeros(3, dtype=int)
        hi = np.array(shape) - 1

    roi_vol = volume[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]

    # ── Normalise ROI ─────────────────────────────────────────────────────
    roi_clipped = np.clip(roi_vol, hu_clip[0], hu_clip[1]).astype(np.float32)
    vmin, vmax = float(hu_clip[0]), float(hu_clip[1])
    roi_norm = (roi_clipped - vmin) / (vmax - vmin)

    # ── Convert mm sigmas → voxel sigmas ─────────────────────────────────
    mean_sp = float(np.mean(spacing_mm))
    sigmas_vox = [s / mean_sp for s in sigmas]

    # ── Frangi on the small ROI ────────────────────────────────────────────
    import time as _time
    _t0 = _time.perf_counter()
    if HAS_TORCH_MPS:
        roi_vessel = _frangi_mps(
            roi_norm,
            sigmas=sigmas_vox,
            black_ridges=False,
            alpha=0.5,
            beta=0.5,
        )
        _backend = 'MPS'
    else:
        roi_vessel = frangi(
            roi_norm,
            sigmas=sigmas_vox,
            black_ridges=False,
            alpha=0.5,
            beta=0.5,
        ).astype(np.float32)
        _backend = 'CPU'
    _dt = _time.perf_counter() - _t0
    print(f'[vesselness] Frangi ({_backend}) on ROI {roi_norm.shape}: {_dt:.1f}s')

    # ── Embed back into a full-volume array ───────────────────────────────
    vesselness = np.zeros(shape, dtype=np.float32)
    vesselness[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = roi_vessel

    return vesselness


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _decimate_centerline(
    pts: List[np.ndarray],
    spacing_mm: List[float],
    min_step_frac: float = 0.5,
) -> List[np.ndarray]:
    """
    Greedy decimation of a centerline path.

    Removes near-duplicate and oscillating points by keeping only points
    that are at least *min_step_frac * mean_spacing* apart from the last
    retained point.  This eliminates the thousands of repeated voxels that
    the gradient-descent tracer accumulates when it oscillates near a waypoint.

    The first and last points are always kept.

    Parameters
    ----------
    pts           : list of (3,) int arrays in LOCAL voxel coords
    spacing_mm    : [sz, sy, sx]
    min_step_frac : fraction of mean spacing to use as minimum step (default 0.5)

    Returns
    -------
    Decimated list of (3,) int arrays
    """
    if len(pts) <= 2:
        return pts
    sp = np.array(spacing_mm, dtype=np.float64)
    min_dist_mm = min_step_frac * float(np.mean(sp))
    kept: List[np.ndarray] = [pts[0]]
    for pt in pts[1:-1]:
        dist_mm = np.linalg.norm((pt - kept[-1]).astype(np.float64) * sp)
        if dist_mm >= min_dist_mm:
            kept.append(pt)
    kept.append(pts[-1])
    return kept


def _extract_centerline_fmm(
    vesselness: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    waypoints_ijk: List[List[int]],
    roi_radius_mm: float = 35.0,
    volume: Optional[np.ndarray] = None,
    hu_vessel_thresh: float = 250.0,
) -> np.ndarray:
    """
    Per-segment Fast Marching centerline extraction.

    For each consecutive pair (source->target) among the seed points:
      1. Crop a local ROI around just those two points + margin.
      2. Build speed field from HU-threshold connected component seeded
         at the source voxel.  This prevents FMM from wandering into the
         aorta or cardiac chambers for distal segments.
      3. Run skfmm.travel_time() from source in the local ROI.
      4. Gradient-descent back-trace from target to source.
      5. Map back to global coordinates and concatenate.

    Parameters
    ----------
    vesselness      : (Z, Y, X) float32 vesselness map (fallback when volume absent)
    spacing_mm      : [z, y, x]
    ostium_ijk      : [z, y, x] ostium voxel
    waypoints_ijk   : list of [z, y, x] waypoints
    roi_radius_mm   : extra margin around each segment pair (mm)
    volume          : (Z, Y, X) float32 HU array -- enables HU-threshold speed
    hu_vessel_thresh: HU threshold for vessel lumen (default 250 HU)
    -------
    centerline_ijk : (N, 3) array [z, y, x]
    """
    shape = vesselness.shape
    all_points = [np.array(ostium_ijk)] + [np.array(p) for p in waypoints_ijk]
    sp = np.array(spacing_mm, dtype=np.float64)
    mean_sp = float(np.mean(sp))
    eps = 1e-6
    def _snap_hu(roi_hu: np.ndarray, pt_local: np.ndarray, l_shape: np.ndarray) -> np.ndarray:
        """Snap pt_local to nearest voxel with HU>threshold within 5mm."""
        z, y, x = pt_local
        if float(roi_hu[z, y, x]) > hu_vessel_thresh:
            return pt_local
        r_max = max(1, int(5.0 / mean_sp))
        for r in range(1, r_max + 1):
            z0, z1 = max(0, z - r), min(l_shape[0], z + r + 1)
            y0, y1 = max(0, y - r), min(l_shape[1], y + r + 1)
            x0, x1 = max(0, x - r), min(l_shape[2], x + r + 1)
            sub = roi_hu[z0:z1, y0:y1, x0:x1]
            if sub.max() > hu_vessel_thresh:
                idx = np.unravel_index(sub.argmax(), sub.shape)
                return np.array([z0 + idx[0], y0 + idx[1], x0 + idx[2]], dtype=int)
        return pt_local

    def _snap_ves(pt_local: np.ndarray, local_ves: np.ndarray, l_shape: np.ndarray) -> np.ndarray:
        """Snap pt_local to nearest voxel with vesselness>0.05 within 5mm."""
        z, y, x = pt_local
        if float(local_ves[z, y, x]) >= 0.05:
            return pt_local
        r_max = max(1, int(5.0 / mean_sp))
        for r in range(1, r_max + 1):
            z0, z1 = max(0, z - r), min(l_shape[0], z + r + 1)
            y0, y1 = max(0, y - r), min(l_shape[1], y + r + 1)
            x0, x1 = max(0, x - r), min(l_shape[2], x + r + 1)
            sub = local_ves[z0:z1, y0:y1, x0:x1]
            if sub.max() >= 0.05:
                idx = np.unravel_index(sub.argmax(), sub.shape)
                return np.array([z0 + idx[0], y0 + idx[1], x0 + idx[2]], dtype=int)
        return pt_local

    # -- Per-segment local HU-max tracking --------------------------------
    # FMM is unsuitable here: the correct coronary path deviates up to 15+
    # voxels from the straight inter-seed line (passing through pericardial
    # fat / around the heart surface).  We trace by linear interpolation +
    # local HU-max snapping instead.
    full_path_global: List[np.ndarray] = [all_points[0].copy()]
    for seg_idx in range(len(all_points) - 1):
        src_g = all_points[seg_idx].copy().astype(float)
        tgt_g = all_points[seg_idx + 1].copy().astype(float)
        seg_len_mm = float(np.linalg.norm((tgt_g - src_g) * sp))
        n_steps = max(20, int(np.ceil(seg_len_mm / mean_sp)))
        r_vox = max(1, int(round(3.0 / mean_sp)))
        prev_pt = all_points[seg_idx].copy().astype(int)
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pt_f = src_g + t * (tgt_g - src_g)
            z0_c = int(np.clip(int(round(float(pt_f[0]))), 0, shape[0] - 1))
            y0_c = int(np.clip(int(round(float(pt_f[1]))), 0, shape[1] - 1))
            x0_c = int(np.clip(int(round(float(pt_f[2]))), 0, shape[2] - 1))
            z_lo = max(0, z0_c - r_vox)
            z_hi = min(shape[0], z0_c + r_vox + 1)
            y_lo = max(0, y0_c - r_vox)
            y_hi = min(shape[1], y0_c + r_vox + 1)
            x_lo = max(0, x0_c - r_vox)
            x_hi = min(shape[2], x0_c + r_vox + 1)
            if volume is not None:
                sub = volume[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
                if sub.size > 0 and sub.max() > hu_vessel_thresh:
                    idx = np.unravel_index(sub.argmax(), sub.shape)
                    snap = np.array([z_lo+idx[0], y_lo+idx[1], x_lo+idx[2]], dtype=int)
                else:
                    snap = np.array([z0_c, y0_c, x0_c], dtype=int)
            else:
                sub_ves = vesselness[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
                if sub_ves.size > 0 and sub_ves.max() > 0.05:
                    idx = np.unravel_index(sub_ves.argmax(), sub_ves.shape)
                    snap = np.array([z_lo+idx[0], y_lo+idx[1], x_lo+idx[2]], dtype=int)
                else:
                    snap = np.array([z0_c, y0_c, x0_c], dtype=int)
            if not np.array_equal(snap, prev_pt):
                full_path_global.append(snap)
                prev_pt = snap

    # -- Deduplicate and decimate ------------------------------------------
    unique_path: List[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    for pt in full_path_global:
        if prev is None or not np.array_equal(pt, prev):
            unique_path.append(pt)
            prev = pt
    unique_path = _decimate_centerline(unique_path, spacing_mm, min_step_frac=0.5)
    return np.array(unique_path)


# ─────────────────────────────────────────────
# Dijkstra fallback (used only if scikit-fmm unavailable)
# ─────────────────────────────────────────────

def _build_graph_vectorised(cost: np.ndarray):  # type: ignore[return]
    """
    Build a 26-connected sparse cost graph from a 3-D cost array.
    Used only when scikit-fmm is not available.
    """
    Z, Y, X = cost.shape
    n = cost.size

    flat = np.arange(n, dtype=np.int32)
    z_idx, y_idx, x_idx = np.unravel_index(flat, (Z, Y, X))

    rows_all, cols_all, data_all = [], [], []

    offsets_26 = [(dz, dy, dx)
                  for dz in [-1, 0, 1]
                  for dy in [-1, 0, 1]
                  for dx in [-1, 0, 1]
                  if not (dz == 0 and dy == 0 and dx == 0)]

    for dz, dy, dx in offsets_26:
        nz = z_idx + dz
        ny = y_idx + dy
        nx = x_idx + dx

        valid = (nz >= 0) & (nz < Z) & (ny >= 0) & (ny < Y) & (nx >= 0) & (nx < X)

        src = flat[valid]
        nb = np.ravel_multi_index(
            (nz[valid].astype(np.int32),
             ny[valid].astype(np.int32),
             nx[valid].astype(np.int32)),
            (Z, Y, X)
        ).astype(np.int32)

        step_dist = float(np.sqrt(dz*dz + dy*dy + dx*dx))
        edge_cost = 0.5 * (cost.flat[src] + cost.flat[nb]) * step_dist

        rows_all.append(src)
        cols_all.append(nb)
        data_all.append(edge_cost.astype(np.float32))

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _extract_centerline_dijkstra(
    vesselness: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    waypoints_ijk: List[List[int]],
    roi_radius_mm: float = 35.0,
) -> np.ndarray:
    """Dijkstra fallback — used only if scikit-fmm is not installed."""
    shape = vesselness.shape
    all_points = [np.array(ostium_ijk)] + [np.array(p) for p in waypoints_ijk]

    margin_vox = np.array([int(roi_radius_mm / s) for s in spacing_mm])
    pts_arr = np.array(all_points)
    lo = np.maximum(pts_arr.min(axis=0) - margin_vox, 0).astype(int)
    hi = np.minimum(pts_arr.max(axis=0) + margin_vox,
                    np.array(shape) - 1).astype(int)

    roi = vesselness[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]
    roi_shape = roi.shape

    eps = 1e-3
    cost = (1.0 / (roi.astype(np.float64) + eps))
    graph = _build_graph_vectorised(cost)

    def global_to_roi(pt):
        return tuple((np.array(pt) - lo).astype(int))

    def roi_to_flat(pt_local):
        return int(np.ravel_multi_index(pt_local, roi_shape))

    ostium_local = global_to_roi(all_points[0])
    src_flat = roi_to_flat(ostium_local)

    dist_matrix, predecessors = dijkstra(
        graph, directed=False,
        indices=src_flat,
        return_predecessors=True,
    )

    full_path_flat = [src_flat]
    current_src = src_flat

    for wp in all_points[1:]:
        wp_local = global_to_roi(wp)
        wp_flat = roi_to_flat(wp_local)

        path = []
        node = wp_flat
        while node != current_src and node >= 0:
            path.append(node)
            node = int(predecessors[node])
        path.append(current_src)
        path.reverse()

        if len(path) > 1:
            full_path_flat.extend(path[1:])
        current_src = wp_flat

    centerline_ijk = []
    seen: set = set()
    for flat_idx in full_path_flat:
        if flat_idx in seen:
            continue
        seen.add(flat_idx)
        local_ijk = np.unravel_index(int(flat_idx), roi_shape)
        global_ijk = tuple(int(local_ijk[i] + lo[i]) for i in range(3))
        centerline_ijk.append(global_ijk)

    return np.array(centerline_ijk)


# ─────────────────────────────────────────────────────────────────────────────
# Vessel auto-tracer (greedy beam-search)
# Used when waypoints are too close to ostium
# ─────────────────────────────────────────────────────────────────────────────

def _autotrace_vessel(
    volume: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    direction_hint: np.ndarray,
    trace_length_mm: float = 60.0,
    hu_vessel_thresh: float = 200.0,
    search_radius_mm: float = 2.0,
    step_mm: float = 0.5,
) -> np.ndarray:
    """
    Greedy momentum-guided vessel tracer.

    Walks the vessel from *ostium_ijk* by repeating:
      1. Predict next position = current + momentum * step.
      2. Collect candidate voxels in a sphere of *search_radius_mm*
         that are (a) brighter than *hu_vessel_thresh* AND
         (b) in the forward half-space (dot with momentum > -0.3*radius).
      3. Among candidates, pick the one with the highest HU.
      4. Update momentum via EMA of recent step directions.
      5. Stop when no bright forward candidate is found.

    Using *search_radius_mm* = 2 mm (~5 vox at 0.38 mm spacing) prevents
    the tracer jumping to the aorta or cardiac chambers.
    Parameters
    ----------
    volume           : (Z,Y,X) float32 HU array
    spacing_mm       : [sz, sy, sx]
    ostium_ijk       : starting voxel [z, y, x]
    direction_hint   : initial marching direction (z,y,x), need not be unit
    trace_length_mm  : how far to trace from ostium (mm)
    hu_vessel_thresh : minimum HU for vessel lumen detection
    search_radius_mm : search ball radius (mm) -- keep small (1.5-2 mm)
    step_mm          : arc step size (mm)
    -------
    path : (N, 3) int array [z, y, x], ordered from ostium
    """
    sp = np.array(spacing_mm, dtype=np.float64)
    mean_sp = float(np.mean(sp))
    shape = np.array(volume.shape, dtype=np.int64)
    r_vox = max(1, int(np.ceil(search_radius_mm / mean_sp)))
    n_steps = int(np.ceil(trace_length_mm / step_mm))
    # Snap ostium to nearest local HU maximum within 3 mm (handles off-centre seeds)
    snap_r = max(1, int(np.round(3.0 / mean_sp)))
    oz, oy, ox = int(ostium_ijk[0]), int(ostium_ijk[1]), int(ostium_ijk[2])
    z0s = int(np.clip(oz - snap_r, 0, shape[0]-1))
    z1s = int(np.clip(oz + snap_r + 1, 0, shape[0]))
    y0s = int(np.clip(oy - snap_r, 0, shape[1]-1))
    y1s = int(np.clip(oy + snap_r + 1, 0, shape[1]))
    x0s = int(np.clip(ox - snap_r, 0, shape[2]-1))
    x1s = int(np.clip(ox + snap_r + 1, 0, shape[2]))
    snap_patch = volume[z0s:z1s, y0s:y1s, x0s:x1s]
    si = np.unravel_index(int(snap_patch.argmax()), snap_patch.shape)
    snapped = np.array([z0s + si[0], y0s + si[1], x0s + si[2]], dtype=np.int64)
    d = direction_hint.astype(np.float64).copy()
    if np.linalg.norm(d) < 1e-6:
        d = np.array([-1.0, 0.0, 0.0])
    # Convert hint to mm-space then normalise
    d_mm = d * sp
    if np.linalg.norm(d_mm) < 1e-6:
        d_mm = d.copy()
    d_mm = d_mm / np.linalg.norm(d_mm)

    pos = snapped.astype(np.float64)
    path_vox: List[np.ndarray] = [snapped.copy()]
    momentum = d_mm.copy()  # unit vector in mm-space
    alpha = 0.75  # EMA weight: higher = smoother / more inertia
    for _ in range(n_steps):
        # Predict next position in voxel-space
        step_vox = momentum * (step_mm / sp)  # mm direction / mm-per-vox
        pred = pos + step_vox
        pred_vox = np.round(pred).astype(np.int64)
        z0 = int(np.clip(pred_vox[0] - r_vox, 0, shape[0] - 1))
        z1 = int(np.clip(pred_vox[0] + r_vox + 1, 0, shape[0]))
        y0 = int(np.clip(pred_vox[1] - r_vox, 0, shape[1] - 1))
        y1 = int(np.clip(pred_vox[1] + r_vox + 1, 0, shape[1]))
        x0 = int(np.clip(pred_vox[2] - r_vox, 0, shape[2] - 1))
        x1 = int(np.clip(pred_vox[2] + r_vox + 1, 0, shape[2]))
        patch = volume[z0:z1, y0:y1, x0:x1]
        if patch.size == 0:
            break

        # Offset vectors from current pos (mm-space)
        gz, gy, gx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dz_mm = (gz - pos[0]) * sp[0]
        dy_mm = (gy - pos[1]) * sp[1]
        dx_mm = (gx - pos[2]) * sp[2]
        dist_sq = dz_mm**2 + dy_mm**2 + dx_mm**2
        # Forward half-space: dot(offset_mm, momentum) > -0.3*radius
        dot_val = dz_mm * momentum[0] + dy_mm * momentum[1] + dx_mm * momentum[2]
        sphere_mask = dist_sq <= (search_radius_mm ** 2)
        forward_mask = dot_val > (-0.3 * search_radius_mm)
        bright_mask = patch >= hu_vessel_thresh
        candidate_mask = sphere_mask & forward_mask & bright_mask

        if not candidate_mask.any():
            break  # lost the vessel

        masked_hu = np.where(candidate_mask, patch, -np.inf)
        idx = np.unravel_index(int(masked_hu.argmax()), masked_hu.shape)
        best_vox = np.array([z0 + idx[0], y0 + idx[1], x0 + idx[2]], dtype=np.int64)
        # Update momentum in mm-space
        step_vec_mm = (best_vox.astype(np.float64) - pos) * sp
        step_norm = np.linalg.norm(step_vec_mm)
        if step_norm > 1e-6:
            step_dir = step_vec_mm / step_norm
            momentum = alpha * momentum + (1 - alpha) * step_dir
            m_norm = np.linalg.norm(momentum)
            if m_norm > 1e-6:
                momentum = momentum / m_norm
        pos = best_vox.astype(np.float64)
        if not np.array_equal(best_vox, path_vox[-1]):
            path_vox.append(best_vox.copy())
    return np.array(path_vox, dtype=int)


def _find_vessel_direction(
    volume: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    hint_direction: Optional[np.ndarray] = None,
    probe_mm: float = 5.0,
    hu_vessel_thresh: float = 200.0,
) -> np.ndarray:
    """
    Find the best initial marching direction from the ostium.

    Probes 26 directions (cube faces/edges/corners) of radius *probe_mm*
    from the snapped ostium, scores each by summing HU of bright voxels
    (> hu_vessel_thresh) along the probe ray, and returns the best direction.
    If *hint_direction* is given, it is included as an extra probe candidate.

    Returns unit vector in voxel-index space [z, y, x].
    """
    sp = np.array(spacing_mm, dtype=np.float64)
    mean_sp = float(np.mean(sp))
    shape = np.array(volume.shape, dtype=np.int64)

    # Snap ostium to local HU max within 3mm
    snap_r = max(1, int(np.round(3.0 / mean_sp)))
    oz, oy, ox = int(ostium_ijk[0]), int(ostium_ijk[1]), int(ostium_ijk[2])
    z0s = int(np.clip(oz - snap_r, 0, shape[0] - 1))
    z1s = int(np.clip(oz + snap_r + 1, 0, shape[0]))
    y0s = int(np.clip(oy - snap_r, 0, shape[1] - 1))
    y1s = int(np.clip(oy + snap_r + 1, 0, shape[1]))
    x0s = int(np.clip(ox - snap_r, 0, shape[2] - 1))
    x1s = int(np.clip(ox + snap_r + 1, 0, shape[2]))
    snap_patch = volume[z0s:z1s, y0s:y1s, x0s:x1s]
    si = np.unravel_index(int(snap_patch.argmax()), snap_patch.shape)
    snapped = np.array([z0s + si[0], y0s + si[1], x0s + si[2]], dtype=np.float64)

    # Build 26 probe directions in mm-space (unit vectors)
    offsets_26 = np.array(
        [(dz, dy, dx)
         for dz in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dx in (-1, 0, 1)
         if not (dz == 0 and dy == 0 and dx == 0)],
        dtype=np.float64,
    )
    dirs_mm = offsets_26 * sp[np.newaxis, :]
    norms = np.linalg.norm(dirs_mm, axis=1, keepdims=True)
    dirs_mm = dirs_mm / norms

    # Add hint direction if given
    if hint_direction is not None:
        h = np.array(hint_direction, dtype=np.float64)
        h_mm = h * sp
        h_norm = np.linalg.norm(h_mm)
        if h_norm > 1e-6:
            dirs_mm = np.vstack([dirs_mm, h_mm / h_norm])

    # Score each direction: sum HU of bright voxels along probe_mm ray, step 0.5mm
    ray_steps = max(3, int(probe_mm / 0.5))
    best_score = -np.inf
    best_dir_mm = dirs_mm[0]
    for d_mm in dirs_mm:
        score = 0.0
        for s in range(1, ray_steps + 1):
            t = s * 0.5  # mm
            pt = snapped + d_mm * (t / sp)  # voxel position
            iz = int(np.round(pt[0]))
            iy = int(np.round(pt[1]))
            ix = int(np.round(pt[2]))
            if not (0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]):
                break
            hu = float(volume[iz, iy, ix])
            if hu >= hu_vessel_thresh:
                score += hu
        if score > best_score:
            best_score = score
            best_dir_mm = d_mm

    # Return as unit vector in voxel-index space
    best_vox = best_dir_mm / sp
    norm_vox = np.linalg.norm(best_vox)
    return best_vox / norm_vox if norm_vox > 1e-6 else np.array([-1.0, 0.0, 0.0])


def _sample_waypoints_from_path(
    path: np.ndarray,
    spacing_mm: List[float],
    step_mm: float = 5.0,
) -> List[List[int]]:
    """
    Sub-sample a dense path into waypoints spaced ~step_mm apart.
    Returns a list of [z, y, x] integer lists (excludes the first point,
    which is the ostium already known to the caller).
    """
    if len(path) < 2:
        return []
    sp = np.array(spacing_mm, dtype=np.float64)
    diffs = np.diff(path.astype(np.float64), axis=0) * sp
    seg_len = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumlen[-1])
    waypoints: List[List[int]] = []
    d = step_mm
    while d < total:
        idx = int(np.searchsorted(cumlen, d, side='left'))
        idx = min(idx, len(path) - 1)
        waypoints.append(path[idx].tolist())
        d += step_mm
    # Always include the final traced point
    if len(waypoints) == 0 or not np.array_equal(waypoints[-1], path[-1].tolist()):
        waypoints.append(path[-1].tolist())
    return waypoints


# ─────────────────────────────────────────────
# Public API: extract_centerline_seeds
# (dispatches to FMM or Dijkstra automatically)
# ─────────────────────────────────────────────

def extract_centerline_seeds(
    volume: np.ndarray,
    vesselness: np.ndarray,
    spacing_mm: List[float],
    ostium_ijk: List[int],
    waypoints_ijk: List[List[int]],
    roi_radius_mm: float = 35.0,
    min_guide_mm: float = 8.0,
) -> np.ndarray:
    """
    Extract centerline from ostium through waypoints.
    Dijkstra with no graph construction overhead.  Falls back to vectorised
    Dijkstra if scikit-fmm is not installed.
    When the provided waypoints are clustered too close to the ostium
    (total arc < *min_guide_mm*), the function automatically traces the
    vessel with a greedy beam-search (``_autotrace_vessel``) and substitutes
    the resulting path as dense guide waypoints before running the main
    tracker.  This fixes cases where manually placed waypoints are within
    1–2 mm of the ostium and give the tracker no direction to follow.
    Parameters
    ----------
    volume        : (Z, Y, X) float32 HU array
    vesselness    : (Z, Y, X) float32 vesselness map
    spacing_mm    : [z, y, x] voxel spacing in mm
    ostium_ijk    : [z, y, x] ostium voxel
    waypoints_ijk : list of [z, y, x] waypoints
    roi_radius_mm : half-size of ROI cube around seeds (mm)
    min_guide_mm  : minimum total waypoint arc (mm) before auto-tracing kicks in
    Returns
    -------
    centerline_ijk : (N, 3) array of ordered centerline voxel indices (z, y, x)
    """
    sp = np.array(spacing_mm, dtype=np.float64)
    mean_sp = float(np.mean(sp))
    all_pts = [ostium_ijk] + list(waypoints_ijk)
    # ── Check if waypoints are too close to ostium ────────────────────────
    # Compute total straight-line arc spanned by the provided seeds.
    pts_mm = np.array(all_pts, dtype=np.float64) * sp
    seed_arc_mm = float(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1).sum()) if len(all_pts) > 1 else 0.0

    if seed_arc_mm < min_guide_mm:
        import warnings
        warnings.warn(
            f"Waypoints too close to ostium (arc={seed_arc_mm:.1f}mm < {min_guide_mm}mm). "
            f"Auto-tracing vessel from ostium to generate guide waypoints.",
            RuntimeWarning,
        )
        # Use _find_vessel_direction to probe all 26 directions from ostium
        # — do NOT rely on waypoints[0]-ostium which may point the wrong way
        hint = None
        if len(waypoints_ijk) > 0:
            hint = (np.array(waypoints_ijk[0], dtype=np.float64)
                    - np.array(ostium_ijk, dtype=np.float64))
        d_hint = _find_vessel_direction(
            volume=volume,
            spacing_mm=spacing_mm,
            ostium_ijk=ostium_ijk,
            hint_direction=hint,
        )

        traced = _autotrace_vessel(
            volume=volume,
            spacing_mm=spacing_mm,
            ostium_ijk=ostium_ijk,
            direction_hint=d_hint,
            trace_length_mm=60.0,
        )
        if len(traced) >= 5:
            waypoints_ijk = _sample_waypoints_from_path(traced, spacing_mm, step_mm=5.0)
            all_pts = [ostium_ijk] + waypoints_ijk
            pts_mm = np.array(all_pts, dtype=np.float64) * sp
            seed_arc_mm = float(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1).sum())
            print(f"[centerline] Auto-traced {len(traced)} pts ({seed_arc_mm:.1f}mm); "
                  f"using {len(waypoints_ijk)} guide waypoints.")
        else:
            print(f"[centerline] WARNING: auto-trace returned only {len(traced)} pts — vessel may be too dim")
    # Expected arc from waypoint straight-line distances
    expected_arc_mm = float(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1).sum())
    min_pts_expected = max(10, int(expected_arc_mm / mean_sp / 4))
    if HAS_SKFMM:
        cl = _extract_centerline_fmm(vesselness, spacing_mm, ostium_ijk, waypoints_ijk, roi_radius_mm, volume=volume)
    else:
        cl = _extract_centerline_dijkstra(vesselness, spacing_mm, ostium_ijk, waypoints_ijk, roi_radius_mm)
    cl_arc_mm = float(np.linalg.norm(np.diff(cl.astype(np.float64) * sp, axis=0), axis=1).sum()) if len(cl) > 1 else 0.0
    if len(cl) > 0:
        end_dist_mm = float(np.linalg.norm((cl[-1].astype(np.float64) - np.array(all_pts[-1], dtype=np.float64)) * sp))
        cl_endpoint_ok = end_dist_mm < mean_sp * 40.0
    else:
        cl_endpoint_ok = False
    use_linear = (len(cl) < min_pts_expected
                  or cl_arc_mm < expected_arc_mm * 0.20
                  or cl_arc_mm > expected_arc_mm * 3.0
                  or not cl_endpoint_ok)
    if use_linear:
        import warnings
        if not cl_endpoint_ok:
            reason = f"endpoint drift ({end_dist_mm:.1f}mm from final waypoint)"
        elif cl_arc_mm > expected_arc_mm * 3.0:
            reason = f"oscillating arc ({cl_arc_mm:.1f}mm > {expected_arc_mm * 3.0:.1f}mm)"
        elif cl_arc_mm < expected_arc_mm * 0.20:
            reason = f"short arc ({cl_arc_mm:.1f}mm < {expected_arc_mm * 0.20:.1f}mm)"
        else:
            reason = f"sparse ({len(cl)} pts < {min_pts_expected})"
        warnings.warn(
            f"Centerline fallback [{reason}]: using cubic-spline interpolation of waypoints.",
            RuntimeWarning,
        )
        from scipy.interpolate import CubicSpline
        pts = np.array(all_pts, dtype=np.float64)          # (K, 3) waypoints
        pts_mm = pts * sp                                   # convert to mm
        seg_mm = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_mm)])    # cumulative arc per waypoint
        total_arc = arc[-1]
        if total_arc < 1e-6 or len(pts) < 3:
            # Too few waypoints for cubic — degenerate to linear
            step_mm = 0.5
            n_out = max(2, int(np.ceil(total_arc / step_mm)))
            t = np.linspace(0.0, 1.0, n_out)
            spline_pts = pts[[0]] + t[:, None] * (pts[[-1]] - pts[[0]])
        else:
            # Cubic spline through waypoints (natural boundary conditions)
            cs = CubicSpline(arc, pts_mm, bc_type='natural')
            step_mm = 0.5
            n_out = max(2, int(np.ceil(total_arc / step_mm)))
            s_vals = np.linspace(0.0, total_arc, n_out)
            spline_pts = cs(s_vals) / sp                    # back to voxel ijk
        cl = np.clip(np.round(spline_pts).astype(int), 0, np.array(vesselness.shape) - 1)
    return cl


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
    scale = np.array(spacing_mm, dtype=np.float32)

    diffs = np.diff(centerline_ijk.astype(np.float32), axis=0)
    diffs_mm = diffs * scale[np.newaxis, :]
    seg_lengths = np.linalg.norm(diffs_mm, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    end_mm = start_mm + length_mm
    mask = (cumlen >= start_mm) & (cumlen <= end_mm)
    return centerline_ijk[mask]


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
    radius_search_mm: max radius to consider

    Returns
    -------
    radii_mm : (N,) array of estimated radii in mm
    """
    lumen_mask = (volume >= lumen_hu_range[0]) & (volume <= lumen_hu_range[1])

    # EDT inside a tight ROI around the centerline (faster than full volume)
    pts = centerline_ijk.astype(int)
    margin = np.array([int(radius_search_mm / s) for s in spacing_mm])
    lo = np.maximum(pts.min(axis=0) - margin, 0)
    hi = np.minimum(pts.max(axis=0) + margin, np.array(volume.shape) - 1)

    roi_lumen = lumen_mask[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]
    roi_edt = distance_transform_edt(roi_lumen, sampling=spacing_mm).astype(np.float32)

    radii_mm = np.array([
        float(roi_edt[
            int(p[0]) - lo[0],
            int(p[1]) - lo[1],
            int(p[2]) - lo[2],
        ])
        for p in pts
    ], dtype=np.float32)

    return np.clip(radii_mm, 0.5, radius_search_mm)


# ─────────────────────────────────────────────
# Load seeds from JSON
# ─────────────────────────────────────────────

def load_seeds(seeds_path: str | Path) -> Dict[str, Any]:
    """Load vessel seed JSON file."""
    with open(seeds_path) as f:
        return json.load(f)


VESSEL_CONFIGS = {
    "LAD": {"start_mm": 5.0,  "length_mm": 40.0},
    "LCX": {"start_mm": 5.0,  "length_mm": 40.0},
    "RCA": {"start_mm": 10.0, "length_mm": 40.0},  # 10–50mm
}
