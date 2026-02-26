"""
visualize.py
All PCAT analysis visualizations:

  Output 1: 3D volume render of PCAT VOI + artery centerlines (pyvista)
  Output 3: Curved Planar Reformat (CPR) of FAI values — yellow→red colormap
            for each vessel (RCA, LAD, LCX)
  Output 4: HU distribution histogram of fat voxels in the VOI
  Output 5: Radial HU profile — 1mm-step concentric rings 0→20mm from vessel wall

All functions save PNG files to output_dir and optionally display interactively.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import distance_transform_edt, map_coordinates

# Guard pyvista import — it may not be installed in all environments
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn(
        "pyvista not installed. 3D rendering (Output 1) will be skipped. "
        "Install with: pip install pyvista"
    )

matplotlib.use("Agg")   # Non-interactive backend for batch rendering

FAI_HU_MIN = -190.0
FAI_HU_MAX = -30.0


# ─────────────────────────────────────────────────────────────────────────────
# Output 1: 3D Volume Render (pyvista)
# ─────────────────────────────────────────────────────────────────────────────

def render_3d_voi(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    vessel_centerlines: Dict[str, np.ndarray],
    vessel_radii: Dict[str, np.ndarray],
    spacing_mm: List[float],
    output_dir: str | Path,
    prefix: str = "pcat",
    screenshot: bool = True,
    interactive: bool = False,
) -> Optional[Path]:
    """
    Render the PCAT VOI as a 3D semi-transparent surface with colored centerlines.

    Uses pyvista marching cubes on the VOI mask and tube glyphs for centerlines.

    Parameters
    ----------
    volume            : (Z, Y, X) HU float32 — full CCTA
    voi_mask          : (Z, Y, X) bool — PCAT VOI (union of all vessels)
    vessel_centerlines: dict {vessel_name: (N, 3) centerline voxel indices}
    vessel_radii      : dict {vessel_name: (N,) radii in mm}
    spacing_mm        : [sz, sy, sx]
    output_dir        : directory to save PNG
    prefix            : filename prefix
    screenshot        : save a PNG screenshot
    interactive       : open interactive window (False for batch/headless mode)

    Returns
    -------
    path to saved PNG, or None if pyvista unavailable
    """
    if not HAS_PYVISTA:
        print("[visualize] Skipping 3D render: pyvista not installed.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sz, sy, sx = spacing_mm

    # ── Build pyvista ImageData grid ──────────────────────────────────────
    pv_grid = pv.ImageData()
    pv_grid.dimensions = np.array(volume.shape)[::-1] + 1  # (X+1, Y+1, Z+1)
    pv_grid.spacing = (sx, sy, sz)   # pyvista uses (x, y, z) order
    pv_grid.origin = (0.0, 0.0, 0.0)

    # Attach VOI mask as scalar
    pv_grid.cell_data["voi"] = voi_mask.flatten(order="F").astype(np.uint8)

    # Extract VOI surface via threshold
    voi_surface = pv_grid.threshold(0.5, scalars="voi").extract_surface()

    # ── Attach HU scalars on the VOI surface ────────────────────────────
    # Clip to fat range for coloring
    pv_grid.cell_data["hu"] = np.clip(
        volume.flatten(order="F"), FAI_HU_MIN, FAI_HU_MAX
    ).astype(np.float32)
    voi_fat_cells = pv_grid.threshold(0.5, scalars="voi")
    hu_scalars = voi_fat_cells.cell_data["hu"]

    # ── Build centerline tubes ─────────────────────────────────────────
    VESSEL_COLORS = {"LAD": "red", "LCX": "blue", "RCA": "green"}
    tube_meshes = []

    for vessel_name, cl_ijk in vessel_centerlines.items():
        if len(cl_ijk) < 2:
            continue
        # Convert ijk (z, y, x) → mm coords (x, y, z) in pyvista space
        pts_xyz = cl_ijk[:, [2, 1, 0]] * np.array([sx, sy, sz])
        spline = pv.Spline(pts_xyz, n_points=max(len(pts_xyz) * 2, 100))
        mean_r = float(np.mean(vessel_radii.get(vessel_name, [1.5])))
        tube = spline.tube(radius=mean_r * 0.5)
        tube_meshes.append((vessel_name, tube, VESSEL_COLORS.get(vessel_name, "white")))

    # ── Render ────────────────────────────────────────────────────────
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(off_screen=not interactive, window_size=(1200, 900))

    # FAI-colored VOI surface
    fai_cmap = _fai_colormap()
    plotter.add_mesh(
        voi_fat_cells.extract_surface(),
        scalars="hu",
        clim=[FAI_HU_MIN, FAI_HU_MAX],
        cmap=fai_cmap,
        opacity=0.4,
        show_scalar_bar=True,
        scalar_bar_args={"title": "HU (FAI range)", "fmt": "%.0f"},
    )

    for vessel_name, tube, color in tube_meshes:
        plotter.add_mesh(tube, color=color, opacity=0.9, label=vessel_name)

    plotter.add_legend(face="rectangle", bcolor=[0.1, 0.1, 0.1], size=(0.18, 0.12))
    plotter.add_text(f"PCAT 3D — {prefix}", font_size=12, position="upper_edge")
    plotter.camera_position = "iso"

    if interactive:
        plotter.show()

    out_path = None
    if screenshot:
        out_path = output_dir / f"{prefix}_3d_voi.png"
        plotter.screenshot(str(out_path))
        print(f"[visualize] 3D render saved: {out_path.name}")

    plotter.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Output 3: Curved Planar Reformat (CPR) with FAI colormap
# ─────────────────────────────────────────────────────────────────────────────

def render_cpr_fai(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
) -> Path:
    """
    Compute and render a Curved Planar Reformat (CPR) of the FAI signal.

    For each centerline point:
      - Compute the tangent vector (finite difference)
      - Build a 2D sampling plane perpendicular to the tangent
      - Sample the volume on that plane
      - Stack planes to form a 2D CPR image

    FAI voxels (-190 to -30 HU) are colored yellow→red.
    Non-fat voxels are shown in grayscale (anatomic context).

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii
    spacing_mm      : [sz, sy, sx]
    vessel_name     : e.g. "LAD"
    output_dir      : output directory
    prefix          : filename prefix
    slab_thickness_mm: CPR plane half-width in the depth direction (for MIP)
    width_mm        : half-width of the CPR plane (lateral extent)

    Returns
    -------
    Path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sz, sy, sx = spacing_mm
    vox_size = np.array([sz, sy, sx])   # (z, y, x) mm per voxel
    shape = np.array(volume.shape)      # (Z, Y, X)

    N = len(centerline_ijk)
    if N < 3:
        print(f"[visualize] CPR: too few centerline points for {vessel_name}, skipping.")
        return None

    # Number of pixels across the CPR width
    n_width = int(np.ceil(2.0 * width_mm / float(np.mean(vox_size[1:]))))
    if n_width < 10:
        n_width = 50

    # Build tangent vectors (finite differences, normalized)
    cl = centerline_ijk.astype(float)   # (N, 3) in voxel units [z, y, x]
    tangents = np.zeros_like(cl)
    tangents[1:-1] = cl[2:] - cl[:-2]
    tangents[0] = cl[1] - cl[0]
    tangents[-1] = cl[-1] - cl[-2]

    # Scale to mm for proper normalization
    tangents_mm = tangents * vox_size[np.newaxis, :]
    norms = np.linalg.norm(tangents_mm, axis=1, keepdims=True) + 1e-12
    tangents_mm /= norms

    # Build per-point sampling planes
    # We need two orthogonal vectors (u, v) perpendicular to the tangent
    cpr_image = np.full((N, n_width), np.nan, dtype=np.float32)

    for i in range(N):
        t = tangents_mm[i]  # tangent in mm [z, y, x]

        # Find orthogonal vector u: use Gram-Schmidt against a reference
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(t, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, t) * t
        u /= np.linalg.norm(u) + 1e-12

        # Sample positions along u direction (width)
        w_range = np.linspace(-width_mm, width_mm, n_width)  # mm
        sample_pts_mm = cl[i] * vox_size + u * w_range[:, np.newaxis]  # (n_width, 3) mm [z,y,x]

        # Convert mm → voxel indices
        sample_vox = sample_pts_mm / vox_size[np.newaxis, :]  # (n_width, 3)

        # Extract coordinates for map_coordinates (z, y, x)
        coords = [
            sample_vox[:, 0],
            sample_vox[:, 1],
            sample_vox[:, 2],
        ]

        # Clip to volume bounds
        valid = (
            (coords[0] >= 0) & (coords[0] < shape[0] - 1) &
            (coords[1] >= 0) & (coords[1] < shape[1] - 1) &
            (coords[2] >= 0) & (coords[2] < shape[2] - 1)
        )

        vals = map_coordinates(volume, coords, order=1, mode="constant", cval=np.nan)
        vals[~valid] = np.nan
        cpr_image[i] = vals

    # Build composite image: grayscale base + FAI overlay
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # ── Grayscale base (anatomical context) ──────────────────────────────
    gray_img = np.clip(cpr_image, -200, 400)  # window for soft tissue
    gray_norm = (gray_img + 200) / 600.0       # normalize to [0, 1]
    gray_norm = np.nan_to_num(gray_norm, nan=0.0)

    # Show as grayscale background
    ax.imshow(
        gray_norm.T,
        aspect="auto",
        origin="lower",
        cmap="gray",
        vmin=0, vmax=1,
        interpolation="bilinear",
    )

    # ── FAI overlay (only -190 to -30 HU) ───────────────────────────────
    fai_img = np.where(
        (cpr_image >= FAI_HU_MIN) & (cpr_image <= FAI_HU_MAX),
        cpr_image,
        np.nan,
    )
    fai_cmap = _fai_colormap()
    fai_im = ax.imshow(
        fai_img.T,
        aspect="auto",
        origin="lower",
        cmap=fai_cmap,
        vmin=FAI_HU_MIN,
        vmax=FAI_HU_MAX,
        alpha=0.85,
        interpolation="bilinear",
    )

    # Colorbar
    cbar = plt.colorbar(fai_im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("HU (FAI)", fontsize=10)
    cbar.set_ticks([FAI_HU_MIN, -150, -100, -50, FAI_HU_MAX])

    # Axis labels
    arclengths = _compute_arclengths(centerline_ijk, spacing_mm)
    x_ticks_mm = np.arange(0, arclengths[-1] + 1, 10.0)
    x_tick_idxs = [int(np.argmin(np.abs(arclengths - t))) for t in x_ticks_mm]
    ax.set_xticks(x_tick_idxs)
    ax.set_xticklabels([f"{t:.0f}" for t in x_ticks_mm], fontsize=9)
    ax.set_xlabel("Distance along vessel (mm)", fontsize=11)

    w_ticks_mm = np.arange(-width_mm, width_mm + 1, 5.0)
    w_tick_idxs = ((w_ticks_mm + width_mm) / (2 * width_mm) * (n_width - 1)).astype(int)
    ax.set_yticks(np.clip(w_tick_idxs, 0, n_width - 1))
    ax.set_yticklabels([f"{t:.0f}" for t in w_ticks_mm], fontsize=9)
    ax.set_ylabel("Lateral distance from centerline (mm)", fontsize=11)

    ax.set_title(
        f"CPR — {vessel_name} — FAI overlay (HU {FAI_HU_MIN:.0f} to {FAI_HU_MAX:.0f})",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_fai.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] CPR FAI saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Output 4: HU Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_hu_histogram(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    n_bins: int = 80,
) -> Path:
    """
    Plot HU distribution of all voxels within the VOI, with FAI range highlighted.

    Parameters
    ----------
    volume      : (Z, Y, X) HU float32
    voi_mask    : (Z, Y, X) bool — pericoronary VOI
    vessel_name : label for the plot
    output_dir  : output directory
    prefix      : filename prefix
    n_bins      : number of histogram bins

    Returns
    -------
    Path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hu_vals = volume[voi_mask].astype(np.float64)
    fat_vals = hu_vals[(hu_vals >= FAI_HU_MIN) & (hu_vals <= FAI_HU_MAX)]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Full VOI distribution
    ax.hist(
        hu_vals,
        bins=n_bins,
        range=(-250, 500),
        color="#4A90D9",
        alpha=0.6,
        edgecolor="none",
        label=f"All VOI voxels (n={len(hu_vals):,})",
        density=True,
    )

    # FAI (fat) voxels
    ax.hist(
        fat_vals,
        bins=n_bins // 2,
        range=(FAI_HU_MIN, FAI_HU_MAX),
        color="#E8A020",
        alpha=0.85,
        edgecolor="none",
        label=f"FAI fat voxels (n={len(fat_vals):,})",
        density=True,
    )

    # Shade FAI region
    ax.axvspan(FAI_HU_MIN, FAI_HU_MAX, alpha=0.08, color="orange", label="FAI range")

    # Vertical lines
    ax.axvline(FAI_HU_MIN, color="orange", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(FAI_HU_MAX, color="red", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(float(np.mean(fat_vals)) if len(fat_vals) > 0 else 0,
               color="red", linewidth=1.5, linestyle="-",
               label=f"Mean FAI HU = {np.mean(fat_vals):.1f}" if len(fat_vals) > 0 else "")

    ax.set_xlabel("Hounsfield Unit (HU)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"PCAT HU Distribution — {vessel_name}\n"
        f"FAI fraction = {100 * len(fat_vals) / max(len(hu_vals), 1):.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-250, 500)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_hu_histogram.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] HU histogram saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Output 5: Radial HU Profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_radial_hu_profile(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    max_radial_mm: float = 20.0,
    ring_step_mm: float = 1.0,
) -> Path:
    """
    Plot radial HU profile: mean HU in 1mm-step concentric rings from vessel outer wall.

    X-axis: distance from coronary artery outer wall (mm) — 0 = outer vessel wall
    Y-axis: mean HU of FAI-filtered voxels (-190 to -30) in each ring

    The "outer vessel wall" is at distance = mean_radius from the centerline.
    Rings extend from there outward to max_radial_mm.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) centerline voxels [z, y, x]
    radii_mm        : (N,) per-point vessel radius in mm
    spacing_mm      : [sz, sy, sx]
    vessel_name     : label
    output_dir      : output directory
    prefix          : filename prefix
    max_radial_mm   : maximum radial distance from outer vessel wall
    ring_step_mm    : ring width in mm

    Returns
    -------
    Path to saved PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sz, sy, sx = spacing_mm
    mean_radius_mm = float(np.mean(radii_mm))

    # ── Build centerline distance transform in a local bounding box ──────
    max_outer_mm = mean_radius_mm + max_radial_mm + ring_step_mm
    margin_vox = np.array([
        int(np.ceil(max_outer_mm / sz)) + 3,
        int(np.ceil(max_outer_mm / sy)) + 3,
        int(np.ceil(max_outer_mm / sx)) + 3,
    ])

    lo = np.maximum(centerline_ijk.min(axis=0) - margin_vox, 0).astype(int)
    hi = np.minimum(centerline_ijk.max(axis=0) + margin_vox,
                    np.array(volume.shape) - 1).astype(int)
    sub_shape = tuple((hi - lo + 1).tolist())

    # Build centerline mask in subvolume
    cl_local = centerline_ijk - lo
    cl_mask_sub = np.zeros(sub_shape, dtype=bool)
    for pt in cl_local:
        z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
        if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
            cl_mask_sub[z, y, x] = True

    # Distance from each voxel to nearest centerline point (in mm)
    dist_from_centerline_mm = distance_transform_edt(~cl_mask_sub, sampling=spacing_mm)

    # Subvolume HU
    vol_sub = volume[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]

    # ── Compute mean HU per ring ─────────────────────────────────────────
    # Rings defined relative to vessel outer wall: ring_inner/outer from OUTER WALL
    # distance from outer wall = dist_from_centerline - mean_radius
    dist_from_wall_mm = dist_from_centerline_mm - mean_radius_mm  # negative inside vessel

    ring_edges = np.arange(0.0, max_radial_mm + ring_step_mm, ring_step_mm)
    ring_centers = ring_edges[:-1] + ring_step_mm / 2.0

    mean_hus = []
    std_hus = []
    n_voxels = []

    for r_inner, r_outer in zip(ring_edges[:-1], ring_edges[1:]):
        ring_mask = (
            (dist_from_wall_mm >= r_inner) &
            (dist_from_wall_mm < r_outer)
        )
        hu_ring = vol_sub[ring_mask]
        fat_ring = hu_ring[(hu_ring >= FAI_HU_MIN) & (hu_ring <= FAI_HU_MAX)]

        if len(fat_ring) > 0:
            mean_hus.append(float(np.mean(fat_ring)))
            std_hus.append(float(np.std(fat_ring)))
        else:
            mean_hus.append(np.nan)
            std_hus.append(np.nan)
        n_voxels.append(len(fat_ring))

    mean_hus = np.array(mean_hus)
    std_hus = np.array(std_hus)
    valid = ~np.isnan(mean_hus)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax1, ax2 = axes

    # — Left panel: Mean HU radial profile ——————————————————————————————
    colors = plt.cm.RdYlGn_r(np.linspace(0.0, 1.0, len(ring_centers)))

    # Shade the background with FAI range context
    ax1.axhspan(FAI_HU_MIN, FAI_HU_MAX, alpha=0.08, color="orange")
    ax1.axhline(FAI_HU_MIN, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(FAI_HU_MAX, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    if valid.any():
        ax1.plot(
            ring_centers[valid],
            mean_hus[valid],
            marker="o",
            markersize=5,
            linewidth=2,
            color="#D94040",
            zorder=3,
            label="Mean FAI HU",
        )
        ax1.fill_between(
            ring_centers[valid],
            mean_hus[valid] - std_hus[valid],
            mean_hus[valid] + std_hus[valid],
            alpha=0.25,
            color="#D94040",
            label="±1 SD",
        )
    else:
        ax1.text(0.5, 0.5, "No fat voxels found\nin this VOI",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=12, color="gray")

    ax1.set_xlabel("Distance from coronary outer wall (mm)", fontsize=11)
    ax1.set_ylabel("Mean HU (FAI range)", fontsize=11)
    ax1.set_xlim(0, max_radial_mm)
    ax1.set_ylim(FAI_HU_MIN - 20, FAI_HU_MAX + 20)
    ax1.set_title(f"Radial HU Profile — {vessel_name}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(10))

    # — Right panel: Voxel count per ring (data density) ——————————————
    bar_colors = [
        "#D94040" if n > 0 else "#AAAAAA"
        for n in n_voxels
    ]
    ax2.bar(ring_centers, n_voxels, width=ring_step_mm * 0.85,
            color=bar_colors, edgecolor="none", alpha=0.8)
    ax2.set_xlabel("Distance from coronary outer wall (mm)", fontsize=11)
    ax2.set_ylabel("Fat voxels per ring (n)", fontsize=11)
    ax2.set_title(f"FAI Voxel Count per Ring — {vessel_name}", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, max_radial_mm)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax2.xaxis.set_major_locator(MaxNLocator(10))

    # Annotation: mean radius
    for ax in [ax1, ax2]:
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.0,
                   label=f"vessel wall (r={mean_radius_mm:.1f}mm)")

    plt.suptitle(
        f"PCAT Radial Analysis — {vessel_name} — mean vessel radius = {mean_radius_mm:.2f} mm",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_radial_profile.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Radial profile saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure: all vessels on one page
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(
    vessel_stats: Dict[str, Dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "pcat",
) -> Path:
    """
    Generate a summary table/bar chart of PCAT statistics for all vessels.

    vessel_stats: dict {vessel_name: stats_dict from pcat_segment.compute_pcat_stats()}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vessel_names = list(vessel_stats.keys())
    hu_means = [vessel_stats[v].get("hu_mean", np.nan) for v in vessel_names]
    fat_fracs = [vessel_stats[v].get("fat_fraction", 0) * 100 for v in vessel_names]
    n_voxels = [vessel_stats[v].get("n_fat_voxels", 0) for v in vessel_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    COLORS = {"LAD": "#E8533A", "LCX": "#4A90D9", "RCA": "#2ECC71"}
    bar_colors = [COLORS.get(v, "#888888") for v in vessel_names]

    # Mean HU
    axes[0].bar(vessel_names, hu_means, color=bar_colors, edgecolor="none")
    axes[0].set_ylabel("Mean FAI HU", fontsize=11)
    axes[0].set_title("Mean PCAT HU", fontsize=11, fontweight="bold")
    axes[0].set_ylim(FAI_HU_MIN - 10, FAI_HU_MAX + 10)
    axes[0].axhline(FAI_HU_MIN, color="orange", linestyle="--", linewidth=0.8)
    axes[0].axhline(FAI_HU_MAX, color="red", linestyle="--", linewidth=0.8)
    _add_value_labels(axes[0], hu_means, fmt="{:.1f}")

    # Fat fraction
    axes[1].bar(vessel_names, fat_fracs, color=bar_colors, edgecolor="none")
    axes[1].set_ylabel("Fat voxel fraction (%)", fontsize=11)
    axes[1].set_title("PCAT Fat Fraction", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, max(fat_fracs) * 1.25 + 5 if max(fat_fracs) > 0 else 20)
    _add_value_labels(axes[1], fat_fracs, fmt="{:.1f}%")

    # n fat voxels
    axes[2].bar(vessel_names, n_voxels, color=bar_colors, edgecolor="none")
    axes[2].set_ylabel("Fat voxel count", fontsize=11)
    axes[2].set_title("PCAT Voxel Count", fontsize=11, fontweight="bold")
    _add_value_labels(axes[2], n_voxels, fmt="{:,}")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Vessel", fontsize=10)

    plt.suptitle(f"PCAT Summary — {prefix}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / f"{prefix}_summary.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Summary chart saved: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fai_colormap() -> mcolors.LinearSegmentedColormap:
    """
    Custom FAI colormap: yellow at -190 HU → red at -30 HU.
    Values outside range are transparent (set alpha=0 via bad color).
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fai",
        [
            (0.0, "#FFEE00"),   # yellow  — most negative fat (-190 HU)
            (0.4, "#FF8800"),   # orange
            (0.7, "#FF4400"),   # orange-red
            (1.0, "#CC0000"),   # dark red — least negative fat (-30 HU)
        ],
    )
    cmap.set_bad(alpha=0.0)   # NaN → transparent
    cmap.set_under(alpha=0.0)
    cmap.set_over(alpha=0.0)
    return cmap


def _compute_arclengths(centerline_ijk: np.ndarray, spacing_mm: List[float]) -> np.ndarray:
    """Cumulative arc-length along centerline in mm."""
    scale = np.array(spacing_mm)
    diffs = np.diff(centerline_ijk.astype(float), axis=0) * scale
    seglens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seglens)])


def _add_value_labels(ax: plt.Axes, values: List, fmt: str = "{:.1f}") -> None:
    """Add text labels on top of bar chart bars."""
    for patch, val in zip(ax.patches, values):
        if np.isnan(float(val)):
            continue
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        label = fmt.format(val)
        ax.text(x, y + abs(y) * 0.02 + 0.5, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
