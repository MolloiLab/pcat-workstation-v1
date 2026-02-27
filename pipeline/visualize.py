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
FAI_RISK_THRESHOLD = -70.1  # Oikonomou 2018 / CRISP-CT: FAI > -70.1 HU = HIGH risk


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
    Render the PCAT VOI + coronary artery tubes in VRT style matching the
    Syngo.via reference: beige/off-white vessels on a pure-black background,
    semi-transparent FAI-coloured pericoronary fat cloud, LAO-Cranial camera.
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
    Z, Y, X = volume.shape
    pv_grid = pv.ImageData()
    pv_grid.dimensions = (X+1, Y+1, Z+1)
    pv_grid.spacing = (sx, sy, sz)   # pyvista uses (x, y, z) order
    pv_grid.origin = (0.0, 0.0, 0.0)

    # Cell data: one value per cell = one per voxel
    # Cells are indexed [x, y, z] in VTK/pyvista order
    # voi_mask is [z, y, x] numpy order
    # To match: transpose to (X, Y, Z) then flatten in C order
    voi_vtk = voi_mask.transpose(2, 1, 0).flatten(order="C").astype(np.uint8)  # (X*Y*Z,)
    pv_grid.cell_data["voi"] = voi_vtk

    # Extract VOI surface via threshold
    voi_surface = pv_grid.threshold(0.5, scalars="voi").extract_surface(algorithm="dataset_surface")

    # ── Attach HU scalars on the VOI surface ────────────────────────────
    # Clip to fat range for coloring
    hu_vtk = np.clip(volume.transpose(2, 1, 0).flatten(order="C"), FAI_HU_MIN, FAI_HU_MAX).astype(np.float32)
    pv_grid.cell_data["hu"] = hu_vtk
    voi_fat_cells = pv_grid.threshold(0.5, scalars="voi")
    hu_scalars = voi_fat_cells.cell_data["hu"]

    # ── Build centerline tubes (beige/off-white, VRT style) ─────────────────
    # All vessels rendered in a single warm beige colour matching the
    # Syngo.via VRT reference (mean vessel RGB ≈ 224, 204, 173).
    # Pass as float [0,1] RGB tuple — pyvista hex parsing shifts the hue
    # under default lighting, while a float tuple is applied directly.
    VESSEL_COLOR = (224 / 255, 204 / 255, 173 / 255)  # RGB(224,204,173) warm beige
    tube_meshes = []
    for vessel_name, cl_ijk in vessel_centerlines.items():
        if len(cl_ijk) < 2:
            continue
        # Convert ijk (z, y, x) -> mm coords (x, y, z) in pyvista space
        pts_xyz = cl_ijk[:, [2, 1, 0]] * np.array([sx, sy, sz])
        spline = pv.Spline(pts_xyz, n_points=max(len(pts_xyz) * 2, 100))
        mean_r = float(np.mean(vessel_radii.get(vessel_name, [1.5])))
        # VRT-style: inflate tubes to 3x vessel radius so they are clearly
        # visible at this volume scale; minimum 2 mm to ensure visibility.
        tube = spline.tube(radius=max(mean_r * 3.0, 2.0))
        tube_meshes.append((vessel_name, tube, VESSEL_COLOR))

    # ── Render ─────────────────────────────────────────────────────────────
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(off_screen=not interactive, window_size=(1200, 900))
    plotter.set_background("black")   # pure black — matches VRT reference

    # FAI-coloured pericoronary fat VOI (semi-transparent cloud)
    fai_cmap = _fai_colormap()
    plotter.add_mesh(
        voi_fat_cells.extract_surface(algorithm="dataset_surface"),
        scalars="hu",
        clim=[FAI_HU_MIN, FAI_HU_MAX],
        cmap=fai_cmap,
        opacity=0.25,   # translucent so vessels show through clearly
        show_scalar_bar=True,
        scalar_bar_args={"title": "PCAT HU", "fmt": "%.0f"},
    )

    # Vessel tubes: beige/off-white with specular highlights for VRT gloss
    for vessel_name, tube, color in tube_meshes:
        plotter.add_mesh(
            tube,
            color=color,
            opacity=1.0,
            smooth_shading=True,
            specular=0.6,
            specular_power=40,
            ambient=0.15,
            diffuse=0.85,
            label=vessel_name,
        )

    plotter.add_legend(face="rectangle", bcolor=[0.05, 0.05, 0.05], size=(0.18, 0.12))
    plotter.add_text(f"PCAT 3D — {prefix}", font_size=12, position="upper_edge")
    # ── LAO-Cranial camera (Left Anterior Oblique ~30°, Cranial ~25°) ─────────
    # Matches "Coronaries VRT LAO Cran" series from Syngo.via.
    # pyvista/VTK axes: x=left, y=posterior, z=superior.
    center = np.array(pv_grid.center)
    cx, cy, cz = center
    bounds = pv_grid.bounds   # (xmin,xmax,ymin,ymax,zmin,zmax)
    diag = np.sqrt(
        (bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2
    )
    dist = diag * 1.4   # camera distance from focal point
    lao_rad  = np.radians(30)   # Left Anterior Oblique
    cran_rad = np.radians(25)   # Cranial tilt
    cam_x =  dist * np.sin(lao_rad) * np.cos(cran_rad)   # patient-left offset
    cam_y = -dist * np.cos(lao_rad) * np.cos(cran_rad)   # anterior offset
    cam_z =  dist * np.sin(cran_rad)                     # cranial offset
    cam_pos = (cx + cam_x, cy + cam_y, cz + cam_z)
    plotter.camera_position = [cam_pos, (cx, cy, cz), (0.0, 0.0, 1.0)]

    if interactive:
        plotter.show()

    out_path = None
    if screenshot:
        out_path = output_dir / f"{prefix}_3d_voi.png"
        plotter.screenshot(str(out_path))
        print(f"[visualize] 3D render saved: {out_path.name}")

    plotter.close()
    return out_path


def render_3d_voi_dicom(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    vessel_centerlines: Dict[str, np.ndarray],
    vessel_radii: Dict[str, np.ndarray],
    spacing_mm: List[float],
    output_dir: str | Path,
    prefix: str = "pcat",
    n_frames: int = 37,
    frame_size: int = 529,
) -> Optional[Path]:
    """
    Render n_frames rotating VRT frames of the PCAT VOI + artery centerlines as
    DICOM Secondary Capture files (RGB 529×529, series desc 'Coronaries VRT LAO Cran').
    Returns path to the output DICOM directory, or None if pyvista unavailable.

    Parameters
    ----------
    volume            : (Z, Y, X) HU float32
    voi_mask          : (Z, Y, X) bool — PCAT VOI union
    vessel_centerlines: {vessel_name: (N, 3) voxel ijk}
    vessel_radii      : {vessel_name: (N,) radii mm}
    spacing_mm        : [sz, sy, sx]
    output_dir        : base output directory (DICOM subfolder is created inside)
    prefix            : filename prefix
    n_frames          : number of rotation frames (default 37, full 360°)
    frame_size        : pixel dimension of each frame (default 529)

    Returns
    -------
    Path to the DICOM folder, or None if pyvista unavailable
    """
    if not HAS_PYVISTA:
        print("[visualize] Skipping 3D DICOM render: pyvista not installed.")
        return None

    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import generate_uid, ExplicitVRLittleEndian
        SecondaryCaptureImageStorage = "1.2.840.10008.5.1.4.1.1.7"
    except ImportError:
        print("[visualize] Skipping 3D DICOM render: pydicom not installed. pip install pydicom")
        return None

    output_dir = Path(output_dir)
    dicom_dir = output_dir / f"{prefix}_3d_dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    sz, sy, sx = spacing_mm
    VESSEL_COLOR = (224 / 255, 204 / 255, 173 / 255)

    # ── Build pyvista scene (same as render_3d_voi) ──────────────────────
    Z, Y, X = volume.shape
    pv_grid = pv.ImageData()
    pv_grid.dimensions = (X + 1, Y + 1, Z + 1)
    pv_grid.spacing = (sx, sy, sz)
    pv_grid.origin = (0.0, 0.0, 0.0)

    voi_vtk = voi_mask.transpose(2, 1, 0).flatten(order="C").astype(np.uint8)
    pv_grid.cell_data["voi"] = voi_vtk

    hu_vtk = np.clip(volume.transpose(2, 1, 0).flatten(order="C"), FAI_HU_MIN, FAI_HU_MAX).astype(np.float32)
    pv_grid.cell_data["hu"] = hu_vtk
    voi_fat_cells = pv_grid.threshold(0.5, scalars="voi")

    tube_meshes = []
    for vessel_name, cl_ijk in vessel_centerlines.items():
        if len(cl_ijk) < 2:
            continue
        pts_xyz = cl_ijk[:, [2, 1, 0]] * np.array([sx, sy, sz])
        spline = pv.Spline(pts_xyz, n_points=max(len(pts_xyz) * 2, 100))
        mean_r = float(np.mean(vessel_radii.get(vessel_name, [1.5])))
        tube = spline.tube(radius=max(mean_r * 3.0, 2.0))
        tube_meshes.append((vessel_name, tube, VESSEL_COLOR))

    # ── LAO-Cranial base camera parameters ───────────────────────────────
    center = np.array(pv_grid.center)
    cx, cy, cz = center
    bounds = pv_grid.bounds
    diag = np.sqrt(
        (bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2 + (bounds[5] - bounds[4]) ** 2
    )
    cam_dist = diag * 1.4
    cran_rad = np.radians(25)

    fai_cmap = _fai_colormap()
    series_uid = generate_uid()
    series_number = 900

    print(f"[visualize] Rendering {n_frames} 3D DICOM frames to {dicom_dir.name}/")

    for frame_idx in range(n_frames):
        # Rotate camera 360° around the cranial (Z) axis
        angle_offset = np.radians(360.0 * frame_idx / n_frames)
        base_az = np.radians(30) + angle_offset   # LAO base + rotation offset

        cam_x = cam_dist * np.sin(base_az) * np.cos(cran_rad)
        cam_y = -cam_dist * np.cos(base_az) * np.cos(cran_rad)
        cam_z = cam_dist * np.sin(cran_rad)
        cam_pos = (cx + cam_x, cy + cam_y, cz + cam_z)

        plotter = pv.Plotter(off_screen=True, window_size=(frame_size, frame_size))
        plotter.set_background("black")
        pv.set_plot_theme("dark")

        plotter.add_mesh(
            voi_fat_cells.extract_surface(algorithm="dataset_surface"),
            scalars="hu",
            clim=[FAI_HU_MIN, FAI_HU_MAX],
            cmap=fai_cmap,
            opacity=0.25,
            show_scalar_bar=False,
        )

        for _vname, tube, color in tube_meshes:
            plotter.add_mesh(
                tube,
                color=color,
                opacity=1.0,
                smooth_shading=True,
                specular=0.6,
                specular_power=40,
                ambient=0.15,
                diffuse=0.85,
            )

        plotter.camera_position = [cam_pos, (cx, cy, cz), (0.0, 0.0, 1.0)]
        img = plotter.screenshot(return_img=True)  # numpy RGB uint8
        plotter.close()

        # Resize to exactly frame_size x frame_size if needed
        if img.shape[0] != frame_size or img.shape[1] != frame_size:
            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.fromarray(img).resize((frame_size, frame_size), _PILImage.LANCZOS)
                img = np.array(pil_img)
            except ImportError:
                # PIL not available: use numpy slicing (crop/pad) as fallback
                import warnings as _w
                _w.warn("PIL not installed; 3D DICOM frames may not be exactly frame_size px. pip install Pillow")

        # ── Write DICOM Secondary Capture ────────────────────────────────
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_implicit_VR = False
        ds.is_little_endian = True
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "OT"
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.InstanceNumber = frame_idx + 1
        ds.SeriesDescription = "Coronaries VRT LAO Cran"
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL", "VRT"]
        ds.BurnedInAnnotation = "NO"
        ds.Rows = frame_size
        ds.Columns = frame_size
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0  # RGB interleaved
        ds.PixelData = img.astype(np.uint8).tobytes()

        dcm_path = dicom_dir / f"{prefix}_3d_frame{frame_idx + 1:03d}.dcm"
        pydicom.dcmwrite(str(dcm_path), ds, write_like_original=False)

    print(f"[visualize] 3D DICOM series written: {n_frames} frames -> {dicom_dir}")
    return dicom_dir


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
    Compute and render a Curved Planar Reformat (CPR) — straightened CPR per
    Kanitsar et al. (2002) — of the FAI signal along a coronary artery.
      1. Compute tangent T[i] at each centerline point (finite difference, mm-normalised).
      2. Build a stable Bishop frame (parallel transport / rotation-minimising frame):
         - Initialise N[0] = any vector perpendicular to T[0].
         - Propagate: N[i+1] = N[i] - dot(N[i], T[i+1]) * T[i+1]  (re-normalised).
         - B[i] = T[i] x N[i]  (binormal).
         This avoids the frame-flip artefact of naive per-point Gram-Schmidt against
         a fixed reference vector, which flips 180° when the tangent aligns with
         that reference.
      3. At each centerline point i, sample the volume on a 2-D grid in the (N, B)
         plane centred on the centerline point:
             sample_pts[j,k] = cl_mm[i] + u[k]*N[i] + v[j]*B[i]
         where u in [-width_mm, +width_mm]  and  v in [-height_mm, +height_mm].
      4. Apply a thin-slab MIP along the tangent (+/-slab_thickness_mm/2) by
         sampling a small number of planes offset along T and taking the maximum.
      5. Stack the 2-D slices -> CPR volume of shape (N_pts, n_height, n_width).
      6. Display the centre row for the standard straightened CPR:
         x-axis = arc-length along the vessel (index i),
         y-axis = lateral distance from centreline (index k along N).
    Non-fat voxels are shown in grayscale (anatomic context).
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii
    spacing_mm      : [sz, sy, sx]
    vessel_name     : e.g. "LAD"
    output_dir      : output directory
    prefix          : filename prefix
    slab_thickness_mm : total slab thickness for MIP along tangent (mm); 0 = single plane
    width_mm        : half-width of the CPR plane (lateral extent from centreline)
    Returns
    -------
    Path to saved PNG, or None if too few centerline points
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N_pts = len(centerline_ijk)
    if N_pts < 3:
        print(f"[visualize] CPR: too few centerline points for {vessel_name}, skipping.")
        return None
    (cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_height, n_width) = _compute_cpr_data(
        volume, centerline_ijk, spacing_mm,
        slab_thickness_mm=slab_thickness_mm, width_mm=width_mm,
    )
    # ── Build the display CPR image ───────────────────────────────────────
    # Straightened CPR: take the centre row (v=0, i.e. B=0 plane).
    # Shape: (N_pts, n_width).  Transpose -> (n_width, N_pts) for imshow so that
    # x-axis = arc-length and y-axis = lateral distance.
    centre_row = n_height // 2
    cpr_image  = cpr_volume[:, centre_row, :]   # (N_pts, n_width)
    fig, ax = plt.subplots(figsize=(7, 14), dpi=150)
    # Grayscale background (soft-tissue window -200...+400 HU)
    gray_img  = np.clip(cpr_image, -200.0, 400.0)
    gray_norm = (gray_img + 200.0) / 600.0
    gray_norm = np.nan_to_num(gray_norm, nan=0.0)
    # Display WITHOUT transpose: shape (N_pts, n_width)
    # origin="upper" → row 0 (arc-length=0 = ostium) at top
    # x-axis = lateral distance (n_width columns)
    # y-axis = arc-length (N_pts rows, top=0mm=ostium, bottom=40mm=distal)
    ax.imshow(
        gray_norm,
        aspect="auto",
        origin="upper",
        cmap="gray",
        vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )
    # FAI overlay (-190 to -30 HU only)
    fai_img = np.where(
        (cpr_image >= FAI_HU_MIN) & (cpr_image <= FAI_HU_MAX),
        cpr_image,
        np.nan,
    )
    fai_cmap = _fai_colormap()
    fai_im = ax.imshow(
        fai_img,
        aspect="auto",
        origin="upper",
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
    # X-axis: lateral distance ticks (mm)
    y_ticks_mm  = np.arange(-width_mm, width_mm + 1, 5.0)
    y_tick_idxs = ((y_ticks_mm + width_mm) / (2 * width_mm) * (n_width - 1)).astype(int)
    ax.set_xticks(np.clip(y_tick_idxs, 0, n_width - 1))
    ax.set_xticklabels([f"{t:.0f}" for t in y_ticks_mm], fontsize=9)
    ax.set_xlabel("Lateral distance from centreline (mm)", fontsize=11)
    # Y-axis: arc-length ticks (mm) — arclengths go top to bottom
    x_ticks_mm  = np.arange(0, arclengths[-1] + 1, 10.0)
    x_tick_idxs = [int(np.argmin(np.abs(arclengths - t))) for t in x_ticks_mm]
    ax.set_yticks(x_tick_idxs)
    ax.set_yticklabels([f"{t:.0f}" for t in x_ticks_mm], fontsize=9)
    ax.set_ylabel("Distance along vessel (mm)", fontsize=11)
    ax.axvline(n_width // 2, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"CPR — {vessel_name} — FAI overlay (HU {FAI_HU_MIN:.0f} to {FAI_HU_MAX:.0f})\n"
        f"Bishop-frame straightened CPR (vessel top→bottom, ostium at top)  |  slab MIP {slab_thickness_mm:.0f} mm  |  width ±{width_mm:.0f} mm",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_fai.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] CPR FAI saved: {out_path.name}")
    return out_path


def render_cpr_grayscale(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
    n_rotations: int = 6,
    window_center: float = 200.0,
    window_width: float = 600.0,
) -> Optional[Path]:
    """
    Render multi-rotation grayscale CPR images at 6 different viewing angles.

    Generates a 2x3 panel figure showing the CPR at rotation angles
    0°, 60°, 120°, 180°, 240°, 300° around the vessel axis.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) centerline voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii
    spacing_mm      : [sz, sy, sx]
    vessel_name     : e.g. "LAD"
    output_dir      : output directory
    prefix          : filename prefix
    slab_thickness_mm : total slab thickness for MIP along tangent (mm)
    width_mm        : half-width of the CPR plane (lateral extent from centreline)
    n_rotations     : number of rotation angles (default 6: 0°, 60°, 120°, 180°, 240°, 300°)
    window_center   : window center for HU display (default 200)
    window_width    : window width for HU display (default 600)

    Returns
    -------
    Path to saved PNG, or None if too few centerline points
    """
    from scipy.ndimage import rotate as ndimage_rotate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N_pts = len(centerline_ijk)
    if N_pts < 3:
        print(f"[visualize] CPR grayscale: too few centerline points for {vessel_name}, skipping.")
        return None

    # Compute CPR data using shared helper
    (cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_height, n_width) = _compute_cpr_data(
        volume, centerline_ijk, spacing_mm,
        slab_thickness_mm=slab_thickness_mm, width_mm=width_mm,
    )

    # Compute rotation angles
    angles = np.linspace(0, 360, n_rotations, endpoint=False)

    # HU windowing parameters
    HU_min = window_center - window_width / 2.0
    HU_max = window_center + window_width / 2.0

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes = axes.flatten()

    for idx, theta in enumerate(angles):
        ax = axes[idx]

        # Rotate CPR volume around the centerline axis (in the N-B plane)
        # cpr_volume has shape (N_pts, n_height, n_width)
        # Rotate each cross-section slice by theta degrees in the (N, B) plane
        rotated_view = ndimage_rotate(
            cpr_volume,
            angle=theta,
            axes=(1, 2),
            reshape=False,
            order=1,
            mode='nearest'
        )

        # Extract the centre row (v=0, i.e. B=0 plane after rotation)
        centre_row = n_height // 2
        cpr_rotated = rotated_view[:, centre_row, :]  # (N_pts, n_width)

        # Apply HU windowing
        windowed = np.clip(cpr_rotated, HU_min, HU_max)
        windowed_norm = (windowed - HU_min) / window_width  # -> [0, 1]
        windowed_norm = np.nan_to_num(windowed_norm, nan=0.0)

        # Display
        ax.imshow(
            windowed_norm,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            origin="upper",
            interpolation="bilinear",
        )
        ax.set_title(f"{int(theta)}°", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lateral distance (mm)", fontsize=9)
        ax.set_ylabel("Arc-length (mm)", fontsize=9)

    # Overall title
    fig.suptitle(
        f"CPR Grayscale — {vessel_name} — WC={int(window_center)} WW={int(window_width)}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_grayscale.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] CPR grayscale saved: {out_path.name}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# Output 3c: Native / Curved CPR (Syngo.via style)
# ─────────────────────────────────────────────────────────────────────────────

def render_cpr_native(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    slab_thickness_mm: float = 3.0,
    n_rotations: int = 3,
    output_size: int = 512,
    half_width_mm: float = 15.0,
    window_center: float = 200.0,
    window_width: float = 600.0,
) -> List[Path]:
    """
    Render true curved CPR images matching the Syngo.via clinical reference.

    Each output image (output_size x output_size) is a curved-MPR: the X axis
    runs along the vessel arc-length (vessel unrolled horizontally), and the Y
    axis runs radially outward from the centerline in a rotated direction.
    Pixels outside the volume bounds are set to NaN -> rendered black (sentinel).

    This matches the Syngo.via 'Curved Range Radial MPR_CURVED' reference DICOMs:
    - 512x512, dark/black background, vessel unrolled along the frame.
    - n_rotations evenly spaced radial cut-plane angles (0deg, 120deg, 240deg).
    Parameters
    ----------
    volume            : (Z, Y, X) HU float32
    centerline_ijk    : (N, 3) centerline voxel indices [z, y, x]
    radii_mm          : (N,) per-point vessel radii in mm
    spacing_mm        : [sz, sy, sx]
    vessel_name       : e.g. 'RCA'
    output_dir        : directory to write PNGs
    prefix            : filename prefix
    slab_thickness_mm : half-slab thickness around the radial cut plane (mm), default 3
    n_rotations       : number of rotation views, default 3
    output_size       : output image size in pixels (default 512)
    half_width_mm     : radial extent from centerline on each side (mm), default 15
    window_center     : HU window center (default 200, matching Syngo.via reference)
    window_width      : HU window width  (default 600, matching Syngo.via reference)
    -------
    List of Paths to saved PNGs (one per rotation angle)
    """
    from scipy.ndimage import map_coordinates as _map_coords
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sz, sy, sx = float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])
    N_pts = centerline_ijk.shape[0]
    if N_pts < 3:
        print(f"[visualize] CPR native: too few centerline points for {vessel_name}, skipping.")
        return []
    vox_size = np.array([sz, sy, sx], dtype=np.float64)  # (z, y, x)
    shape = np.array(volume.shape, dtype=int)             # (Z, Y, X)
    cl_mm = centerline_ijk.astype(np.float64) * vox_size  # (N, 3)
    # ── Resample centerline to output_size evenly spaced arc-length steps ────
    diffs = np.diff(cl_mm, axis=0)                  # (N-1, 3)
    seg_len = np.linalg.norm(diffs, axis=1)         # (N-1,)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])  # (N,)
    total_len = cumlen[-1]
    if total_len < 1.0:
        print(f"[visualize] CPR native: degenerate centerline ({total_len:.2f}mm) for {vessel_name}, skipping.")
        return []
    s_out = np.linspace(0.0, total_len, output_size)  # (W,) arc-length samples
    cl_rs = np.stack([
        np.interp(s_out, cumlen, cl_mm[:, dim]) for dim in range(3)
    ], axis=1)  # (W, 3) -- resampled centerline, one entry per column

    # ── Per-column tangent vectors (gradient, normalised) ────────────────────
    # Smooth the resampled centerline before tangent estimation to eliminate
    # staircase artifacts when the input centerline is sparse.
    from scipy.ndimage import gaussian_filter1d as _gf1d
    # Smooth over 5mm of arc-length regardless of input centerline quality.
    # This removes waypoint kinks and staircase artifacts while preserving vessel curvature.
    # Each column spans total_len/output_size mm, so sigma_mm maps to sigma_cols columns.
    sigma_mm = 5.0
    sigma_cols = max(4, int(sigma_mm / (total_len / output_size)))
    cl_rs_smooth = np.stack([_gf1d(cl_rs[:, d], sigma=sigma_cols) for d in range(3)], axis=1)
    tangents = np.gradient(cl_rs_smooth, axis=0)  # (W, 3)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    tangents /= norms  # (W, 3), unit tangents

    # ── Build a stable perpendicular frame at each column ────────────────────
    # Use Bishop (parallel-transport) framing to avoid discontinuous flips.
    def _init_perp(t0):
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(t0, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        v = ref - np.dot(ref, t0) * t0
        return v / (np.linalg.norm(v) + 1e-12)

    perp = np.zeros_like(tangents)  # (W, 3)
    perp[0] = _init_perp(tangents[0])
    for i in range(1, output_size):
        v = perp[i - 1] - np.dot(perp[i - 1], tangents[i]) * tangents[i]
        nv = np.linalg.norm(v)
        perp[i] = v / (nv + 1e-12) if nv > 1e-8 else _init_perp(tangents[i])

    # ── For each rotation angle: build the 2D curved-CPR image ───────────────
    out_paths: List[Path] = []
    angles_deg = np.linspace(0.0, 360.0, n_rotations, endpoint=False)
    mean_sp = float(np.mean(vox_size))
    n_slab = max(1, int(np.ceil(2.0 * slab_thickness_mm / mean_sp)))
    for rot_idx, theta_deg in enumerate(angles_deg):
        theta = np.radians(theta_deg)
        # Rotate perpendicular frame by theta around tangent (Rodrigues, simplified:
        # since perp is already perpendicular to tangent, dot(perp, tangent) = 0)
        U_rot = (
            perp * np.cos(theta)
            + np.cross(tangents, perp) * np.sin(theta)
        )  # (W, 3) unit radial direction per column
        norms2 = np.linalg.norm(U_rot, axis=1, keepdims=True) + 1e-12
        U_rot /= norms2

        # Row coordinates: radial offsets from centerline
        r_coords = np.linspace(half_width_mm, -half_width_mm, output_size)  # (H,)

        # Build 3D sample points: (H, W, 3)
        pts_base_mm = (
            cl_rs[np.newaxis, :, :]                             # (1, W, 3)
            + r_coords[:, np.newaxis, np.newaxis]               # (H, 1, 1)
              * U_rot[np.newaxis, :, :]                         # (1, W, 3)
        )  # (H, W, 3)
        # ── Thin-slab MaxIP around the radial cut plane ─────────────────────
        # Slab direction = tangent x U_rot (the remaining orthogonal axis)
        slab_dirs = np.cross(tangents, U_rot)  # (W, 3) slab normal per column
        slab_norms = np.linalg.norm(slab_dirs, axis=1, keepdims=True) + 1e-12
        slab_dirs /= slab_norms  # (W, 3) unit

        slab_offsets = np.linspace(-slab_thickness_mm, slab_thickness_mm, n_slab)
        projection = np.full((output_size, output_size), -np.inf, dtype=np.float32)
        for s_off in slab_offsets:
            pts_mm = (
                pts_base_mm
                + s_off * slab_dirs[np.newaxis, :, :]  # (1, W, 3)
            )  # (H, W, 3)
            pts_vox = pts_mm / vox_size[np.newaxis, np.newaxis, :]  # (H, W, 3)
            z_v = pts_vox[:, :, 0].ravel()
            y_v = pts_vox[:, :, 1].ravel()
            x_v = pts_vox[:, :, 2].ravel()
            valid = (
                (z_v >= 0) & (z_v < shape[0] - 1) &
                (y_v >= 0) & (y_v < shape[1] - 1) &
                (x_v >= 0) & (x_v < shape[2] - 1)
            )
            vals = _map_coords(
                volume,
                [z_v, y_v, x_v],
                order=1,
                mode="constant",
                cval=-np.inf,
            ).astype(np.float32)
            vals[~valid] = -np.inf
            vals_2d = vals.reshape(output_size, output_size)
            better = vals_2d > projection
            projection[better] = vals_2d[better]
        projection[np.isinf(projection) & (projection < 0)] = np.nan
        # Transpose so arc-length is rows (top=ostium, bottom=distal end)
        # and radial offset is columns — vessel runs top-to-bottom like Syngo.via
        projection = projection.T  # (output_size arc-length, output_size radial)

        # ── HU windowing ─────────────────────────────────────────────────────
        HU_lo = window_center - window_width / 2.0
        windowed = np.clip(projection, HU_lo, HU_lo + window_width)
        windowed_norm = (windowed - HU_lo) / window_width  # [0, 1]
        windowed_norm = np.nan_to_num(windowed_norm, nan=0.0)  # black for sentinel

        # ── Save clean 512x512 PNG ────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(5.12, 5.12), dpi=100)
        ax.imshow(
            windowed_norm,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            origin="upper",
            interpolation="bilinear",
        )
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out_path = output_dir / f"{prefix}_{vessel_name}_cpr_native_rot{rot_idx:02d}.png"
        plt.savefig(str(out_path), dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[visualize] CPR native saved: {out_path.name}  (rotation {int(theta_deg)}deg)")
        out_paths.append(out_path)
    return out_paths

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
    ax.axvline(FAI_RISK_THRESHOLD, color="#CC2200", linewidth=1.8, linestyle=":",
               label=f"FAI risk cut-off ({FAI_RISK_THRESHOLD} HU)")
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
    ax1.axhspan(-90, -65, alpha=0.12, color="lightblue", label="Typical FAI range (-90 to -65 HU)")
    ax1.axhline(FAI_RISK_THRESHOLD, color="#CC2200", linewidth=1.6, linestyle=":",
               alpha=0.9, label=f"FAI risk cut-off ({FAI_RISK_THRESHOLD} HU)")

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
    ax1.set_ylim(-105, -50)   # covers -90 to -65 with margin, matches paper
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


def _compute_cpr_data(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Compute Bishop-frame CPR volume and associated frame data.

    Used by both render_cpr_fai() (batch/Agg) and cpr_browser.py (interactive/TkAgg).

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) centerline voxel indices [z, y, x]
    spacing_mm      : [sz, sy, sx]
    slab_thickness_mm : total slab thickness for MIP along tangent (mm)
    width_mm        : half-width of the CPR / cross-section plane (mm)

    Returns
    -------
    cpr_volume  : (N_pts, n_height, n_width) float32 — sampled HU values
    N_frame     : (N_pts, 3) Bishop normal vectors
    B_frame     : (N_pts, 3) Bishop binormal vectors
    cl_mm       : (N_pts, 3) centerline in mm [z, y, x]
    arclengths  : (N_pts,) cumulative arc-length in mm
    n_height    : int
    n_width     : int
    """
    sz, sy, sx = spacing_mm
    vox_size = np.array([sz, sy, sx], dtype=np.float64)  # (z, y, x) mm/voxel
    shape = np.array(volume.shape, dtype=int)             # (Z, Y, X)

    N_pts = len(centerline_ijk)

    # ── Pixel grid dimensions ──────────────────────────────────────────────
    mean_sp_xy = float(np.mean(vox_size[1:]))   # lateral voxel size (mm)
    n_width  = max(int(np.ceil(2.0 * width_mm  / mean_sp_xy)), 50)
    n_height = n_width   # square cross-section plane
    height_mm = width_mm  # symmetric about centreline

    # ── Centreline in physical mm space [z, y, x] ──────────────────────────
    cl_mm = centerline_ijk.astype(np.float64) * vox_size[np.newaxis, :]  # (N, 3)

    # ── Tangent vectors (finite differences, mm-normalised) ─────────────────
    T = np.zeros((N_pts, 3), dtype=np.float64)
    T[1:-1] = cl_mm[2:] - cl_mm[:-2]
    T[0]    = cl_mm[1]  - cl_mm[0]
    T[-1]   = cl_mm[-1] - cl_mm[-2]
    norms   = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    T      /= norms   # unit tangents in mm space

    # ── Bishop frame (parallel transport — rotation-minimising) ────────────
    N_frame = np.zeros((N_pts, 3), dtype=np.float64)  # normal
    B_frame = np.zeros((N_pts, 3), dtype=np.float64)  # binormal

    ref0 = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T[0], ref0)) > 0.9:
        ref0 = np.array([0.0, 1.0, 0.0])
    n0 = np.cross(T[0], ref0)
    n0 /= np.linalg.norm(n0) + 1e-12
    N_frame[0] = n0
    B_frame[0] = np.cross(T[0], N_frame[0])

    for i in range(1, N_pts):
        ni = N_frame[i - 1] - np.dot(N_frame[i - 1], T[i]) * T[i]
        norm_ni = np.linalg.norm(ni)
        if norm_ni > 1e-8:
            N_frame[i] = ni / norm_ni
        else:
            N_frame[i] = N_frame[i - 1]
        B_frame[i] = np.cross(T[i], N_frame[i])
        bnorm = np.linalg.norm(B_frame[i])
        if bnorm > 1e-8:
            B_frame[i] /= bnorm

    # ── Slab MIP offsets along tangent ───────────────────────────────────
    n_slab = max(1, int(np.ceil(slab_thickness_mm / mean_sp_xy)))
    if n_slab > 1:
        slab_offsets_mm = np.linspace(-slab_thickness_mm / 2.0,
                                       slab_thickness_mm / 2.0,
                                       n_slab)
    else:
        slab_offsets_mm = np.array([0.0])

    # Sampling grids: u along N (width), v along B (height)
    u_range = np.linspace(-width_mm,  width_mm,  n_width)   # (n_width,)
    v_range = np.linspace(-height_mm, height_mm, n_height)  # (n_height,)
    UU, VV = np.meshgrid(u_range, v_range, indexing="xy")  # (n_height, n_width)

    # ── Sample volume at each centreline point ─────────────────────────────
    cpr_volume = np.full((N_pts, n_height, n_width), np.nan, dtype=np.float32)

    for i in range(N_pts):
        pts_base_mm = (
            cl_mm[i][np.newaxis, np.newaxis, :]                               # (1,1,3)
            + UU[:, :, np.newaxis] * N_frame[i][np.newaxis, np.newaxis, :]   # (H,W,3)
            + VV[:, :, np.newaxis] * B_frame[i][np.newaxis, np.newaxis, :]   # (H,W,3)
        )  # -> (n_height, n_width, 3)

        slab_max = np.full((n_height, n_width), -np.inf, dtype=np.float32)

        for s_off in slab_offsets_mm:
            pts_mm = pts_base_mm + s_off * T[i][np.newaxis, np.newaxis, :]
            pts_vox = pts_mm / vox_size[np.newaxis, np.newaxis, :]

            z_v = pts_vox[:, :, 0].ravel()
            y_v = pts_vox[:, :, 1].ravel()
            x_v = pts_vox[:, :, 2].ravel()
            valid = (
                (z_v >= 0) & (z_v < shape[0] - 1) &
                (y_v >= 0) & (y_v < shape[1] - 1) &
                (x_v >= 0) & (x_v < shape[2] - 1)
            )
            vals = map_coordinates(
                volume,
                [z_v, y_v, x_v],
                order=1,
                mode="constant",
                cval=np.nan,
            ).astype(np.float32)
            vals[~valid] = np.nan
            vals_2d = vals.reshape(n_height, n_width)
            better    = vals_2d > slab_max
            not_nan   = ~np.isnan(vals_2d)
            slab_max[better & not_nan] = vals_2d[better & not_nan]
            slab_max[np.isinf(slab_max) & (slab_max < 0)] = np.nan

        cpr_volume[i] = slab_max

    # ── Arc-lengths ─────────────────────────────────────────────────────────────
    arclengths = _compute_arclengths(centerline_ijk, spacing_mm)

    return cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_height, n_width


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
