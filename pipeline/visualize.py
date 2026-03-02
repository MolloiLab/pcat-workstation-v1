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
from scipy.ndimage import distance_transform_edt, map_coordinates, gaussian_filter1d
from scipy.interpolate import CubicSpline

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

# NOTE: Backend is NOT set here. Callers that need non-interactive
# rendering (run_pipeline.py) set matplotlib.use('Agg') before importing
# this module. Interactive tools (cpr_browser.py) set TkAgg instead.

FAI_HU_MIN = -190.0
FAI_HU_MAX = -30.0
FAI_RISK_THRESHOLD = -70.1  # Oikonomou 2018 / CRISP-CT: FAI > -70.1 HU = HIGH risk



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
    VESSEL_COLOR = (1.0, 0.15, 0.15)  # Bright red for vessels

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
    
    # Create fat-only mask: voi AND hu in fat range
    fat_only_vtk = ((voi_mask & (volume >= FAI_HU_MIN) & (volume <= FAI_HU_MAX))
                    .transpose(2, 1, 0).flatten(order="C").astype(np.uint8))
    pv_grid.cell_data["fat_only"] = fat_only_vtk
    voi_fat_cells = pv_grid.threshold(0.5, scalars="voi")
    voi_fat_only_cells = pv_grid.threshold(0.5, scalars="fat_only")

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

        # Layer 1: Full VOI structure in dark gray (opacity=0.1)
        plotter.add_mesh(
            voi_fat_cells.extract_surface(algorithm="dataset_surface"),
            color=(0.2, 0.2, 0.2),  # dark gray
            opacity=0.1,
            show_scalar_bar=False,
        )
        
        # Layer 2: Fat-only cells with FAI colormap (opacity=0.6)
        plotter.add_mesh(
            voi_fat_only_cells.extract_surface(algorithm="dataset_surface"),
            scalars="hu",
            clim=[FAI_HU_MIN, FAI_HU_MAX],
            cmap=fai_cmap,
            opacity=0.6,
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
    # cpr_volume is (pixels_wide, pixels_high) = (n_width, n_height)
    # Transpose → (n_height, n_width) for imshow:
    #   rows = lateral axis  (y = lateral mm from centreline)
    #   cols = arc-length axis (x = distance along vessel, ostium at left)
    cpr_image = cpr_volume.T   # (n_height, n_width)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
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
    cbar = plt.colorbar(fai_im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("HU (FAI)", fontsize=10)
    cbar.set_ticks([FAI_HU_MIN, -150, -100, -50, FAI_HU_MAX])
    # After transpose: cols = arc-length, rows = lateral
    # X-axis: arc-length ticks (mm)
    total_mm = float(arclengths[-1]) if len(arclengths) > 0 else 0.0
    x_ticks_mm  = np.arange(0, total_mm + 1, 10.0)
    x_tick_idxs = ((x_ticks_mm / total_mm) * (n_width - 1)).astype(int) if total_mm > 0 else [0]
    ax.set_xticks(np.clip(x_tick_idxs, 0, n_width - 1))
    ax.set_xticklabels([f"{t:.0f}" for t in x_ticks_mm], fontsize=9)
    ax.set_xlabel("Distance along vessel (mm)", fontsize=11)
    # Y-axis: lateral distance ticks (mm)
    y_ticks_mm  = np.arange(-width_mm, width_mm + 1, 5.0)
    y_tick_idxs = (((y_ticks_mm + width_mm) / (2 * width_mm)) * (n_height - 1)).astype(int)
    ax.set_yticks(np.clip(y_tick_idxs, 0, n_height - 1))
    ax.set_yticklabels([f"{t:.0f}" for t in y_ticks_mm], fontsize=9)
    ax.set_ylabel("Lateral distance from centreline (mm)", fontsize=11)
    ax.axhline(n_height // 2, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"CPR — {vessel_name} — FAI overlay (HU {FAI_HU_MIN:.0f} to {FAI_HU_MAX:.0f})\n"
        f"Horos-exact straightened CPR | slab MIP {slab_thickness_mm:.0f} mm | width ±{width_mm:.0f} mm",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_fai.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] CPR FAI saved: {out_path.name}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# Output 3b: CPR DICOM — true HU int16 pixel data
# ─────────────────────────────────────────────────────────────────────────────

def render_cpr_dicom(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
    patient_meta: Optional[dict] = None,
) -> Optional[Path]:
    """
    Render a Curved Planar Reformat (CPR) as a DICOM CT Image Storage file
    containing real HU values (int16 pixels, no RGB, no normalization).

    Pixel data stored as int16 with RescaleIntercept=0 / RescaleSlope=1 so
    that the HU value of every pixel equals the stored integer directly.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii
    spacing_mm      : [sz, sy, sx]
    vessel_name     : e.g. "LAD"
    output_dir      : output directory
    prefix          : filename prefix
    slab_thickness_mm : total slab thickness for MIP along binormal (mm)
    width_mm        : half-width of lateral axis in mm
    patient_meta    : optional patient metadata from DICOM

    Returns
    -------
    Path to saved DICOM file, or None if too few centerline points
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import generate_uid, ExplicitVRLittleEndian
        CTImageStorage = "1.2.840.10008.5.1.4.1.1.2"
    except ImportError:
        print("[visualize] Skipping CPR DICOM: pydicom not installed. pip install pydicom")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(centerline_ijk) < 3:
        print(f"[visualize] CPR DICOM: too few centerline points for {vessel_name}, skipping.")
        return None

    (cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_height, n_width) = _compute_cpr_data(
        volume, centerline_ijk, spacing_mm,
        slab_thickness_mm=slab_thickness_mm, width_mm=width_mm,
    )
    # cpr_volume is (pixels_wide, pixels_high); transpose → (rows=lateral, cols=arc-length)
    cpr_image = cpr_volume.T   # (n_height, n_width) — real HU float32

    # Replace NaN (OOB voxels) with -1024 (standard DICOM air HU fill)
    cpr_image = np.nan_to_num(cpr_image, nan=-1024.0)

    # Clip to valid int16 range before casting
    cpr_int16 = np.clip(cpr_image, -32768, 32767).astype(np.int16)

    rows, cols = cpr_int16.shape

    # ── DICOM header ─────────────────────────────────────────────────────────
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    sop_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 901
    ds.SeriesDescription = f"CPR {vessel_name} HU"
    ds.ImageType = ["DERIVED", "SECONDARY", "REFORMAT", "CPR"]
    ds.BurnedInAnnotation = "NO"

    # True HU storage: 16-bit signed, slope=1, intercept=0
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1   # signed int16
    ds.RescaleIntercept = 0.0
    ds.RescaleSlope = 1.0
    ds.RescaleType = "HU"
    ds.WindowCenter = 50
    ds.WindowWidth = 400

    # Pixel spacing: [row_spacing_mm, col_spacing_mm]
    # row spacing = lateral mm per pixel;  col spacing = arc-length mm per pixel
    lat_mm_per_px = (2.0 * width_mm) / n_height
    arc_total_mm  = float(arclengths[-1]) if len(arclengths) > 0 else float(2.0 * width_mm)
    arc_mm_per_px = arc_total_mm / n_width if n_width > 0 else lat_mm_per_px
    ds.PixelSpacing = [f"{lat_mm_per_px:.6f}", f"{arc_mm_per_px:.6f}"]

    # Optional patient metadata
    if patient_meta:
        for tag in ['PatientID', 'StudyInstanceUID', 'AccessionNumber',
                    'PatientName', 'StudyDate', 'StudyTime']:
            if tag in patient_meta:
                setattr(ds, tag, patient_meta[tag])

    ds.PixelData = cpr_int16.tobytes()

    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_hu.dcm"
    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)
    print(f"[visualize] CPR HU DICOM saved: {out_path.name}  "
          f"HU range: [{cpr_int16.min()}, {cpr_int16.max()}]")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Output 3c: CPR PNG — vessel wall detection + 4 curved green lines + FAI
# ─────────────────────────────────────────────────────────────────────────────

def render_cpr_png(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    radii_mm: np.ndarray,
    spacing_mm: List[float],
    vessel_name: str,
    output_dir: str | Path,
    prefix: str = "pcat",
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
) -> Optional[Path]:
    """
    Render a clinical-quality CPR PNG with:
      - Grayscale background (soft-tissue window)
      - 4 curved green boundary lines derived from vessel radius per column:
          1: left vessel wall  (row = centre - radius_px)
          2: right vessel wall (row = centre + radius_px)
          3: left PCAT boundary  (row = centre - 2*radius_px)
          4: right PCAT boundary (row = centre + 2*radius_px)
      - FAI colour map (HU -190 → -30, yellow→red) ONLY inside the PCAT band
        (between green lines 1–3 on the left and 2–4 on the right).
      - Horizontal orientation: arc-length on X-axis, ostium at left.

    Parameters
    ----------
    volume          : (Z, Y, X) HU float32
    centerline_ijk  : (N, 3) voxel indices [z, y, x]
    radii_mm        : (N,) vessel radii in mm — one per seed waypoint
    spacing_mm      : [sz, sy, sx]
    vessel_name     : e.g. "LAD"
    output_dir      : output directory
    prefix          : filename prefix
    slab_thickness_mm : slab MIP thickness along binormal (mm)
    width_mm        : half-width of CPR (lateral axis extent in mm)

    Returns
    -------
    Path to saved PNG, or None on failure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(centerline_ijk) < 3:
        print(f"[visualize] CPR PNG: too few centerline points for {vessel_name}, skipping.")
        return None

    (cpr_volume, N_frame, B_frame, cl_mm, arclengths, n_height, n_width) = _compute_cpr_data(
        volume, centerline_ijk, spacing_mm,
        slab_thickness_mm=slab_thickness_mm, width_mm=width_mm,
    )
    cpr_image = cpr_volume.T   # (n_height, n_width): rows=lateral, cols=arc-length

    # ── Compute per-column vessel radius in pixel units ───────────────────────
    # radii_mm has one value per centerline point; interpolate to n_width columns,
    # then Gaussian-smooth to eliminate per-point EDT noise (jagged green lines).
    if radii_mm is not None and len(radii_mm) >= 2:
        src_x = np.linspace(0, n_width - 1, len(radii_mm))
        col_x = np.arange(n_width, dtype=float)
        r_mm  = np.interp(col_x, src_x, radii_mm)          # (n_width,) mm
        r_mm  = gaussian_filter1d(r_mm, sigma=8.0)          # smooth out per-point EDT noise
    else:
        r_mm = np.full(n_width, float(np.mean(radii_mm)) if radii_mm is not None else 2.0)
    lat_mm_per_px = (2.0 * width_mm) / n_height
    r_px = np.clip(r_mm / lat_mm_per_px, 1.0, n_height / 2.0 - 1.0)  # (n_width,)

    # Vessel centre row (pixels) — centreline is at the geometric centre
    centre_row = (n_height - 1) / 2.0

    # Row coordinates of the 4 boundary curves (one float per column)
    wall_top = centre_row - r_px          # inner top    (vessel wall, upper)
    wall_bot = centre_row + r_px          # inner bottom (vessel wall, lower)
    pcat_top = centre_row - 2.0 * r_px   # outer top    (PCAT boundary)
    pcat_bot = centre_row + 2.0 * r_px   # outer bottom (PCAT boundary)

    # ── Grayscale background (soft-tissue window -200 … +400 HU) ─────────────
    gray = np.clip(cpr_image, -200.0, 400.0)
    gray = np.nan_to_num((gray + 200.0) / 600.0, nan=0.0)

    # ── FAI mask: only within PCAT region AND HU in [-190, -30] ─────────────
    # Build a pixel-level boolean mask for the PCAT annular band.
    # The band has TWO parts:
    #   upper pcat band: pcat_top <= row <= wall_top
    #   lower pcat band: wall_bot <= row <= pcat_bot
    row_idx  = np.arange(n_height, dtype=float)[:, np.newaxis]  # (n_height, 1)
    upper_band = (row_idx >= pcat_top[np.newaxis, :]) & (row_idx <= wall_top[np.newaxis, :])
    lower_band = (row_idx >= wall_bot[np.newaxis, :]) & (row_idx <= pcat_bot[np.newaxis, :])
    pcat_mask  = upper_band | lower_band
    fat_mask   = (cpr_image >= FAI_HU_MIN) & (cpr_image <= FAI_HU_MAX)
    fai_hu     = np.where(pcat_mask & fat_mask, cpr_image, np.nan).astype(np.float32)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    ax.imshow(
        gray,
        aspect="auto", origin="upper", cmap="gray",
        vmin=0.0, vmax=1.0, interpolation="bilinear",
    )

    fai_cmap = _fai_colormap()
    fai_im = ax.imshow(
        fai_hu,
        aspect="auto", origin="upper",
        cmap=fai_cmap, vmin=FAI_HU_MIN, vmax=FAI_HU_MAX,
        alpha=0.85, interpolation="bilinear",
    )

    cbar = plt.colorbar(fai_im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("HU (FAI)", fontsize=10)
    cbar.set_ticks([FAI_HU_MIN, -150, -100, -50, FAI_HU_MAX])

    # ── Draw 4 curved green boundary lines ────────────────────────────────────
    col_coords = np.arange(n_width)
    green_hi = "#00ff00"
    green_lo = "#00cc00"
    ax.plot(col_coords, wall_top, color=green_hi, linewidth=1.2, alpha=0.85, label="vessel wall")
    ax.plot(col_coords, wall_bot, color=green_hi, linewidth=1.2, alpha=0.85)
    ax.plot(col_coords, pcat_top, color=green_lo, linewidth=1.0, alpha=0.7, linestyle="--", label="PCAT boundary")
    ax.plot(col_coords, pcat_bot, color=green_lo, linewidth=1.0, alpha=0.7, linestyle="--")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)

    # ── Axes ticks ────────────────────────────────────────────────────────────
    total_mm = float(arclengths[-1]) if len(arclengths) > 0 else 0.0
    x_ticks_mm  = np.arange(0, total_mm + 1, 10.0)
    x_tick_idxs = ((x_ticks_mm / total_mm) * (n_width - 1)).astype(int) if total_mm > 0 else [0]
    ax.set_xticks(np.clip(x_tick_idxs, 0, n_width - 1))
    ax.set_xticklabels([f"{t:.0f}" for t in x_ticks_mm], fontsize=9)
    ax.set_xlabel("Distance along vessel (mm)", fontsize=11)

    y_ticks_mm  = np.arange(-width_mm, width_mm + 1, 5.0)
    y_tick_idxs = (((y_ticks_mm + width_mm) / (2 * width_mm)) * (n_height - 1)).astype(int)
    ax.set_yticks(np.clip(y_tick_idxs, 0, n_height - 1))
    ax.set_yticklabels([f"{t:.0f}" for t in y_ticks_mm], fontsize=9)
    ax.set_ylabel("Lateral distance from centreline (mm)", fontsize=11)

    ax.axhline(n_height // 2, color="white", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.set_title(
        f"CPR — {vessel_name} — vessel wall + FAI overlay\n"
        f"green solid = vessel wall  |  green dashed = PCAT boundary (±2r)  |  "
        f"colour = FAI HU {FAI_HU_MIN:.0f}→{FAI_HU_MAX:.0f}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{vessel_name}_cpr_wall.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] CPR wall PNG saved: {out_path.name}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Horos-exact CPR helpers
# Mirrors: N3BezierCoreCreateCurveWithNodes → N3BezierCoreGetVectorInfo →
#          CPRVolumeDataLinearInterpolatedFloatAtDicomVector →
#          CPRStraightenedOperation
# ─────────────────────────────────────────────────────────────────────────────

def _bezier_fit_centerline(
    waypoints_mm: np.ndarray,
) -> Tuple["CubicSpline", float]:
    """
    Fit a C2-smooth cubic spline through centerline waypoints in DICOM mm-space.
    Mirrors N3BezierCoreCreateCurveWithNodes(..., N3BezierNodeOpenEndsStyle).

    Parameters
    ----------
    waypoints_mm : (N, 3) float64 — centerline points in DICOM mm [z, y, x]

    Returns
    -------
    cs         : CubicSpline parameterised by arc-length s in [0, total_len]
    total_len  : float — total arc-length of the fitted curve in mm
    """
    if len(waypoints_mm) < 2:
        raise ValueError("Need at least 2 waypoints")
    # Remove duplicate consecutive points to avoid zero-length segments
    unique = [waypoints_mm[0]]
    for p in waypoints_mm[1:]:
        if np.linalg.norm(p - unique[-1]) > 1e-6:
            unique.append(p)
    pts = np.array(unique, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("Degenerate centerline — all points are co-located")
    # Chord-length parameterisation (arc-length approximation on raw points)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(s[-1])
    if total_len < 1e-3:
        raise ValueError(f"Degenerate centerline — total length {total_len:.4f} mm")
    # 'not-a-knot' boundary: matches Horos open-ends style (tangents extrapolated,
    # not clamped), giving natural curve continuation past end waypoints.
    cs = CubicSpline(s, pts, bc_type='not-a-knot')
    return cs, total_len


def _sample_bezier_frame(
    cs: "CubicSpline",
    total_len: float,
    n_cols: int,
    initial_normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the spline at n_cols uniformly-spaced arc-length positions and
    compute a rotation-minimising (Bishop / parallel-transport) frame at each.
    Mirrors N3BezierCoreGetVectorInfo(bezierCore, spacing, 0, initialNormal, ...).

    Parameters
    ----------
    cs             : CubicSpline from _bezier_fit_centerline
    total_len      : arc-length of the curve in mm
    n_cols         : number of output columns (= pixelsWide in Horos)
    initial_normal : (3,) unit vector — initial N direction (perpendicular to T[0]).
                     If None, a stable default is chosen automatically.

    Returns
    -------
    s          : (n_cols,) arc-length sample positions
    positions  : (n_cols, 3) DICOM mm positions
    tangents   : (n_cols, 3) unit tangent vectors
    normals    : (n_cols, 3) Bishop normal vectors (parallel-transported)
    binormals  : (n_cols, 3) B = T × N
    """
    s = np.linspace(0.0, total_len, n_cols)
    positions = cs(s)                        # (n_cols, 3)
    T_raw     = cs(s, 1)                     # analytic first derivative
    norms     = np.linalg.norm(T_raw, axis=1, keepdims=True) + 1e-15
    tangents  = T_raw / norms                # unit tangents (n_cols, 3)

    # — Initial normal — perpendicular to T[0] ─────────────────────────────
    if initial_normal is not None:
        # Project onto plane perpendicular to T[0], re-normalise
        n0 = initial_normal - np.dot(initial_normal, tangents[0]) * tangents[0]
        n_norm = np.linalg.norm(n0)
        if n_norm < 1e-8:
            initial_normal = None  # degenerate — fall back
        else:
            n0 = n0 / n_norm
    if initial_normal is None:
        # Choose a stable reference axis (Horos uses world 'up' = +Y)
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(tangents[0], ref)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        n0 = np.cross(tangents[0], ref)
        n0 /= np.linalg.norm(n0) + 1e-15

    # — Parallel transport (Bishop frame) ─────────────────────────────────
    # Each normal is the previous normal projected onto the plane perpendicular
    # to the new tangent — the rotation-minimising frame used by Horos.
    normals   = np.empty((n_cols, 3), dtype=np.float64)
    binormals = np.empty((n_cols, 3), dtype=np.float64)
    normals[0] = n0
    b0 = np.cross(tangents[0], n0)
    b0_norm = np.linalg.norm(b0)
    binormals[0] = b0 / b0_norm if b0_norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    for i in range(1, n_cols):
        # Project previous normal onto plane perp to new tangent
        ni = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        ni_norm = np.linalg.norm(ni)
        if ni_norm > 1e-8:
            normals[i] = ni / ni_norm
        else:
            normals[i] = normals[i - 1]
        bi = np.cross(tangents[i], normals[i])
        bi_norm = np.linalg.norm(bi)
        binormals[i] = bi / bi_norm if bi_norm > 1e-8 else binormals[i - 1]

    return s, positions, tangents, normals, binormals


def _sample_volume_trilinear(
    volume: np.ndarray,
    vox_size: np.ndarray,
    pts_mm: np.ndarray,
) -> np.ndarray:
    """
    Trilinear interpolation of the CT volume at arbitrary DICOM mm positions.
    Mirrors CPRVolumeDataLinearInterpolatedFloatAtDicomVector.

    Parameters
    ----------
    volume   : (Z, Y, X) float32 CT HU volume
    vox_size : (3,) [sz, sy, sx] mm per voxel
    pts_mm   : (..., 3) float64 sample points in DICOM mm [z, y, x]

    Returns
    -------
    vals : (...) float32 — HU values; NaN for out-of-bounds points
    """
    shape_in = pts_mm.shape[:-1]
    pts_flat = pts_mm.reshape(-1, 3)              # (M, 3)
    # Convert mm → voxel coordinates (volumeTransform in Horos)
    pts_vox = pts_flat / vox_size[np.newaxis, :]  # (M, 3)
    z_v = pts_vox[:, 0]
    y_v = pts_vox[:, 1]
    x_v = pts_vox[:, 2]
    vol_shape = np.array(volume.shape, dtype=np.float64)
    valid = (
        (z_v >= 0) & (z_v <= vol_shape[0] - 1) &
        (y_v >= 0) & (y_v <= vol_shape[1] - 1) &
        (x_v >= 0) & (x_v <= vol_shape[2] - 1)
    )
    # map_coordinates order=1 = trilinear — exact equivalent of Horos
    # CPRVolumeDataLinearInterpolatedFloatAtVolumeCoordinate
    vals = map_coordinates(
        volume,
        [z_v, y_v, x_v],
        order=1,
        mode="constant",
        cval=np.nan,
    ).astype(np.float32)
    vals[~valid] = np.nan
    return vals.reshape(shape_in)


def _build_cpr_image(
    volume: np.ndarray,
    vox_size: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    n_rows: int,
    row_extent_mm: float,
    slab_mm: float = 3.0,
) -> np.ndarray:
    """
    Build a 2-D straightened CPR image by sampling the CT volume on a
    rectangular grid in the (normal, binormal) plane at each centerline point.
    Mirrors CPRStraightenedOperation execution.

    Layout (matches Horos CPRStraightenedView):
      - columns  = arc-length axis (one column per centerline sample)
      - rows     = lateral axis along normals direction
      - slab MIP = thin-slab MaxIP along binormal direction
                   (slabWidth in CPRGeneratorRequest)

    Parameters
    ----------
    volume        : (Z, Y, X) float32 CT HU
    vox_size      : (3,) [sz, sy, sx] mm/voxel
    positions     : (n_cols, 3) centerline positions in mm
    normals       : (n_cols, 3) Bishop normal — defines the row axis
    binormals     : (n_cols, 3) Bishop binormal — defines the slab direction
    n_rows        : int — number of output rows (= pixelsHigh in Horos)
    row_extent_mm : float — half-width of the row axis in mm
    slab_mm       : float — total slab thickness in mm for MIP (slabWidth)

    Returns
    -------
    img : (n_rows, n_cols) float32 — HU image, NaN for out-of-bounds
    """
    n_cols = len(positions)
    mean_sp = float(np.mean(vox_size))
    n_slab  = max(1, int(np.round(slab_mm / mean_sp)))
    slab_offsets = np.linspace(-slab_mm / 2.0, slab_mm / 2.0, n_slab)  # mm

    # Row sample coords: [+row_extent_mm … 0 … -row_extent_mm]
    # positive = displayed at top when origin='upper' in imshow
    row_offsets = np.linspace(row_extent_mm, -row_extent_mm, n_rows)   # (n_rows,)

    # Base sample grid: (n_rows, n_cols, 3)
    pts_base = (
        positions[np.newaxis, :, :]                           # (1, n_cols, 3)
        + row_offsets[:, np.newaxis, np.newaxis]              # (n_rows, 1, 1)
        * normals[np.newaxis, :, :]                           # (1, n_cols, 3)
    )  # → (n_rows, n_cols, 3)

    if n_slab == 1:
        img = _sample_volume_trilinear(volume, vox_size, pts_base)  # (n_rows, n_cols)
        return img

    # MIP over slab (along binormal)
    slab_max = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
    for s_off in slab_offsets:
        pts = pts_base + s_off * binormals[np.newaxis, :, :]  # (n_rows, n_cols, 3)
        vals = _sample_volume_trilinear(volume, vox_size, pts)  # (n_rows, n_cols)
        better   = vals > slab_max
        not_nan  = ~np.isnan(vals)
        slab_max[better & not_nan] = vals[better & not_nan]

    slab_max[np.isneginf(slab_max)] = np.nan
    return slab_max


def _compute_cpr_data(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_mm: List[float],
    slab_thickness_mm: float = 3.0,
    width_mm: float = 25.0,
    initial_normal: Optional[np.ndarray] = None,
    pixels_wide: int = 512,
    pixels_high: int = 512,
    aorta_prepend_mm: float = 5.0,
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Straightened CPR pipeline (Kanitsar et al. 2002 / Horos CPRStraightenedOperation).
    Steps:
      1. _bezier_fit_centerline  — C2-smooth spline through centerline
      2. _sample_bezier_frame    — Bishop / parallel-transport frame
      3. Rotate frame by rotation_deg around tangent (rotational CPR)
      4. _build_cpr_image        — sample CT volume on perpendicular planes
    Parameters
    ----------
    volume           : (Z, Y, X) HU float32
    centerline_ijk   : (N, 3) centerline voxel indices [z, y, x]
    spacing_mm       : [sz, sy, sx]
    slab_thickness_mm: total slab thickness for MIP along binormal (mm)
    width_mm         : half-width of lateral (row) axis in mm
    initial_normal   : (3,) optional initial normal for the Bishop frame
    pixels_wide      : output columns (arc-length axis)  — default 512
    pixels_high      : output rows    (lateral axis)     — default 512
    aorta_prepend_mm : mm to extrapolate backward from ostium (aortic root)
    rotation_deg     : rotation angle (degrees) of the cutting plane around
                       the vessel centerline.  0° = default Bishop frame.
                       Enables rotational CPR à la Kanitsar / Stanford 3DQ.
    
    Returns
    -------
    cpr_volume : (pixels_wide, pixels_high) float32
                 Indexed as [arc_col, lateral_row]; transpose for imshow.
    N_frame    : (pixels_wide, 3) (rotated) normal vectors
    B_frame    : (pixels_wide, 3) (rotated) binormal vectors
    cl_mm      : (pixels_wide, 3) sampled centerline positions in mm
    arclengths : (pixels_wide,) cumulative arc-length in mm
    n_height   : pixels_high
    n_width    : pixels_wide
    """
    vox_size = np.array(spacing_mm, dtype=np.float64)  # [sz, sy, sx]
    cl_mm_raw = centerline_ijk.astype(np.float64) * vox_size[np.newaxis, :]
    # Prepend aortic approach segment (extrapolate backward from ostium)
    if aorta_prepend_mm > 0.0 and len(cl_mm_raw) >= 2:
        mean_sp_mm = float(np.mean(vox_size))
        n_prepend  = max(1, int(np.round(aorta_prepend_mm / mean_sp_mm)))
        d0 = cl_mm_raw[1] - cl_mm_raw[0]
        d0_norm = np.linalg.norm(d0)
        if d0_norm > 1e-9:
            d0 /= d0_norm
            offsets = np.arange(n_prepend, 0, -1, dtype=np.float64)[:, np.newaxis] * mean_sp_mm
            prepend_pts = cl_mm_raw[0][np.newaxis, :] - d0[np.newaxis, :] * offsets
            # Clip prepended points to volume bounds to avoid OOB stripes
            vol_max_mm = (np.array(volume.shape, dtype=np.float64) - 1) * vox_size
            prepend_pts = np.clip(prepend_pts, 0.0, vol_max_mm)
            cl_mm_raw = np.vstack([prepend_pts, cl_mm_raw])
    try:
        cs, total_len = _bezier_fit_centerline(cl_mm_raw)
    except ValueError as e:
        print(f"[visualize] _compute_cpr_data: {e}")
        empty = np.full((pixels_wide, pixels_high), np.nan, dtype=np.float32)
        z = np.zeros((pixels_wide, 3))
        return empty, z, z, cl_mm_raw[:pixels_wide], np.linspace(0, 1, pixels_wide), pixels_high, pixels_wide
    # Step 2: Sample frame at pixels_wide equally-spaced arc-length positions
    s, positions, tangents, N_frame, B_frame = _sample_bezier_frame(
        cs, total_len, pixels_wide, initial_normal=initial_normal,
    )

    # Step 3: Rotational CPR — rotate N, B around T by rotation_deg
    if abs(rotation_deg) > 1e-6:
        theta = np.deg2rad(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        N_rot =  cos_t * N_frame + sin_t * B_frame
        B_rot = -sin_t * N_frame + cos_t * B_frame
        N_frame = N_rot
        B_frame = B_rot

    # Step 4: Build CPR image — (pixels_high, pixels_wide) float32
    cpr_img = _build_cpr_image(
        volume, vox_size,
        positions, N_frame, B_frame,
        n_rows=pixels_high,
        row_extent_mm=width_mm,
        slab_mm=slab_thickness_mm,
    )  # (pixels_high, pixels_wide)
    # ── Fix vertical stripe artifacts ──────────────────────────────────────
    # Strategy: for each column with >30% NaN rows, replace NaN values using
    # per-row linear interpolation from nearest valid columns.  This is more
    # robust than the previous per-column copy because it preserves the lateral
    # structure at each row independently.
    col_nan_frac = np.mean(np.isnan(cpr_img), axis=0)  # (pixels_wide,)
    bad_cols  = np.where(col_nan_frac > 0.30)[0]
    good_cols = np.where(col_nan_frac <= 0.30)[0]
    if bad_cols.size > 0 and good_cols.size >= 2:
        # Row-wise interpolation: for each row, interpolate NaN columns
        # from the nearest good columns on either side
        for row in range(cpr_img.shape[0]):
            row_data = cpr_img[row, :]
            nan_mask = np.isnan(row_data)
            if nan_mask.all() or (~nan_mask).all():
                continue
            valid_idx = np.where(~nan_mask)[0]
            # np.interp extrapolates with edge values by default
            row_data[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, row_data[valid_idx]
            )
            cpr_img[row, :] = row_data
    elif bad_cols.size > 0 and good_cols.size > 0:
        # Fallback: too few good columns — use nearest-neighbour fill
        for col in bad_cols:
            nearest = good_cols[np.argmin(np.abs(good_cols - col))]
            cpr_img[:, col] = cpr_img[:, nearest]
    # Transpose → (pixels_wide, pixels_high) for indexed [col, row] access
    # render functions transpose back to (n_rows, n_cols) for imshow
    cpr_volume = cpr_img.T  # (pixels_wide, pixels_high)



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
