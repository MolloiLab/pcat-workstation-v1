"""
dicom_loader.py
Load a DICOM series folder → 3D HU numpy array + spatial metadata dict.

Handles Siemens syngo.via exports:
  - RescaleIntercept = -8192  (Siemens FOV-fill sentinel, clamped to -1024 air on load)
  - Axial orientation [1,0,0,0,1,0]
  - Sub-mm isotropic voxels
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pydicom


def load_dicom_series(dicom_dir: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load all .dcm files in dicom_dir, sort by ImagePositionPatient Z,
    apply RescaleSlope/Intercept to get Hounsfield Units.

    Returns
    -------
    volume : np.ndarray  float32, shape (Z, Y, X), in HU
    meta   : dict with keys:
        patient_dir, n_slices, rows, cols,
        spacing_mm  [z, y, x],
        origin_mm   [x, y, z]  (ImagePositionPatient of first slice),
        orientation (list),
        rescale_intercept, rescale_slope,
        z_positions (list of floats, mm)
    """
    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    # Read all slices (headers only first for sorting)
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        slices.append(ds)

    # Sort by Z position
    def _z(ds):
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None:
            return float(pos[2])
        inst = getattr(ds, "InstanceNumber", 0)
        return float(inst)

    slices.sort(key=_z)

    # Reference slice for metadata
    ref = slices[0]
    rows = int(ref.Rows)
    cols = int(ref.Columns)
    pixel_spacing = [float(x) for x in ref.PixelSpacing]  # [row_spacing, col_spacing] in mm

    # Compute Z spacing from consecutive slice positions
    z_positions = [_z(s) for s in slices]
    if len(z_positions) > 1:
        z_diffs = np.diff(z_positions)
        z_spacing = float(np.median(np.abs(z_diffs)))
    else:
        z_spacing = float(getattr(ref, "SliceThickness", 1.0))

    # spacing_mm: (z, y, x)  — row = y direction, col = x direction
    spacing_mm = [z_spacing, pixel_spacing[0], pixel_spacing[1]]

    # Origin = ImagePositionPatient of first slice (patient coords in mm)
    origin = getattr(ref, "ImagePositionPatient", [0.0, 0.0, 0.0])
    origin_mm = [float(v) for v in origin]

    orientation = getattr(ref, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    orientation = [float(v) for v in orientation]

    rescale_slope = float(getattr(ref, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ref, "RescaleIntercept", -1024.0))

    # Siemens sentinel value (-8192 HU) marks pixels outside the scan FOV.
    # These are not real tissue — clamp them to -1024 HU (air) so they don't
    # corrupt VOI stats, fat thresholding, or vessel detection.
    # Also hard-clip extreme high HU (> 3095) which are scanner artefacts.
    SENTINEL_HU  = -8192.0
    HU_AIR       = -1024.0
    HU_MAX_VALID =  3095.0
    # Build 3D array
    volume = np.zeros((len(slices), rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        raw = ds.pixel_array.astype(np.float32)
        hu  = raw * rescale_slope + rescale_intercept
        # Replace FOV-fill sentinel with air
        hu[hu <= SENTINEL_HU + 1] = HU_AIR
        # Clip implausibly high values (metal artefact limit)
        hu = np.clip(hu, HU_AIR, HU_MAX_VALID)
        volume[i] = hu

    meta = {
        "patient_dir": str(dicom_dir),
        "patient_id": str(getattr(ref, "PatientID", "unknown")),
        "study_description": str(getattr(ref, "StudyDescription", "")),
        "series_description": str(getattr(ref, "SeriesDescription", "")),
        "n_slices": len(slices),
        "rows": rows,
        "cols": cols,
        "spacing_mm": spacing_mm,   # [z, y, x]
        "origin_mm": origin_mm,     # [x, y, z] in patient coords
        "orientation": orientation,
        "rescale_intercept": rescale_intercept,
        "rescale_slope": rescale_slope,
        "z_positions": z_positions,
        "shape": list(volume.shape),  # (Z, Y, X)
    }

    return volume, meta
