"""
export_raw.py
Export the unfiltered pericoronary VOI as a .raw binary file (int16, same spatial
dimensions as the original CCTA volume) plus a companion metadata JSON.

Non-VOI voxels are set to a sentinel value (default 0 HU) so the volume shape
exactly matches the source CCTA and can be loaded in any tool that accepts .raw.

File layout:
    <output_dir>/
        <prefix>_voi.raw          — int16, Fortran order Z-fast (or C-order Z-first)
        <prefix>_voi_metadata.json — shape, spacing, origin, dtype, sentinel, etc.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
# New NIfTI export function
# ─────────────────────────────────────────────

def export_voi_nifti(
    voi_mask: np.ndarray,
    spacing_mm: list,
    output_dir: str | Path,
    prefix: str = "pcat",
) -> Path:
    """
    Save binary VOI mask as NIfTI .nii.gz file.

    Parameters
    ----------
    voi_mask    : (Z, Y, X) bool array — True inside pericoronary VOI
    spacing_mm  : [sz, sy, sx] voxel spacing in mm
    output_dir  : directory to write file
    prefix      : filename prefix

    Returns
    -------
    Path to saved .nii.gz file
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI export. Install with: pip install nibabel"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sz, sy, sx = spacing_mm

    # NIfTI affine: diagonal voxel spacing (RAS convention, mm)
    # Array is (Z, Y, X) but NIfTI expects (X, Y, Z) — transpose
    data = voi_mask.astype(np.int8)  # 1=VOI, 0=outside

    # Affine: scale matrix (sz, sy, sx diagonal)
    # We keep the numpy (Z,Y,X) order — nibabel handles it via the affine
    affine = np.diag([sx, sy, sz, 1.0])

    img = nib.Nifti1Image(data, affine=affine)
    img.header.set_zooms((sx, sy, sz))
    img.header['qform_code'] = 1
    img.header['sform_code'] = 1

    out_path = output_dir / f"{prefix}_voi.nii.gz"
    nib.save(img, str(out_path))

    n_voi = int(voi_mask.sum())
    print(f"[export] NIfTI VOI saved: {out_path.name}  ({n_voi:,} VOI voxels)")
    return out_path


def export_combined_voi_nifti(
    vessel_masks: dict,
    spacing_mm: list,
    output_dir: str | Path,
    prefix: str = "pcat_all",
) -> Path:
    """Export union of all vessel VOI masks as a single NIfTI file."""
    shapes = [m.shape for m in vessel_masks.values()]
    combined = np.zeros(shapes[0], dtype=bool)
    for mask in vessel_masks.values():
        combined |= mask
    return export_voi_nifti(combined, spacing_mm, output_dir, prefix)


# ─────────────────────────────────────────────
# Main export function
# ─────────────────────────────────────────────

# Deprecated: use export_voi_nifti() instead
def export_voi_raw(
    volume: np.ndarray,
    voi_mask: np.ndarray,
    meta: Dict[str, Any],
    output_dir: str | Path,
    prefix: str = "pcat",
    sentinel_hu: int = 0,
    dtype: np.dtype = np.int16,
) -> Tuple[Path, Path]:
    """
    Save unfiltered VOI voxels to a .raw file with the same dimensions as the CCTA.

    Voxels outside the VOI mask are replaced with `sentinel_hu`.
    Voxels inside the VOI retain their original HU values (clamped to int16 range).

    Parameters
    ----------
    volume      : (Z, Y, X) float32 HU array — full CCTA volume
    voi_mask    : (Z, Y, X) bool — True inside the pericoronary VOI
    meta        : metadata dict from dicom_loader.load_dicom_series()
    output_dir  : directory to write files into (created if needed)
    prefix      : filename prefix (e.g. "patient1200_LAD")
    sentinel_hu : HU value for non-VOI voxels in the output (default 0)
    dtype       : output dtype (default int16; covers -32768..32767, sufficient for HU)

    Returns
    -------
    raw_path   : Path to the .raw file
    json_path  : Path to the metadata JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output volume: sentinel everywhere, VOI values preserved
    out = np.full(volume.shape, sentinel_hu, dtype=np.float32)
    out[voi_mask] = volume[voi_mask]

    # Convert to target dtype with clamping
    info = np.iinfo(dtype)
    out_clipped = np.clip(out, info.min, info.max).astype(dtype)

    # Write raw binary (C-order, Z-first — axial slices contiguous)
    raw_path = output_dir / f"{prefix}_voi.raw"
    out_clipped.tofile(str(raw_path))

    # Write companion metadata JSON
    json_path = output_dir / f"{prefix}_voi_metadata.json"
    export_meta = {
        "prefix": prefix,
        "raw_file": raw_path.name,
        "shape_zyx": list(volume.shape),          # (Z, Y, X)
        "dtype": str(dtype(0).dtype),              # e.g. "int16"
        "byte_order": "C",                         # row-major / Z-first
        "sentinel_hu": int(sentinel_hu),
        "n_voi_voxels": int(voi_mask.sum()),
        "spacing_mm_zyx": meta["spacing_mm"],      # [z, y, x]
        "origin_mm_xyz": meta["origin_mm"],        # [x, y, z] patient coords
        "orientation": meta["orientation"],
        "patient_id": meta.get("patient_id", ""),
        "series_description": meta.get("series_description", ""),
        "rescale_slope": meta.get("rescale_slope", 1.0),
        "rescale_intercept": meta.get("rescale_intercept", -1024.0),
        "z_positions_mm": meta["z_positions"],
        "load_instructions": (
            "np.fromfile(raw_file, dtype=dtype).reshape(shape_zyx)"
        ),
    }

    with open(json_path, "w") as f:
        json.dump(export_meta, f, indent=2)

    n_voi = int(voi_mask.sum())
    file_mb = raw_path.stat().st_size / 1e6
    print(f"[export_raw] Saved {raw_path.name}  ({file_mb:.1f} MB, {n_voi:,} VOI voxels)")
    print(f"[export_raw] Metadata: {json_path.name}")

    return raw_path, json_path


# ─────────────────────────────────────────────
# Multi-vessel combined export
# ─────────────────────────────────────────────

# Deprecated: use export_combined_voi_nifti() instead
def export_combined_voi_raw(
    volume: np.ndarray,
    vessel_masks: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_dir: str | Path,
    prefix: str = "pcat_all",
    sentinel_hu: int = 0,
    dtype: np.dtype = np.int16,
) -> Tuple[Path, Path]:
    """
    Export a combined VOI mask (union of all vessel VOIs) as a single .raw file.

    Parameters
    ----------
    vessel_masks : dict mapping vessel name → voi_mask bool array

    Returns
    -------
    raw_path, json_path
    """
    combined_mask = np.zeros(volume.shape, dtype=bool)
    for name, mask in vessel_masks.items():
        combined_mask |= mask

    return export_voi_raw(
        volume=volume,
        voi_mask=combined_mask,
        meta=meta,
        output_dir=output_dir,
        prefix=prefix,
        sentinel_hu=sentinel_hu,
        dtype=dtype,
    )


# ─────────────────────────────────────────────
# Load helper (for downstream analysis)
# ─────────────────────────────────────────────

def load_voi_raw(raw_path: str | Path, json_path: str | Path) -> Tuple[np.ndarray, Dict]:
    """
    Load a previously exported VOI .raw file back into numpy.

    Returns
    -------
    volume : int16 (Z, Y, X) array
    meta   : the companion metadata dict
    """
    with open(json_path) as f:
        meta = json.load(f)

    dtype = np.dtype(meta["dtype"])
    shape = tuple(meta["shape_zyx"])
    volume = np.fromfile(str(raw_path), dtype=dtype).reshape(shape)

    return volume, meta
