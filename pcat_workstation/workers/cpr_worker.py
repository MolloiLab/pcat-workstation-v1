"""Lightweight async CPR generator for live seed editing.

Mirrors Horos' CPRGenerator pattern: when the curved path (spline
through seeds) changes, a new CPR image is generated asynchronously
and delivered via signal.  Only the CPR computation runs here — no
vesselness, VOI, or statistics.
"""

from __future__ import annotations

import traceback
from typing import List

import numpy as np
from PySide6.QtCore import QThread, Signal


def build_cpr_fast(
    volume: np.ndarray,
    centerline_ijk: np.ndarray,
    spacing_mm: list,
    pixels_wide: int = 512,
    pixels_high: int = 256,
    width_mm: float = 25.0,
    slab_mm: float = 3.0,
) -> dict | None:
    """Fast CPR generation (~100ms) for live editing preview.

    Bypasses _compute_cpr_data's slow aorta-prepend logic.
    Calls the building blocks directly:
      1. _bezier_fit_centerline (spline fit)
      2. _sample_bezier_frame (Bishop frame)
      3. _build_cpr_image_fast (trilinear sampling)

    Returns dict with keys:
      cpr_image: (pixels_wide, pixels_high) float32 array
      N_frame, B_frame: (pixels_wide, 3) Bishop frame vectors
      positions_mm: (pixels_wide, 3) centerline positions in mm
      arclengths: (pixels_wide,) arc-length at each position
    Or None if centerline too short.
    """
    from pipeline.visualize import (
        _bezier_fit_centerline,
        _build_cpr_image_fast,
        _sample_bezier_frame,
    )

    # Convert centerline from voxel indices to mm
    cl_mm = centerline_ijk * np.array(spacing_mm, dtype=np.float64)

    # Fit cubic spline
    try:
        cs, total_len = _bezier_fit_centerline(cl_mm)
    except ValueError:
        return None

    # Sample Bishop frame at pixels_wide positions
    s, positions, tangents, normals, binormals = _sample_bezier_frame(
        cs, total_len, pixels_wide,
    )

    # Build CPR image via trilinear interpolation
    vox_size = np.array(spacing_mm, dtype=np.float64)
    cpr_image = _build_cpr_image_fast(
        volume, vox_size, positions, normals, binormals,
        n_rows=pixels_high, row_extent_mm=width_mm, slab_mm=slab_mm,
    )

    # cpr_image from _build_cpr_image_fast is (n_rows, n_cols) = (pixels_high, pixels_wide).
    # The caller expects (pixels_wide, pixels_high) to match _compute_cpr_data's convention.
    cpr_image = cpr_image.T

    return {
        "cpr_image": cpr_image,
        "N_frame": normals,
        "B_frame": binormals,
        "positions_mm": positions,
        "arclengths": s,
    }


class CPRWorker(QThread):
    """Generate a CPR image from a spline centerline on a background thread.

    Signals
    -------
    cpr_ready       : str, object, float  – vessel, cpr_image, row_extent_mm
    cpr_frame_ready : str, object         – vessel, frame_data dict
    """

    cpr_ready = Signal(str, object, float)
    cpr_frame_ready = Signal(str, object)

    def __init__(
        self,
        vessel: str,
        volume: np.ndarray,
        centerline_ijk: np.ndarray,
        spacing_mm: List[float],
        parent=None,
    ):
        super().__init__(parent)
        self.vessel = vessel
        self.volume = volume
        self.centerline_ijk = centerline_ijk
        self.spacing_mm = spacing_mm

    def run(self) -> None:
        try:
            result = build_cpr_fast(
                self.volume,
                self.centerline_ijk,
                self.spacing_mm,
                pixels_wide=512,
                pixels_high=256,
                width_mm=25.0,
                slab_mm=3.0,
            )
            if result is None:
                return
            self.cpr_ready.emit(self.vessel, result["cpr_image"], 25.0)
            self.cpr_frame_ready.emit(self.vessel, {
                "N_frame": result["N_frame"],
                "B_frame": result["B_frame"],
                "positions_mm": result["positions_mm"],
                "arclengths": result["arclengths"],
                "volume": self.volume,
                "spacing": self.spacing_mm,
            })
        except Exception:
            traceback.print_exc()
