"""QThread worker that runs the full PCAT pipeline in the background.

Wraps the pipeline modules (auto_seeds, centerline, contour_extraction,
pcat_segment) and emits per-stage progress signals so the GUI stays
responsive.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from pcat_workstation.models.patient_session import PIPELINE_STAGES, PatientSession


# Default vessels when none are specified
_ALL_VESSELS = ["LAD", "LCx", "RCA"]


class PipelineWorker(QThread):
    """Run the PCAT analysis pipeline on a background thread.

    Signals
    -------
    stage_started      : str           – stage name beginning execution
    stage_completed    : str, float    – stage name, elapsed seconds
    stage_failed       : str, str      – stage name, error message
    pipeline_completed : dict          – full results dict
    pipeline_failed    : str           – fatal/unrecoverable error message
    progress_message   : str           – free-form status text
    """

    stage_started = Signal(str)
    stage_completed = Signal(str, float)
    stage_failed = Signal(str, str)
    pipeline_completed = Signal(dict)
    pipeline_failed = Signal(str)
    progress_message = Signal(str)

    seeds_ready = Signal(object)
    centerlines_ready = Signal(object)
    contours_ready = Signal(object)
    cpr_ready = Signal(str, object)  # (vessel_name, cpr_image_2d)
    voi_masks_ready = Signal(object)

    def __init__(
        self,
        session: PatientSession,
        vessels: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.vessels = vessels or list(_ALL_VESSELS)
        self.resume_from = resume_from

        # Intermediate results – accessible after completion
        self.vessel_centerlines: Dict[str, np.ndarray] = {}
        self.vessel_contour_results: Dict[str, Any] = {}
        self.vessel_voi_masks: Dict[str, np.ndarray] = {}
        self.vessel_stats: Dict[str, Any] = {}

        # Private helpers
        self._seeds_path: Optional[Path] = None
        self._centerlines_npz: Optional[Path] = None

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------

    def _should_skip(self, stage: str) -> bool:
        """Return True if *stage* should be skipped (already complete or
        before *resume_from*)."""
        if self.resume_from is not None:
            try:
                resume_idx = PIPELINE_STAGES.index(self.resume_from)
                stage_idx = PIPELINE_STAGES.index(stage)
                if stage_idx < resume_idx:
                    return True
            except ValueError:
                pass
        return self.session.stage_status.get(stage) == "complete"

    def _emit(self, msg: str) -> None:
        self.progress_message.emit(msg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 – sequential pipeline stages
        """Execute the pipeline stages sequentially."""
        # Late imports to avoid import-time side effects (TotalSegmentator,
        # matplotlib, etc.) and keep the GUI import graph lightweight.
        try:
            from pipeline.auto_seeds import generate_seeds
            from pipeline.centerline import (
                VESSEL_CONFIGS,
                clip_centerline_by_arclength,
                estimate_vessel_radii,
                load_seeds,
            )
            from pipeline.contour_extraction import (
                build_contour_based_voi,
                extract_vessel_contours,
            )
            from pipeline.pcat_segment import compute_pcat_stats
        except Exception as exc:
            self.pipeline_failed.emit(
                f"Failed to import pipeline modules: {exc}"
            )
            return

        volume = self.session.get_volume()
        meta = self.session.get_meta()
        if volume is None or meta is None:
            self.pipeline_failed.emit(
                "Session has no loaded volume/meta. Call session.load_dicom() first."
            )
            return

        spacing_mm = meta["spacing_mm"]
        session_dir = self.session.session_dir
        prefix = self.session._prefix
        results: Dict[str, Any] = {"vessels": {}, "errors": []}

        # ── Stage: import ────────────────────────────────────────────
        if not self._should_skip("import"):
            t0 = time.time()
            self.stage_started.emit("import")
            self._emit("Volume already loaded")
            self.session.set_stage_status("import", "complete")
            self.stage_completed.emit("import", time.time() - t0)

        # ── Stage: seeds ─────────────────────────────────────────────
        if not self._should_skip("seeds"):
            t0 = time.time()
            self.stage_started.emit("seeds")
            self.session.set_stage_status("seeds", "running")
            try:
                self._emit("Generating coronary seeds via TotalSegmentator...")
                seeds_json = session_dir / f"{prefix}_seeds.json"
                if not seeds_json.exists():
                    generate_seeds(
                        dicom_dir=self.session.dicom_dir,
                        output_json=seeds_json,
                    )
                self._seeds_path = seeds_json
                self.session.set_stage_status("seeds", "complete")
                self.stage_completed.emit("seeds", time.time() - t0)

                import json
                seeds_data_raw = json.loads(self._seeds_path.read_text())
                seed_points = {}
                for v in self.vessels:
                    for key in (v, v.upper(), v.replace("x", "X")):
                        if key in seeds_data_raw and seeds_data_raw[key].get("ostium_ijk"):
                            ijk = seeds_data_raw[key]["ostium_ijk"]
                            if all(c is not None for c in ijk):
                                seed_points[v] = ijk
                                break
                if seed_points:
                    self.seeds_ready.emit(seed_points)
            except Exception as exc:
                self._handle_stage_failure("seeds", exc)
                self.pipeline_failed.emit(f"Seeds generation failed: {exc}")
                return
        else:
            # Locate existing seeds file for later stages
            self._seeds_path = session_dir / f"{prefix}_seeds.json"

        # ── Stage: vesselness ────────────────────────────────────────
        # In auto mode the seed_editor produces an NPZ with centerlines;
        # we load those rather than running a separate vesselness filter.
        if not self._should_skip("vesselness"):
            t0 = time.time()
            self.stage_started.emit("vesselness")
            self.session.set_stage_status("vesselness", "running")
            try:
                self._emit("Loading centerlines from seed data...")
                raw_dir = session_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                npz_path = raw_dir / f"{prefix}_centerlines.npz"

                if not npz_path.exists():
                    # If no NPZ yet, generate seeds which will produce it
                    seeds_json = self._seeds_path or (
                        session_dir / f"{prefix}_seeds.json"
                    )
                    if not seeds_json.exists():
                        generate_seeds(
                            dicom_dir=self.session.dicom_dir,
                            output_json=seeds_json,
                        )
                    self._seeds_path = seeds_json

                self._centerlines_npz = npz_path if npz_path.exists() else None
                self.session.set_stage_status("vesselness", "complete")
                self.stage_completed.emit("vesselness", time.time() - t0)
            except Exception as exc:
                self._handle_stage_failure("vesselness", exc)
                self.pipeline_failed.emit(
                    f"Vesselness/centerline loading failed: {exc}"
                )
                return
        else:
            raw_dir = session_dir / "raw"
            npz_path = raw_dir / f"{prefix}_centerlines.npz"
            self._centerlines_npz = npz_path if npz_path.exists() else None

        # ── Stage: centerlines ───────────────────────────────────────
        if not self._should_skip("centerlines"):
            t0 = time.time()
            self.stage_started.emit("centerlines")
            self.session.set_stage_status("centerlines", "running")
            try:
                seeds_data = load_seeds(self._seeds_path)

                for vessel in self.vessels:
                    self._emit(f"Processing {vessel} centerline...")
                    if vessel not in seeds_data:
                        self._emit(f"  {vessel} not in seeds file -- skipping")
                        continue

                    vsd = seeds_data[vessel]
                    ostium = vsd.get("ostium_ijk")
                    if not ostium or any(v is None for v in ostium):
                        self._emit(f"  {vessel} has null seeds -- skipping")
                        continue

                    # Load full centerline from NPZ
                    centerline_full = None
                    if self._centerlines_npz and self._centerlines_npz.exists():
                        cl_data = np.load(
                            str(self._centerlines_npz), allow_pickle=True
                        )
                        cl_key = f"{vessel}_centerline_ijk"
                        if cl_key in cl_data:
                            centerline_full = cl_data[cl_key]

                    if centerline_full is None or len(centerline_full) < 3:
                        self._emit(
                            f"  {vessel} centerline not found or too short"
                        )
                        continue

                    vcfg = VESSEL_CONFIGS.get(vessel, {})
                    start_mm = float(vcfg.get("start_mm", 0.0))
                    length_mm = float(vcfg.get("length_mm", 40.0))

                    centerline = clip_centerline_by_arclength(
                        centerline_full,
                        spacing_mm,
                        start_mm=start_mm,
                        length_mm=length_mm,
                    )
                    if len(centerline) < 5:
                        self._emit(
                            f"  {vessel} clipped centerline too short "
                            f"({len(centerline)} pts)"
                        )
                        continue

                    # Estimate radii
                    self._emit(f"  Estimating {vessel} vessel radii...")
                    radii_mm = estimate_vessel_radii(
                        volume, centerline, spacing_mm
                    )

                    # Store results
                    self.vessel_centerlines[vessel] = {
                        "full": centerline_full,
                        "proximal": centerline,
                        "radii": radii_mm,
                    }
                    self._emit(
                        f"  {vessel}: {len(centerline)} pts, "
                        f"mean radius {float(np.mean(radii_mm)):.2f} mm"
                    )

                self.session.set_stage_status("centerlines", "complete")
                self.stage_completed.emit("centerlines", time.time() - t0)

                cl_viz = {v: d["proximal"] for v, d in self.vessel_centerlines.items()}
                self.centerlines_ready.emit(cl_viz)
            except Exception as exc:
                self._handle_stage_failure("centerlines", exc)
                self.pipeline_failed.emit(
                    f"Centerline extraction failed: {exc}"
                )
                return

        # ── Stage: contours ──────────────────────────────────────────
        if not self._should_skip("contours"):
            t0 = time.time()
            self.stage_started.emit("contours")
            self.session.set_stage_status("contours", "running")

            for vessel in self.vessels:
                if vessel not in self.vessel_centerlines:
                    continue
                try:
                    self._emit(f"Extracting {vessel} vessel wall contours...")
                    centerline = self.vessel_centerlines[vessel]["proximal"]
                    contour_result = extract_vessel_contours(
                        volume,
                        centerline,
                        spacing_mm,
                        vessel_name=vessel,
                    )
                    self.vessel_contour_results[vessel] = contour_result
                    self._emit(
                        f"  {vessel} contours: r_eq mean="
                        f"{np.mean(contour_result.r_eq):.2f} mm"
                    )
                except Exception as exc:
                    self.stage_failed.emit(
                        "contours", f"{vessel}: {exc}"
                    )
                    self._emit(f"  {vessel} contour extraction failed: {exc}")

            self.session.set_stage_status("contours", "complete")
            self.stage_completed.emit("contours", time.time() - t0)

            self.contours_ready.emit(dict(self.vessel_contour_results))

            # Generate CPR images from contour data
            for vessel, cr in self.vessel_contour_results.items():
                try:
                    from pipeline.visualize import _build_cpr_image_fast
                    cpr_img = _build_cpr_image_fast(
                        volume, np.array(spacing_mm),
                        cr.positions_mm, cr.N_frame, cr.B_frame,
                        n_rows=128, row_extent_mm=10.0, slab_mm=3.0,
                    )
                    self.cpr_ready.emit(vessel, cpr_img)
                    self._emit(f"  {vessel} CPR generated")
                except Exception as exc:
                    self._emit(f"  {vessel} CPR failed: {exc}")

        # ── Stage: pcat_voi ──────────────────────────────────────────
        if not self._should_skip("pcat_voi"):
            t0 = time.time()
            self.stage_started.emit("pcat_voi")
            self.session.set_stage_status("pcat_voi", "running")

            for vessel, cr in self.vessel_contour_results.items():
                try:
                    self._emit(f"Building {vessel} PCAT VOI mask...")
                    voi_mask = build_contour_based_voi(
                        volume_shape=volume.shape,
                        contours=cr.contours,
                        centerline_mm=cr.positions_mm,
                        N_frame=cr.N_frame,
                        B_frame=cr.B_frame,
                        r_eq=cr.r_eq,
                        spacing_mm=spacing_mm,
                        pcat_scale=3.0,
                    )
                    self.vessel_voi_masks[vessel] = voi_mask
                    self._emit(
                        f"  {vessel} VOI: {int(voi_mask.sum()):,} voxels"
                    )
                except Exception as exc:
                    self.stage_failed.emit("pcat_voi", f"{vessel}: {exc}")
                    self._emit(f"  {vessel} VOI build failed: {exc}")

            self.session.set_stage_status("pcat_voi", "complete")
            self.stage_completed.emit("pcat_voi", time.time() - t0)

            self.voi_masks_ready.emit(dict(self.vessel_voi_masks))

        # ── Stage: statistics ────────────────────────────────────────
        if not self._should_skip("statistics"):
            t0 = time.time()
            self.stage_started.emit("statistics")
            self.session.set_stage_status("statistics", "running")

            for vessel, voi_mask in self.vessel_voi_masks.items():
                try:
                    self._emit(f"Computing {vessel} PCAT statistics...")
                    stats = compute_pcat_stats(volume, voi_mask, vessel)
                    self.vessel_stats[vessel] = stats
                    self.session.set_vessel_stats(vessel, stats)
                    self._emit(
                        f"  {vessel}: mean_HU={stats['hu_mean']:.1f}, "
                        f"fat_fraction={100*stats['fat_fraction']:.1f}%"
                    )
                except Exception as exc:
                    self.stage_failed.emit(
                        "statistics", f"{vessel}: {exc}"
                    )
                    self._emit(f"  {vessel} stats failed: {exc}")

            self.session.set_stage_status("statistics", "complete")
            self.stage_completed.emit("statistics", time.time() - t0)

        # ── Done ─────────────────────────────────────────────────────
        results["vessels"] = dict(self.vessel_stats)
        self._emit("Pipeline complete")
        self.pipeline_completed.emit(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_stage_failure(self, stage: str, exc: Exception) -> None:
        """Log a stage failure and update session status."""
        tb = traceback.format_exc()
        msg = f"{stage} failed: {exc}\n{tb}"
        self.stage_failed.emit(stage, str(exc))
        self.session.set_stage_status(stage, "failed")
        self._emit(msg)
