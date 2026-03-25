"""QThread worker that runs the full PCAT pipeline in the background.

Wraps the pipeline modules (centerline FMM+vesselness auto-trace, tubular VOI,
pcat_segment FAI + angular asymmetry) and emits per-stage progress signals so
the GUI stays responsive.

Pipeline flow:
  Manual ostium seed → FMM+Vesselness auto-trace centerline → tubular VOI → FAI + angular asymmetry
"""

from __future__ import annotations

import io
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from pcat_workstation.models.patient_session import PIPELINE_STAGES, PatientSession


class _TeeWriter(io.TextIOBase):
    """Tee stdout/stderr: writes to the original stream AND emits a Qt signal.

    Buffers partial lines so that each signal emission is a complete line.
    Filters out tqdm control characters (\\r, ANSI escapes) for clean display.
    """

    def __init__(self, original, signal_fn):
        super().__init__()
        self._original = original
        self._signal_fn = signal_fn
        self._buf = ""

    def write(self, text: str) -> int:
        # Always write to original (terminal)
        if self._original is not None:
            self._original.write(text)
            self._original.flush()
        # Buffer and emit complete lines to the UI
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line and not line.startswith("\x1b"):  # skip ANSI escapes
                self._signal_fn(line)
        # Handle \r (tqdm progress bar overwrite)
        if "\r" in self._buf:
            parts = self._buf.split("\r")
            last = parts[-1].strip()
            if last and not last.startswith("\x1b"):
                self._signal_fn(last)
            self._buf = ""
        return len(text)

    def flush(self):
        if self._original is not None:
            self._original.flush()


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
    cpr_ready = Signal(str, object, float)  # (vessel_name, cpr_image_2d, row_extent_mm)
    cpr_frame_ready = Signal(str, object)  # (vessel, dict with N_frame, B_frame, positions_mm, arclengths)
    radii_ready = Signal(object)  # {vessel: radii_mm_array}
    voi_masks_ready = Signal(object)
    analysis_data_ready = Signal(str, object)  # (vessel, dict with hu_values, distances_mm, mean_hu)

    def __init__(
        self,
        session: PatientSession,
        vessels: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
        stop_after: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.vessels = vessels or list(_ALL_VESSELS)
        self.resume_from = resume_from
        self.stop_after = stop_after

        # Intermediate results – accessible after completion
        self.vessel_centerlines: Dict[str, np.ndarray] = {}
        self.vessel_voi_masks: Dict[str, np.ndarray] = {}
        self.vessel_stats: Dict[str, Any] = {}

        # Private helpers
        self._seed_points: Dict[str, Any] = {}

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
        """Emit a progress message to the UI (clinician-facing)."""
        self.progress_message.emit(msg)

    def _debug(self, msg: str) -> None:
        """Print debug info to terminal only (developer-facing)."""
        print(msg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 – sequential pipeline stages
        """Execute the pipeline stages sequentially."""
        # Redirect stdout/stderr so that pipeline print() output
        # (progress bars, status messages) appears in the Progress panel.
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee_out = _TeeWriter(old_stdout, self._emit)
        tee_err = _TeeWriter(old_stderr, self._emit)
        sys.stdout = tee_out
        sys.stderr = tee_err
        try:
            self._run_pipeline()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _run_pipeline(self) -> None:  # noqa: C901
        """Internal pipeline execution (stdout/stderr already tee'd)."""
        # Late imports to avoid import-time side effects and keep the GUI
        # import graph lightweight.
        try:
            from pipeline.centerline import (
                VESSEL_CONFIGS,
                clip_centerline_by_arclength,
                estimate_vessel_radii,
            )
            from pipeline.pcat_segment import build_tubular_voi, compute_pcat_stats
            from pipeline.visualize import _compute_cpr_data
            from pcat_workstation.app.config import (
                VOI_MODE,
                CRISP_GAP_MM,
                CRISP_RING_MM,
                DEFAULT_PCAT_SCALE,
                N_ANGULAR_SECTORS,
            )
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
        # Seeds come from user interaction (SeedEditState), not auto-generation.
        # This stage validates that seeds exist in the session.
        if not self._should_skip("seeds"):
            t0 = time.time()
            self.stage_started.emit("seeds")
            self.session.set_stage_status("seeds", "running")
            try:
                seeds_data = self.session.seeds_data
                # SeedEditState saves as {"flat": {...}, "extended": {...}}
                # Extract the extended format if present
                if isinstance(seeds_data, dict) and "extended" in seeds_data:
                    seeds_data = seeds_data["extended"]
                if not seeds_data:
                    # Check for seeds JSON file as fallback
                    seeds_json = session_dir / f"{prefix}_seeds.json"
                    if seeds_json.exists():
                        import json
                        seeds_data = json.loads(seeds_json.read_text())

                if not seeds_data:
                    self._handle_stage_failure("seeds", RuntimeError(
                        "No seeds found. Place ostium seeds manually before running."
                    ))
                    self.pipeline_failed.emit(
                        "No seeds found. Place ostium seeds manually before running."
                    )
                    return

                seed_points = {}
                for v in self.vessels:
                    for key in (v, v.upper(), v.replace("x", "X")):
                        if key in seeds_data:
                            entry = seeds_data[key]
                            ostium = entry.get("ostium") or entry.get("ostium_ijk")
                            if ostium and all(c is not None for c in ostium):
                                seed_points[v] = {
                                    "ostium": ostium,
                                    "waypoints": entry.get("waypoints", entry.get("waypoints_ijk", [])),
                                }
                                break

                if not seed_points:
                    self._handle_stage_failure("seeds", RuntimeError(
                        "No valid ostium seeds found. Place seeds for at least one vessel."
                    ))
                    self.pipeline_failed.emit(
                        "No valid ostium seeds found. Place seeds for at least one vessel."
                    )
                    return

                self._seed_points = seed_points
                vessels_found = list(seed_points.keys())
                self._emit(f"Seeds validated: {', '.join(vessels_found)}")
                self.seeds_ready.emit(seed_points)
                self.session.set_stage_status("seeds", "complete")
                self.stage_completed.emit("seeds", time.time() - t0)
            except Exception as exc:
                self._handle_stage_failure("seeds", exc)
                self.pipeline_failed.emit(f"Seed validation failed: {exc}")
                return
        else:
            # Load existing seeds for later stages
            self._seed_points = {}
            seeds_data = self.session.seeds_data
            if isinstance(seeds_data, dict) and "extended" in seeds_data:
                seeds_data = seeds_data["extended"]
            if seeds_data:
                for v in self.vessels:
                    for key in (v, v.upper(), v.replace("x", "X")):
                        if key in seeds_data:
                            entry = seeds_data[key]
                            ostium = entry.get("ostium") or entry.get("ostium_ijk")
                            if ostium and all(c is not None for c in ostium):
                                self._seed_points[v] = {
                                    "ostium": ostium,
                                    "waypoints": entry.get("waypoints", entry.get("waypoints_ijk", [])),
                                }
                                break

        # ── stop_after check ──────────────────────────────────────────
        if self.stop_after == "seeds" and self.session.stage_status.get("seeds") == "complete":
            results["vessels"] = dict(self.vessel_stats)
            self._emit("Stopped after seeds")
            self.pipeline_completed.emit(results)
            return

        # ── Stage: centerlines ───────────────────────────────────────
        # Manual centerline: cubic spline through user-placed seeds.
        # No FMM or vesselness — the user's clicks ARE the centerline.
        if not self._should_skip("centerlines"):
            t0 = time.time()
            self.stage_started.emit("centerlines")
            self.session.set_stage_status("centerlines", "running")
            try:
                from pcat_workstation.models.seed_edit_state import _fit_spline_centerline

                for vessel in self.vessels:
                    if vessel not in self._seed_points:
                        continue

                    sp = self._seed_points[vessel]
                    ostium = sp["ostium"]
                    waypoints = sp.get("waypoints", [])

                    # Build ordered seed list: ostium → waypoints
                    ordered_seeds = [ostium] + [wp for wp in waypoints if wp]
                    if len(ordered_seeds) < 2:
                        self._emit(f"  {vessel}: need ostium + at least 1 waypoint — skipping")
                        continue

                    # Fit dense cubic spline through seeds (0.5mm step)
                    self._emit(f"Fitting {vessel} spline centerline...")
                    centerline_full = _fit_spline_centerline(
                        ordered_seeds, spacing_mm, volume.shape, step_mm=0.5
                    )

                    if centerline_full is None or len(centerline_full) < 3:
                        self._emit(f"  {vessel}: spline fitting failed — skipping")
                        continue

                    # Estimate radii on the FULL centerline
                    self._emit(f"Estimating {vessel} vessel radii...")
                    radii_full = estimate_vessel_radii(volume, centerline_full, spacing_mm)

                    # Clip to proximal segment for VOI/stats measurement only
                    vcfg = VESSEL_CONFIGS.get(vessel, {})
                    start_mm = float(vcfg.get("start_mm", 0.0))
                    length_mm = float(vcfg.get("length_mm", 40.0))
                    centerline_prox = clip_centerline_by_arclength(
                        centerline_full, spacing_mm,
                        start_mm=start_mm, length_mm=length_mm,
                    )
                    radii_prox = radii_full[:len(centerline_prox)]

                    if len(centerline_prox) < 5:
                        self._debug(f"  {vessel} proximal centerline too short ({len(centerline_prox)} pts)")
                        continue

                    self.vessel_centerlines[vessel] = {
                        "full": centerline_full,
                        "proximal": centerline_prox,
                        "radii_full": radii_full,
                        "radii": radii_prox,
                    }
                    arc_mm = len(centerline_full) * 0.5
                    self._debug(
                        f"  {vessel}: {len(centerline_full)} pts (~{arc_mm:.0f} mm), "
                        f"proximal={len(centerline_prox)} pts, "
                        f"mean radius {float(np.mean(radii_prox)):.2f} mm"
                    )

                    # Generate CPR from the FULL centerline
                    try:
                        self._emit(f"Generating {vessel} CPR image...")
                        cpr_vol, N_frame, B_frame, positions, arclengths, n_h, n_w = _compute_cpr_data(
                            volume, centerline_full, spacing_mm,
                            slab_thickness_mm=3.0,
                            width_mm=25.0,
                            pixels_wide=512,
                            pixels_high=256,
                        )
                        self.cpr_ready.emit(vessel, cpr_vol, 25.0)
                        self.cpr_frame_ready.emit(vessel, {
                            "N_frame": N_frame,
                            "B_frame": B_frame,
                            "positions_mm": positions,
                            "arclengths": arclengths,
                            "volume": volume,
                            "spacing": spacing_mm,
                        })
                        self._debug(f"  {vessel} CPR generated ({n_w}x{n_h})")
                    except Exception as exc:
                        self._emit(f"  {vessel} CPR failed: {exc}")

                self.session.set_stage_status("centerlines", "complete")
                self.stage_completed.emit("centerlines", time.time() - t0)

                cl_viz = {v: d["full"] for v, d in self.vessel_centerlines.items()}
                self.centerlines_ready.emit(cl_viz)

                radii_viz = {v: d["radii"] for v, d in self.vessel_centerlines.items() if "radii" in d}
                if radii_viz:
                    self.radii_ready.emit(radii_viz)
            except Exception as exc:
                self._handle_stage_failure("centerlines", exc)
                self.pipeline_failed.emit(f"Centerline extraction failed: {exc}")
                return

        # ── stop_after check ──────────────────────────────────────────
        if self.stop_after == "centerlines" and self.session.stage_status.get("centerlines") == "complete":
            results["vessels"] = dict(self.vessel_stats)
            self._emit("Stopped after centerlines")
            self.pipeline_completed.emit(results)
            return

        # ── Stage: pcat_voi ──────────────────────────────────────────
        if not self._should_skip("pcat_voi"):
            t0 = time.time()
            self.stage_started.emit("pcat_voi")
            self.session.set_stage_status("pcat_voi", "running")

            for vessel, cl_data in self.vessel_centerlines.items():
                try:
                    self._emit(f"Building {vessel} PCAT VOI mask...")
                    voi_mask = build_tubular_voi(
                        volume_shape=volume.shape,
                        centerline_ijk=cl_data["proximal"],
                        spacing_mm=spacing_mm,
                        radii_mm=cl_data["radii"],
                        voi_mode=VOI_MODE,
                        crisp_gap_mm=CRISP_GAP_MM,
                        crisp_ring_mm=CRISP_RING_MM,
                        radius_multiplier=DEFAULT_PCAT_SCALE,
                    )
                    self.vessel_voi_masks[vessel] = voi_mask
                    self._debug(f"  {vessel} VOI: {int(voi_mask.sum()):,} voxels")
                except Exception as exc:
                    self.stage_failed.emit("pcat_voi", f"{vessel}: {exc}")
                    self._emit(f"  {vessel} VOI build failed: {exc}")

            self.session.set_stage_status("pcat_voi", "complete")
            self.stage_completed.emit("pcat_voi", time.time() - t0)
            self.voi_masks_ready.emit(dict(self.vessel_voi_masks))

        # ── stop_after check ──────────────────────────────────────────
        if self.stop_after == "pcat_voi" and self.session.stage_status.get("pcat_voi") == "complete":
            results["vessels"] = dict(self.vessel_stats)
            self._emit("Stopped after pcat_voi")
            self.pipeline_completed.emit(results)
            return

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

                    # Compute angular asymmetry
                    cl_data = self.vessel_centerlines.get(vessel)
                    if cl_data is not None:
                        try:
                            from pipeline.pcat_segment import compute_angular_asymmetry
                            octant_data = compute_angular_asymmetry(
                                volume=volume,
                                centerline_ijk=cl_data["proximal"],
                                radii_mm=cl_data["radii"],
                                spacing_mm=spacing_mm,
                                n_sectors=N_ANGULAR_SECTORS,
                                voi_mode=VOI_MODE,
                                crisp_gap_mm=CRISP_GAP_MM,
                                crisp_ring_mm=CRISP_RING_MM,
                                radius_multiplier=DEFAULT_PCAT_SCALE,
                            )
                            stats["octants"] = octant_data
                            self._emit(f"  {vessel}: angular asymmetry computed ({N_ANGULAR_SECTORS} sectors)")
                        except Exception as ang_exc:
                            self._emit(f"  {vessel} angular asymmetry failed: {ang_exc}")

                    # Compute radial profile and emit analysis data
                    try:
                        from pipeline.radial_profile import compute_radial_profile
                        hu_values = volume[voi_mask].astype(np.float32)
                        distances_mm, mean_hu, std_hu = compute_radial_profile(
                            volume, voi_mask, spacing_mm=spacing_mm,
                        )
                        analysis_payload = {
                            "hu_values": hu_values,
                            "distances_mm": distances_mm,
                            "mean_hu": mean_hu,
                            "std_hu": std_hu,
                        }
                        if "octants" in stats:
                            analysis_payload["octants"] = stats["octants"]
                        self.analysis_data_ready.emit(vessel, analysis_payload)
                    except Exception as prof_exc:
                        self._emit(f"  {vessel} radial profile failed: {prof_exc}")
                except Exception as exc:
                    self.stage_failed.emit("statistics", f"{vessel}: {exc}")
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
