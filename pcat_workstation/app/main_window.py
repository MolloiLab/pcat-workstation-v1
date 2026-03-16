"""Main application window for the PCAT Workstation."""

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QStatusBar, QMessageBox, QProgressDialog,
    QStackedWidget, QSplitter,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QKeySequence, QAction
from pathlib import Path
from typing import Optional

from pcat_workstation.widgets.dicom_browser import DicomBrowser
from pcat_workstation.widgets.progress_panel import ProgressPanel
from pcat_workstation.widgets.toolbar import MainToolBar
from pcat_workstation.widgets.mpr_panel import MPRPanel
from pcat_workstation.widgets.results_summary import ResultsSummary
from pcat_workstation.widgets.analysis_dashboard import AnalysisDashboard
from pcat_workstation.workers.pipeline_worker import PipelineWorker
from pcat_workstation.workers.dicom_loader_worker import DicomLoaderWorker
from pcat_workstation.models.patient_session import PatientSession
from pcat_workstation.models.dicom_index import DicomIndex
from pcat_workstation.app.config import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_LEVEL


class MainWindow(QMainWindow):
    """Top-level window that orchestrates DICOM browser, VTK viewer,
    progress panel, and toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCAT Workstation")
        self.setMinimumSize(1200, 800)

        # State
        self._dicom_index = DicomIndex()
        self._session: Optional[PatientSession] = None
        self._current_vessel: str = "LAD"
        self._pipeline_worker: Optional[PipelineWorker] = None
        self._loader_worker: Optional[DicomLoaderWorker] = None

        # Build UI
        self._setup_central_widget()
        self._setup_docks()
        self._setup_toolbar()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._connect_signals()

        # Populate recent projects
        self._load_recent_projects()

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def _setup_central_widget(self) -> None:
        # Stacked widget: page 0 = MPR viewer, page 1 = results summary
        self._central_stack = QStackedWidget()

        self._mpr_panel = MPRPanel()
        self._results_summary = ResultsSummary()

        self._central_stack.addWidget(self._mpr_panel)       # index 0
        self._central_stack.addWidget(self._results_summary)  # index 1
        self._central_stack.setCurrentIndex(0)

        # Analysis dashboard (collapsible, below the stacked widget)
        self._analysis_dashboard = AnalysisDashboard()

        # Vertical splitter: top = stack, bottom = dashboard
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._central_stack)
        splitter.addWidget(self._analysis_dashboard)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.setCentralWidget(splitter)

    def _setup_docks(self) -> None:
        # Left dock — DICOM Browser
        self._dicom_browser = DicomBrowser()
        left_dock = QDockWidget("DICOM Browser", self)
        left_dock.setWidget(self._dicom_browser)
        left_dock.setFeatures(
            QDockWidget.DockWidgetFeature(0)  # not closable, not movable, not floatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # Right dock — Progress Panel
        self._progress_panel = ProgressPanel()
        right_dock = QDockWidget("Pipeline", self)
        right_dock.setWidget(self._progress_panel)
        right_dock.setFeatures(
            QDockWidget.DockWidgetFeature(0)
        )
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

    def _setup_toolbar(self) -> None:
        self._toolbar = MainToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)

    def _setup_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        import_action = QAction("Import DICOM...", self)
        import_action.setShortcut(QKeySequence("Ctrl+O"))
        import_action.triggered.connect(self._dicom_browser._on_import_clicked)
        file_menu.addAction(import_action)

        export_action = QAction("Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._toolbar.export_clicked)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Pipeline menu
        pipeline_menu = menu_bar.addMenu("&Pipeline")

        run_action = QAction("Run", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._on_run_pipeline)
        pipeline_menu.addAction(run_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready \u2014 Import a DICOM folder to begin")

    def _connect_signals(self) -> None:
        # DICOM browser signals
        self._dicom_browser.dicom_imported.connect(self._on_dicom_imported)
        self._dicom_browser.session_selected.connect(self._on_session_selected)
        self._dicom_browser.session_removed.connect(self._on_session_removed)

        # Progress panel
        self._progress_panel.run_clicked.connect(self._on_run_pipeline)

        # Toolbar
        self._toolbar.run_clicked.connect(self._on_run_pipeline)
        self._toolbar.vessel_changed.connect(self._on_vessel_changed)
        self._toolbar.wl_preset_changed.connect(self._on_wl_changed)

        # MPR panel
        self._mpr_panel.window_level_changed.connect(self._on_viewer_wl_changed)

        # Results summary
        self._results_summary.view_cpr_requested.connect(self._on_view_cpr)

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    @Slot(str)
    def _on_dicom_imported(self, dicom_dir: str) -> None:
        """Handle a new DICOM folder import."""
        if self._loader_worker is not None and self._loader_worker.isRunning():
            return

        dicom_path = Path(dicom_dir)

        # Check if we already have a session for this DICOM dir
        existing = self._dicom_index.get_session_dir_for_dicom(dicom_path)
        if existing and existing.exists():
            session_dir = existing
        else:
            session_dir = self._dicom_index.create_session_dir(
                patient_id="unknown", study_date=""
            )

        self._session = PatientSession(session_dir)

        # Disable import while loading
        self._dicom_browser._import_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading DICOM from {dicom_dir}...")

        # Progress dialog
        self._load_progress = QProgressDialog(
            "Reading DICOM files...", None, 0, 100, self
        )
        self._load_progress.setWindowTitle("Loading DICOM")
        self._load_progress.setMinimumDuration(0)
        self._load_progress.setWindowModality(Qt.WindowModal)
        self._load_progress.setValue(0)

        # Launch background loader
        self._loader_worker = DicomLoaderWorker(dicom_path, self._session, parent=self)
        self._loader_worker.progress.connect(self._on_loader_progress)
        self._loader_worker.progress_pct.connect(self._on_loader_pct)
        self._loader_worker.finished.connect(
            lambda vol, meta: self._on_loader_finished(vol, meta, dicom_path, session_dir)
        )
        self._loader_worker.failed.connect(self._on_loader_failed)
        self._loader_worker.start()

    @Slot(str)
    def _on_loader_progress(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if hasattr(self, "_load_progress") and self._load_progress is not None:
            self._load_progress.setLabelText(message)

    @Slot(int, int)
    def _on_loader_pct(self, current: int, total: int) -> None:
        if hasattr(self, "_load_progress") and self._load_progress is not None:
            self._load_progress.setMaximum(total)
            self._load_progress.setValue(current)

    def _on_loader_finished(self, volume, meta, dicom_path: Path, session_dir: Path) -> None:
        """Handle successful DICOM load from background thread."""
        # Close progress dialog
        if hasattr(self, "_load_progress") and self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None

        # Re-enable import
        self._dicom_browser._import_btn.setEnabled(True)

        # Mark import stage complete (must be on main thread for autosave)
        self._session.set_stage_status("import", "complete")

        # VTK setup must happen on main thread
        if volume is not None and meta is not None:
            spacing = meta.get("spacing_mm", [1.0, 1.0, 1.0])
            self._mpr_panel.set_volume(volume, spacing)

        # Show viewer mode
        self._central_stack.setCurrentIndex(0)

        # Update DICOM browser patient info
        if meta:
            self._dicom_browser.set_patient_info(meta)

        # Rename session dir now that we know patient_id/study_date
        if session_dir.name.startswith("unknown_"):
            new_dir = self._dicom_index.create_session_dir(
                patient_id=self._session.patient_id,
                study_date=self._session.study_date,
            )
            # Move session data to properly-named dir
            import shutil
            for item in session_dir.iterdir():
                shutil.move(str(item), str(new_dir / item.name))
            session_dir.rmdir()
            session_dir = new_dir
            self._session.session_dir = new_dir
            self._session._autosave()

        # Update index
        self._dicom_index.add_recent(
            session_dir=session_dir,
            patient_id=self._session.patient_id,
            study_date=self._session.study_date,
            dicom_dir=dicom_path,
            stage_summary="imported",
        )
        self._load_recent_projects()

        # Enable run
        self._progress_panel.set_run_enabled(True)
        self._toolbar.set_run_enabled(True)
        self._progress_panel.set_stage_status("import", "complete")

        shape = meta.get("shape", [0, 0, 0]) if meta else [0, 0, 0]
        self.statusBar().showMessage(
            f"Loaded: {self._session.patient_id} "
            f"({shape[0]}\u00d7{shape[1]}\u00d7{shape[2]})"
        )
        self._loader_worker = None

    @Slot(str)
    def _on_loader_failed(self, error: str) -> None:
        """Handle DICOM load failure from background thread."""
        if hasattr(self, "_load_progress") and self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        self._dicom_browser._import_btn.setEnabled(True)
        QMessageBox.critical(self, "DICOM Load Error", error)
        self.statusBar().showMessage("DICOM load failed")
        self._loader_worker = None

    def _on_reload_finished(self, volume, meta) -> None:
        """Handle volume reload from an existing session."""
        if volume is not None and meta is not None:
            spacing = meta.get("spacing_mm", [1.0, 1.0, 1.0])
            self._mpr_panel.set_volume(volume, spacing)
        if meta:
            self._dicom_browser.set_patient_info(meta)
        self.statusBar().showMessage(f"Resumed: {self._session.patient_id}")
        self._loader_worker = None

    @Slot(str)
    def _on_session_selected(self, session_dir: str) -> None:
        """Handle selection of an existing session from the recent list."""
        session_path = Path(session_dir)
        if not (session_path / "session.json").exists():
            self.statusBar().showMessage("Session not found")
            return

        self._session = PatientSession.load(session_path)

        # Restore stage statuses in progress panel
        self._progress_panel.reset_stages()
        for stage, status in self._session.stage_status.items():
            self._progress_panel.set_stage_status(stage, status)

        # Reload volume if DICOM dir still exists
        if self._session.dicom_dir and self._session.dicom_dir.exists():
            self.statusBar().showMessage("Reloading volume...")
            self._loader_worker = DicomLoaderWorker(
                self._session.dicom_dir, self._session, parent=self
            )
            self._loader_worker.progress.connect(self._on_loader_progress)
            self._loader_worker.finished.connect(self._on_reload_finished)
            self._loader_worker.failed.connect(
                lambda _: self.statusBar().showMessage("Volume reload failed")
            )
            self._loader_worker.start()

        # Restore vessel stats if available
        if self._session.vessel_stats:
            self._progress_panel.set_vessel_summary(self._session.vessel_stats)
            # Show results summary if pipeline completed
            display_stats = {}
            for vessel, stats in self._session.vessel_stats.items():
                display_stats[vessel] = {
                    "mean_fai": stats.get("hu_mean", 0.0),
                    "risk": stats.get("fai_risk", "LOW"),
                }
            self._results_summary.set_results(
                patient_id=self._session.patient_id,
                study_date=self._session.study_date,
                vessel_stats=display_stats,
            )
            self._central_stack.setCurrentIndex(1)  # Show results
        else:
            self._central_stack.setCurrentIndex(0)  # Show viewer

        self._progress_panel.set_run_enabled(True)
        self._toolbar.set_run_enabled(True)

        self.statusBar().showMessage(f"Resumed: {self._session.patient_id}")

    @Slot()
    def _on_run_pipeline(self) -> None:
        """Launch the PCAT analysis pipeline."""
        if self._session is None:
            return
        if (
            self._pipeline_worker is not None
            and self._pipeline_worker.isRunning()
        ):
            return  # Already running

        # Determine resume point
        resume_from = self._session.get_resume_stage()

        self._pipeline_worker = PipelineWorker(
            session=self._session,
            resume_from=resume_from,
        )

        # Connect signals
        self._pipeline_worker.stage_started.connect(self._on_stage_started)
        self._pipeline_worker.stage_completed.connect(self._on_stage_completed)
        self._pipeline_worker.stage_failed.connect(self._on_stage_failed)
        self._pipeline_worker.pipeline_completed.connect(self._on_pipeline_completed)
        self._pipeline_worker.pipeline_failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.progress_message.connect(self._on_progress_message)
        self._pipeline_worker.seeds_ready.connect(self._on_seeds_ready)
        self._pipeline_worker.centerlines_ready.connect(self._on_centerlines_ready)
        self._pipeline_worker.contours_ready.connect(self._on_contours_ready)
        self._pipeline_worker.voi_masks_ready.connect(self._on_voi_masks_ready)
        self._pipeline_worker.cpr_ready.connect(self._on_cpr_ready)

        self._progress_panel.set_running(True)
        self._toolbar.set_run_enabled(False)
        self._central_stack.setCurrentIndex(0)  # Show viewer during pipeline

        self._mpr_panel.clear_overlays()
        self._mpr_panel.clear_cpr()
        self._pipeline_worker.start()

    @Slot(str)
    def _on_stage_started(self, stage: str) -> None:
        self._progress_panel.set_stage_status(stage, "running")
        self.statusBar().showMessage(f"Running: {stage}...")

    @Slot(str, float)
    def _on_stage_completed(self, stage: str, elapsed: float) -> None:
        self._progress_panel.set_stage_status(stage, "complete", elapsed)

    @Slot(str, str)
    def _on_stage_failed(self, stage: str, error: str) -> None:
        self._progress_panel.set_stage_status(stage, "failed")
        self.statusBar().showMessage(f"Failed: {stage} \u2014 {error}")

    @Slot(dict)
    def _on_pipeline_completed(self, results: dict) -> None:
        self._progress_panel.set_running(False)
        self._toolbar.set_run_enabled(True)

        # Update results summary
        if self._session and self._session.vessel_stats:
            # Convert vessel_stats format: need mean_fai key
            display_stats = {}
            for vessel, stats in self._session.vessel_stats.items():
                display_stats[vessel] = {
                    "mean_fai": stats.get("hu_mean", 0.0),
                    "risk": stats.get("fai_risk", "LOW"),
                }
            self._results_summary.set_results(
                patient_id=self._session.patient_id,
                study_date=self._session.study_date,
                vessel_stats=display_stats,
            )
            self._central_stack.setCurrentIndex(1)  # Show results

        # Update progress panel vessel summary
        if self._session.vessel_stats:
            self._progress_panel.set_vessel_summary(self._session.vessel_stats)

        self.statusBar().showMessage("Pipeline complete")
        self._pipeline_worker = None

    @Slot(str)
    def _on_pipeline_failed(self, error: str) -> None:
        self._progress_panel.set_running(False)
        self._toolbar.set_run_enabled(True)
        self.statusBar().showMessage(f"Pipeline failed: {error}")
        QMessageBox.warning(self, "Pipeline Error", error)
        self._pipeline_worker = None

    @Slot(str)
    def _on_progress_message(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._progress_panel.set_progress_message(message)

    @Slot(object)
    def _on_seeds_ready(self, seeds_dict: dict) -> None:
        meta = self._session.get_meta() if self._session else None
        if meta:
            self._mpr_panel.set_seed_overlay(seeds_dict, meta["spacing_mm"])

    @Slot(object)
    def _on_centerlines_ready(self, centerlines_dict: dict) -> None:
        meta = self._session.get_meta() if self._session else None
        if meta:
            self._mpr_panel.set_centerline_overlay(centerlines_dict, meta["spacing_mm"])

    @Slot(object)
    def _on_contours_ready(self, contour_results_dict: dict) -> None:
        self._mpr_panel.set_contour_overlay(contour_results_dict)

    @Slot(object)
    def _on_voi_masks_ready(self, voi_masks_dict: dict) -> None:
        meta = self._session.get_meta() if self._session else None
        if meta:
            self._mpr_panel.set_voi_overlay(voi_masks_dict, meta["spacing_mm"])

    @Slot(str, object)
    def _on_cpr_ready(self, vessel: str, cpr_image) -> None:
        self._mpr_panel.set_cpr_data(vessel, cpr_image)

    @Slot(str)
    def _on_vessel_changed(self, vessel: str) -> None:
        """Handle vessel selection change from toolbar."""
        self._current_vessel = vessel
        self._mpr_panel.set_cpr_vessel(vessel)
        self.statusBar().showMessage(f"Vessel: {vessel}")

    @Slot(float, float)
    def _on_wl_changed(self, window: float, level: float) -> None:
        """Handle W/L preset change from toolbar."""
        self._mpr_panel.set_window_level(window, level)

    @Slot(float, float)
    def _on_viewer_wl_changed(self, window: float, level: float) -> None:
        """Handle W/L change from viewer interaction."""
        pass  # Toolbar W/L display sync reserved for future

    @Slot(str)
    def _on_view_cpr(self, vessel: str) -> None:
        """Handle CPR view request from results summary."""
        self._central_stack.setCurrentIndex(0)  # Back to viewer
        self._toolbar.set_vessel(vessel)

    @Slot(str)
    def _on_session_removed(self, session_dir: str) -> None:
        """Remove a session from the recent projects list."""
        self._dicom_index.remove_recent(Path(session_dir))
        self._load_recent_projects()

    @Slot()
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About PCAT Workstation",
            "PCAT Workstation\n\n"
            "Pericoronary Adipose Tissue Analysis\n"
            "Molloi Lab \u2014 UC Irvine",
        )

    # ------------------------------------------------------------------ #
    #  Events
    # ------------------------------------------------------------------ #

    def showEvent(self, event):
        super().showEvent(event)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _load_recent_projects(self) -> None:
        """Populate the DICOM browser sidebar with recent projects."""
        recent = self._dicom_index.get_recent()
        self._dicom_browser.load_recent(recent)
