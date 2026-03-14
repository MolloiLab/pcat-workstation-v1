"""Worker for loading DICOM data in a subprocess (bypasses Python GIL)."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal


def _load_in_subprocess(
    dicom_dir: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run in a child process — imports are local to avoid pickling issues."""
    from pipeline.dicom_loader import load_dicom_series

    volume, meta = load_dicom_series(dicom_dir)
    return volume, meta


class DicomLoaderWorker(QThread):
    """Load DICOM files in a separate *process* so the UI never freezes.

    Uses multiprocessing.Process + a pipe to fully bypass the Python GIL.
    The QThread just waits on the pipe and forwards signals to Qt.

    Signals
    -------
    progress : str              – status message
    finished : object, object   – (volume, meta) on success
    failed   : str              – error message on failure
    """

    progress = Signal(str)
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, dicom_dir: Path, session: Any, parent=None):
        super().__init__(parent)
        self.dicom_dir = dicom_dir
        self.session = session

    def run(self) -> None:
        try:
            self.progress.emit("Reading DICOM files...")

            # Run the heavy DICOM I/O + numpy building in a child process
            with mp.Pool(1) as pool:
                result = pool.apply(_load_in_subprocess, (str(self.dicom_dir),))

            volume, meta = result
            self.progress.emit("Finalizing...")

            # Update session state (lightweight, OK on this thread)
            self.session._volume = volume
            self.session._meta = meta
            self.session.dicom_dir = self.dicom_dir
            self.session.patient_id = str(meta.get("patient_id", "unknown"))
            self.session.study_date = str(meta.get("study_date", ""))
            self.session.series_description = str(
                meta.get("series_description", "")
            )
            self.session.kVp = float(meta.get("kVp", 0.0))

            self.finished.emit(volume, meta)
        except Exception as exc:
            self.failed.emit(str(exc))
