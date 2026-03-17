from PySide6.QtCore import QObject, Signal
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


# Ordered pipeline stages
PIPELINE_STAGES = [
    "import",
    "seeds",
    "centerlines",
    "contours",
    "pcat_voi",
    "statistics",
]


class PatientSession(QObject):
    """Tracks one patient analysis session through the PCAT pipeline.

    Wraps pipeline stage outputs with autosave and session management.
    The CT volume itself is kept in memory but never serialized to
    session.json -- only metadata, stage status, and vessel stats are
    persisted.
    """

    stage_changed = Signal(str, str)  # (stage_name, status)
    data_changed = Signal()
    session_saved = Signal(str)  # save path

    def __init__(self, session_dir: Path, parent=None):
        super().__init__(parent)
        self.session_dir: Path = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.patient_id: str = ""
        self.study_date: str = ""
        self.series_description: str = ""
        self.kVp: float = 0.0
        self.dicom_dir: Optional[Path] = None
        self.created_at: datetime = datetime.now()
        self.modified_at: datetime = datetime.now()

        self.stage_status: Dict[str, str] = {
            stage: "pending" for stage in PIPELINE_STAGES
        }
        self.vessel_stats: Dict[str, Dict[str, Any]] = {}

        self._volume: Optional[np.ndarray] = None
        self._meta: Optional[Dict] = None
        self._prefix: str = "pcat"
        self.seeds_data: Optional[Dict] = None  # Extended seeds for edit mode

    # ------------------------------------------------------------------
    # DICOM loading
    # ------------------------------------------------------------------

    def load_dicom(self, dicom_dir: Path) -> None:
        """Load a DICOM series and extract relevant metadata."""
        from pipeline.dicom_loader import load_dicom_series

        self.dicom_dir = Path(dicom_dir)
        volume, meta = load_dicom_series(dicom_dir)
        self._volume = volume
        self._meta = meta

        self.patient_id = str(meta.get("patient_id", self.session_dir.name))
        self.study_date = str(meta.get("study_date", ""))
        self.series_description = str(meta.get("series_description", ""))
        self.kVp = float(meta.get("kVp", 0.0))

        self.set_stage_status("import", "complete")

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def set_stage_status(self, stage: str, status: str) -> None:
        """Update the status of a pipeline stage.

        Args:
            stage: One of PIPELINE_STAGES.
            status: One of "pending", "running", "complete", "failed",
                    "skipped".
        """
        self.stage_status[stage] = status
        self.stage_changed.emit(stage, status)
        self._autosave()

    def get_resume_stage(self) -> Optional[str]:
        """Return the next stage to run after the last completed one.

        Returns None if no stage has been completed yet.
        """
        last_complete_idx = -1
        for idx, stage in enumerate(PIPELINE_STAGES):
            if self.stage_status.get(stage) == "complete":
                last_complete_idx = idx

        if last_complete_idx < 0:
            return None

        next_idx = last_complete_idx + 1
        if next_idx < len(PIPELINE_STAGES):
            return PIPELINE_STAGES[next_idx]
        return None  # all stages complete

    # ------------------------------------------------------------------
    # Volume / meta access
    # ------------------------------------------------------------------

    def get_volume(self) -> Optional[np.ndarray]:
        """Return the cached CT volume, attempting reload if absent."""
        if self._volume is None:
            vol_path = self.session_dir / f"{self._prefix}_volume.npz"
            if vol_path.exists():
                data = np.load(vol_path)
                self._volume = data["volume"]
        return self._volume

    def get_meta(self) -> Optional[Dict]:
        """Return cached DICOM metadata."""
        return self._meta

    # ------------------------------------------------------------------
    # Vessel stats
    # ------------------------------------------------------------------

    def set_vessel_stats(self, vessel: str, stats: Dict[str, Any]) -> None:
        """Store per-vessel FAI results."""
        self.vessel_stats[vessel] = stats
        self.data_changed.emit()
        self._autosave()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _autosave(self) -> None:
        """Save session state to session.json (no volume data)."""
        self.modified_at = datetime.now()
        save_path = self.session_dir / "session.json"
        save_path.write_text(
            json.dumps(self.to_dict(), indent=4, default=str),
            encoding="utf-8",
        )
        self.session_saved.emit(str(save_path))

    def save(self) -> None:
        """Explicit save -- delegates to _autosave."""
        self._autosave()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the session."""
        return {
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "series_description": self.series_description,
            "kVp": self.kVp,
            "dicom_dir": str(self.dicom_dir) if self.dicom_dir else None,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "stage_status": dict(self.stage_status),
            "vessel_stats": dict(self.vessel_stats),
            "seeds_data": self.seeds_data,
        }

    # ------------------------------------------------------------------
    # Class-level loaders
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, session_dir: Path, parent=None) -> "PatientSession":
        """Load a session from an existing session.json.

        The CT volume is NOT loaded here -- it will be lazy-loaded on
        first call to get_volume().
        """
        session_dir = Path(session_dir)
        json_path = session_dir / "session.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return cls.from_dict(data, session_dir, parent=parent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        session_dir: Path,
        parent=None,
    ) -> "PatientSession":
        """Reconstruct a PatientSession from a serialized dict."""
        session = cls(session_dir, parent=parent)
        session.patient_id = data.get("patient_id", "")
        session.study_date = data.get("study_date", "")
        session.series_description = data.get("series_description", "")
        session.kVp = float(data.get("kVp", 0.0))

        dicom_dir = data.get("dicom_dir")
        session.dicom_dir = Path(dicom_dir) if dicom_dir else None

        created = data.get("created_at")
        if created:
            session.created_at = datetime.fromisoformat(created)
        modified = data.get("modified_at")
        if modified:
            session.modified_at = datetime.fromisoformat(modified)

        session.stage_status.update(data.get("stage_status", {}))
        # Backward compat: drop vesselness from old session.json files
        session.stage_status.pop("vesselness", None)
        session.vessel_stats = data.get("vessel_stats", {})
        session.seeds_data = data.get("seeds_data")

        return session
