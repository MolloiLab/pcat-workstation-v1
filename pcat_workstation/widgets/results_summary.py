"""Results summary landing page shown after pipeline completes."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from pcat_workstation.app.config import FAI_RISK_THRESHOLD, VESSEL_CONFIGS

_VESSELS = ["LAD", "LCx", "RCA"]
_COLUMNS = ["Vessel", "FAI (HU)", "Risk", "Confidence", "Flags"]

_BG = "#1c1c1e"
_ROW_ALT = "#2c2c2e"
_TEXT = "#e5e5e7"
_MUTED = "#98989d"
_GREEN = "#30d158"
_AMBER = "#ff9f0a"
_MONO = "'Menlo', 'Courier New', monospace"


class ResultsSummary(QWidget):
    """Scrollable results landing page for PCAT analysis."""

    view_cpr_requested = Signal(str)
    review_flags_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {_BG}; border: none; }}")
        outer.addWidget(scroll)

        container = QWidget()
        container.setStyleSheet(f"background: {_BG};")
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setSpacing(12)
        scroll.setWidget(container)

        # title
        self._title = QLabel("PCAT Analysis Results")
        self._title.setStyleSheet(f"font-size: 22pt; font-weight: bold; color: {_TEXT};")
        self._layout.addWidget(self._title)

        # patient info
        self._patient_label = QLabel()
        self._patient_label.setStyleSheet(f"font-size: 15pt; color: {_MUTED};")
        self._layout.addWidget(self._patient_label)

        # table container
        self._table_frame = QFrame()
        self._table_frame.setStyleSheet("border: none;")
        self._table_layout = QVBoxLayout(self._table_frame)
        self._table_layout.setContentsMargins(0, 8, 0, 8)
        self._table_layout.setSpacing(0)
        self._layout.addWidget(self._table_frame)

        # clinical caption
        self._caption = QLabel()
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(
            f"font-size: 15pt; color: {_TEXT}; font-style: italic;"
        )
        self._layout.addWidget(self._caption)

        # navigation buttons
        nav = QHBoxLayout()
        nav.setSpacing(12)
        self._nav_buttons: dict[str, QPushButton] = {}
        for vessel in _VESSELS:
            btn = QPushButton(f"View {vessel} CPR")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._secondary_btn_css())
            btn.clicked.connect(lambda _=False, v=vessel: self.view_cpr_requested.emit(v))
            nav.addWidget(btn)
            self._nav_buttons[vessel] = btn
        nav.addStretch()
        self._layout.addLayout(nav)

        # flags section
        self._flags_frame = QFrame()
        self._flags_frame.setVisible(False)
        flags_lay = QHBoxLayout(self._flags_frame)
        flags_lay.setContentsMargins(0, 4, 0, 0)
        self._flags_label = QLabel()
        self._flags_label.setStyleSheet(f"font-size: 15pt; color: {_AMBER};")
        flags_lay.addWidget(self._flags_label)
        review_btn = QPushButton("Review All \u2192")
        review_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        review_btn.setStyleSheet(self._secondary_btn_css())
        review_btn.clicked.connect(self.review_flags_requested.emit)
        flags_lay.addWidget(review_btn)
        flags_lay.addStretch()
        self._layout.addWidget(self._flags_frame)

        self._layout.addStretch()

    # --------------------------------------------------------------- public
    def set_results(
        self,
        patient_id: str,
        study_date: str,
        vessel_stats: dict,
    ):
        """Populate the results page.

        vessel_stats format per vessel::

            {"mean_fai": -74.2, "risk": "LOW", "confidence": "High",
             "flags": 2}

        Only ``mean_fai`` is required; other keys fall back to defaults.
        """
        self._patient_label.setText(f"Patient {patient_id} \u2014 {study_date}")

        # rebuild table rows
        self._clear_table()
        self._add_header_row()

        elevated_vessels: list[str] = []
        total_flags = 0

        for idx, vessel in enumerate(_VESSELS):
            stats = vessel_stats.get(vessel, {})
            mean_fai = stats.get("mean_fai")
            fai_str = f"{mean_fai:.1f}" if mean_fai is not None else "\u2014"

            is_elevated = mean_fai is not None and mean_fai >= FAI_RISK_THRESHOLD
            risk_text = "Elevated" if is_elevated else "Normal"
            risk_color = _AMBER if is_elevated else _GREEN

            if is_elevated:
                elevated_vessels.append(vessel)

            confidence = stats.get("confidence", "\u2014")
            flags = stats.get("flags", 0)
            total_flags += flags

            bg = _ROW_ALT if idx % 2 == 1 else _BG
            vessel_color = VESSEL_CONFIGS[vessel]["color"]

            self._add_data_row(
                vessel, vessel_color, fai_str, risk_text, risk_color,
                str(confidence), str(flags), bg,
            )

        # caption
        if elevated_vessels:
            parts = []
            for v in elevated_vessels:
                parts.append(
                    f"{v}: Elevated pericoronary inflammation. "
                    f"Above {FAI_RISK_THRESHOLD} HU threshold "
                    "(Oikonomou et al., Lancet 2018)."
                )
            self._caption.setText(" ".join(parts))
        else:
            self._caption.setText("No elevated pericoronary inflammation detected.")

        # flags section
        if total_flags > 0:
            self._flags_label.setText(
                f"\u26A0 {total_flags} position{'s' if total_flags != 1 else ''} "
                "flagged for review"
            )
            self._flags_frame.setVisible(True)
        else:
            self._flags_frame.setVisible(False)

    def clear(self):
        """Reset to empty state."""
        self._patient_label.setText("")
        self._caption.setText("")
        self._clear_table()
        self._flags_frame.setVisible(False)

    # --------------------------------------------------------------- table helpers
    def _clear_table(self):
        while self._table_layout.count():
            item = self._table_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_header_row(self):
        row = QFrame()
        row.setStyleSheet(f"background: {_BG}; border: none;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(0)
        for col in _COLUMNS:
            lbl = QLabel(col)
            lbl.setStyleSheet(
                f"font-size: 13pt; font-weight: bold; color: {_MUTED}; border: none;"
            )
            lbl.setMinimumWidth(120)
            lay.addWidget(lbl, stretch=1)
        self._table_layout.addWidget(row)

    def _add_data_row(
        self,
        vessel: str,
        vessel_color: str,
        fai_str: str,
        risk_text: str,
        risk_color: str,
        confidence: str,
        flags: str,
        bg: str,
    ):
        row = QFrame()
        row.setStyleSheet(f"background: {bg}; border: none;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(0)

        # vessel
        v_lbl = QLabel(vessel)
        v_lbl.setStyleSheet(
            f"font-size: 15pt; font-weight: bold; color: {vessel_color}; border: none;"
        )
        v_lbl.setMinimumWidth(120)
        lay.addWidget(v_lbl, stretch=1)

        # FAI
        f_lbl = QLabel(fai_str)
        f_lbl.setStyleSheet(
            f"font-size: 15pt; color: {_TEXT}; font-family: {_MONO}; border: none;"
        )
        f_lbl.setMinimumWidth(120)
        lay.addWidget(f_lbl, stretch=1)

        # Risk
        r_lbl = QLabel(risk_text)
        r_lbl.setStyleSheet(
            f"font-size: 15pt; font-weight: 600; color: {risk_color}; border: none;"
        )
        r_lbl.setMinimumWidth(120)
        lay.addWidget(r_lbl, stretch=1)

        # Confidence
        c_lbl = QLabel(confidence)
        c_lbl.setStyleSheet(f"font-size: 15pt; color: {_TEXT}; border: none;")
        c_lbl.setMinimumWidth(120)
        lay.addWidget(c_lbl, stretch=1)

        # Flags
        fl_lbl = QLabel(flags)
        fl_lbl.setStyleSheet(f"font-size: 15pt; color: {_TEXT}; border: none;")
        fl_lbl.setMinimumWidth(120)
        lay.addWidget(fl_lbl, stretch=1)

        self._table_layout.addWidget(row)

    @staticmethod
    def _secondary_btn_css() -> str:
        return (
            "QPushButton {"
            f"  background: {_BG};"
            f"  color: {_MUTED};"
            "  border: 1px solid #38383a;"
            "  border-radius: 6px;"
            "  padding: 8px 16px;"
            "  font-size: 13pt;"
            "}"
            "QPushButton:hover {"
            f"  background: {_ROW_ALT};"
            f"  color: {_TEXT};"
            "}"
        )
