from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QFrame,
    QHeaderView, QAbstractItemView, QMenu,
)
from PySide6.QtCore import Signal, Qt, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
from typing import Optional, Dict, List


class DicomBrowser(QWidget):
    """Left sidebar widget for importing DICOM folders and browsing recent projects."""

    dicom_imported = Signal(str)
    session_selected = Signal(str)
    session_removed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumWidth(220)
        self.setMaximumWidth(350)

        self._build_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- Header ---
        header = QLabel("DICOM Browser")
        header.setStyleSheet("font-weight: bold; font-size: 18pt; color: #e5e5e7;")
        layout.addWidget(header)

        # --- Import button ---
        self._import_btn = QPushButton("Import DICOM Folder")
        self._import_btn.setCursor(Qt.PointingHandCursor)
        self._import_btn.clicked.connect(self._on_import_clicked)
        layout.addWidget(self._import_btn)

        # --- Drop zone ---
        self._drop_zone = QFrame()
        self._drop_zone.setFixedHeight(80)
        drop_layout = QVBoxLayout(self._drop_zone)
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_label = QLabel("Drop DICOM folder here")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setStyleSheet("color: #636366; font-size: 13pt; border: none;")
        drop_layout.addWidget(drop_label)
        layout.addWidget(self._drop_zone)

        # --- Separator ---
        layout.addWidget(self._separator())

        # --- Patient info section ---
        self._patient_info_frame = QFrame()
        info_layout = QVBoxLayout(self._patient_info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(3)

        self._lbl_patient_id = QLabel()
        self._lbl_patient_id.setStyleSheet("font-weight: bold; color: #e5e5e7;")
        self._lbl_study_date = QLabel()
        self._lbl_series_desc = QLabel()
        self._lbl_kvp = QLabel()
        self._lbl_dimensions = QLabel()
        self._lbl_dimensions.setStyleSheet("font-family: 'Menlo', 'Courier New', monospace; color: #e5e5e7;")
        self._lbl_spacing = QLabel()
        self._lbl_spacing.setStyleSheet("font-family: 'Menlo', 'Courier New', monospace; color: #e5e5e7;")

        for lbl in (
            self._lbl_patient_id,
            self._lbl_study_date,
            self._lbl_series_desc,
            self._lbl_kvp,
            self._lbl_dimensions,
            self._lbl_spacing,
        ):
            info_layout.addWidget(lbl)

        self._patient_info_frame.setVisible(False)
        layout.addWidget(self._patient_info_frame)

        # --- Separator ---
        self._patient_sep = self._separator()
        self._patient_sep.setVisible(False)
        layout.addWidget(self._patient_sep)

        # --- Recent projects ---
        recent_label = QLabel("Recent")
        recent_label.setStyleSheet("font-weight: bold; font-size: 15pt; color: #e5e5e7;")
        layout.addWidget(recent_label)

        self._recent_tree = QTreeWidget()
        self._recent_tree.setHeaderLabels(["Patient", "Date", "Status"])
        self._recent_tree.setColumnCount(3)
        self._recent_tree.setRootIsDecorated(False)
        self._recent_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._recent_tree.setAlternatingRowColors(True)
        self._recent_tree.setFrameShape(QFrame.NoFrame)
        self._recent_tree.itemDoubleClicked.connect(self._on_recent_double_clicked)
        self._recent_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._recent_tree.customContextMenuRequested.connect(self._on_recent_context_menu)

        header_view = self._recent_tree.header()
        header_view.setStretchLastSection(True)
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        layout.addWidget(self._recent_tree, stretch=1)

    # ------------------------------------------------------------------ #
    #  Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self):
        self.setStyleSheet(
            """
            DicomBrowser {
                background: #1c1c1e;
            }
            QPushButton#importBtn {
                background: #0a84ff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 15pt;
            }
            QPushButton#importBtn:hover {
                background: #0070e0;
            }
            QTreeWidget {
                background: #2c2c2e;
                alternate-background-color: #1c1c1e;
                color: #e5e5e7;
                border: none;
                font-size: 13pt;
            }
            QTreeWidget::item:selected {
                background: #464649;
            }
            QHeaderView::section {
                background: #2c2c2e;
                color: #98989d;
                border: none;
                padding: 4px;
                font-size: 13pt;
            }
            QLabel {
                color: #e5e5e7;
            }
            """
        )
        self._import_btn.setObjectName("importBtn")
        self._drop_zone.setStyleSheet(
            "QFrame { border: 2px dashed #38383a; border-radius: 8px; background: transparent; }"
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #38383a;")
        line.setFixedHeight(2)
        return line

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_patient_info(self, meta: Dict) -> None:
        """Update patient info labels from a DICOM metadata dictionary."""
        self._lbl_patient_id.setText(f"Patient: {meta.get('patient_id', 'N/A')}")
        self._lbl_study_date.setText(f"Study: {meta.get('study_description', 'N/A')}")
        self._lbl_series_desc.setText(f"Series: {meta.get('series_description', 'N/A')}")

        kvp = meta.get("kVp") or meta.get("kvp")
        self._lbl_kvp.setText(f"kVp: {kvp}" if kvp else "kVp: N/A")

        shape = meta.get("shape")
        if shape and len(shape) == 3:
            self._lbl_dimensions.setText(f"Dims: {shape[0]}\u00d7{shape[1]}\u00d7{shape[2]}")
        else:
            self._lbl_dimensions.setText("Dims: N/A")

        spacing = meta.get("spacing_mm")
        if spacing and len(spacing) >= 3:
            self._lbl_spacing.setText(
                f"Spacing: {spacing[0]:.2f} \u00d7 {spacing[1]:.2f} \u00d7 {spacing[2]:.2f} mm"
            )
        else:
            self._lbl_spacing.setText("Spacing: N/A")

        self._patient_info_frame.setVisible(True)
        self._patient_sep.setVisible(True)

    def load_recent(self, recent_list: List[Dict]) -> None:
        """Populate the recent projects tree (max 10 entries)."""
        self._recent_tree.clear()
        for entry in recent_list[:10]:
            item = QTreeWidgetItem([
                entry.get("patient_id", ""),
                entry.get("study_date", ""),
                entry.get("stage_summary", ""),
            ])
            item.setData(0, Qt.UserRole, entry.get("session_dir", ""))
            self._recent_tree.addTopLevelItem(item)

    # ------------------------------------------------------------------ #
    #  Slots / event handlers
    # ------------------------------------------------------------------ #

    def _on_import_clicked(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.dicom_imported.emit(dir_path)

    def _on_recent_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        session_dir = item.data(0, Qt.UserRole)
        if session_dir:
            self.session_selected.emit(session_dir)

    def _on_recent_context_menu(self, pos) -> None:
        item = self._recent_tree.itemAt(pos)
        if item is None:
            return
        session_dir = item.data(0, Qt.UserRole)
        if not session_dir:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove from History")
        action = menu.exec(self._recent_tree.viewport().mapToGlobal(pos))
        if action is remove_action:
            idx = self._recent_tree.indexOfTopLevelItem(item)
            self._recent_tree.takeTopLevelItem(idx)
            self.session_removed.emit(session_dir)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime.hasUrls() and mime.urls():
            url = mime.urls()[0]
            path = Path(url.toLocalFile())
            if path.is_dir():
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime.hasUrls() and mime.urls():
            url = mime.urls()[0]
            dir_path = url.toLocalFile()
            if Path(dir_path).is_dir():
                self.dicom_imported.emit(dir_path)
                event.acceptProposedAction()
                return
        event.ignore()
