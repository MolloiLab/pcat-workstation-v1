"""Straightened CPR (Curved Planar Reformation) view widget."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from typing import Optional


class CPRView(QWidget):
    """Displays a straightened CPR image for a selected vessel.

    The CPR is a 2D image where:
    - Columns = arc-length along the vessel (ostium at left)
    - Rows = cross-section width perpendicular to the vessel
    """

    vessel_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_vessel: str = "LAD"
        self._cpr_images: dict = {}  # vessel -> (n_rows, n_cols) float32 HU
        self._window: float = 800.0
        self._level: float = 200.0
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "CPRView { background-color: #0f0f0f; border: 1px solid #2a2a2a; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QLabel("CPR — run pipeline to generate")
        self._header.setStyleSheet(
            "QLabel { color: #e0e0e0; background-color: transparent; "
            "padding: 4px 8px; font-size: 13pt; font-weight: bold; }"
        )
        self._header.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self._header)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: #0f0f0f;")
        layout.addWidget(self._image_label, stretch=1)

    # ── Public API ───────────────────────────────────────────────

    def set_cpr_data(self, vessel: str, cpr_image: np.ndarray) -> None:
        """Store a CPR image (float32 HU) for a vessel."""
        self._cpr_images[vessel] = cpr_image
        if vessel == self._current_vessel:
            self._refresh()

    def set_vessel(self, vessel: str) -> None:
        """Switch which vessel's CPR is displayed."""
        self._current_vessel = vessel
        self._refresh()

    def set_window_level(self, window: float, level: float) -> None:
        """Update W/L and re-render."""
        self._window = max(1.0, window)
        self._level = level
        self._refresh()

    def clear(self) -> None:
        """Clear all CPR data."""
        self._cpr_images.clear()
        self._image_label.setPixmap(QPixmap())
        self._header.setText("CPR — run pipeline to generate")

    # ── Internal ─────────────────────────────────────────────────

    def _refresh(self) -> None:
        img = self._cpr_images.get(self._current_vessel)
        if img is None:
            self._image_label.setPixmap(QPixmap())
            self._header.setText(f"CPR: {self._current_vessel} — not available")
            return

        self._header.setText(
            f"CPR: {self._current_vessel}  "
            f"({img.shape[1]} samples, {img.shape[0]}px wide)"
        )

        # Apply window/level to convert HU -> 8-bit grayscale
        low = self._level - self._window / 2
        high = self._level + self._window / 2
        scaled = np.clip((img - low) / (high - low), 0, 1)
        gray = (scaled * 255).astype(np.uint8)

        # Replace NaN with black
        gray[np.isnan(img)] = 0

        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit the label while keeping aspect ratio
        label_size = self._image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            pixmap = pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self._image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()
