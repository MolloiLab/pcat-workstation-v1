"""Interactive CPR (Curved Planar Reformation) view widget.

Provides a split-panel CPR viewer replicating the Horos coronary artery
workflow: left panel shows the straightened CPR image with a movable needle
line, right panel shows the perpendicular cross-section at the needle
position with vessel lumen contour and PCAT VOI ring.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
    QKeyEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget


# ---------------------------------------------------------------------------
# Cross-section sampling helpers (lazy-imported for speed)
# ---------------------------------------------------------------------------

def _sample_cross_section(
    volume: np.ndarray,
    vox_size: np.ndarray,
    center: np.ndarray,
    N_vec: np.ndarray,
    B_vec: np.ndarray,
    width_mm: float = 15.0,
    n_cs: int = 128,
) -> np.ndarray:
    """Sample a perpendicular cross-section from the volume."""
    from scipy.ndimage import map_coordinates

    offsets = np.linspace(-width_mm, width_mm, n_cs)
    nn, bb = np.meshgrid(offsets, offsets)
    pts = (
        center[None, None, :]
        + nn[:, :, None] * N_vec[None, None, :]
        + bb[:, :, None] * B_vec[None, None, :]
    )
    pts_vox = pts / vox_size[None, None, :]
    z_v = pts_vox[..., 0].ravel()
    y_v = pts_vox[..., 1].ravel()
    x_v = pts_vox[..., 2].ravel()
    vals = map_coordinates(volume, [z_v, y_v, x_v], order=1, mode="nearest")
    return vals.reshape(n_cs, n_cs)


def _find_lumen_contours(cs_img: np.ndarray) -> list:
    """Find vessel lumen contours from a cross-section image (HU > 150)."""
    from skimage import measure as skimage_measure

    lumen_mask = cs_img > 150.0
    contours = skimage_measure.find_contours(lumen_mask.astype(float), 0.5)
    return [c for c in contours if len(c) >= 20]


# ---------------------------------------------------------------------------
# HU -> 8-bit grayscale
# ---------------------------------------------------------------------------

def _apply_wl(img: np.ndarray, window: float, level: float) -> np.ndarray:
    """Apply window/level to float32 HU array, return uint8 grayscale."""
    low = level - window / 2.0
    high = level + window / 2.0
    scaled = np.clip((img - low) / (high - low), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    gray[np.isnan(img)] = 0
    return gray


def _gray_to_qimage(gray: np.ndarray) -> QImage:
    """Convert uint8 grayscale numpy array to QImage (copies data)."""
    h, w = gray.shape
    # Must copy because numpy may reuse buffer
    gray_c = np.ascontiguousarray(gray)
    qimg = QImage(gray_c.data, w, h, w, QImage.Format_Grayscale8).copy()
    return qimg


# ---------------------------------------------------------------------------
# Vessel data container
# ---------------------------------------------------------------------------

class _VesselData:
    """Holds all data for one vessel."""
    __slots__ = (
        "cpr_image",
        "contour_result",
        "volume",
        "spacing",
        "row_extent_mm",
        "cpr_N_frame",
        "cpr_B_frame",
        "cpr_positions_mm",
        "cpr_arclengths",
    )

    def __init__(self) -> None:
        self.cpr_image: Optional[np.ndarray] = None
        self.contour_result = None  # ContourResult or None
        self.volume: Optional[np.ndarray] = None
        self.spacing: Optional[np.ndarray] = None
        self.row_extent_mm: Optional[float] = None
        self.cpr_N_frame: Optional[np.ndarray] = None
        self.cpr_B_frame: Optional[np.ndarray] = None
        self.cpr_positions_mm: Optional[np.ndarray] = None
        self.cpr_arclengths: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════
# CPR Image Panel (left)
# ═══════════════════════════════════════════════════════════════════════════

class _CPRPanel(QWidget):
    """Renders the CPR image with overlays: needle, centerline, wall contours."""

    needle_index_changed = Signal(int)  # emitted when user clicks/scrolls
    wl_drag = Signal(float, float)  # dx, dy from right-drag

    def __init__(self, parent: "CPRView") -> None:
        super().__init__(parent)
        self._root = parent
        self._pixmap: Optional[QPixmap] = None
        self._needle_idx: int = 0
        self._n_positions: int = 0

        # Right-drag W/L state
        self._right_dragging = False
        self._last_drag_pos: Optional[QPointF] = None

        self.setMinimumWidth(80)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ── painting ─────────────────────────────────────────────────────

    def set_pixmap(self, pm: Optional[QPixmap], n_positions: int) -> None:
        self._pixmap = pm
        self._n_positions = n_positions
        self.update()

    def set_needle(self, idx: int) -> None:
        self._needle_idx = max(0, min(idx, self._n_positions - 1)) if self._n_positions else 0
        self.update()

    @property
    def needle_idx(self) -> int:
        return self._needle_idx

    # ── coordinate helpers ───────────────────────────────────────────

    def _image_rect(self) -> QRectF:
        """Rectangle within widget where the image is drawn (aspect-fit)."""
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        sw, sh = pw * scale, ph * scale
        x0 = (ww - sw) / 2.0
        y0 = (wh - sh) / 2.0
        return QRectF(x0, y0, sw, sh)

    def _y_for_index(self, idx: int) -> float:
        """Widget Y coordinate for a given arc-length index."""
        rect = self._image_rect()
        if self._n_positions <= 1:
            return rect.top() + rect.height() / 2.0
        frac = idx / (self._n_positions - 1)
        return rect.top() + frac * rect.height()

    def _index_for_y(self, y: float) -> int:
        """Arc-length index for a widget Y coordinate."""
        rect = self._image_rect()
        if rect.height() <= 0 or self._n_positions <= 1:
            return 0
        frac = (y - rect.top()) / rect.height()
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * (self._n_positions - 1)))

    # ── paint ────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0f0f0f"))

        if self._pixmap is None or self._pixmap.isNull():
            p.setPen(QColor("#888888"))
            p.setFont(QFont("Helvetica", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "CPR — run pipeline to generate")
            p.end()
            return

        rect = self._image_rect()

        # Draw CPR image
        p.drawPixmap(rect.toRect(), self._pixmap)

        # Vertical dashed centerline
        pen_cl = QPen(QColor(255, 255, 255, 128), 0.8, Qt.DashLine)
        p.setPen(pen_cl)
        cx = rect.left() + rect.width() / 2.0
        p.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))

        # Arc-length tick marks on left edge
        self._draw_arc_ticks(p, rect)

        # Vessel wall boundaries (green solid)
        self._draw_wall_boundaries(p, rect)

        # PCAT boundary (green dashed)
        self._draw_pcat_boundary(p, rect)

        # Needle line (cyan)
        if self._n_positions > 0:
            ny = self._y_for_index(self._needle_idx)
            pen_needle = QPen(QColor("#00ffcc"), 1.6)
            p.setPen(pen_needle)
            p.drawLine(QPointF(rect.left(), ny), QPointF(rect.right(), ny))

        p.end()

    def _draw_arc_ticks(self, p: QPainter, rect: QRectF) -> None:
        """Draw arc-length ticks (0 mm, 10 mm, 20 mm ...) along the left Y-axis."""
        vdata = self._root._current_vdata()
        if vdata is None or vdata.contour_result is None:
            return

        arclengths = vdata.contour_result.arclengths
        if arclengths is None or len(arclengths) == 0:
            return

        max_arc = arclengths[-1]
        p.setPen(QColor("#cccccc"))
        p.setFont(QFont("monospace", 8))

        step_mm = 10.0
        arc_val = 0.0
        while arc_val <= max_arc + 0.01:
            # Find the index closest to this arc-length
            idx = int(np.searchsorted(arclengths, arc_val))
            idx = min(idx, len(arclengths) - 1)
            y = self._y_for_index(idx)
            # Tick
            p.drawLine(QPointF(rect.left() - 4, y), QPointF(rect.left(), y))
            p.drawText(QPointF(rect.left() - 36, y + 4), f"{int(arc_val)}")
            arc_val += step_mm

    def _draw_wall_boundaries(self, p: QPainter, rect: QRectF) -> None:
        """Draw vessel wall boundaries as green lines along the CPR."""
        vdata = self._root._current_vdata()
        if vdata is None or vdata.contour_result is None:
            return

        cr = vdata.contour_result
        r_eq = cr.r_eq
        if r_eq is None or len(r_eq) == 0:
            return

        cpr_img = vdata.cpr_image
        if cpr_img is None:
            return

        n_pos = len(r_eq)
        cpr_w = cpr_img.shape[1]  # pixel width of the CPR image

        pen = QPen(QColor("#00ff00"), 1.2)
        p.setPen(pen)

        # Wall boundaries: center +/- r_eq mapped to pixel column -> widget x
        # half_width_mm is the physical half-width of the CPR lateral axis (row_extent_mm).
        half_width_mm = vdata.row_extent_mm if vdata.row_extent_mm is not None else max(cpr_w * 0.15, 10.0)

        for i in range(n_pos - 1):
            y0 = self._y_for_index(i)
            y1 = self._y_for_index(i + 1)
            for sign in (-1.0, 1.0):
                frac0 = 0.5 + sign * r_eq[i] / (2.0 * half_width_mm)
                frac1 = 0.5 + sign * r_eq[i + 1] / (2.0 * half_width_mm)
                x0 = rect.left() + frac0 * rect.width()
                x1 = rect.left() + frac1 * rect.width()
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    def _draw_pcat_boundary(self, p: QPainter, rect: QRectF) -> None:
        """Draw PCAT boundary (3x r_eq) as green dashed lines."""
        vdata = self._root._current_vdata()
        if vdata is None or vdata.contour_result is None:
            return

        cr = vdata.contour_result
        r_eq = cr.r_eq
        if r_eq is None or len(r_eq) == 0:
            return

        cpr_img = vdata.cpr_image
        if cpr_img is None:
            return

        n_pos = len(r_eq)
        cpr_w = cpr_img.shape[1]
        half_width_mm = vdata.row_extent_mm if vdata.row_extent_mm is not None else max(cpr_w * 0.15, 10.0)

        pen = QPen(QColor("#00cc00"), 1.0, Qt.DashLine)
        p.setPen(pen)

        pcat_r = r_eq * 3.0

        for i in range(n_pos - 1):
            y0 = self._y_for_index(i)
            y1 = self._y_for_index(i + 1)
            for sign in (-1.0, 1.0):
                frac0 = 0.5 + sign * pcat_r[i] / (2.0 * half_width_mm)
                frac1 = 0.5 + sign * pcat_r[i + 1] / (2.0 * half_width_mm)
                x0 = rect.left() + frac0 * rect.width()
                x1 = rect.left() + frac1 * rect.width()
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    # ── mouse / keyboard ─────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton and self._n_positions > 0:
            idx = self._index_for_y(ev.position().y())
            self.needle_index_changed.emit(idx)
        elif ev.button() == Qt.RightButton:
            self._right_dragging = True
            self._last_drag_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._right_dragging and self._last_drag_pos is not None:
            pos = ev.position()
            dx = pos.x() - self._last_drag_pos.x()
            dy = pos.y() - self._last_drag_pos.y()
            self._last_drag_pos = pos
            self.wl_drag.emit(dx, dy)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.RightButton:
            self._right_dragging = False
            self._last_drag_pos = None
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if self._n_positions > 0:
            delta = -1 if ev.angleDelta().y() > 0 else 1
            new_idx = max(0, min(self._needle_idx + delta, self._n_positions - 1))
            self.needle_index_changed.emit(new_idx)
        ev.accept()

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if self._n_positions > 0:
            if ev.key() == Qt.Key_Up:
                new_idx = max(0, self._needle_idx - 1)
                self.needle_index_changed.emit(new_idx)
                ev.accept()
                return
            elif ev.key() == Qt.Key_Down:
                new_idx = min(self._n_positions - 1, self._needle_idx + 1)
                self.needle_index_changed.emit(new_idx)
                ev.accept()
                return
        super().keyPressEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Section Panel (right)
# ═══════════════════════════════════════════════════════════════════════════

class _CrossSectionPanel(QWidget):
    """Renders the perpendicular cross-section at the needle position."""

    wl_drag = Signal(float, float)

    def __init__(self, parent: "CPRView") -> None:
        super().__init__(parent)
        self._root = parent
        self._pixmap: Optional[QPixmap] = None
        self._lumen_contours: list = []  # list of (N,2) arrays in pixel coords
        self._r_eq_mm: float = 0.0
        self._arc_mm: float = 0.0
        self._width_mm: float = 15.0
        self._n_cs: int = 128

        self._right_dragging = False
        self._last_drag_pos: Optional[QPointF] = None

        self.setMinimumWidth(60)
        self.setMouseTracking(True)

    def set_cross_section(
        self,
        cs_img: Optional[np.ndarray],
        window: float,
        level: float,
        r_eq_mm: float,
        arc_mm: float,
    ) -> None:
        self._r_eq_mm = r_eq_mm
        self._arc_mm = arc_mm
        if cs_img is not None:
            gray = _apply_wl(cs_img, window, level)
            qimg = _gray_to_qimage(gray)
            self._pixmap = QPixmap.fromImage(qimg)
            self._lumen_contours = _find_lumen_contours(cs_img)
            self._n_cs = cs_img.shape[0]
        else:
            self._pixmap = None
            self._lumen_contours = []
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._lumen_contours = []
        self.update()

    def _image_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
        sw, sh = pw * scale, ph * scale
        x0 = (ww - sw) / 2.0
        y0 = (wh - sh) / 2.0
        return QRectF(x0, y0, sw, sh)

    def _pix_to_widget(self, px: float, py: float) -> QPointF:
        """Convert pixel coords in the cross-section image to widget coords."""
        rect = self._image_rect()
        sx = rect.width() / self._n_cs if self._n_cs > 0 else 1.0
        sy = rect.height() / self._n_cs if self._n_cs > 0 else 1.0
        return QPointF(rect.left() + px * sx, rect.top() + py * sy)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0f0f0f"))

        if self._pixmap is None or self._pixmap.isNull():
            p.setPen(QColor("#888888"))
            p.setFont(QFont("Helvetica", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Cross-section")
            p.end()
            return

        rect = self._image_rect()
        p.drawPixmap(rect.toRect(), self._pixmap)

        # Crosshair at center
        cx = rect.left() + rect.width() / 2.0
        cy = rect.top() + rect.height() / 2.0
        pen_ch = QPen(QColor(255, 255, 255, 102), 0.6)  # alpha=0.4
        p.setPen(pen_ch)
        p.drawLine(QPointF(rect.left(), cy), QPointF(rect.right(), cy))
        p.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))

        # Lumen contours (white, lw=2.0)
        if self._lumen_contours:
            pen_lumen = QPen(QColor("#ffffff"), 2.0)
            p.setPen(pen_lumen)
            for contour in self._lumen_contours:
                for j in range(len(contour) - 1):
                    pt0 = self._pix_to_widget(contour[j, 1], contour[j, 0])
                    pt1 = self._pix_to_widget(contour[j + 1, 1], contour[j + 1, 0])
                    p.drawLine(pt0, pt1)
                # Close the contour
                if len(contour) > 2:
                    pt0 = self._pix_to_widget(contour[-1, 1], contour[-1, 0])
                    pt1 = self._pix_to_widget(contour[0, 1], contour[0, 0])
                    p.drawLine(pt0, pt1)

        # VOI ring (yellow dashed, radius = 3 * r_eq)
        if self._r_eq_mm > 0 and self._width_mm > 0:
            pen_voi = QPen(QColor("#ffee00"), 1.5, Qt.DashLine)
            p.setPen(pen_voi)
            p.setBrush(Qt.NoBrush)
            voi_r_mm = 3.0 * self._r_eq_mm
            # Convert mm radius to widget pixels
            pix_per_mm = self._n_cs / (2.0 * self._width_mm)
            voi_r_pix = voi_r_mm * pix_per_mm
            scale = rect.width() / self._n_cs if self._n_cs > 0 else 1.0
            voi_r_widget = voi_r_pix * scale
            p.drawEllipse(QPointF(cx, cy), voi_r_widget, voi_r_widget)

        # Annotation text
        p.setPen(QColor("#e0e0e0"))
        p.setFont(QFont("monospace", 9))
        text_x = rect.left() + 4
        text_y = rect.bottom() - 6
        p.drawText(QPointF(text_x, text_y - 14), f"arc: {self._arc_mm:.1f} mm")
        p.drawText(QPointF(text_x, text_y), f"r_eq: {self._r_eq_mm:.2f} mm")

        p.end()

    # ── mouse (right-drag W/L) ───────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.RightButton:
            self._right_dragging = True
            self._last_drag_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._right_dragging and self._last_drag_pos is not None:
            pos = ev.position()
            dx = pos.x() - self._last_drag_pos.x()
            dy = pos.y() - self._last_drag_pos.y()
            self._last_drag_pos = pos
            self.wl_drag.emit(dx, dy)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.RightButton:
            self._right_dragging = False
            self._last_drag_pos = None
        super().mouseReleaseEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════
# Main CPRView widget
# ═══════════════════════════════════════════════════════════════════════════

class CPRView(QWidget):
    """Interactive CPR viewer with needle navigation and cross-section display.

    Drop-in replacement for the previous static CPRView. All original API
    methods (set_cpr_data, set_vessel, set_window_level, clear) are preserved.
    """

    # Signals
    needle_moved = Signal(float, float, float)  # x_mm, y_mm, z_mm
    vessel_changed = Signal(str)
    window_level_changed = Signal(float, float)

    # Cross-section parameters
    _CS_WIDTH_MM: float = 15.0
    _CS_RESOLUTION: int = 128

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_vessel: str = "LAD"
        self._vessels: Dict[str, _VesselData] = {}
        self._window: float = 800.0
        self._level: float = 200.0
        self._needle_idx: int = 0

        # Cross-section cache: (vessel, idx) -> np.ndarray
        self._cs_cache: Dict[Tuple[str, int], np.ndarray] = {}

        self._build_ui()
        self._connect_signals()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "CPRView { background-color: #0f0f0f; border: 1px solid #2a2a2a; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setStyleSheet(
            "QSplitter { background-color: #0f0f0f; }"
            "QSplitter::handle { background-color: #2a2a2a; width: 2px; }"
        )

        self._cpr_panel = _CPRPanel(self)
        self._cs_panel = _CrossSectionPanel(self)

        self._splitter.addWidget(self._cpr_panel)
        self._splitter.addWidget(self._cs_panel)
        self._splitter.setStretchFactor(0, 7)
        self._splitter.setStretchFactor(1, 3)

        layout.addWidget(self._splitter)

    def _connect_signals(self) -> None:
        self._cpr_panel.needle_index_changed.connect(self._on_needle_moved)
        self._cpr_panel.wl_drag.connect(self._on_wl_drag)
        self._cs_panel.wl_drag.connect(self._on_wl_drag)

    # ── Internal data access ─────────────────────────────────────────

    def _current_vdata(self) -> Optional[_VesselData]:
        return self._vessels.get(self._current_vessel)

    def _get_or_create_vdata(self, vessel: str) -> _VesselData:
        if vessel not in self._vessels:
            self._vessels[vessel] = _VesselData()
        return self._vessels[vessel]

    # ── Public API (original, preserved) ─────────────────────────────

    def set_cpr_data(self, vessel: str, cpr_image: np.ndarray, row_extent_mm: float = 25.0) -> None:
        """Store a CPR image (float32 HU, rows x cols) for a vessel."""
        vd = self._get_or_create_vdata(vessel)
        vd.cpr_image = cpr_image
        vd.row_extent_mm = row_extent_mm
        # Invalidate cross-section cache for this vessel
        self._cs_cache = {k: v for k, v in self._cs_cache.items() if k[0] != vessel}
        if vessel == self._current_vessel:
            self._refresh_cpr()
            self._refresh_cs()

    def set_cpr_frame(self, vessel: str, frame_data: dict) -> None:
        """Store the Bishop frame used to generate the CPR image.

        This ensures cross-section sampling uses the same frame orientation
        as the CPR image, avoiding rotational mismatch with the contour
        extraction's independently computed Bishop frame.
        """
        vd = self._get_or_create_vdata(vessel)
        vd.cpr_N_frame = frame_data["N_frame"]
        vd.cpr_B_frame = frame_data["B_frame"]
        vd.cpr_positions_mm = frame_data["positions_mm"]
        vd.cpr_arclengths = frame_data["arclengths"]
        # Invalidate cross-section cache for this vessel
        self._cs_cache = {k: v for k, v in self._cs_cache.items() if k[0] != vessel}

    def set_vessel(self, vessel: str) -> None:
        """Switch which vessel is displayed."""
        if vessel == self._current_vessel:
            return
        self._current_vessel = vessel
        self._needle_idx = 0
        self._refresh_cpr()
        self._refresh_cs()
        self.vessel_changed.emit(vessel)

    def set_window_level(self, window: float, level: float) -> None:
        """Update window/level and re-render both panels."""
        self._window = max(1.0, window)
        self._level = level
        self._refresh_cpr()
        self._refresh_cs()

    def clear(self) -> None:
        """Clear all vessel data and reset display."""
        self._vessels.clear()
        self._cs_cache.clear()
        self._needle_idx = 0
        self._cpr_panel.set_pixmap(None, 0)
        self._cs_panel.clear()

    # ── Public API (new) ─────────────────────────────────────────────

    def set_contour_data(
        self,
        vessel: str,
        contour_result,
        volume: np.ndarray,
        spacing,
    ) -> None:
        """Store ContourResult and volume reference for cross-section sampling.

        Parameters
        ----------
        vessel : str
            Vessel name (e.g. "LAD", "LCx", "RCA").
        contour_result : ContourResult
            Has: positions_mm (N,3), N_frame (N,3), B_frame (N,3),
            contours, r_eq (N,), arclengths (N,).
        volume : np.ndarray
            3D CT volume (float32 HU).
        spacing : array-like
            Voxel spacing [z, y, x] in mm.
        """
        vd = self._get_or_create_vdata(vessel)
        vd.contour_result = contour_result
        vd.volume = volume
        vd.spacing = np.asarray(spacing, dtype=np.float64)
        # Invalidate cache
        self._cs_cache = {k: v for k, v in self._cs_cache.items() if k[0] != vessel}
        if vessel == self._current_vessel:
            self._refresh_cpr()
            self._refresh_cs()

    # ── Needle movement ──────────────────────────────────────────────

    def _map_needle_to_frame_idx(self, needle_idx: int, vd: _VesselData) -> int:
        """Map a needle index (0..n_positions-1) to a CPR frame index.

        The CPR frame has ``pixels_wide`` entries (e.g. 512) while the
        displayed image may have a different row count (e.g. 256).  We
        linearly interpolate to find the matching frame index.
        """
        n_pos = self._n_positions()
        n_frame = len(vd.cpr_N_frame)
        if n_pos <= 1 or n_frame <= 1:
            return min(needle_idx, n_frame - 1)
        frac = needle_idx / (n_pos - 1)
        return int(round(frac * (n_frame - 1)))

    def _on_needle_moved(self, idx: int) -> None:
        vd = self._current_vdata()
        if vd is None:
            return

        n_pos = self._n_positions()
        idx = max(0, min(idx, n_pos - 1))
        if idx == self._needle_idx:
            return

        self._needle_idx = idx
        self._cpr_panel.set_needle(idx)

        # Emit patient coordinates in VTK order (x_mm, y_mm, z_mm)
        # positions_mm is in numpy order [z, y, x] so swap to VTK [x, y, z]
        if vd.cpr_positions_mm is not None:
            fi = self._map_needle_to_frame_idx(idx, vd)
            pos = vd.cpr_positions_mm[fi]
            self.needle_moved.emit(float(pos[2]), float(pos[1]), float(pos[0]))
        elif vd.contour_result is not None and idx < len(vd.contour_result.positions_mm):
            pos = vd.contour_result.positions_mm[idx]
            self.needle_moved.emit(float(pos[2]), float(pos[1]), float(pos[0]))

        # Debounced cross-section update
        QTimer.singleShot(30, self._refresh_cs)

    def _n_positions(self) -> int:
        """Number of arc-length positions for current vessel."""
        vd = self._current_vdata()
        if vd is None:
            return 0
        if vd.cpr_image is not None:
            return vd.cpr_image.shape[0]
        if vd.contour_result is not None:
            return len(vd.contour_result.positions_mm)
        return 0

    # ── W/L drag ─────────────────────────────────────────────────────

    def _on_wl_drag(self, dx: float, dy: float) -> None:
        self._window = max(1.0, self._window + dx * 2.0)
        self._level = self._level - dy * 2.0
        self._refresh_cpr()
        self._refresh_cs()
        self.window_level_changed.emit(self._window, self._level)

    # ── Refresh rendering ────────────────────────────────────────────

    def _refresh_cpr(self) -> None:
        vd = self._current_vdata()
        if vd is None or vd.cpr_image is None:
            self._cpr_panel.set_pixmap(None, 0)
            return

        img = vd.cpr_image
        gray = _apply_wl(img, self._window, self._level)
        qimg = _gray_to_qimage(gray)
        pm = QPixmap.fromImage(qimg)

        n_pos = img.shape[0]
        self._cpr_panel.set_pixmap(pm, n_pos)
        self._cpr_panel.set_needle(self._needle_idx)

    def _refresh_cs(self) -> None:
        """Recompute and display the cross-section at current needle index."""
        vd = self._current_vdata()
        if vd is None:
            self._cs_panel.clear()
            return

        idx = self._needle_idx
        cr = vd.contour_result
        r_eq_mm = 0.0
        arc_mm = 0.0

        # Prefer CPR frame arclengths (matches image orientation)
        if vd.cpr_arclengths is not None:
            fi = self._map_needle_to_frame_idx(idx, vd)
            arc_mm = float(vd.cpr_arclengths[fi])
        elif cr is not None and cr.arclengths is not None and idx < len(cr.arclengths):
            arc_mm = float(cr.arclengths[idx])

        # r_eq always comes from contour extraction (correct source)
        if cr is not None and cr.r_eq is not None and idx < len(cr.r_eq):
            r_eq_mm = float(cr.r_eq[idx])

        # Try to get cross-section from cache or compute
        cs_img = self._get_cross_section(idx)
        self._cs_panel.set_cross_section(cs_img, self._window, self._level, r_eq_mm, arc_mm)

    def _get_cross_section(self, idx: int) -> Optional[np.ndarray]:
        """Get cross-section image at given index, with caching."""
        vd = self._current_vdata()
        if vd is None or vd.volume is None or vd.spacing is None:
            return None

        # Need either CPR frame or contour extraction frame
        has_cpr_frame = vd.cpr_N_frame is not None
        has_contour = vd.contour_result is not None
        if not has_cpr_frame and not has_contour:
            return None

        cache_key = (self._current_vessel, idx)
        if cache_key in self._cs_cache:
            return self._cs_cache[cache_key]

        # Prefer CPR frame (matches CPR image orientation)
        if has_cpr_frame:
            fi = self._map_needle_to_frame_idx(idx, vd)
            N_vec = vd.cpr_N_frame[fi]
            B_vec = vd.cpr_B_frame[fi]
            center = vd.cpr_positions_mm[fi]
        else:
            cr = vd.contour_result
            if idx >= len(cr.positions_mm):
                return None
            N_vec = cr.N_frame[idx]
            B_vec = cr.B_frame[idx]
            center = cr.positions_mm[idx]

        cs_img = _sample_cross_section(
            volume=vd.volume,
            vox_size=vd.spacing,
            center=center,
            N_vec=N_vec,
            B_vec=B_vec,
            width_mm=self._CS_WIDTH_MM,
            n_cs=self._CS_RESOLUTION,
        )

        # Keep cache bounded
        if len(self._cs_cache) > 500:
            # Remove oldest entries (arbitrary, just trim)
            keys = list(self._cs_cache.keys())
            for k in keys[:250]:
                del self._cs_cache[k]

        self._cs_cache[cache_key] = cs_img
        return cs_img

    # ── Qt overrides ─────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_cpr()

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        # Forward arrow keys to CPR panel
        if ev.key() in (Qt.Key_Up, Qt.Key_Down):
            self._cpr_panel.keyPressEvent(ev)
            return
        super().keyPressEvent(ev)
