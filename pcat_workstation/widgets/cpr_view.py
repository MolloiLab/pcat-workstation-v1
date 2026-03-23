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
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


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
        "_cpr_N_frame_orig",
        "_cpr_B_frame_orig",
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
        self._cpr_N_frame_orig: Optional[np.ndarray] = None
        self._cpr_B_frame_orig: Optional[np.ndarray] = None


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

        # A/B/C needle lines: B is the main needle, A and C are offset by interval
        self._needle_interval: int = 20  # index spacing between A/B and B/C

        # Line dragging state
        self._dragging_line: Optional[str] = None  # "A", "B", or "C"

        # Right-drag W/L state
        self._right_dragging = False
        self._last_drag_pos: Optional[QPointF] = None

        # Pan / Zoom / W-L tool state
        self._pan_offset = QPointF(0, 0)
        self._zoom_factor = 1.0
        self._pan_dragging = False
        self._pan_start: Optional[QPointF] = None
        self._zoom_dragging = False
        self._zoom_start: Optional[QPointF] = None
        self._wl_dragging = False
        self._last_wl_pos: Optional[QPointF] = None

        # Measurement state
        self._measuring = False
        self._measure_start: Optional[QPointF] = None
        self._measure_end: Optional[QPointF] = None
        self._measurements: list = []  # list of (start_pt, end_pt, distance_mm) tuples

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
        """Rectangle within widget where the image is drawn.

        In **straightened** mode the image is stretched to fill the panel
        (independent X/Y scaling).  In **stretched** mode the physical
        aspect ratio is preserved so that 1 mm on the arc-length axis
        equals 1 mm on the lateral axis, enabling accurate distance
        measurement in any direction.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()

        if self._root._stretched_mode:
            # Preserve physical aspect ratio (aspect-fit)
            vdata = self._root._current_vdata()
            if (
                vdata is not None
                and vdata.cpr_arclengths is not None
                and len(vdata.cpr_arclengths) > 1
            ):
                arc_total_mm = float(vdata.cpr_arclengths[-1])
                lateral_mm = 2.0 * (vdata.row_extent_mm or 25.0)
                # Physical aspect: height / width in mm
                physical_aspect = arc_total_mm / lateral_mm
                # Fit within widget preserving physical aspect
                if physical_aspect > (wh / ww):
                    # Height-limited
                    sh = wh
                    sw = wh / physical_aspect
                else:
                    # Width-limited
                    sw = ww
                    sh = ww * physical_aspect
            else:
                # Fallback: pixel aspect-fit
                scale = min(ww / pw, wh / ph) if pw > 0 and ph > 0 else 1.0
                sw, sh = pw * scale, ph * scale
        else:
            # Straightened: stretch to fill panel (independent scaling)
            sw, sh = ww, wh

        # Apply zoom
        sw *= self._zoom_factor
        sh *= self._zoom_factor
        x0 = (ww - sw) / 2.0 + self._pan_offset.x()
        y0 = (wh - sh) / 2.0 + self._pan_offset.y()
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

        # Needle lines: A (yellow), B (cyan, main), C (yellow)
        if self._n_positions > 0:
            idx_a = max(0, self._needle_idx - self._needle_interval)
            idx_b = self._needle_idx
            idx_c = min(self._n_positions - 1, self._needle_idx + self._needle_interval)

            for idx, color, label in [
                (idx_a, QColor("#ffee00"), "A"),
                (idx_b, QColor("#00ffcc"), "B"),
                (idx_c, QColor("#ffee00"), "C"),
            ]:
                ny = self._y_for_index(idx)
                pen = QPen(color, 1.6)
                p.setPen(pen)
                p.drawLine(QPointF(rect.left(), ny), QPointF(rect.right(), ny))
                # Label at right edge
                p.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
                p.drawText(QPointF(rect.right() - 14, ny - 3), label)

        # ── Measurements overlay ─────────────────────────────────────
        pen_measure = QPen(QColor("#ff9f0a"), 2.0)
        p.setPen(pen_measure)
        p.setFont(QFont("Helvetica", 9, QFont.Bold))
        for start, end, dist_mm in self._measurements:
            p.setPen(QPen(QColor("#ff9f0a"), 2.0))
            p.drawLine(start, end)
            p.drawEllipse(start, 3, 3)
            p.drawEllipse(end, 3, 3)
            mid = (start + end) / 2.0
            p.drawText(mid + QPointF(5, -5), f"{dist_mm:.1f} mm")

        # Current measurement in progress
        if self._measuring and self._measure_start and self._measure_end:
            p.setPen(QPen(QColor("#ff9f0a"), 2.0, Qt.DashLine))
            p.drawLine(self._measure_start, self._measure_end)

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
        tool = self._root._current_tool
        if ev.button() == Qt.LeftButton:
            if tool == "Navigate" and self._n_positions > 0:
                y = ev.position().y()
                # Check proximity to each line (within 8 pixels)
                idx_a = max(0, self._needle_idx - self._needle_interval)
                idx_b = self._needle_idx
                idx_c = min(self._n_positions - 1, self._needle_idx + self._needle_interval)
                lines = {
                    "A": self._y_for_index(idx_a),
                    "B": self._y_for_index(idx_b),
                    "C": self._y_for_index(idx_c),
                }
                closest = min(lines.items(), key=lambda kv: abs(kv[1] - y))
                if abs(closest[1] - y) < 8:
                    self._dragging_line = closest[0]
                else:
                    # Click not near any line - move B to click position
                    self._dragging_line = None
                    idx = self._index_for_y(y)
                    self.needle_index_changed.emit(idx)
            elif tool == "W/L":
                self._wl_dragging = True
                self._last_wl_pos = ev.position()
            elif tool == "Pan":
                self._pan_start = ev.position()
                self._pan_dragging = True
            elif tool == "Zoom":
                self._zoom_start = ev.position()
                self._zoom_dragging = True
            elif tool == "Measure":
                self._measure_start = ev.position()
                self._measure_end = ev.position()
                self._measuring = True
        elif ev.button() == Qt.RightButton:
            # Right-click always = W/L (unchanged)
            self._right_dragging = True
            self._last_drag_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        tool = self._root._current_tool
        if self._right_dragging and self._last_drag_pos is not None:
            # Right-drag W/L (always active regardless of tool)
            pos = ev.position()
            dx = pos.x() - self._last_drag_pos.x()
            dy = pos.y() - self._last_drag_pos.y()
            self._last_drag_pos = pos
            self.wl_drag.emit(dx, dy)
        elif tool == "Navigate" and self._dragging_line and ev.buttons() & Qt.LeftButton:
            idx = self._index_for_y(ev.position().y())
            if self._dragging_line == "B":
                self.needle_index_changed.emit(idx)
            elif self._dragging_line == "A":
                new_interval = max(1, self._needle_idx - idx)
                self._needle_interval = new_interval
                self.update()
                self._root._refresh_all_cs()
            elif self._dragging_line == "C":
                new_interval = max(1, idx - self._needle_idx)
                self._needle_interval = new_interval
                self.update()
                self._root._refresh_all_cs()
        elif tool == "W/L" and self._wl_dragging and self._last_wl_pos is not None:
            pos = ev.position()
            dx = pos.x() - self._last_wl_pos.x()
            dy = pos.y() - self._last_wl_pos.y()
            self._last_wl_pos = pos
            self.wl_drag.emit(dx, dy)
        elif tool == "Pan" and self._pan_dragging and self._pan_start is not None:
            delta = ev.position() - self._pan_start
            self._pan_offset += delta
            self._pan_start = ev.position()
            self.update()
        elif tool == "Zoom" and self._zoom_dragging and self._zoom_start is not None:
            dy = ev.position().y() - self._zoom_start.y()
            self._zoom_factor = max(0.2, min(5.0, self._zoom_factor * (1.0 + dy * 0.005)))
            self._zoom_start = ev.position()
            self.update()
        elif tool == "Measure" and self._measuring:
            self._measure_end = ev.position()
            self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        tool = self._root._current_tool
        if ev.button() == Qt.LeftButton:
            if tool == "Measure" and self._measuring:
                self._measuring = False
                if self._measure_start and self._measure_end:
                    dist_mm = self._compute_measurement_mm(
                        self._measure_start, self._measure_end
                    )
                    if dist_mm > 0.5:  # ignore tiny accidental clicks
                        self._measurements.append((
                            QPointF(self._measure_start),
                            QPointF(self._measure_end),
                            dist_mm,
                        ))
                self._measure_start = None
                self._measure_end = None
                self.update()
            self._dragging_line = None
            self._wl_dragging = False
            self._last_wl_pos = None
            self._pan_dragging = False
            self._pan_start = None
            self._zoom_dragging = False
            self._zoom_start = None
        elif ev.button() == Qt.RightButton:
            self._right_dragging = False
            self._last_drag_pos = None
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if ev.modifiers() & Qt.ControlModifier:
            # Ctrl+scroll: change interval between A, B, C lines
            delta = 2 if ev.angleDelta().y() > 0 else -2
            self._needle_interval = max(1, self._needle_interval + delta)
            self.update()
            self._root._refresh_all_cs()
            ev.accept()
            return
        if ev.modifiers() & Qt.ShiftModifier:
            # Shift+scroll: rotate the CPR cutting plane
            current = self._root._rotation_slider.value()
            delta = 5 if ev.angleDelta().y() > 0 else -5
            self._root._rotation_slider.setValue((current + delta) % 361)
            ev.accept()
            return
        if self._n_positions > 0:
            delta = -1 if ev.angleDelta().y() > 0 else 1
            new_idx = max(0, min(self._needle_idx + delta, self._n_positions - 1))
            self.needle_index_changed.emit(new_idx)
        ev.accept()

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key_Delete and self._measurements:
            self._measurements.pop()
            self.update()
            ev.accept()
            return
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

    def _compute_measurement_mm(self, p1: QPointF, p2: QPointF) -> float:
        """Convert pixel distance to mm using CPR's physical scale."""
        rect = self._image_rect()
        vdata = self._root._current_vdata()
        if vdata is None or vdata.cpr_image is None:
            return 0.0

        n_arc, n_lateral = vdata.cpr_image.shape
        row_extent_mm = vdata.row_extent_mm or 25.0

        # Pixels to image coordinates
        img_x1 = (p1.x() - rect.left()) / rect.width() * n_lateral
        img_y1 = (p1.y() - rect.top()) / rect.height() * n_arc
        img_x2 = (p2.x() - rect.left()) / rect.width() * n_lateral
        img_y2 = (p2.y() - rect.top()) / rect.height() * n_arc

        # Image coords to mm
        # Lateral: n_lateral pixels span 2 * row_extent_mm
        mm_per_pixel_lateral = 2.0 * row_extent_mm / n_lateral
        # Arc-length: need total arc-length
        if vdata.cpr_arclengths is not None and len(vdata.cpr_arclengths) > 0:
            arc_total_mm = float(vdata.cpr_arclengths[-1])
        else:
            arc_total_mm = n_arc * 0.1  # fallback estimate
        mm_per_pixel_arc = arc_total_mm / n_arc

        dx_mm = (img_x2 - img_x1) * mm_per_pixel_lateral
        dy_mm = (img_y2 - img_y1) * mm_per_pixel_arc

        return float(np.sqrt(dx_mm**2 + dy_mm**2))

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            self._root.fullscreen_requested.emit(self._root)
        ev.accept()


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Section Panel (right)
# ═══════════════════════════════════════════════════════════════════════════

class _CrossSectionPanel(QWidget):
    """Renders the perpendicular cross-section at the needle position."""

    wl_drag = Signal(float, float)

    def __init__(self, parent: "CPRView", label: str = "B") -> None:
        super().__init__(parent)
        self._root = parent
        self._label: str = label
        self._pixmap: Optional[QPixmap] = None
        self._lumen_contours: list = []  # list of (N,2) arrays in pixel coords
        self._r_eq_mm: float = 0.0
        self._arc_mm: float = 0.0
        self._dist_from_b_mm: float = 0.0  # signed distance from B in mm
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
        dist_from_b_mm: float = 0.0,
    ) -> None:
        self._r_eq_mm = r_eq_mm
        self._arc_mm = arc_mm
        self._dist_from_b_mm = dist_from_b_mm
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

        # Label in top-left corner (A, B, or C)
        label_color = QColor("#00ffcc") if self._label == "B" else QColor("#ffee00")
        p.setPen(label_color)
        p.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        p.drawText(QPointF(rect.left() + 6, rect.top() + 18), self._label)

        # Annotation text
        p.setPen(QColor("#e0e0e0"))
        p.setFont(QFont("monospace", 9))
        text_x = rect.left() + 4
        text_y = rect.bottom() - 6
        lines = []
        lines.append(f"arc: {self._arc_mm:.1f} mm")
        lines.append(f"r_eq: {self._r_eq_mm:.2f} mm")
        if self._label != "B" and abs(self._dist_from_b_mm) > 0.01:
            sign = "+" if self._dist_from_b_mm > 0 else ""
            lines.append(f"{sign}{self._dist_from_b_mm:.1f} mm from B")
        for i, line in enumerate(reversed(lines)):
            p.drawText(QPointF(text_x, text_y - i * 14), line)

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

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            self._root.fullscreen_requested.emit(self._root)
        ev.accept()


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
    fullscreen_requested = Signal(object)  # emits self

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

        # Rotation angle toolbar
        self._cpr_toolbar = QWidget()
        self._cpr_toolbar.setStyleSheet("background-color: #1a1a1a;")
        toolbar_layout = QHBoxLayout(self._cpr_toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(8)

        angle_label = QLabel("Angle:")
        angle_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        toolbar_layout.addWidget(angle_label)

        self._rotation_slider = QSlider(Qt.Horizontal)
        self._rotation_slider.setRange(0, 360)
        self._rotation_slider.setValue(0)
        self._rotation_slider.setTickInterval(45)
        self._rotation_slider.setTickPosition(QSlider.TicksBelow)
        self._rotation_slider.valueChanged.connect(self._on_rotation_changed)
        toolbar_layout.addWidget(self._rotation_slider)

        self._rotation_label = QLabel("0\u00b0")
        self._rotation_label.setFixedWidth(35)
        self._rotation_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        toolbar_layout.addWidget(self._rotation_label)

        # Straightened / Stretched toggle
        self._stretched_mode = False
        self._straightened_btn = QRadioButton("Straightened")
        self._straightened_btn.setChecked(True)
        self._straightened_btn.setStyleSheet("color: #e5e5e7; font-size: 10pt;")
        self._stretched_btn = QRadioButton("Stretched")
        self._stretched_btn.setStyleSheet("color: #e5e5e7; font-size: 10pt;")
        self._cpr_mode_group = QButtonGroup(self)
        self._cpr_mode_group.addButton(self._straightened_btn, 0)
        self._cpr_mode_group.addButton(self._stretched_btn, 1)
        self._cpr_mode_group.idToggled.connect(self._on_cpr_mode_changed)
        toolbar_layout.addWidget(self._straightened_btn)
        toolbar_layout.addWidget(self._stretched_btn)

        # Mouse tool selector
        self._current_tool = "Navigate"
        toolbar_layout.addWidget(QLabel("Tool:"))
        self._tool_combo = QComboBox()
        self._tool_combo.addItems(["Navigate", "W/L", "Pan", "Zoom", "Measure"])
        self._tool_combo.setCurrentIndex(0)
        self._tool_combo.setFixedWidth(100)
        self._tool_combo.setStyleSheet(
            "QComboBox { background: #2c2c2e; color: #e5e5e7; border: 1px solid #38383a; "
            "border-radius: 3px; padding: 2px 8px; font-size: 10pt; }"
        )
        self._tool_combo.currentTextChanged.connect(
            lambda t: setattr(self, '_current_tool', t)
        )
        toolbar_layout.addWidget(self._tool_combo)

        layout.addWidget(self._cpr_toolbar)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setStyleSheet(
            "QSplitter { background-color: #0f0f0f; }"
            "QSplitter::handle { background-color: #2a2a2a; width: 2px; }"
        )

        self._cpr_panel = _CPRPanel(self)

        # Right side: 3 stacked cross-sections (A, B, C)
        self._cs_container = QWidget()
        cs_layout = QVBoxLayout(self._cs_container)
        cs_layout.setContentsMargins(0, 0, 0, 0)
        cs_layout.setSpacing(2)

        self._cs_panels: list[_CrossSectionPanel] = []
        for label in ("A", "B", "C"):
            panel = _CrossSectionPanel(self, label=label)
            self._cs_panels.append(panel)
            cs_layout.addWidget(panel)

        self._splitter.addWidget(self._cpr_panel)
        self._splitter.addWidget(self._cs_container)
        self._splitter.setStretchFactor(0, 7)
        self._splitter.setStretchFactor(1, 3)

        layout.addWidget(self._splitter)

    def _connect_signals(self) -> None:
        self._cpr_panel.needle_index_changed.connect(self._on_needle_moved)
        self._cpr_panel.wl_drag.connect(self._on_wl_drag)
        for panel in self._cs_panels:
            panel.wl_drag.connect(self._on_wl_drag)

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
            self._cpr_panel._measurements = []
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
        # Store the original (unrotated) Bishop frame for interactive rotation
        vd._cpr_N_frame_orig = frame_data["N_frame"].copy()
        vd._cpr_B_frame_orig = frame_data["B_frame"].copy()
        # Invalidate cross-section cache for this vessel
        self._cs_cache = {k: v for k, v in self._cs_cache.items() if k[0] != vessel}

    def set_vessel(self, vessel: str) -> None:
        """Switch which vessel is displayed."""
        if vessel == self._current_vessel:
            return
        self._current_vessel = vessel
        self._needle_idx = 0
        # Reset rotation slider when switching vessels
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(0)
        self._rotation_label.setText("0\u00b0")
        self._rotation_slider.blockSignals(False)
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
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(0)
        self._rotation_label.setText("0\u00b0")
        self._rotation_slider.blockSignals(False)
        self._cpr_panel.set_pixmap(None, 0)
        for panel in self._cs_panels:
            panel.clear()

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

    # ── Rotation ─────────────────────────────────────────────────────

    def _on_rotation_changed(self, angle: int) -> None:
        """Regenerate CPR image at the new rotation angle using fast trilinear."""
        self._rotation_label.setText(f"{angle}\u00b0")
        vd = self._current_vdata()
        if vd is None or vd.volume is None or vd.spacing is None:
            return
        if vd.cpr_positions_mm is None or vd._cpr_N_frame_orig is None:
            return
        if vd.cpr_image is None:
            return

        # Rotate the ORIGINAL (unrotated) Bishop frame by the new angle
        theta = np.deg2rad(float(angle))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        N_orig = vd._cpr_N_frame_orig
        B_orig = vd._cpr_B_frame_orig
        N_rot = cos_t * N_orig + sin_t * B_orig
        B_rot = -sin_t * N_orig + cos_t * B_orig

        # Rebuild CPR image using fast trilinear method
        from pipeline.visualize import _build_cpr_image_fast

        n_lateral = vd.cpr_image.shape[1]  # lateral pixel count
        cpr_img_raw = _build_cpr_image_fast(
            vd.volume, vd.spacing,
            vd.cpr_positions_mm, N_rot, B_rot,
            n_rows=n_lateral,
            row_extent_mm=vd.row_extent_mm or 25.0,
            slab_mm=0.0,  # thin slab for interactive speed
        )
        # _build_cpr_image_fast returns (n_lateral, n_arc); transpose to (n_arc, n_lateral)
        cpr_img = cpr_img_raw.T

        # Update stored data
        vd.cpr_image = cpr_img
        vd.cpr_N_frame = N_rot
        vd.cpr_B_frame = B_rot
        self._cs_cache.clear()
        self._cpr_panel._measurements = []
        self._refresh_cpr()
        self._refresh_cs()

    # ── CPR mode toggle ────────────────────────────────────────────

    def _on_cpr_mode_changed(self, id: int, checked: bool) -> None:
        """Switch between Straightened (fill panel) and Stretched (preserve aspect)."""
        if checked:
            self._stretched_mode = (id == 1)
            self._cpr_panel.update()

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
        """Backward-compatible alias for _refresh_all_cs."""
        self._refresh_all_cs()

    def _refresh_all_cs(self) -> None:
        """Recompute and display cross-sections for all 3 panels (A, B, C)."""
        vd = self._current_vdata()
        if vd is None:
            for panel in self._cs_panels:
                panel.clear()
            return

        n_pos = self._n_positions()
        if n_pos == 0:
            for panel in self._cs_panels:
                panel.clear()
            return

        interval = self._cpr_panel._needle_interval
        idx_b = self._needle_idx
        idx_a = max(0, idx_b - interval)
        idx_c = min(n_pos - 1, idx_b + interval)

        cr = vd.contour_result

        # Get arc-length at B for distance calculations
        arc_b = 0.0
        if vd.cpr_arclengths is not None:
            fi_b = self._map_needle_to_frame_idx(idx_b, vd)
            arc_b = float(vd.cpr_arclengths[fi_b])
        elif cr is not None and cr.arclengths is not None and idx_b < len(cr.arclengths):
            arc_b = float(cr.arclengths[idx_b])

        for panel, idx in zip(self._cs_panels, [idx_a, idx_b, idx_c]):
            r_eq_mm = 0.0
            arc_mm = 0.0

            # Prefer CPR frame arclengths
            if vd.cpr_arclengths is not None:
                fi = self._map_needle_to_frame_idx(idx, vd)
                arc_mm = float(vd.cpr_arclengths[fi])
            elif cr is not None and cr.arclengths is not None and idx < len(cr.arclengths):
                arc_mm = float(cr.arclengths[idx])

            if cr is not None and cr.r_eq is not None and idx < len(cr.r_eq):
                r_eq_mm = float(cr.r_eq[idx])

            dist_from_b_mm = arc_mm - arc_b

            cs_img = self._get_cross_section(idx)
            panel.set_cross_section(
                cs_img, self._window, self._level,
                r_eq_mm, arc_mm, dist_from_b_mm,
            )

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
