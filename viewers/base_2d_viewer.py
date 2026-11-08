"""
Base interactive 2D medical viewport widget.
Provides high-performance slice rendering, zoom/pan, window/level interaction,
crosshair linking, distance caliper measurements, and anatomical HUD overlays.
"""
from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QFont, QMouseEvent, QWheelEvent, QPaintEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QToolButton

from imaging.window_level import WindowLevelManager


class Base2DViewport(QWidget):
    """Core 2D viewport rendering canvas."""
    crosshair_clicked = Signal(float, float)  # (col, row) on current slice
    pixel_inspected = Signal(float, float, object)  # (col, row, hu_value or None)
    window_level_changed = Signal(float, float)  # window, level
    slice_nav_requested = Signal(int)  # delta slice (+1 or -1)
    view_updated = Signal()

    def __init__(self, plane_name: str = "AXIAL", parent=None):
        super().__init__(parent)
        self.plane_name = plane_name.upper()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #000000;")

        # Active slice data
        self.raw_slice: Optional[np.ndarray] = None  # 2D float32
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)  # (row_spacing, col_spacing) in mm
        self.qimage: Optional[QImage] = None

        # Interaction tools: 'navigate', 'window_level', 'pan', 'zoom', 'measure', 'crosshair'
        self.active_tool: str = "window_level"

        # Transform state
        self.zoom_factor: float = 1.0
        self.pan_offset: QPointF = QPointF(0, 0)
        self.flip_h: bool = False
        self.flip_v: bool = False

        # Interaction tracking
        self._last_mouse_pos: Optional[QPointF] = None
        self._is_dragging: bool = False
        self._mouse_button: Optional[Qt.MouseButton] = None

        # Crosshairs and measurements
        self.show_crosshair: bool = True
        self.crosshair_pos: Optional[QPointF] = None  # in slice pixel coordinates (col, row)
        self.measure_points: List[QPointF] = []       # list of QPointF in slice coordinates
        self.hover_pos: Optional[QPointF] = None
        self.hover_value: Optional[float] = None

        # Window/Level
        self.wl_manager = WindowLevelManager(window=400.0, level=40.0)

        # Segmentation manager and active slice
        self.segmentation_manager = None
        self.current_slice_index: int = 0

        # Anatomical labels
        self._init_anatomical_labels()

    def set_segmentation_manager(self, manager):
        self.segmentation_manager = manager
        if manager:
            manager.segment_updated.connect(lambda _: self.update())
            manager.model_changed.connect(self.update)
            manager.preview_updated.connect(self.update)

    def _init_anatomical_labels(self):
        """Standard radiological orientation indicators."""
        if self.plane_name == "AXIAL":
            self.lbl_top = "A"      # Anterior
            self.lbl_bottom = "P"   # Posterior
            self.lbl_left = "R"     # Patient Right
            self.lbl_right = "L"    # Patient Left
        elif self.plane_name == "SAGITTAL":
            self.lbl_top = "S"      # Superior
            self.lbl_bottom = "I"   # Inferior
            self.lbl_left = "A"     # Anterior
            self.lbl_right = "P"    # Posterior
        elif self.plane_name == "CORONAL":
            self.lbl_top = "S"      # Superior
            self.lbl_bottom = "I"   # Inferior
            self.lbl_left = "R"     # Patient Right
            self.lbl_right = "L"    # Patient Left
        else:
            self.lbl_top = ""
            self.lbl_bottom = ""
            self.lbl_left = ""
            self.lbl_right = ""

    def set_slice_data(
        self,
        slice_data: np.ndarray,
        pixel_spacing: Tuple[float, float],
        preserve_transform: bool = True
    ):
        """Sets new 2D slice data, generates display QImage, and requests redraw."""
        self.raw_slice = slice_data.astype(np.float32)
        self.pixel_spacing = pixel_spacing
        self._generate_qimage()

        if not preserve_transform:
            self.fit_to_window()
        self.update()

    def set_window_level(self, window: float, level: float):
        self.wl_manager.set_window_level(window, level)
        self._generate_qimage()
        self.update()

    def _generate_qimage(self):
        if self.raw_slice is None or self.raw_slice.size == 0:
            self.qimage = None
            return

        # Map to 8-bit uint8 grayscale [0, 255]
        u8_data = self.wl_manager.apply(self.raw_slice)
        h, w = u8_data.shape
        # Create QImage from buffer
        bytes_per_line = w
        # QImage requires persistent buffer, copy is safe
        img = QImage(u8_data.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        
        if self.flip_h or self.flip_v:
            img = img.mirrored(self.flip_h, self.flip_v)

        self.qimage = img

    def fit_to_window(self):
        """Fits and centers the slice in the viewport while preserving physical aspect ratio."""
        if self.raw_slice is None or self.width() <= 10 or self.height() <= 10:
            return

        h_slice, w_slice = self.raw_slice.shape
        row_sp, col_sp = self.pixel_spacing

        # Physical slice dimensions in mm
        phys_w = w_slice * col_sp
        phys_h = h_slice * row_sp

        scale_x = (self.width() * 0.90) / max(phys_w, 1.0)
        scale_y = (self.height() * 0.90) / max(phys_h, 1.0)
        scale = min(scale_x, scale_y)

        self.zoom_factor = scale
        # Center in viewport
        display_w = phys_w * scale
        display_h = phys_h * scale
        self.pan_offset = QPointF(
            (self.width() - display_w) / 2.0,
            (self.height() - display_h) / 2.0
        )
        self.update()

    def reset_view(self):
        self.fit_to_window()

    def toggle_flip_h(self):
        self.flip_h = not self.flip_h
        self._generate_qimage()
        self.update()

    def toggle_flip_v(self):
        self.flip_v = not self.flip_v
        self._generate_qimage()
        self.update()

    def clear_measurements(self):
        self.measure_points.clear()
        self.update()

    # Coordinate mapping between screen and slice pixels
    def screen_to_slice(self, screen_pt: QPointF) -> Optional[QPointF]:
        if self.raw_slice is None or self.zoom_factor <= 0:
            return None
        row_sp, col_sp = self.pixel_spacing
        # screen = pan_offset + slice_coord * spacing * zoom
        x_mm = (screen_pt.x() - self.pan_offset.x()) / self.zoom_factor
        y_mm = (screen_pt.y() - self.pan_offset.y()) / self.zoom_factor

        col = x_mm / max(col_sp, 1e-4)
        row = y_mm / max(row_sp, 1e-4)

        h_slice, w_slice = self.raw_slice.shape
        if self.flip_h:
            col = w_slice - 1 - col
        if self.flip_v:
            row = h_slice - 1 - row

        return QPointF(col, row)

    def slice_to_screen(self, slice_pt: QPointF) -> QPointF:
        if self.raw_slice is None:
            return QPointF(0, 0)
        h_slice, w_slice = self.raw_slice.shape
        row_sp, col_sp = self.pixel_spacing

        col = slice_pt.x()
        row = slice_pt.y()
        if self.flip_h:
            col = w_slice - 1 - col
        if self.flip_v:
            row = h_slice - 1 - row

        x_screen = self.pan_offset.x() + (col * col_sp * self.zoom_factor)
        y_screen = self.pan_offset.y() + (row * row_sp * self.zoom_factor)
        return QPointF(x_screen, y_screen)

    # Mouse interaction handlers
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.ControlModifier:
            # Zoom at mouse position
            zoom_step = 1.15 if delta > 0 else 0.85
            self._zoom_at(event.position(), zoom_step)
        else:
            # Slice navigation
            steps = 1 if delta > 0 else -1
            self.slice_nav_requested.emit(steps)
        event.accept()

    def _zoom_at(self, anchor: QPointF, factor: float):
        new_zoom = np.clip(self.zoom_factor * factor, 0.05, 50.0)
        # Keep point under cursor fixed
        self.pan_offset = anchor - (anchor - self.pan_offset) * (new_zoom / self.zoom_factor)
        self.zoom_factor = new_zoom
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.active_tool == "segment_contour" and self.segmentation_manager:
            self.segmentation_manager.finish_contour()
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.position()
        self._mouse_button = event.button()
        self._is_dragging = True

        slice_pt = self.screen_to_slice(event.position())
        if slice_pt and self.raw_slice is not None:
            h, w = self.raw_slice.shape
            if 0 <= slice_pt.x() < w and 0 <= slice_pt.y() < h:
                if self.active_tool == "measure" and event.button() == Qt.LeftButton:
                    self.measure_points.append(slice_pt)
                    if len(self.measure_points) > 2:
                        self.measure_points = [slice_pt]
                    self.update()
                elif self.active_tool == "crosshair" or (event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier)):
                    self.crosshair_pos = slice_pt
                    self.crosshair_clicked.emit(slice_pt.x(), slice_pt.y())
                    self.update()
                elif self.active_tool in ("segment_brush", "segment_eraser") and event.button() == Qt.LeftButton:
                    if self.segmentation_manager:
                        self.segmentation_manager.begin_stroke()
                        self.segmentation_manager.continue_stroke(
                            self.plane_name, self.current_slice_index, slice_pt.x(), slice_pt.y()
                        )
                        self.update()
                elif self.active_tool == "segment_contour" and event.button() == Qt.LeftButton:
                    if self.segmentation_manager:
                        c_tool = self.segmentation_manager.contour_tool
                        if c_tool.is_active() and len(c_tool.points) >= 3:
                            p0 = c_tool.points[0]
                            if np.hypot(slice_pt.x() - p0[0], slice_pt.y() - p0[1]) <= 6.0:
                                self.segmentation_manager.finish_contour()
                                self.update()
                                event.accept()
                                return
                        self.segmentation_manager.add_contour_point(
                            self.plane_name, self.current_slice_index, slice_pt.x(), slice_pt.y()
                        )
                        self.update()
                elif self.active_tool == "segment_seed_picker" and event.button() == Qt.LeftButton:
                    if self.segmentation_manager and self.segmentation_manager.model:
                        from segmentation.tools.brush_tool import BrushTool
                        c_int = int(round(slice_pt.x()))
                        r_int = int(round(slice_pt.y()))
                        val = float(self.raw_slice[r_int, c_int])
                        geom = self.segmentation_manager.model.geometry
                        z, y, x = BrushTool.slice_to_voxel_coord(
                            self.plane_name, self.current_slice_index, slice_pt.x(), slice_pt.y(), geom
                        )
                        self.segmentation_manager.set_seed_point(z, y, x, val)
                        self.update()

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        curr_pos = event.position()
        slice_pt = self.screen_to_slice(curr_pos)

        # Inspect voxel value at hover
        if slice_pt and self.raw_slice is not None:
            h, w = self.raw_slice.shape
            c_int = int(round(slice_pt.x()))
            r_int = int(round(slice_pt.y()))
            if 0 <= c_int < w and 0 <= r_int < h:
                val = float(self.raw_slice[r_int, c_int])
                self.hover_pos = slice_pt
                self.hover_value = val
                self.pixel_inspected.emit(c_int, r_int, val)
            else:
                self.hover_value = None
                self.pixel_inspected.emit(0, 0, None)

        if self._is_dragging and self._last_mouse_pos:
            dx = curr_pos.x() - self._last_mouse_pos.x()
            dy = curr_pos.y() - self._last_mouse_pos.y()

            # Pan with middle button or Pan tool
            if self._mouse_button == Qt.MiddleButton or (self.active_tool == "pan" and self._mouse_button == Qt.LeftButton):
                self.pan_offset += QPointF(dx, dy)
                self.update()

            # Window / Level with LeftButton in default mode
            elif self._mouse_button == Qt.LeftButton and self.active_tool == "window_level":
                # Horizontal drag: Window width, Vertical drag: Level center
                self.wl_manager.window += dx * 2.0
                self.wl_manager.level -= dy * 2.0
                self._generate_qimage()
                self.window_level_changed.emit(self.wl_manager.window, self.wl_manager.level)
                self.update()

            # Zoom with RightButton or Zoom tool
            elif self._mouse_button == Qt.RightButton or (self.active_tool == "zoom" and self._mouse_button == Qt.LeftButton):
                factor = 1.0 + (dx - dy) * 0.008
                self._zoom_at(curr_pos, factor)

            # Continuous crosshair dragging
            elif self.active_tool == "crosshair" and self._mouse_button == Qt.LeftButton and slice_pt:
                self.crosshair_pos = slice_pt
                self.crosshair_clicked.emit(slice_pt.x(), slice_pt.y())
                self.update()

            # Continuous brush painting / erasing
            elif self.active_tool in ("segment_brush", "segment_eraser") and self._mouse_button == Qt.LeftButton and slice_pt:
                if self.segmentation_manager:
                    self.segmentation_manager.continue_stroke(
                        self.plane_name, self.current_slice_index, slice_pt.x(), slice_pt.y()
                    )
                    self.update()

            self._last_mouse_pos = curr_pos

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mouse_button == Qt.LeftButton and self.active_tool in ("segment_brush", "segment_eraser"):
            if self.segmentation_manager:
                self.segmentation_manager.end_stroke()
                self.update()
        self._is_dragging = False
        self._mouse_button = None
        event.accept()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Draw medical image
        if self.qimage and not self.qimage.isNull() and self.raw_slice is not None:
            h_slice, w_slice = self.raw_slice.shape
            row_sp, col_sp = self.pixel_spacing

            disp_w = w_slice * col_sp * self.zoom_factor
            disp_h = h_slice * row_sp * self.zoom_factor
            target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), disp_w, disp_h)
            painter.drawImage(target_rect, self.qimage)

            # Draw Segmentation Overlays
            if self.segmentation_manager and self.segmentation_manager.model:
                preview_slice = None
                if self.segmentation_manager.preview_mask is not None:
                    nz, ny, nx = self.segmentation_manager.model.geometry.shape
                    pm = self.segmentation_manager.preview_mask
                    pl = self.plane_name.lower()
                    if pl == "axial":
                        preview_slice = pm[int(np.clip(self.current_slice_index, 0, nz - 1)), :, :]
                    elif pl == "sagittal":
                        preview_slice = pm[::-1, :, int(np.clip(self.current_slice_index, 0, nx - 1))]
                    elif pl == "coronal":
                        preview_slice = pm[::-1, int(np.clip(self.current_slice_index, 0, ny - 1)), :]

                overlay_rgba = self.segmentation_manager.model.extract_slice_overlay(
                    plane=self.plane_name,
                    slice_idx=self.current_slice_index,
                    global_opacity=self.segmentation_manager.global_opacity,
                    preview_mask_slice=preview_slice
                )
                if overlay_rgba is not None:
                    oh, ow, _ = overlay_rgba.shape
                    overlay_qimg = QImage(
                        overlay_rgba.data, ow, oh, ow * 4, QImage.Format_RGBA8888
                    ).copy()
                    if self.flip_h or self.flip_v:
                        overlay_qimg = overlay_qimg.mirrored(self.flip_h, self.flip_v)
                    painter.drawImage(target_rect, overlay_qimg)

            # Draw Contour Path
            if self.segmentation_manager and self.segmentation_manager.contour_tool.is_active():
                c_tool = self.segmentation_manager.contour_tool
                if c_tool.active_plane == self.plane_name.lower() and c_tool.active_slice_idx == self.current_slice_index:
                    pts = [self.slice_to_screen(QPointF(c, r)) for c, r in c_tool.points]
                    painter.setPen(QPen(QColor(255, 200, 0, 240), 2))
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i], pts[i + 1])
                    if self.hover_pos:
                        scr_h = self.slice_to_screen(self.hover_pos)
                        painter.setPen(QPen(QColor(255, 200, 0, 160), 1, Qt.DashLine))
                        painter.drawLine(pts[-1], scr_h)
                    painter.setBrush(QColor(255, 255, 255, 240))
                    for p in pts:
                        painter.drawEllipse(p, 3, 3)

            # Draw Seed Point indicator
            if self.segmentation_manager and self.segmentation_manager.seed_point and self.segmentation_manager.model:
                sz, sy, sx = self.segmentation_manager.seed_point
                nz, ny, nx = self.segmentation_manager.model.geometry.shape
                pl = self.plane_name.lower()
                is_on_slice = False
                col_pt, row_pt = 0.0, 0.0
                if pl == "axial" and sz == self.current_slice_index:
                    col_pt, row_pt = float(sx), float(sy)
                    is_on_slice = True
                elif pl == "sagittal" and sx == self.current_slice_index:
                    col_pt, row_pt = float(sy), float((nz - 1) - sz)
                    is_on_slice = True
                elif pl == "coronal" and sy == self.current_slice_index:
                    col_pt, row_pt = float(sx), float((nz - 1) - sz)
                    is_on_slice = True

                if is_on_slice:
                    scr_seed = self.slice_to_screen(QPointF(col_pt, row_pt))
                    painter.setPen(QPen(QColor(255, 50, 50, 240), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(scr_seed, 8, 8)
                    painter.drawLine(int(scr_seed.x() - 12), int(scr_seed.y()), int(scr_seed.x() + 12), int(scr_seed.y()))
                    painter.drawLine(int(scr_seed.x()), int(scr_seed.y() - 12), int(scr_seed.x()), int(scr_seed.y() + 12))

            # Draw Brush Cursor ring
            if self.active_tool in ("segment_brush", "segment_eraser") and self.hover_pos and self.segmentation_manager:
                r_vox = self.segmentation_manager.brush_tool.radius
                scr_center = self.slice_to_screen(self.hover_pos)
                r_screen = r_vox * col_sp * self.zoom_factor
                ring_color = QColor(0, 255, 180, 200) if self.active_tool == "segment_brush" else QColor(255, 80, 80, 200)
                painter.setPen(QPen(ring_color, 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(scr_center, r_screen, r_screen)

            # Draw Crosshairs
            if self.show_crosshair and self.crosshair_pos:
                screen_ch = self.slice_to_screen(self.crosshair_pos)
                # Crosshair lines in subtle cyan
                painter.setPen(QPen(QColor(0, 230, 255, 180), 1, Qt.DashLine))
                # Horizontal line across viewport
                painter.drawLine(0, int(screen_ch.y()), self.width(), int(screen_ch.y()))
                # Vertical line across viewport
                painter.drawLine(int(screen_ch.x()), 0, int(screen_ch.x()), self.height())

            # Draw Distance Measurement Caliper
            if self.measure_points:
                painter.setPen(QPen(QColor(255, 215, 0, 230), 2))
                painter.setBrush(QColor(255, 215, 0, 255))
                scr_pts = [self.slice_to_screen(p) for p in self.measure_points]

                for p in scr_pts:
                    painter.drawEllipse(p, 4, 4)

                if len(scr_pts) == 2:
                    painter.drawLine(scr_pts[0], scr_pts[1])
                    # Calculate physical distance
                    pt1 = (self.measure_points[0].x(), self.measure_points[0].y())
                    pt2 = (self.measure_points[1].x(), self.measure_points[1].y())
                    dist_mm = np.sqrt(
                        ((pt2[0] - pt1[0]) * col_sp) ** 2 +
                        ((pt2[1] - pt1[1]) * row_sp) ** 2
                    )
                    mid_pt = QPointF(
                        (scr_pts[0].x() + scr_pts[1].x()) / 2.0 + 8,
                        (scr_pts[0].y() + scr_pts[1].y()) / 2.0 - 8
                    )
                    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                    painter.drawText(mid_pt, f"{dist_mm:.2f} mm")

        # Draw HUD Overlays (Anatomical labels, W/L, Cursor HU)
        self._paint_hud(painter)

    def _paint_hud(self, painter: QPainter):
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(0, 210, 255))

        margin = 12
        # Top-Left: Plane name
        painter.drawText(margin, margin + 14, self.plane_name)

        # Top-Right: Window / Level
        wl_text = f"W: {int(self.wl_manager.window)}  L: {int(self.wl_manager.level)}"
        fm = painter.fontMetrics()
        w_width = fm.horizontalAdvance(wl_text)
        painter.drawText(self.width() - margin - w_width, margin + 14, wl_text)

        # Bottom-Left: Zoom & Cursor inspection
        zoom_text = f"Zoom: {int(self.zoom_factor * 100)}%"
        if self.hover_value is not None:
            zoom_text += f" | Val: {self.hover_value:.1f} HU"
        painter.drawText(margin, self.height() - margin, zoom_text)

        # Anatomical direction labels (yellow/amber for clinical clarity)
        painter.setPen(QColor(255, 190, 0))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))

        # Top
        if self.lbl_top:
            w_top = painter.fontMetrics().horizontalAdvance(self.lbl_top)
            painter.drawText(int((self.width() - w_top) / 2), margin + 14, self.lbl_top)
        # Bottom
        if self.lbl_bottom:
            w_bot = painter.fontMetrics().horizontalAdvance(self.lbl_bottom)
            painter.drawText(int((self.width() - w_bot) / 2), self.height() - margin, self.lbl_bottom)
        # Left
        if self.lbl_left:
            painter.drawText(margin, int(self.height() / 2), self.lbl_left)
        # Right
        if self.lbl_right:
            painter.drawText(self.width() - margin - 14, int(self.height() / 2), self.lbl_right)


class ViewportPanel(QWidget):
    """Encapsulates a 2D Viewport with title header, toolbar controls, and slice slider footer."""
    slice_changed = Signal(str, int)  # plane_name, slice_index
    activated = Signal(str)           # plane_name

    def __init__(self, plane_name: str, parent=None):
        super().__init__(parent)
        self.plane_name = plane_name
        self.total_slices = 1
        self.current_slice = 0
        self.is_active = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)

        # Top Header Bar
        self.header_bar = QWidget()
        self.header_bar.setFixedHeight(28)
        self.header_bar.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #333333;")
        h_layout = QHBoxLayout(self.header_bar)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(6)

        self.title_label = QLabel(self.plane_name.upper())
        self.title_label.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.title_label)

        h_layout.addStretch()

        # Viewport Action Buttons
        self.btn_fit = self._create_mini_btn("Fit", "Fit Image to Window", self._on_fit_clicked)
        self.btn_reset = self._create_mini_btn("Reset", "Reset View", self._on_reset_clicked)
        self.btn_fliph = self._create_mini_btn("Flip H", "Flip Horizontally", self._on_fliph_clicked)
        self.btn_flipv = self._create_mini_btn("Flip V", "Flip Vertically", self._on_flipv_clicked)
        self.btn_clear_meas = self._create_mini_btn("Clear", "Clear Measurements", self._on_clear_meas_clicked)

        h_layout.addWidget(self.btn_fit)
        h_layout.addWidget(self.btn_reset)
        h_layout.addWidget(self.btn_fliph)
        h_layout.addWidget(self.btn_flipv)
        h_layout.addWidget(self.btn_clear_meas)

        layout.addWidget(self.header_bar)

        # 2D Canvas Viewport
        self.viewport = Base2DViewport(self.plane_name)
        self.viewport.slice_nav_requested.connect(self._on_slice_nav_requested)
        layout.addWidget(self.viewport, stretch=1)

        # Bottom Slice Navigation Bar
        self.footer_bar = QWidget()
        self.footer_bar.setFixedHeight(32)
        self.footer_bar.setStyleSheet("background-color: #1a1a1a; border-top: 1px solid #2a2a2a;")
        f_layout = QHBoxLayout(self.footer_bar)
        f_layout.setContentsMargins(8, 0, 8, 0)
        f_layout.setSpacing(8)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setValue(0)
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #333333;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #007acc;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d2ff;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
        """)
        self.slice_slider.valueChanged.connect(self._on_slider_value_changed)
        f_layout.addWidget(self.slice_slider, stretch=1)

        self.slice_info_label = QLabel("Slice: 0 / 0 (0.0 mm)")
        self.slice_info_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        f_layout.addWidget(self.slice_info_label)

        layout.addWidget(self.footer_bar)
        self.set_active(False)

    def _create_mini_btn(self, text: str, tooltip: str, callback) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(20)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 4px;
            }
            QToolButton:hover {
                background-color: #383838;
                border-color: #007acc;
            }
        """)
        btn.clicked.connect(callback)
        return btn

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.setStyleSheet("border: 2px solid #00aaff;")
        else:
            self.setStyleSheet("border: 1px solid #2d2d2d;")

    def mousePressEvent(self, event: QMouseEvent):
        self.activated.emit(self.plane_name)
        super().mousePressEvent(event)

    def set_segmentation_manager(self, manager):
        self.viewport.set_segmentation_manager(manager)

    def set_slice_count(self, count: int):
        self.total_slices = max(1, count)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, self.total_slices - 1)
        self.slice_slider.blockSignals(False)
        self.update_slice_label()

    def set_current_slice(self, index: int, phys_pos_mm: float = 0.0):
        self.current_slice = int(np.clip(index, 0, self.total_slices - 1))
        self.viewport.current_slice_index = self.current_slice
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.blockSignals(False)
        self.update_slice_label(phys_pos_mm)

    def update_slice_label(self, phys_pos_mm: float = 0.0):
        self.slice_info_label.setText(
            f"Slice: {self.current_slice + 1}/{self.total_slices} ({phys_pos_mm:+.1f} mm)"
        )

    def _on_slider_value_changed(self, val: int):
        self.current_slice = val
        self.viewport.current_slice_index = val
        self.slice_changed.emit(self.plane_name, val)

    def _on_slice_nav_requested(self, delta: int):
        new_val = int(np.clip(self.current_slice + delta, 0, self.total_slices - 1))
        if new_val != self.current_slice:
            self.slice_slider.setValue(new_val)

    def _on_fit_clicked(self):
        self.viewport.fit_to_window()

    def _on_reset_clicked(self):
        self.viewport.reset_view()

    def _on_fliph_clicked(self):
        self.viewport.toggle_flip_h()

    def _on_flipv_clicked(self):
        self.viewport.toggle_flip_v()

    def _on_clear_meas_clicked(self):
        self.viewport.clear_measurements()
