"""
Draw Contour Tool for Creating Closed 2D Segmentation Regions.
Allows users to define polygon vertices on 2D slices and rasterize closed
regions into the 3D volume mask with anatomical coordinate fidelity.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter, QPolygonF, QColor


class ContourTool:
    """Manages interactive contour point collection and polygon filling."""

    def __init__(self):
        self.points: List[Tuple[float, float]] = []  # List of (col, row) on current slice
        self.active_plane: Optional[str] = None
        self.active_slice_idx: Optional[int] = None
        self.mode: str = "paint"  # 'paint' or 'erase'

    def is_active(self) -> bool:
        return len(self.points) > 0

    def start_contour(self, plane: str, slice_idx: int, initial_point: Tuple[float, float]):
        self.active_plane = plane.lower()
        self.active_slice_idx = slice_idx
        self.points = [initial_point]

    def add_point(self, col: float, row: float):
        self.points.append((col, row))

    def clear(self):
        self.points.clear()
        self.active_plane = None
        self.active_slice_idx = None

    def close_and_fill(
        self,
        mask: np.ndarray,
        mode: Optional[str] = None
    ) -> bool:
        """
        Closes the active polygon contour, fills its interior, and writes
        to the 3D volume mask at the active slice index.
        Returns True if successful, False if insufficient points (< 3).
        """
        if len(self.points) < 3 or self.active_plane is None or self.active_slice_idx is None:
            self.clear()
            return False

        fill_mode = (mode or self.mode).lower()
        plane = self.active_plane
        slice_idx = self.active_slice_idx
        nz, ny, nx = mask.shape

        if plane == "axial":
            w, h = nx, ny
            z = int(np.clip(slice_idx, 0, nz - 1))
        elif plane == "sagittal":
            w, h = ny, nz
            x = int(np.clip(slice_idx, 0, nx - 1))
        elif plane == "coronal":
            w, h = nx, nz
            y = int(np.clip(slice_idx, 0, ny - 1))
        else:
            self.clear()
            return False

        # Rasterize polygon into 2D uint8 image using QPainter
        qimg = QImage(w, h, QImage.Format_Grayscale8)
        qimg.fill(0)

        painter = QPainter(qimg)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QColor(255, 255, 255))

        qpoly = QPolygonF([QPointF(c, r) for c, r in self.points])
        painter.drawPolygon(qpoly)
        painter.end()

        # Extract 2D numpy array from QImage buffer
        ptr = qimg.bits()
        # QImage memory layout: (h, bytes_per_line)
        bpl = qimg.bytesPerLine()
        arr_2d = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))[:, :w]
        fill_mask = (arr_2d > 127)

        val_to_set = 1 if fill_mode == "paint" else 0

        # Apply to 3D mask
        if plane == "axial":
            if val_to_set == 1:
                mask[z, :, :][fill_mask] = 1
            else:
                mask[z, :, :][fill_mask] = 0
        elif plane == "sagittal":
            # In resliced sagittal view, row is inverted Z: raw_slice is volume[::-1, :, x]
            sag_slice = mask[::-1, :, x]
            if val_to_set == 1:
                sag_slice[fill_mask] = 1
            else:
                sag_slice[fill_mask] = 0
            mask[::-1, :, x] = sag_slice
        elif plane == "coronal":
            # In resliced coronal view, row is inverted Z: raw_slice is volume[::-1, y, :]
            cor_slice = mask[::-1, y, :]
            if val_to_set == 1:
                cor_slice[fill_mask] = 1
            else:
                cor_slice[fill_mask] = 0
            mask[::-1, y, :] = cor_slice

        self.clear()
        return True
