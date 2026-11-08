"""
Sagittal Viewport for 2D longitudinal slice visualization (Left-Right cross-sections).
"""
from __future__ import annotations
from typing import Optional
from viewers.base_2d_viewer import ViewportPanel
from imaging.volume_processor import VolumeProcessor


class SagittalViewer(ViewportPanel):
    """Specialized sagittal slice viewport."""

    def __init__(self, parent=None):
        super().__init__(plane_name="SAGITTAL", parent=parent)
        self.processor: Optional[VolumeProcessor] = None

    def set_processor(self, processor: VolumeProcessor):
        self.processor = processor
        if not processor.has_data():
            return
        nx = processor.geometry.num_x
        self.set_slice_count(nx)
        mid = nx // 2
        self.show_slice(mid, preserve_transform=False)

    def show_slice(self, slice_idx: int, preserve_transform: bool = True):
        if not self.processor or not self.processor.has_data():
            return
        data, spacing = self.processor.reslice_manager.extract_sagittal_slice(slice_idx)
        phys_pos = self.processor.coordinate_manager.get_slice_physical_position("sagittal", slice_idx)
        self.viewport.set_slice_data(data, spacing, preserve_transform=preserve_transform)
        self.set_current_slice(slice_idx, phys_pos)

    def update_crosshair(self, y_vox: float, z_vox: float):
        """
        Updates crosshair in sagittal view given (Y, Z) volume coords.
        In resliced sagittal view, column is Y, row is inverted Z (Superior at top).
        """
        from PySide6.QtCore import QPointF
        if not self.processor or not self.processor.has_data():
            return
        nz = self.processor.geometry.num_z
        row = (nz - 1) - z_vox
        col = y_vox
        self.viewport.crosshair_pos = QPointF(col, row)
        self.viewport.update()
