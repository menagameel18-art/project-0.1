"""
Axial Viewport for 2D transverse medical slice visualization.
"""
from __future__ import annotations
from typing import Optional, Tuple
from viewers.base_2d_viewer import ViewportPanel
from imaging.volume_processor import VolumeProcessor


class AxialViewer(ViewportPanel):
    """Specialized axial slice viewport."""

    def __init__(self, parent=None):
        super().__init__(plane_name="AXIAL", parent=parent)
        self.processor: Optional[VolumeProcessor] = None

    def set_processor(self, processor: VolumeProcessor):
        self.processor = processor
        if not processor.has_data():
            return
        nz = processor.geometry.num_z
        self.set_slice_count(nz)
        # Default to middle slice
        mid = nz // 2
        self.show_slice(mid, preserve_transform=False)

    def show_slice(self, slice_idx: int, preserve_transform: bool = True):
        if not self.processor or not self.processor.has_data():
            return
        data, spacing = self.processor.reslice_manager.extract_axial_slice(slice_idx)
        phys_pos = self.processor.coordinate_manager.get_slice_physical_position("axial", slice_idx)
        self.viewport.set_slice_data(data, spacing, preserve_transform=preserve_transform)
        self.set_current_slice(slice_idx, phys_pos)

    def update_crosshair(self, x_vox: float, y_vox: float):
        """Updates crosshair position in axial view given volume voxel coords (X, Y)."""
        from PySide6.QtCore import QPointF
        self.viewport.crosshair_pos = QPointF(x_vox, y_vox)
        self.viewport.update()
