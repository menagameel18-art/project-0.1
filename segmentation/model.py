"""
Segmentation Data Model.
Manages individual segments, multi-segment labelmaps, binary masks,
and anatomical coordinate alignment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from imaging.coordinate_manager import VolumeGeometry


DEFAULT_PALETTE: List[Tuple[int, int, int, int]] = [
    (244, 67, 54, 200),    # Red
    (76, 175, 80, 200),    # Green
    (0, 210, 255, 200),    # Cyan
    (255, 235, 59, 200),   # Yellow
    (233, 30, 99, 200),    # Magenta
    (255, 152, 0, 200),    # Orange
    (156, 39, 176, 200),   # Purple
    (0, 150, 136, 200),    # Teal
    (33, 150, 243, 200),   # Blue
    (205, 220, 57, 200),   # Lime
]


@dataclass
class Segment:
    """Individual anatomical or pathological segment."""
    id: int
    name: str
    color: Tuple[int, int, int, int] = (244, 67, 54, 200)  # RGBA [0-255]
    visible: bool = True
    opacity: float = 0.65  # [0.0 - 1.0]
    mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.uint8))

    @property
    def voxel_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    def calculate_volume_cm3(self, geometry: VolumeGeometry) -> float:
        """Calculates volume in cubic centimeters (cm³ = mL)."""
        voxel_vol_mm3 = geometry.dx * geometry.dy * geometry.dz
        vol_mm3 = self.voxel_count * voxel_vol_mm3
        return vol_mm3 / 1000.0


class SegmentationModel:
    """Central data store for all segments associated with a medical volume."""

    def __init__(self, geometry: VolumeGeometry):
        self.geometry = geometry
        self.segments: Dict[int, Segment] = {}
        self._next_id: int = 1

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.geometry.shape

    def add_segment(
        self,
        name: Optional[str] = None,
        color: Optional[Tuple[int, int, int, int]] = None,
        opacity: float = 0.65,
        mask: Optional[np.ndarray] = None
    ) -> Segment:
        """Creates and registers a new segment with its own 3D binary mask."""
        seg_id = self._next_id
        self._next_id += 1

        if not name:
            name = f"Segment {seg_id}"

        if color is None:
            color = DEFAULT_PALETTE[(seg_id - 1) % len(DEFAULT_PALETTE)]

        if mask is None:
            seg_mask = np.zeros(self.geometry.shape, dtype=np.uint8)
        else:
            if mask.shape != self.geometry.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match geometry shape {self.geometry.shape}"
                )
            seg_mask = (mask > 0).astype(np.uint8)

        seg = Segment(
            id=seg_id,
            name=name,
            color=color,
            visible=True,
            opacity=float(np.clip(opacity, 0.0, 1.0)),
            mask=seg_mask
        )
        self.segments[seg_id] = seg
        return seg

    def remove_segment(self, segment_id: int) -> bool:
        """Removes a segment from the model."""
        if segment_id in self.segments:
            del self.segments[segment_id]
            return True
        return False

    def clear_segment(self, segment_id: int):
        """Clears the binary mask of a segment to zeros."""
        if segment_id in self.segments:
            self.segments[segment_id].mask.fill(0)

    def get_segment(self, segment_id: int) -> Optional[Segment]:
        return self.segments.get(segment_id)

    def get_labelmap(self) -> np.ndarray:
        """
        Generates a consolidated integer labelmap (Z, Y, X).
        Background is 0; each voxel contains the ID of the segment.
        """
        labelmap = np.zeros(self.geometry.shape, dtype=np.uint16)
        for seg_id, seg in sorted(self.segments.items()):
            labelmap[seg.mask > 0] = seg_id
        return labelmap

    def extract_slice_overlay(
        self,
        plane: str,
        slice_idx: int,
        global_opacity: float = 1.0,
        preview_mask_slice: Optional[np.ndarray] = None,
        preview_color: Tuple[int, int, int, int] = (255, 170, 0, 220)
    ) -> Optional[np.ndarray]:
        """
        Extracts composite RGBA image overlay for a 2D slice on axial, sagittal, or coronal view.
        Returns:
            rgba_array: shape (H, W, 4), dtype=uint8, or None if no segments are visible.
        """
        nz, ny, nx = self.geometry.shape
        plane_lower = plane.lower()

        if plane_lower == "axial":
            h, w = ny, nx
            z = int(np.clip(slice_idx, 0, nz - 1))
        elif plane_lower == "sagittal":
            h, w = nz, ny
            x = int(np.clip(slice_idx, 0, nx - 1))
        elif plane_lower == "coronal":
            h, w = nz, nx
            y = int(np.clip(slice_idx, 0, ny - 1))
        else:
            return None

        # RGBA canvas
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        has_content = False

        # Draw visible segments
        for seg in self.segments.values():
            if not seg.visible:
                continue

            # Extract 2D slice mask following radiological slice conventions
            if plane_lower == "axial":
                slice_mask = seg.mask[z, :, :]
            elif plane_lower == "sagittal":
                # In resliced sagittal view, Z is inverted [::-1, :, x]
                slice_mask = seg.mask[::-1, :, x]
            elif plane_lower == "coronal":
                # In resliced coronal view, Z is inverted [::-1, y, :]
                slice_mask = seg.mask[::-1, y, :]

            hit_idx = slice_mask > 0
            if np.any(hit_idx):
                has_content = True
                r, g, b, a_base = seg.color
                effective_alpha = int(np.clip(a_base * seg.opacity * global_opacity, 0, 255))
                if effective_alpha > 0:
                    rgba[hit_idx, 0] = r
                    rgba[hit_idx, 1] = g
                    rgba[hit_idx, 2] = b
                    rgba[hit_idx, 3] = effective_alpha

        # Draw semi-automatic preview overlay if provided
        if preview_mask_slice is not None and np.any(preview_mask_slice > 0):
            has_content = True
            hit_prev = preview_mask_slice > 0
            pr, pg, pb, pa = preview_color
            prev_alpha = int(np.clip(pa * global_opacity, 0, 255))
            rgba[hit_prev, 0] = pr
            rgba[hit_prev, 1] = pg
            rgba[hit_prev, 2] = pb
            rgba[hit_prev, 3] = prev_alpha

        return rgba if has_content else None
