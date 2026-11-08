"""
Undo / Redo command history for segmentation operations.
Uses memory-efficient subvolume bounding-box differential storage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from segmentation.model import SegmentationModel


@dataclass
class SubvolumeDiffCommand:
    """Stores the minimal bounding-box subvolume modified by a segmentation action."""
    segment_id: int
    description: str
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    old_submask: np.ndarray  # uint8 subvolume
    new_submask: np.ndarray  # uint8 subvolume

    def apply_undo(self, model: SegmentationModel) -> bool:
        seg = model.get_segment(self.segment_id)
        if not seg:
            return False
        seg.mask[
            self.z_min:self.z_max,
            self.y_min:self.y_max,
            self.x_min:self.x_max
        ] = self.old_submask
        return True

    def apply_redo(self, model: SegmentationModel) -> bool:
        seg = model.get_segment(self.segment_id)
        if not seg:
            return False
        seg.mask[
            self.z_min:self.z_max,
            self.y_min:self.y_max,
            self.x_min:self.x_max
        ] = self.new_submask
        return True


class UndoRedoStack:
    """Manages undo and redo stacks with configurable history depth."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self.undo_stack: List[SubvolumeDiffCommand] = []
        self.redo_stack: List[SubvolumeDiffCommand] = []

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def record_diff(
        self,
        segment_id: int,
        description: str,
        old_mask: np.ndarray,
        new_mask: np.ndarray
    ) -> Optional[SubvolumeDiffCommand]:
        """
        Calculates the 3D bounding box of differences between old and new masks
        and pushes a command onto the undo stack.
        """
        diff = (old_mask != new_mask)
        if not np.any(diff):
            return None

        z_indices, y_indices, x_indices = np.where(diff)
        z_min, z_max = int(z_indices.min()), int(z_indices.max() + 1)
        y_min, y_max = int(y_indices.min()), int(y_indices.max() + 1)
        x_min, x_max = int(x_indices.min()), int(x_indices.max() + 1)

        old_sub = old_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        new_sub = new_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()

        cmd = SubvolumeDiffCommand(
            segment_id=segment_id,
            description=description,
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            old_submask=old_sub,
            new_submask=new_sub
        )

        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)

        # Clear redo stack on new action
        self.redo_stack.clear()
        return cmd

    def undo(self, model: SegmentationModel) -> Optional[SubvolumeDiffCommand]:
        if not self.can_undo():
            return None
        cmd = self.undo_stack.pop()
        if cmd.apply_undo(model):
            self.redo_stack.append(cmd)
            return cmd
        return None

    def redo(self, model: SegmentationModel) -> Optional[SubvolumeDiffCommand]:
        if not self.can_redo():
            return None
        cmd = self.redo_stack.pop()
        if cmd.apply_redo(model):
            self.undo_stack.append(cmd)
            return cmd
        return None
