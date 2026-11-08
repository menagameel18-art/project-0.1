"""
Segmentation Manager.
Central orchestrator for manual tools, semi-automatic segmentation workflows,
and AI-assisted model results (TotalSegmentator). Coordinates the active
segmentation model, undo/redo history, background workers, and viewport synchronization.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QObject, Signal

from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import Segment, SegmentationModel, DEFAULT_PALETTE
from segmentation.undo_redo import UndoRedoStack
from segmentation.tools.brush_tool import BrushTool
from segmentation.tools.contour_tool import ContourTool
from segmentation.algorithms.threshold import ThresholdSegmentation
from segmentation.algorithms.region_growing import RegionGrowingSegmentation
from segmentation.algorithms.worker import SemiAutoWorker
from segmentation.io import SegmentationIO


class SegmentationManager(QObject):
    """Manages the lifecycle and manipulation of segmentation models and tools."""

    segment_list_changed = Signal()
    segment_updated = Signal(int)             # segment_id
    active_segment_changed = Signal(object)   # Optional[int]
    undo_redo_state_changed = Signal(bool, bool)  # can_undo, can_redo
    preview_updated = Signal()
    worker_progress = Signal(int, str)        # pct (0-100), message
    worker_finished = Signal()
    worker_error = Signal(str)
    model_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.model: Optional[SegmentationModel] = None
        self.active_segment_id: Optional[int] = None
        self.active_tool: str = "none"
        self.global_opacity: float = 0.70

        # Interactive states
        self.seed_point: Optional[Tuple[int, int, int]] = None
        self.seed_intensity: Optional[float] = None
        self.preview_mask: Optional[np.ndarray] = None

        # Tools & History
        self.undo_stack = UndoRedoStack(max_depth=50)
        self.brush_tool = BrushTool(radius=6, mode="paint")
        self.contour_tool = ContourTool()

        # Algorithms & Workers
        self.threshold_algo = ThresholdSegmentation()
        self.region_growing_algo = RegionGrowingSegmentation()
        self.worker: Optional[SemiAutoWorker] = None

        # Temporary stroke tracking
        self._stroke_old_mask: Optional[np.ndarray] = None
        self._prev_stroke_pt: Optional[Tuple[float, float]] = None

    # -------------------------------------------------------------------------
    # Geometry & Model Lifecycle
    # -------------------------------------------------------------------------
    def set_geometry(self, geometry: VolumeGeometry):
        """Initializes or updates model geometry to match the active medical study."""
        if self.model is None or self.model.geometry.shape != geometry.shape:
            self.model = SegmentationModel(geometry)
            self.undo_stack.clear()
            self.clear_preview()
            if not self.model.segments:
                seg = self.model.add_segment("Segment 1")
                self.active_segment_id = seg.id
            self.model_changed.emit()
            self.segment_list_changed.emit()
            self.active_segment_changed.emit(self.active_segment_id)
            self.undo_redo_state_changed.emit(False, False)

    # -------------------------------------------------------------------------
    # Segment Management
    # -------------------------------------------------------------------------
    def get_active_segment(self) -> Optional[Segment]:
        if not self.model:
            return None
        return self.model.get_segment(self.active_segment_id)

    def set_active_segment(self, segment_id: Optional[int]):
        if self.active_segment_id != segment_id:
            self.active_segment_id = segment_id
            self.active_segment_changed.emit(segment_id)

    def add_segment(
        self,
        name: Optional[str] = None,
        color: Optional[Tuple[int, int, int, int]] = None,
        opacity: float = 0.65,
        mask: Optional[np.ndarray] = None
    ) -> Optional[Segment]:
        if not self.model:
            return None
        seg = self.model.add_segment(name=name, color=color, opacity=opacity, mask=mask)
        self.active_segment_id = seg.id
        self.segment_list_changed.emit()
        self.active_segment_changed.emit(seg.id)
        self.segment_updated.emit(seg.id)
        return seg

    def remove_segment(self, segment_id: int) -> bool:
        if not self.model:
            return False
        res = self.model.remove_segment(segment_id)
        if res:
            if self.active_segment_id == segment_id:
                self.active_segment_id = next(iter(self.model.segments.keys()), None)
                self.active_segment_changed.emit(self.active_segment_id)
            self.segment_list_changed.emit()
            self.segment_updated.emit(segment_id)
        return res

    def clear_segment(self, segment_id: int):
        if not self.model:
            return
        seg = self.model.get_segment(segment_id)
        if not seg:
            return
        old_mask = seg.mask.copy()
        seg.mask.fill(0)
        self.undo_stack.record_diff(segment_id, "Clear segment", old_mask, seg.mask)
        self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
        self.segment_updated.emit(segment_id)

    def set_segment_visibility(self, segment_id: int, visible: bool):
        if not self.model:
            return
        seg = self.model.get_segment(segment_id)
        if seg and seg.visible != visible:
            seg.visible = visible
            self.segment_updated.emit(segment_id)

    def set_segment_color(self, segment_id: int, color: Tuple[int, int, int, int]):
        if not self.model:
            return
        seg = self.model.get_segment(segment_id)
        if seg:
            seg.color = color
            self.segment_updated.emit(segment_id)

    def set_global_opacity(self, opacity: float):
        self.global_opacity = float(np.clip(opacity, 0.0, 1.0))
        self.preview_updated.emit()

    def set_active_tool(self, tool_name: str):
        self.active_tool = tool_name
        if tool_name == "brush":
            self.brush_tool.set_mode("paint")
        elif tool_name == "eraser":
            self.brush_tool.set_mode("erase")

    # -------------------------------------------------------------------------
    # Manual Painting Strokes
    # -------------------------------------------------------------------------
    def begin_stroke(self):
        seg = self.get_active_segment()
        if seg is not None:
            self._stroke_old_mask = seg.mask.copy()
            self._prev_stroke_pt = None

    def continue_stroke(self, plane: str, slice_idx: int, col: float, row: float):
        seg = self.get_active_segment()
        if seg is not None and self.model is not None:
            curr_pt = (col, row)
            self.brush_tool.apply_stroke(
                mask=seg.mask,
                plane=plane,
                slice_idx=slice_idx,
                curr_pt=curr_pt,
                prev_pt=self._prev_stroke_pt,
                geometry=self.model.geometry
            )
            self._prev_stroke_pt = curr_pt
            self.segment_updated.emit(seg.id)

    def end_stroke(self):
        seg = self.get_active_segment()
        if seg is not None and self._stroke_old_mask is not None:
            self.undo_stack.record_diff(
                seg.id,
                f"{self.brush_tool.mode.title()} stroke",
                self._stroke_old_mask,
                seg.mask
            )
            self._stroke_old_mask = None
            self._prev_stroke_pt = None
            self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
            self.segment_updated.emit(seg.id)

    # -------------------------------------------------------------------------
    # Contour Polygon Tool
    # -------------------------------------------------------------------------
    def add_contour_point(self, plane: str, slice_idx: int, col: float, row: float):
        if not self.contour_tool.is_active():
            self.contour_tool.start_contour(plane, slice_idx, (col, row))
        else:
            self.contour_tool.add_point(col, row)
        self.preview_updated.emit()

    def finish_contour(self):
        seg = self.get_active_segment()
        if seg is not None and self.contour_tool.is_active():
            old_mask = seg.mask.copy()
            if self.contour_tool.close_and_fill(seg.mask, self.brush_tool.mode):
                self.undo_stack.record_diff(seg.id, "Contour Fill", old_mask, seg.mask)
                self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
                self.segment_updated.emit(seg.id)
            self.contour_tool.clear()
            self.preview_updated.emit()

    # -------------------------------------------------------------------------
    # Undo / Redo
    # -------------------------------------------------------------------------
    def undo(self):
        if self.model and self.undo_stack.can_undo():
            cmd = self.undo_stack.undo_stack.pop()
            cmd.apply_undo(self.model)
            self.undo_stack.redo_stack.append(cmd)
            self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
            self.segment_updated.emit(cmd.segment_id)

    def redo(self):
        if self.model and self.undo_stack.can_redo():
            cmd = self.undo_stack.redo_stack.pop()
            cmd.apply_redo(self.model)
            self.undo_stack.undo_stack.append(cmd)
            self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
            self.segment_updated.emit(cmd.segment_id)

    # -------------------------------------------------------------------------
    # Semi-Automatic Workflows (Thresholding & Region Growing)
    # -------------------------------------------------------------------------
    def set_seed_point(self, z: int, y: int, x: int, intensity: Optional[float] = None):
        self.seed_point = (z, y, x)
        self.seed_intensity = intensity
        self.preview_updated.emit()

    def start_threshold_preview(
        self,
        volume: np.ndarray,
        min_val: float,
        max_val: float,
        slice_only: bool = False,
        plane: str = "axial",
        slice_idx: int = 0
    ):
        if not self.model:
            return
        params = {
            "min_val": min_val,
            "max_val": max_val,
            "slice_only": slice_only,
            "plane": plane,
            "slice_idx": slice_idx
        }
        self._start_worker(self.threshold_algo, volume, params)

    def start_region_growing_preview(
        self,
        volume: np.ndarray,
        seed_point: Tuple[int, int, int],
        tolerance: float = 35.0,
        connectivity: int = 6,
        slice_only: bool = False,
        plane: str = "axial",
        slice_idx: int = 0
    ):
        if not self.model:
            return
        params = {
            "seed_point": seed_point,
            "tolerance": tolerance,
            "connectivity": connectivity,
            "slice_only": slice_only,
            "plane": plane,
            "slice_idx": slice_idx
        }
        self._start_worker(self.region_growing_algo, volume, params)

    def _start_worker(self, provider, volume: np.ndarray, params: Dict[str, Any]):
        self.cancel_worker()
        self.worker = SemiAutoWorker(provider, volume, self.model.geometry, params)
        self.worker.progress.connect(self.worker_progress)
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.failed.connect(self.worker_error)
        self.worker.cancelled.connect(self.clear_preview)
        self.worker.start()

    def _on_worker_result(self, result_mask: np.ndarray):
        self.preview_mask = result_mask
        self.preview_updated.emit()
        self.worker_finished.emit()

    def cancel_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.worker = None

    def clear_preview(self):
        self.preview_mask = None
        self.preview_updated.emit()

    def apply_preview(self, combine_mode: str = "union") -> bool:
        if self.preview_mask is None or not np.any(self.preview_mask > 0):
            return False
        seg = self.get_active_segment()
        if not seg:
            return False

        old_mask = seg.mask.copy()
        if combine_mode == "union":
            seg.mask = np.logical_or(seg.mask, self.preview_mask).astype(np.uint8)
        elif combine_mode == "subtract":
            seg.mask[self.preview_mask > 0] = 0
        elif combine_mode == "intersect":
            seg.mask = np.logical_and(seg.mask, self.preview_mask).astype(np.uint8)
        else:
            seg.mask = (self.preview_mask > 0).astype(np.uint8)

        self.undo_stack.record_diff(seg.id, "Apply preview", old_mask, seg.mask)
        self.undo_redo_state_changed.emit(self.undo_stack.can_undo(), self.undo_stack.can_redo())
        self.clear_preview()
        self.segment_updated.emit(seg.id)
        return True

    # -------------------------------------------------------------------------
    # AI & External Masks Integration (TotalSegmentator / NIfTI)
    # -------------------------------------------------------------------------
    def add_nifti_mask(
        self,
        nifti_path: str | Path,
        name: str,
        source: str = "totalsegmentator",
        color: Optional[Tuple[int, int, int] | Tuple[int, int, int, int]] = None
    ) -> Optional[Segment]:
        """Loads a NIfTI segmentation mask, resamples if necessary, and registers it as a segment."""
        if self.model is None:
            return None

        img = sitk.ReadImage(str(nifti_path))
        target_shape = self.model.geometry.shape  # (Z, Y, X)
        img_size = img.GetSize()                  # (X, Y, Z) in ITK
        expected_itk_size = (target_shape[2], target_shape[1], target_shape[0])

        if img_size != expected_itk_size:
            # Resample nearest-neighbor to target geometry
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(expected_itk_size)
            g = self.model.geometry
            resampler.SetOutputSpacing((float(g.dx), float(g.dy), float(g.dz)))
            resampler.SetOutputOrigin((float(g.origin[0]), float(g.origin[1]), float(g.origin[2])))
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            img = resampler.Execute(img)

        mask_np = sitk.GetArrayFromImage(img).astype(np.uint8)
        if mask_np.shape != target_shape:
            # Fallback zero-pad or trim if slight mismatch
            fixed_mask = np.zeros(target_shape, dtype=np.uint8)
            sz = min(fixed_mask.shape[0], mask_np.shape[0])
            sy = min(fixed_mask.shape[1], mask_np.shape[1])
            sx = min(fixed_mask.shape[2], mask_np.shape[2])
            fixed_mask[:sz, :sy, :sx] = mask_np[:sz, :sy, :sx]
            mask_np = fixed_mask

        # Format RGBA color
        if color is None:
            rgba = None
        elif len(color) == 3:
            rgba = (int(color[0]), int(color[1]), int(color[2]), 200)
        else:
            rgba = (int(color[0]), int(color[1]), int(color[2]), int(color[3]))

        seg = self.add_segment(name=name, color=rgba, opacity=0.75, mask=mask_np)
        return seg

    # -------------------------------------------------------------------------
    # I/O Project Persistence
    # -------------------------------------------------------------------------
    def save_project(self, file_path: str, patient_id: str = "Unknown", series_uid: str = "") -> bool:
        if not self.model:
            return False
        return SegmentationIO.save_to_file(self.model, file_path, patient_id, series_uid)

    def load_project(self, file_path: str) -> bool:
        cur_geom = self.model.geometry if self.model else None
        model, meta = SegmentationIO.load_from_file(file_path, cur_geom)
        self.model = model
        self.active_segment_id = next(iter(model.segments.keys()), None)
        self.undo_stack.clear()
        self.clear_preview()
        self.model_changed.emit()
        self.segment_list_changed.emit()
        self.active_segment_changed.emit(self.active_segment_id)
        self.undo_redo_state_changed.emit(False, False)
        return True


# Default instance for legacy/direct module access
manager = SegmentationManager()
