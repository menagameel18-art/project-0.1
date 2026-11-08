"""
Threshold-based Semi-Automatic Segmentation.
Extracts voxels within adjustable intensity ranges.
Respects CT Hounsfield Units and MRI calibrated intensity scales without assuming HU for MRI.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import numpy as np

from imaging.coordinate_manager import VolumeGeometry
from segmentation.provider import SegmentationProvider


class ThresholdSegmentation(SegmentationProvider):
    """Generates segmentation masks based on lower and upper intensity thresholds."""

    @property
    def provider_id(self) -> str:
        return "semi_auto_threshold"

    @property
    def name(self) -> str:
        return "Intensity Thresholding"

    @property
    def category(self) -> str:
        return "semi_automatic"

    @property
    def description(self) -> str:
        return "Segments voxels whose intensity falls between minimum and maximum bounds."

    @property
    def is_asynchronous(self) -> bool:
        return True

    def execute(
        self,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> np.ndarray:
        """
        Executes threshold segmentation.
        Parameters:
            min_val: float (lower threshold)
            max_val: float (upper threshold)
            slice_only: bool (optional, segment only a specific slice)
            plane: str (optional, plane of slice_only)
            slice_idx: int (optional, index of slice_only)
        """
        min_val = float(parameters.get("min_val", 0.0))
        max_val = float(parameters.get("max_val", 1000.0))
        slice_only = bool(parameters.get("slice_only", False))

        if progress_callback:
            progress_callback(10, "Evaluating threshold criteria...")

        if cancel_check and cancel_check():
            return np.zeros(volume.shape, dtype=np.uint8)

        if not slice_only:
            # 3D volume thresholding
            # Compute in chunks if needed or vectorized
            mask = np.zeros(volume.shape, dtype=np.uint8)
            nz = volume.shape[0]

            for z in range(nz):
                if cancel_check and cancel_check():
                    return np.zeros(volume.shape, dtype=np.uint8)
                sl = volume[z, :, :]
                cond = (sl >= min_val) & (sl <= max_val)
                mask[z, :, :][cond] = 1
                if progress_callback and (z % 10 == 0 or z == nz - 1):
                    progress_callback(int(10 + (z / nz) * 85), f"Processing slice {z + 1}/{nz}")

        else:
            # 2D slice thresholding
            mask = np.zeros(volume.shape, dtype=np.uint8)
            plane = str(parameters.get("plane", "axial")).lower()
            slice_idx = int(parameters.get("slice_idx", 0))
            nz, ny, nx = volume.shape

            if plane == "axial":
                z = int(np.clip(slice_idx, 0, nz - 1))
                cond = (volume[z, :, :] >= min_val) & (volume[z, :, :] <= max_val)
                mask[z, :, :][cond] = 1
            elif plane == "sagittal":
                x = int(np.clip(slice_idx, 0, nx - 1))
                cond = (volume[:, :, x] >= min_val) & (volume[:, :, x] <= max_val)
                mask[:, :, x][cond] = 1
            elif plane == "coronal":
                y = int(np.clip(slice_idx, 0, ny - 1))
                cond = (volume[:, y, :] >= min_val) & (volume[:, y, :] <= max_val)
                mask[:, y, :][cond] = 1

        if progress_callback:
            progress_callback(100, "Threshold segmentation complete.")

        return mask
