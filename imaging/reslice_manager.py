"""
Multiplanar Reconstruction (MPR) engine.
Extracts and reconstructs orthogonal 2D anatomical planes (Axial, Sagittal, Coronal)
from 3D volumetric image data, accounting for anisotropic voxel spacing.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from imaging.coordinate_manager import VolumeGeometry


class ResliceManager:
    """Extracts orthogonal slices from a 3D numpy volume with correct anatomical orientation."""

    def __init__(self, volume: Optional[np.ndarray] = None, geometry: Optional[VolumeGeometry] = None):
        self.volume: Optional[np.ndarray] = volume
        self.geometry: Optional[VolumeGeometry] = geometry

    def set_data(self, volume: np.ndarray, geometry: VolumeGeometry):
        """Sets active 3D volume (Z, Y, X) and associated physical geometry."""
        self.volume = volume
        self.geometry = geometry

    def has_volume(self) -> bool:
        return self.volume is not None and self.volume.ndim == 3 and self.volume.size > 0

    def get_max_slices(self) -> Tuple[int, int, int]:
        """Returns total slice counts (axial_count, sagittal_count, coronal_count)."""
        if not self.has_volume() or self.volume is None:
            return (0, 0, 0)
        nz, ny, nx = self.volume.shape
        return (nz, nx, ny)  # Axial (Z), Sagittal (X), Coronal (Y)

    def extract_axial_slice(self, slice_idx: int) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Extracts axial slice at index z.
        Returns:
            slice_data: 2D array of shape (Y, X)
            pixel_spacing: (dy, dx) in mm
        """
        if not self.has_volume() or self.volume is None or self.geometry is None:
            return np.zeros((1, 1), dtype=np.float32), (1.0, 1.0)

        nz = self.volume.shape[0]
        z = int(np.clip(slice_idx, 0, nz - 1))
        slice_data = self.volume[z, :, :]
        return slice_data, (self.geometry.dy, self.geometry.dx)

    def extract_sagittal_slice(self, slice_idx: int) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Extracts sagittal slice at index x (Left-Right plane).
        In radiological orientation:
        - Vertical axis: Superior (top) to Inferior (bottom) -> inverted Z so slice 0 is top or bottom
        - Horizontal axis: Anterior (left) to Posterior (right) -> Y axis
        Returns:
            slice_data: 2D array of shape (Z, Y) oriented [Superior->Inferior, Anterior->Posterior]
            pixel_spacing: (dz, dy) in mm
        """
        if not self.has_volume() or self.volume is None or self.geometry is None:
            return np.zeros((1, 1), dtype=np.float32), (1.0, 1.0)

        nx = self.volume.shape[2]
        x = int(np.clip(slice_idx, 0, nx - 1))
        # Volume shape is (Z, Y, X)
        # Extract along X: result is shape (Z, Y)
        # In DICOM LPS: higher Z is Superior, lower Z is Inferior.
        # For standard display where Top is Superior and Bottom is Inferior, we flip Z axis: [::-1, :]
        raw_slice = self.volume[::-1, :, x]
        return raw_slice, (self.geometry.dz, self.geometry.dy)

    def extract_coronal_slice(self, slice_idx: int) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Extracts coronal slice at index y (Anterior-Posterior plane).
        In radiological orientation:
        - Vertical axis: Superior (top) to Inferior (bottom) -> inverted Z
        - Horizontal axis: Patient Right (left) to Patient Left (right) -> X axis
        Returns:
            slice_data: 2D array of shape (Z, X) oriented [Superior->Inferior, Right->Left]
            pixel_spacing: (dz, dx) in mm
        """
        if not self.has_volume() or self.volume is None or self.geometry is None:
            return np.zeros((1, 1), dtype=np.float32), (1.0, 1.0)

        ny = self.volume.shape[1]
        y = int(np.clip(slice_idx, 0, ny - 1))
        # Volume shape is (Z, Y, X)
        # Extract along Y: result is shape (Z, X)
        # Flip Z so top is Superior:
        raw_slice = self.volume[::-1, y, :]
        return raw_slice, (self.geometry.dz, self.geometry.dx)

    def get_voxel_value(self, z: int, y: int, x: int) -> Optional[float]:
        """Safely inspects voxel intensity at (z, y, x)."""
        if not self.has_volume() or self.volume is None:
            return None
        nz, ny, nx = self.volume.shape
        if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
            return float(self.volume[z, y, x])
        return None
