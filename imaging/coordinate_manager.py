"""
Coordinate and spatial transformation management for medical DICOM volumes.
Follows standard DICOM LPS (Left, Posterior, Superior) coordinate frame conventions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class VolumeGeometry:
    """Stores spatial geometry information of a 3D medical volume."""
    shape: Tuple[int, int, int]  # (Z, Y, X) = (Slices, Rows, Columns)
    spacing: Tuple[float, float, float]  # (dz, dy, dx) in millimeters
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (X0, Y0, Z0) in mm
    row_cosines: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # direction of rows (along X)
    col_cosines: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # direction of columns (along Y)
    slice_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # normal vector (along Z)

    @property
    def num_z(self) -> int:
        return self.shape[0]

    @property
    def num_y(self) -> int:
        return self.shape[1]

    @property
    def num_x(self) -> int:
        return self.shape[2]

    @property
    def dz(self) -> float:
        return self.spacing[0]

    @property
    def dy(self) -> float:
        return self.spacing[1]

    @property
    def dx(self) -> float:
        return self.spacing[2]


class CoordinateManager:
    """Handles bidirectional mapping between voxel indices and physical patient coordinates."""

    def __init__(self, geometry: Optional[VolumeGeometry] = None):
        self.geometry = geometry or VolumeGeometry(
            shape=(1, 1, 1),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0)
        )

    def set_geometry(self, geometry: VolumeGeometry):
        self.geometry = geometry

    def voxel_to_patient(self, z_idx: float, y_idx: float, x_idx: float) -> Tuple[float, float, float]:
        """
        Converts fractional voxel index (z, y, x) to physical patient coordinates (X, Y, Z) in mm.
        """
        g = self.geometry
        x_pt = g.origin[0] + (x_idx * g.dx * g.row_cosines[0]) + (y_idx * g.dy * g.col_cosines[0]) + (z_idx * g.dz * g.slice_normal[0])
        y_pt = g.origin[1] + (x_idx * g.dx * g.row_cosines[1]) + (y_idx * g.dy * g.col_cosines[1]) + (z_idx * g.dz * g.slice_normal[1])
        z_pt = g.origin[2] + (x_idx * g.dx * g.row_cosines[2]) + (y_idx * g.dy * g.col_cosines[2]) + (z_idx * g.dz * g.slice_normal[2])
        return (float(x_pt), float(y_pt), float(z_pt))

    def patient_to_voxel(self, x_pt: float, y_pt: float, z_pt: float) -> Tuple[int, int, int]:
        """
        Converts physical patient coordinates (X, Y, Z) in mm to nearest valid voxel index (z, y, x).
        Clamps to valid volume bounds.
        """
        g = self.geometry
        # Relative vector from origin
        dx_val = x_pt - g.origin[0]
        dy_val = y_pt - g.origin[1]
        dz_val = z_pt - g.origin[2]
        rel = np.array([dx_val, dy_val, dz_val], dtype=np.float64)

        # Direction matrix: columns are [R*dx, C*dy, N*dz]
        r = np.array(g.row_cosines, dtype=np.float64) * g.dx
        c = np.array(g.col_cosines, dtype=np.float64) * g.dy
        n = np.array(g.slice_normal, dtype=np.float64) * g.dz

        M = np.column_stack([r, c, n])
        try:
            vox = np.linalg.solve(M, rel)
            x_idx = int(np.clip(np.round(vox[0]), 0, g.num_x - 1))
            y_idx = int(np.clip(np.round(vox[1]), 0, g.num_y - 1))
            z_idx = int(np.clip(np.round(vox[2]), 0, g.num_z - 1))
            return (z_idx, y_idx, x_idx)
        except np.linalg.LinAlgError:
            # Fallback to axis-aligned calculation
            x_idx = int(np.clip(np.round((x_pt - g.origin[0]) / max(g.dx, 1e-4)), 0, g.num_x - 1))
            y_idx = int(np.clip(np.round((y_pt - g.origin[1]) / max(g.dy, 1e-4)), 0, g.num_y - 1))
            z_idx = int(np.clip(np.round((z_pt - g.origin[2]) / max(g.dz, 1e-4)), 0, g.num_z - 1))
            return (z_idx, y_idx, x_idx)

    def get_slice_physical_position(self, plane: str, index: int) -> float:
        """
        Returns the physical coordinate along the normal of the requested plane in mm.
        plane: 'axial', 'sagittal', or 'coronal'
        """
        g = self.geometry
        if plane.lower() == 'axial':
            idx = int(np.clip(index, 0, g.num_z - 1))
            _, _, z_pos = self.voxel_to_patient(idx, g.num_y / 2.0, g.num_x / 2.0)
            return z_pos
        elif plane.lower() == 'sagittal':
            idx = int(np.clip(index, 0, g.num_x - 1))
            x_pos, _, _ = self.voxel_to_patient(g.num_z / 2.0, g.num_y / 2.0, idx)
            return x_pos
        elif plane.lower() == 'coronal':
            idx = int(np.clip(index, 0, g.num_y - 1))
            _, y_pos, _ = self.voxel_to_patient(g.num_z / 2.0, idx, g.num_x / 2.0)
            return y_pos
        return 0.0

    def calculate_distance_mm(self, plane: str, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """
        Calculates physical distance in millimeters between two 2D slice points (col, row).
        Accounts for pixel spacing on that plane.
        """
        g = self.geometry
        dx_px = pt2[0] - pt1[0]
        dy_px = pt2[1] - pt1[1]

        if plane.lower() == 'axial':
            # pt = (x_px, y_px), spacing = (dx, dy)
            dist = np.sqrt((dx_px * g.dx) ** 2 + (dy_px * g.dy) ** 2)
        elif plane.lower() == 'sagittal':
            # pt = (y_px, z_px), spacing = (dy, dz)
            dist = np.sqrt((dx_px * g.dy) ** 2 + (dy_px * g.dz) ** 2)
        elif plane.lower() == 'coronal':
            # pt = (x_px, z_px), spacing = (dx, dz)
            dist = np.sqrt((dx_px * g.dx) ** 2 + (dy_px * g.dz) ** 2)
        else:
            dist = np.hypot(dx_px, dy_px)

        return float(dist)
