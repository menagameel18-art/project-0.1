"""
Manual Brush and Eraser Tool for 2D Medical Slices.
Maps 2D slice coordinates to original 3D volume voxel coordinates across
Axial, Sagittal, and Coronal views with continuous stroke interpolation.
"""
from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np

from imaging.coordinate_manager import VolumeGeometry


class BrushTool:
    """Applies spherical or circular brush strokes (paint or erase) onto a 3D binary mask."""

    def __init__(self, radius: int = 5, mode: str = "paint"):
        self.radius: int = max(1, radius)
        self.mode: str = mode.lower()  # 'paint' or 'erase'

    def set_radius(self, radius: int):
        self.radius = max(1, int(radius))

    def set_mode(self, mode: str):
        if mode.lower() in ("paint", "erase"):
            self.mode = mode.lower()

    @staticmethod
    def slice_to_voxel_coord(
        plane: str,
        slice_idx: int,
        col: float,
        row: float,
        geometry: VolumeGeometry
    ) -> Tuple[int, int, int]:
        """
        Converts 2D slice coordinates (col, row) on a specific plane to 3D voxel index (z, y, x).
        """
        nz, ny, nx = geometry.shape
        plane_lower = plane.lower()

        if plane_lower == "axial":
            # col is X, row is Y, slice_idx is Z
            x = int(round(col))
            y = int(round(row))
            z = int(round(slice_idx))
        elif plane_lower == "sagittal":
            # col is Y, row is inverted Z, slice_idx is X
            y = int(round(col))
            z = int(round((nz - 1) - row))
            x = int(round(slice_idx))
        elif plane_lower == "coronal":
            # col is X, row is inverted Z, slice_idx is Y
            x = int(round(col))
            z = int(round((nz - 1) - row))
            y = int(round(slice_idx))
        else:
            x, y, z = int(round(col)), int(round(row)), int(round(slice_idx))

        return (
            int(np.clip(z, 0, nz - 1)),
            int(np.clip(y, 0, ny - 1)),
            int(np.clip(x, 0, nx - 1))
        )

    def apply_stroke(
        self,
        mask: np.ndarray,
        plane: str,
        slice_idx: int,
        curr_pt: Tuple[float, float],
        prev_pt: Optional[Tuple[float, float]] = None,
        geometry: Optional[VolumeGeometry] = None
    ):
        """
        Applies a brush stroke to the 3D mask along the active plane.
        Interpolates between prev_pt and curr_pt for continuous coverage.
        """
        col1, row1 = curr_pt
        val_to_write = 1 if self.mode == "paint" else 0

        # Generate interpolated points along line
        points: List[Tuple[float, float]] = []
        if prev_pt is not None:
            col0, row0 = prev_pt
            dist = np.hypot(col1 - col0, row1 - row0)
            step_size = max(1.0, float(self.radius) * 0.5)
            num_steps = max(1, int(np.ceil(dist / step_size)))
            for step in range(num_steps + 1):
                t = step / float(num_steps)
                c = col0 + t * (col1 - col0)
                r = row0 + t * (row1 - row0)
                points.append((c, r))
        else:
            points.append((col1, row1))

        nz, ny, nx = mask.shape
        plane_lower = plane.lower()
        r = self.radius
        r_sq = r * r

        for c, row in points:
            c_int = int(round(c))
            r_int = int(round(row))

            if plane_lower == "axial":
                # Slice is at z = slice_idx
                z = int(np.clip(slice_idx, 0, nz - 1))
                y_min = max(0, r_int - r)
                y_max = min(ny, r_int + r + 1)
                x_min = max(0, c_int - r)
                x_max = min(nx, c_int + r + 1)

                if y_min >= y_max or x_min >= x_max:
                    continue

                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                dist_sq = (yy - r_int) ** 2 + (xx - c_int) ** 2
                circle_mask = dist_sq <= r_sq
                mask[z, y_min:y_max, x_min:x_max][circle_mask] = val_to_write

            elif plane_lower == "sagittal":
                # Slice is at x = slice_idx. In slice: col is Y, row is inverted Z
                x = int(np.clip(slice_idx, 0, nx - 1))
                c_min = max(0, c_int - r)
                c_max = min(ny, c_int + r + 1)
                r_min = max(0, r_int - r)
                r_max = min(nz, r_int + r + 1)

                if c_min >= c_max or r_min >= r_max:
                    continue

                rr, cc = np.ogrid[r_min:r_max, c_min:c_max]
                dist_sq = (rr - r_int) ** 2 + (cc - c_int) ** 2
                circle_mask = dist_sq <= r_sq

                # Convert row to Z: z = (nz - 1) - row
                # We can update voxel indices directly
                r_indices, c_indices = np.where(circle_mask)
                actual_rows = r_min + r_indices
                actual_cols = c_min + c_indices
                z_voxels = (nz - 1) - actual_rows
                y_voxels = actual_cols

                valid = (z_voxels >= 0) & (z_voxels < nz) & (y_voxels >= 0) & (y_voxels < ny)
                mask[z_voxels[valid], y_voxels[valid], x] = val_to_write

            elif plane_lower == "coronal":
                # Slice is at y = slice_idx. In slice: col is X, row is inverted Z
                y = int(np.clip(slice_idx, 0, ny - 1))
                c_min = max(0, c_int - r)
                c_max = min(nx, c_int + r + 1)
                r_min = max(0, r_int - r)
                r_max = min(nz, r_int + r + 1)

                if c_min >= c_max or r_min >= r_max:
                    continue

                rr, cc = np.ogrid[r_min:r_max, c_min:c_max]
                dist_sq = (rr - r_int) ** 2 + (cc - c_int) ** 2
                circle_mask = dist_sq <= r_sq

                r_indices, c_indices = np.where(circle_mask)
                actual_rows = r_min + r_indices
                actual_cols = c_min + c_indices
                z_voxels = (nz - 1) - actual_rows
                x_voxels = actual_cols

                valid = (z_voxels >= 0) & (z_voxels < nz) & (x_voxels >= 0) & (x_voxels < nx)
                mask[z_voxels[valid], y, x_voxels[valid]] = val_to_write
