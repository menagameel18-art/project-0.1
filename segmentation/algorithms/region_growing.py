"""
Region Growing Semi-Automatic Segmentation.
Performs connected component extraction from user-selected seed points
with adjustable intensity tolerance, neighborhood connectivity, and multi-slice expansion.
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np

from imaging.coordinate_manager import VolumeGeometry
from segmentation.provider import SegmentationProvider


class RegionGrowingSegmentation(SegmentationProvider):
    """Connected threshold region growing algorithm."""

    @property
    def provider_id(self) -> str:
        return "semi_auto_region_growing"

    @property
    def name(self) -> str:
        return "Region Growing"

    @property
    def category(self) -> str:
        return "semi_automatic"

    @property
    def description(self) -> str:
        return "Grows a contiguous region from a seed point based on intensity homogeneity."

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
        Executes region growing.
        Parameters:
            seed_point: Tuple[int, int, int] = (z, y, x)
            tolerance: float (delta from seed intensity, e.g. 50 HU)
            min_intensity: Optional[float] (if provided, overrides lower bound)
            max_intensity: Optional[float] (if provided, overrides upper bound)
            connectivity: int (6 or 26 for 3D; 4 or 8 for 2D)
            slice_only: bool (default False = 3D volume, True = current 2D slice only)
            max_voxels: int (safety cap, default 2,000,000)
        """
        seed = parameters.get("seed_point", (0, 0, 0))
        z_seed, y_seed, x_seed = int(seed[0]), int(seed[1]), int(seed[2])

        nz, ny, nx = volume.shape
        if not (0 <= z_seed < nz and 0 <= y_seed < ny and 0 <= x_seed < nx):
            return np.zeros(volume.shape, dtype=np.uint8)

        seed_val = float(volume[z_seed, y_seed, x_seed])
        tolerance = float(parameters.get("tolerance", 40.0))

        if "min_intensity" in parameters and parameters["min_intensity"] is not None:
            lower_bound = float(parameters["min_intensity"])
        else:
            lower_bound = seed_val - tolerance

        if "max_intensity" in parameters and parameters["max_intensity"] is not None:
            upper_bound = float(parameters["max_intensity"])
        else:
            upper_bound = seed_val + tolerance

        connectivity = int(parameters.get("connectivity", 6))
        slice_only = bool(parameters.get("slice_only", False))
        max_voxels = int(parameters.get("max_voxels", 2_000_000))

        if progress_callback:
            progress_callback(5, f"Growing from seed ({z_seed}, {y_seed}, {x_seed}) [{seed_val:.1f}]...")

        # Result mask
        mask = np.zeros(volume.shape, dtype=np.uint8)

        # Neighborhood offsets
        if slice_only:
            # 2D neighbors on slice z_seed
            if connectivity == 8:
                neighbors = [
                    (0, -1, -1), (0, -1, 0), (0, -1, 1),
                    (0, 0, -1),              (0, 0, 1),
                    (0, 1, -1),  (0, 1, 0),  (0, 1, 1)
                ]
            else:  # 4-connected
                neighbors = [
                    (0, -1, 0),
                    (0, 0, -1), (0, 0, 1),
                    (0, 1, 0)
                ]
        else:
            # 3D neighbors
            if connectivity == 26:
                neighbors = []
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            neighbors.append((dz, dy, dx))
            else:  # 6-connected
                neighbors = [
                    (-1, 0, 0), (1, 0, 0),
                    (0, -1, 0), (0, 1, 0),
                    (0, 0, -1), (0, 0, 1)
                ]

        # Fast BFS queue
        queue = deque([(z_seed, y_seed, x_seed)])
        mask[z_seed, y_seed, x_seed] = 1
        count = 1

        check_interval = 2500
        step = 0

        while queue:
            step += 1
            if step % check_interval == 0:
                if cancel_check and cancel_check():
                    return np.zeros(volume.shape, dtype=np.uint8)
                if progress_callback:
                    pct = min(95, int(10 + (count / 100_000) * 80))
                    progress_callback(pct, f"Extracted {count:,} voxels...")

            cz, cy, cx = queue.popleft()

            for dz, dy, dx in neighbors:
                nz_idx = cz + dz
                ny_idx = cy + dy
                nx_idx = cx + dx

                if slice_only and nz_idx != z_seed:
                    continue

                if 0 <= nz_idx < nz and 0 <= ny_idx < ny and 0 <= nx_idx < nx:
                    if mask[nz_idx, ny_idx, nx_idx] == 0:
                        val = float(volume[nz_idx, ny_idx, nx_idx])
                        if lower_bound <= val <= upper_bound:
                            mask[nz_idx, ny_idx, nx_idx] = 1
                            queue.append((nz_idx, ny_idx, nx_idx))
                            count += 1
                            if count >= max_voxels:
                                queue.clear()
                                break

        if progress_callback:
            progress_callback(100, f"Region growing finished: {count:,} voxels segmented.")

        return mask
