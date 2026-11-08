"""
3D Medical Volume Processor.
Handles volume array management, CT Hounsfield conversion, VTK image data conversion,
and procedural medical phantoms for offline testing and instant demonstration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from imaging.coordinate_manager import VolumeGeometry, CoordinateManager
from imaging.reslice_manager import ResliceManager
from imaging.window_level import WindowLevelManager


@dataclass
class SeriesMetadata:
    """Standardized metadata extracted from DICOM series."""
    patient_id: str = "Unknown"
    patient_name: str = "Anonymous"
    study_date: str = ""
    modality: str = "CT"
    study_description: str = ""
    series_description: str = ""
    series_uid: str = ""
    study_uid: str = ""
    num_slices: int = 0
    dimensions: str = ""
    voxel_spacing: str = ""
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    is_ct: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class VolumeProcessor:
    """Central processing unit for medical volume data."""

    def __init__(self):
        self.volume: Optional[np.ndarray] = None  # Shape (Z, Y, X), float32 or int16
        self.geometry: Optional[VolumeGeometry] = None
        self.metadata: SeriesMetadata = SeriesMetadata()
        self.coordinate_manager: CoordinateManager = CoordinateManager()
        self.reslice_manager: ResliceManager = ResliceManager()
        self.window_level_manager: WindowLevelManager = WindowLevelManager()
        self._vtk_image: Optional[vtk.vtkImageData] = None

    def set_volume_data(
        self,
        volume: np.ndarray,
        geometry: VolumeGeometry,
        metadata: Optional[SeriesMetadata] = None
    ):
        """Sets new volume and initializes all dependent managers."""
        self.volume = volume.astype(np.float32)
        self.geometry = geometry
        self.metadata = metadata or SeriesMetadata(
            num_slices=geometry.num_z,
            dimensions=f"{geometry.num_x}x{geometry.num_y}x{geometry.num_z}",
            voxel_spacing=f"{geometry.dx:.2f}x{geometry.dy:.2f}x{geometry.dz:.2f} mm",
            is_ct=(metadata.modality.upper() == "CT") if metadata else True
        )
        self.coordinate_manager.set_geometry(geometry)
        self.reslice_manager.set_data(self.volume, geometry)

        # Set default window/level based on modality
        if self.metadata.is_ct:
            self.window_level_manager.apply_preset("Soft Tissue / Abdomen")
        else:
            self.window_level_manager.auto_window(self.volume)

        # Invalidate cached VTK data
        self._vtk_image = None

    def has_data(self) -> bool:
        return self.volume is not None and self.volume.size > 0

    def get_vtk_image_data(self) -> Optional[vtk.vtkImageData]:
        """
        Converts active numpy 3D volume to vtkImageData with correct physical origin and spacing.
        Caches the VTK image to avoid repeated conversions.
        """
        if not self.has_data() or self.volume is None or self.geometry is None:
            return None

        if self._vtk_image is not None:
            return self._vtk_image

        nz, ny, nx = self.volume.shape
        g = self.geometry

        # In VTK, coordinate order is X, Y, Z (column-major in 3D flat representation)
        # Our numpy volume is (Z, Y, X). A C-contiguous 1D flat buffer of (Z, Y, X)
        # has X varying fastest, then Y, then Z. This corresponds exactly to VTK standard voxel order!
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(g.dx, g.dy, g.dz)
        vtk_image.SetOrigin(g.origin[0], g.origin[1], g.origin[2])

        # Prepare 1D flat array for VTK (float32 or int16)
        flat_data = np.ascontiguousarray(self.volume, dtype=np.float32).ravel()
        vtk_array = numpy_to_vtk(num_array=flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("Scalars")

        vtk_image.GetPointData().SetScalars(vtk_array)
        self._vtk_image = vtk_image
        return self._vtk_image

    @classmethod
    def create_procedural_head_phantom(cls) -> VolumeProcessor:
        """
        Generates a realistic 3D CT Head phantom with true Hounsfield Units:
        Air (-1000 HU), Brain parenchyma (35 HU), Ventricles/CSF (15 HU),
        Cranium / Bone (1000 HU), Eyes, and Sinuses (-800 HU).
        Voxel dimensions: 64 x 128 x 128, spacing: 2.5mm x 1.5mm x 1.5mm.
        """
        nz, ny, nx = 64, 128, 128
        dz, dy, dx = 2.5, 1.5, 1.5
        vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)  # Air base

        # Coordinates normalized
        z_coords = np.linspace(-1.0, 1.0, nz)[:, None, None]
        y_coords = np.linspace(-1.1, 1.1, ny)[None, :, None]
        x_coords = np.linspace(-1.0, 1.0, nx)[None, None, :]

        # Ellipsoid equations for head
        r_head = (x_coords / 0.72) ** 2 + (y_coords / 0.88) ** 2 + (z_coords / 0.85) ** 2
        r_inner = (x_coords / 0.65) ** 2 + (y_coords / 0.80) ** 2 + (z_coords / 0.78) ** 2

        # Scalp / Soft tissue
        scalp_mask = (r_head <= 1.05) & (r_head > 1.0)
        vol[scalp_mask] = 40.0

        # Skull bone
        skull_mask = (r_head <= 1.0) & (r_inner > 1.0)
        vol[skull_mask] = 1100.0

        # Brain tissue
        brain_mask = r_inner <= 1.0
        vol[brain_mask] = 38.0

        # Ventricles (CSF = 15 HU)
        r_vent1 = ((x_coords - 0.12) / 0.08) ** 2 + ((y_coords - 0.05) / 0.28) ** 2 + (z_coords / 0.35) ** 2
        r_vent2 = ((x_coords + 0.12) / 0.08) ** 2 + ((y_coords - 0.05) / 0.28) ** 2 + (z_coords / 0.35) ** 2
        vol[(r_vent1 <= 1.0) | (r_vent2 <= 1.0)] = 15.0

        # Frontal sinuses (air: -800 HU)
        r_sinus = (x_coords / 0.18) ** 2 + ((y_coords + 0.65) / 0.12) ** 2 + ((z_coords + 0.2) / 0.15) ** 2
        vol[r_sinus <= 1.0] = -850.0

        # Eyes (globes: 25 HU)
        r_eye_l = ((x_coords - 0.28) / 0.12) ** 2 + ((y_coords + 0.72) / 0.12) ** 2 + ((z_coords + 0.05) / 0.12) ** 2
        r_eye_r = ((x_coords + 0.28) / 0.12) ** 2 + ((y_coords + 0.72) / 0.12) ** 2 + ((z_coords + 0.05) / 0.12) ** 2
        vol[(r_eye_l <= 1.0) | (r_eye_r <= 1.0)] = 28.0

        # High density lens
        r_lens_l = ((x_coords - 0.28) / 0.04) ** 2 + ((y_coords + 0.81) / 0.04) ** 2 + ((z_coords + 0.05) / 0.04) ** 2
        r_lens_r = ((x_coords + 0.28) / 0.04) ** 2 + ((y_coords + 0.81) / 0.04) ** 2 + ((z_coords + 0.05) / 0.04) ** 2
        vol[(r_lens_l <= 1.0) | (r_lens_r <= 1.0)] = 80.0

        geometry = VolumeGeometry(
            shape=(nz, ny, nx),
            spacing=(dz, dy, dx),
            origin=(-(nx * dx) / 2.0, -(ny * dy) / 2.0, -(nz * dz) / 2.0)
        )
        metadata = SeriesMetadata(
            patient_id="DEMO-001",
            patient_name="DEMO^HEAD_CT",
            study_date="20261018",
            modality="CT",
            study_description="Demonstration Head Study",
            series_description="CT Head 3D Axial Reconstruct",
            series_uid="1.2.826.0.1.3680043.8.498.demo.head",
            study_uid="1.2.826.0.1.3680043.8.498.demo.study",
            num_slices=nz,
            dimensions=f"{nx}x{ny}x{nz}",
            voxel_spacing=f"{dx:.2f}x{dy:.2f}x{dz:.2f} mm",
            rescale_slope=1.0,
            rescale_intercept=0.0,
            is_ct=True
        )

        processor = cls()
        processor.set_volume_data(vol, geometry, metadata)
        return processor

    @classmethod
    def create_procedural_chest_phantom(cls) -> VolumeProcessor:
        """
        Generates a realistic 3D CT Chest phantom:
        Thoracic cage, ribs (1000 HU), spine (800 HU), heart/mediastinum (45 HU),
        lungs (-700 HU), pulmonary vessels (-200 to 40 HU), subcutaneous fat (-90 HU).
        Voxel dimensions: 64 x 128 x 128, spacing: 3.0mm x 1.6mm x 1.6mm.
        """
        nz, ny, nx = 64, 128, 128
        dz, dy, dx = 3.0, 1.6, 1.6
        vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)

        z_coords = np.linspace(-1.0, 1.0, nz)[:, None, None]
        y_coords = np.linspace(-1.0, 1.0, ny)[None, :, None]
        x_coords = np.linspace(-1.0, 1.0, nx)[None, None, :]

        # Body contour
        r_body = (x_coords / 0.85) ** 2 + (y_coords / 0.65) ** 2
        body_mask = (r_body <= 1.0) & (np.abs(z_coords) <= 0.95)
        vol[body_mask] = -90.0  # Subcutaneous fat

        # Muscle layer
        muscle_mask = (r_body <= 0.92) & (np.abs(z_coords) <= 0.95)
        vol[muscle_mask] = 45.0

        # Thoracic cavity
        thorax_mask = (r_body <= 0.82) & (np.abs(z_coords) <= 0.92)
        vol[thorax_mask] = 40.0

        # Spine (posterior, bone = 850 HU)
        spine_mask = ((x_coords / 0.12) ** 2 + ((y_coords - 0.42) / 0.15) ** 2 <= 1.0) & (np.abs(z_coords) <= 0.92)
        vol[spine_mask] = 850.0

        # Sternum (anterior, bone = 800 HU)
        sternum_mask = ((x_coords / 0.10) ** 2 + ((y_coords + 0.52) / 0.08) ** 2 <= 1.0) & (np.abs(z_coords) <= 0.80)
        vol[sternum_mask] = 800.0

        # Lungs (Right & Left, parenchyma = -700 HU)
        r_lung_r = ((x_coords + 0.38) / 0.32) ** 2 + ((y_coords + 0.05) / 0.45) ** 2
        r_lung_l = ((x_coords - 0.38) / 0.32) ** 2 + ((y_coords + 0.05) / 0.45) ** 2
        lung_mask = ((r_lung_r <= 1.0) | (r_lung_l <= 1.0)) & (np.abs(z_coords) <= 0.85)
        vol[lung_mask] = -720.0

        # Heart / Mediastinum (40 HU)
        heart_mask = (((x_coords - 0.1) / 0.28) ** 2 + ((y_coords + 0.1) / 0.30) ** 2 <= 1.0) & (z_coords >= -0.4) & (z_coords <= 0.5)
        vol[heart_mask] = 45.0

        # Ribs (periodic along Z and along body contour)
        rib_phase = np.sin(z_coords * 12.0)
        rib_mask = (r_body > 0.78) & (r_body <= 0.85) & (rib_phase > 0.4)
        vol[rib_mask] = 950.0

        geometry = VolumeGeometry(
            shape=(nz, ny, nx),
            spacing=(dz, dy, dx),
            origin=(-(nx * dx) / 2.0, -(ny * dy) / 2.0, -(nz * dz) / 2.0)
        )
        metadata = SeriesMetadata(
            patient_id="DEMO-002",
            patient_name="DEMO^CHEST_CT",
            study_date="20261018",
            modality="CT",
            study_description="Demonstration Chest CT",
            series_description="CT Chest Thorax 3D",
            series_uid="1.2.826.0.1.3680043.8.498.demo.chest",
            study_uid="1.2.826.0.1.3680043.8.498.demo.study",
            num_slices=nz,
            dimensions=f"{nx}x{ny}x{nz}",
            voxel_spacing=f"{dx:.2f}x{dy:.2f}x{dz:.2f} mm",
            rescale_slope=1.0,
            rescale_intercept=0.0,
            is_ct=True
        )

        processor = cls()
        processor.set_volume_data(vol, geometry, metadata)
        return processor
