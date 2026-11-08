"""
Robust DICOM file scanner, validator, and volume assembler.
Complies with DICOM PS 3.3 and 3.5 standards.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from imaging.coordinate_manager import VolumeGeometry
from imaging.volume_processor import SeriesMetadata, VolumeProcessor


@dataclass
class DicomSliceInfo:
    """Lightweight metadata for a single DICOM slice."""
    file_path: str
    sop_uid: str
    instance_number: int
    position: Optional[Tuple[float, float, float]]
    orientation: Optional[Tuple[float, float, float, float, float, float]]
    pixel_spacing: Tuple[float, float]
    slice_thickness: float
    rescale_slope: float
    rescale_intercept: float
    rows: int
    cols: int
    projection: float = 0.0  # Projection along slice normal for 3D sorting


@dataclass
class DicomSeriesGroup:
    """Group of DICOM slices belonging to the same series."""
    series_uid: str
    study_uid: str
    modality: str
    series_description: str
    study_description: str
    patient_id: str
    patient_name: str
    study_date: str
    slices: List[DicomSliceInfo] = field(default_factory=list)

    @property
    def slice_count(self) -> int:
        return len(self.slices)


class DicomLoader:
    """Scans folders, organizes series, and loads 3D volumes from DICOM files."""

    @staticmethod
    def is_dicom_file(file_path: str) -> bool:
        """Checks if a file has a valid DICOM header (DICM magic bytes or valid preamble)."""
        try:
            if not os.path.isfile(file_path) or os.path.getsize(file_path) < 132:
                return False
            with open(file_path, "rb") as f:
                f.seek(128)
                magic = f.read(4)
                if magic == b"DICM":
                    return True
            # Some DICOM files lack the 128-byte preamble; try pydicom quick read
            pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            return True
        except Exception:
            return False

    @classmethod
    def scan_directory(
        cls,
        directory_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, DicomSeriesGroup]:
        """
        Recursively scans directory for DICOM files and groups them by Series Instance UID.
        """
        series_map: Dict[str, DicomSeriesGroup] = {}
        all_files: List[str] = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                all_files.append(os.path.join(root, file))

        total_files = len(all_files)
        for idx, file_path in enumerate(all_files):
            if progress_callback and (idx % 15 == 0 or idx == total_files - 1):
                progress_callback(idx + 1, total_files, os.path.basename(file_path))

            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=False)
            except Exception:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    if not hasattr(ds, "SeriesInstanceUID"):
                        continue
                except Exception:
                    continue

            series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES"))
            study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN_STUDY"))
            modality = str(getattr(ds, "Modality", "OT"))
            series_desc = str(getattr(ds, "SeriesDescription", "No Description"))
            study_desc = str(getattr(ds, "StudyDescription", "No Description"))
            patient_id = str(getattr(ds, "PatientID", "ANON_ID"))
            
            p_name = getattr(ds, "PatientName", "ANON")
            patient_name = str(p_name) if p_name else "ANON"
            study_date = str(getattr(ds, "StudyDate", ""))

            # Extract spatial metadata
            pos = None
            if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient:
                try:
                    pos = tuple(float(x) for x in ds.ImagePositionPatient)
                except Exception:
                    pos = None

            orient = None
            if hasattr(ds, "ImageOrientationPatient") and ds.ImageOrientationPatient:
                try:
                    orient = tuple(float(x) for x in ds.ImageOrientationPatient)
                except Exception:
                    orient = None

            spacing = (1.0, 1.0)
            if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
                try:
                    spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
                except Exception:
                    spacing = (1.0, 1.0)

            slice_thickness = 1.0
            if hasattr(ds, "SliceThickness") and ds.SliceThickness:
                try:
                    slice_thickness = float(ds.SliceThickness)
                except Exception:
                    slice_thickness = 1.0

            slope = 1.0
            if hasattr(ds, "RescaleSlope") and ds.RescaleSlope is not None:
                try:
                    slope = float(ds.RescaleSlope)
                except Exception:
                    slope = 1.0

            intercept = 0.0
            if hasattr(ds, "RescaleIntercept") and ds.RescaleIntercept is not None:
                try:
                    intercept = float(ds.RescaleIntercept)
                except Exception:
                    intercept = 0.0

            inst_num = 0
            if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
                try:
                    inst_num = int(ds.InstanceNumber)
                except Exception:
                    inst_num = 0

            rows = int(getattr(ds, "Rows", 0))
            cols = int(getattr(ds, "Columns", 0))
            sop_uid = str(getattr(ds, "SOPInstanceUID", os.path.basename(file_path)))

            slice_info = DicomSliceInfo(
                file_path=file_path,
                sop_uid=sop_uid,
                instance_number=inst_num,
                position=pos,
                orientation=orient,
                pixel_spacing=spacing,
                slice_thickness=slice_thickness,
                rescale_slope=slope,
                rescale_intercept=intercept,
                rows=rows,
                cols=cols
            )

            if series_uid not in series_map:
                series_map[series_uid] = DicomSeriesGroup(
                    series_uid=series_uid,
                    study_uid=study_uid,
                    modality=modality,
                    series_description=series_desc,
                    study_description=study_desc,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    study_date=study_date,
                    slices=[]
                )
            series_map[series_uid].slices.append(slice_info)

        return series_map

    @classmethod
    def load_series_volume(
        cls,
        series_group: DicomSeriesGroup,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, VolumeGeometry, SeriesMetadata]:
        """
        Loads all slices of a series group, sorts them using ImagePositionPatient & normal vector,
        converts pixel values to Hounsfield Units, and returns 3D volume, geometry, and metadata.
        """
        if not series_group.slices:
            raise ValueError(f"No slices found in series {series_group.series_uid}")

        slices = series_group.slices[:]

        # Compute slice normal from first available orientation
        row_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        col_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        norm_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for s in slices:
            if s.orientation and len(s.orientation) == 6:
                row_vec = np.array(s.orientation[:3], dtype=np.float64)
                col_vec = np.array(s.orientation[3:], dtype=np.float64)
                norm_vec = np.cross(row_vec, col_vec)
                norm_len = np.linalg.norm(norm_vec)
                if norm_len > 1e-6:
                    norm_vec /= norm_len
                break

        # Calculate projection for each slice
        has_positions = any(s.position is not None for s in slices)
        if has_positions:
            for s in slices:
                if s.position:
                    pos_arr = np.array(s.position, dtype=np.float64)
                    s.projection = float(np.dot(pos_arr, norm_vec))
                else:
                    s.projection = float(s.instance_number)
            # Sort slices along patient slice normal vector
            slices.sort(key=lambda s: s.projection)
        else:
            # Fallback to instance number
            slices.sort(key=lambda s: s.instance_number)

        # Determine reference slice dimensions
        ref_slice = slices[0]
        expected_rows = ref_slice.rows
        expected_cols = ref_slice.cols

        loaded_slices: List[np.ndarray] = []
        valid_projections: List[float] = []
        origin_pos = ref_slice.position or (0.0, 0.0, 0.0)

        total = len(slices)
        for i, s in enumerate(slices):
            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"Reading slice {i + 1}/{total}")

            try:
                ds = pydicom.dcmread(s.file_path, force=True)
                pixel_arr = ds.pixel_array.astype(np.float32)

                # If dimensions match
                if pixel_arr.ndim == 2:
                    if expected_rows == 0 or expected_cols == 0:
                        expected_rows, expected_cols = pixel_arr.shape
                    elif pixel_arr.shape != (expected_rows, expected_cols):
                        # Skip or pad slices with mismatched dimensions
                        continue

                    # Rescale to HU (Hounsfield Units) or physical units
                    slope = getattr(ds, "RescaleSlope", s.rescale_slope) or 1.0
                    intercept = getattr(ds, "RescaleIntercept", s.rescale_intercept) or 0.0
                    hu_arr = pixel_arr * float(slope) + float(intercept)

                    loaded_slices.append(hu_arr)
                    if s.position:
                        valid_projections.append(s.projection)
            except Exception as e:
                # Handle transfer syntax or corrupt slice gracefully
                continue

        if not loaded_slices:
            raise ValueError(f"Failed to decode any valid pixel data in series {series_group.series_uid}")

        # Assemble into 3D volume (Z, Y, X)
        volume_3d = np.stack(loaded_slices, axis=0).astype(np.float32)
        nz, ny, nx = volume_3d.shape

        # Calculate true slice spacing (dz)
        dz = ref_slice.slice_thickness
        if len(valid_projections) > 1:
            diffs = np.abs(np.diff(valid_projections))
            median_diff = float(np.median(diffs))
            if median_diff > 0.01:
                dz = median_diff

        dy, dx = ref_slice.pixel_spacing
        geometry = VolumeGeometry(
            shape=(nz, ny, nx),
            spacing=(max(0.1, dz), max(0.1, dy), max(0.1, dx)),
            origin=origin_pos,
            row_cosines=(float(row_vec[0]), float(row_vec[1]), float(row_vec[2])),
            col_cosines=(float(col_vec[0]), float(col_vec[1]), float(col_vec[2])),
            slice_normal=(float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))
        )

        metadata = SeriesMetadata(
            patient_id=series_group.patient_id,
            patient_name=series_group.patient_name,
            study_date=series_group.study_date,
            modality=series_group.modality,
            study_description=series_group.study_description,
            series_description=series_group.series_description,
            series_uid=series_group.series_uid,
            study_uid=series_group.study_uid,
            num_slices=nz,
            dimensions=f"{nx} x {ny} x {nz}",
            voxel_spacing=f"{dx:.2f} x {dy:.2f} x {dz:.2f} mm",
            rescale_slope=ref_slice.rescale_slope,
            rescale_intercept=ref_slice.rescale_intercept,
            is_ct=(series_group.modality.upper() == "CT")
        )

        return volume_3d, geometry, metadata
