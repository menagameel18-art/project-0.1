"""
Tests for DICOM Loader, slice sorting, and series volume assembly.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import pytest
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from dicom.dicom_loader import DicomLoader, DicomSeriesGroup, DicomSliceInfo


def _create_synthetic_dicom_file(
    file_path: str,
    series_uid: str,
    instance_num: int,
    z_pos: float,
    pixel_data: np.ndarray,
    slope: float = 1.0,
    intercept: float = -1024.0
):
    """Helper to create a valid DICOM file on disk."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = f"{series_uid}.{instance_num}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "TEST-123"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "CT"
    ds.SeriesDescription = "Synthetic CT Series"
    ds.StudyDescription = "Test Study"
    ds.InstanceNumber = instance_num
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept

    # Pixel data
    ds.Rows, ds.Columns = pixel_data.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_data.astype(np.uint16).tobytes()

    ds.save_as(file_path, write_like_original=False)


def test_dicom_file_detection_and_assembly():
    with tempfile.TemporaryDirectory() as tmpdir:
        series_uid = "1.2.826.0.1.test.series"
        
        # Write 5 slices out of order: z = 0, 8, 4, 2, 6
        z_positions = [0.0, 8.0, 4.0, 2.0, 6.0]
        for i, z in enumerate(z_positions):
            fpath = os.path.join(tmpdir, f"slice_{i}.dcm")
            # Create a 32x32 slice where intensity depends on z
            slice_pixels = np.full((32, 32), int(z * 10 + 1024), dtype=np.uint16)
            _create_synthetic_dicom_file(
                file_path=fpath,
                series_uid=series_uid,
                instance_num=i + 1,
                z_pos=z,
                pixel_data=slice_pixels,
                slope=1.0,
                intercept=-1024.0
            )

        # 1. Test directory scanning
        series_map = DicomLoader.scan_directory(tmpdir)
        assert series_uid in series_map
        series_group = series_map[series_uid]
        assert series_group.slice_count == 5

        # 2. Test volume loading and slice sorting along normal
        vol, geom, meta = DicomLoader.load_series_volume(series_group)
        assert vol.shape == (5, 32, 32)
        assert geom.num_z == 5
        assert geom.num_y == 32
        assert geom.num_x == 32

        # Check slices were sorted in ascending order of z: 0, 2, 4, 6, 8
        # Since pixel = z * 10 + 1024 and HU = pixel - 1024 = z * 10:
        expected_hus = [0.0, 20.0, 40.0, 60.0, 80.0]
        for s_idx, exp_hu in enumerate(expected_hus):
            assert pytest.approx(vol[s_idx, 0, 0]) == exp_hu

        # Check slice spacing (dz) derived from delta z
        assert pytest.approx(geom.dz) == 2.0
