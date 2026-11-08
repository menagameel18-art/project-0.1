"""
Tests for CoordinateManager and spatial transformations.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from imaging.coordinate_manager import CoordinateManager, VolumeGeometry


def test_volume_geometry_properties():
    geom = VolumeGeometry(
        shape=(64, 128, 256),
        spacing=(3.0, 1.5, 0.75),
        origin=(10.0, 20.0, 30.0)
    )
    assert geom.num_z == 64
    assert geom.num_y == 128
    assert geom.num_x == 256
    assert geom.dz == 3.0
    assert geom.dy == 1.5
    assert geom.dx == 0.75


def test_coordinate_roundtrip():
    geom = VolumeGeometry(
        shape=(50, 100, 100),
        spacing=(2.5, 0.8, 0.8),
        origin=(-100.0, -100.0, -50.0),
        row_cosines=(1.0, 0.0, 0.0),
        col_cosines=(0.0, 1.0, 0.0),
        slice_normal=(0.0, 0.0, 1.0)
    )
    cm = CoordinateManager(geom)

    # Test origin voxel (0, 0, 0)
    pt = cm.voxel_to_patient(0, 0, 0)
    assert pytest.approx(pt[0]) == -100.0
    assert pytest.approx(pt[1]) == -100.0
    assert pytest.approx(pt[2]) == -50.0

    # Test reverse mapping
    vox = cm.patient_to_voxel(-100.0, -100.0, -50.0)
    assert vox == (0, 0, 0)

    # Test interior voxel (20, 40, 60)
    pt_mid = cm.voxel_to_patient(20, 40, 60)
    vox_mid = cm.patient_to_voxel(*pt_mid)
    assert vox_mid == (20, 40, 60)


def test_physical_distance_calculation():
    geom = VolumeGeometry(
        shape=(30, 100, 100),
        spacing=(2.0, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0)
    )
    cm = CoordinateManager(geom)

    # Axial plane distance: 10 pixels along X -> 10 * 0.5 = 5.0 mm
    dist_ax = cm.calculate_distance_mm("axial", (0, 0), (10, 0))
    assert pytest.approx(dist_ax) == 5.0

    # Sagittal plane distance: 10 pixels along Z -> 10 * 2.0 = 20.0 mm
    dist_sag = cm.calculate_distance_mm("sagittal", (0, 0), (0, 10))
    assert pytest.approx(dist_sag) == 20.0
