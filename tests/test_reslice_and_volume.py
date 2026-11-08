"""
Tests for ResliceManager, VolumeProcessor, and WindowLevelManager.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from imaging.coordinate_manager import VolumeGeometry
from imaging.reslice_manager import ResliceManager
from imaging.volume_processor import VolumeProcessor
from imaging.window_level import WindowLevelManager


def test_window_level_mapping():
    wlm = WindowLevelManager(window=400.0, level=40.0)
    # Low cutoff: 40 - 200 = -160 -> mapped to 0
    # High cutoff: 40 + 200 = 240 -> mapped to 255
    # Center: 40 -> mapped to ~127

    arr = np.array([-200.0, -160.0, 40.0, 240.0, 500.0], dtype=np.float32)
    mapped = wlm.apply(arr)

    assert mapped[0] == 0
    assert mapped[1] == 0
    assert abs(int(mapped[2]) - 127) <= 1
    assert mapped[3] == 255
    assert mapped[4] == 255


def test_reslice_dimensions_and_spacing():
    nz, ny, nx = 40, 80, 100
    dz, dy, dx = 3.0, 0.8, 0.8
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    vol[10, 20, 30] = 500.0  # marked voxel

    geom = VolumeGeometry(shape=(nz, ny, nx), spacing=(dz, dy, dx))
    rm = ResliceManager(vol, geom)

    max_z, max_x, max_y = rm.get_max_slices()
    assert max_z == 40
    assert max_x == 100
    assert max_y == 80

    # Axial slice
    ax_slice, ax_spacing = rm.extract_axial_slice(10)
    assert ax_slice.shape == (80, 100)
    assert ax_spacing == (0.8, 0.8)
    assert ax_slice[20, 30] == 500.0

    # Sagittal slice (along X=30)
    sag_slice, sag_spacing = rm.extract_sagittal_slice(30)
    assert sag_slice.shape == (40, 80)
    assert sag_spacing == (3.0, 0.8)
    # Z inverted in sagittal: z=10 -> row = (40 - 1) - 10 = 29
    assert sag_slice[29, 20] == 500.0

    # Coronal slice (along Y=20)
    cor_slice, cor_spacing = rm.extract_coronal_slice(20)
    assert cor_slice.shape == (40, 100)
    assert cor_spacing == (3.0, 0.8)
    assert cor_slice[29, 30] == 500.0


def test_procedural_head_phantom_generation():
    proc = VolumeProcessor.create_procedural_head_phantom()
    assert proc.has_data()
    assert proc.volume is not None
    assert proc.volume.shape == (64, 128, 128)
    assert proc.metadata.is_ct is True

    # Check CT bone values in skull (> 800 HU)
    assert np.max(proc.volume) >= 1000.0
    # Check air background (-1000 HU)
    assert np.min(proc.volume) <= -990.0

    # Verify VTK image data conversion
    vtk_img = proc.get_vtk_image_data()
    assert vtk_img is not None
    dims = vtk_img.GetDimensions()
    assert dims == (128, 128, 64)  # (X, Y, Z) in VTK


def test_procedural_chest_phantom_generation():
    proc = VolumeProcessor.create_procedural_chest_phantom()
    assert proc.has_data()
    assert proc.volume is not None
    assert proc.volume.shape == (64, 128, 128)
    # Lung parenchyma present (-700 HU)
    assert np.any((proc.volume >= -750) & (proc.volume <= -680))
    # Ribs / Spine present (bone > 800 HU)
    assert np.any(proc.volume >= 800)
