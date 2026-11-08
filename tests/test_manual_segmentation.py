"""
Unit tests for Manual Segmentation tools (Brush, Eraser, and Contour Tool).
Verifies precise 2D slice to 3D voxel coordinate mapping on Axial, Sagittal, and Coronal planes.
"""
import pytest
import numpy as np
from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import SegmentationModel
from segmentation.tools.brush_tool import BrushTool
from segmentation.tools.contour_tool import ContourTool


@pytest.fixture
def test_geometry():
    # nz=32, ny=64, nx=64
    return VolumeGeometry(
        shape=(32, 64, 64),
        spacing=(2.5, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0)
    )


def test_brush_axial_painting_and_erasing(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("TestSeg")

    brush = BrushTool(radius=3, mode="paint")

    # Paint at col=30 (X), row=40 (Y) on axial slice z=15
    brush.apply_stroke(
        mask=seg.mask,
        plane="axial",
        slice_idx=15,
        curr_pt=(30.0, 40.0),
        geometry=test_geometry
    )

    # Center voxel must be 1
    assert seg.mask[15, 40, 30] == 1
    # Adjacent voxels within radius 3 must be 1
    assert seg.mask[15, 40 + 2, 30] == 1
    assert seg.mask[15, 40, 30 + 2] == 1
    # Voxels outside radius 3 must be 0
    assert seg.mask[15, 40 + 5, 30] == 0
    # Other slices along Z must remain untouched
    assert np.all(seg.mask[14, :, :] == 0)
    assert np.all(seg.mask[16, :, :] == 0)

    # Erase center
    brush.set_mode("erase")
    brush.set_radius(1)
    brush.apply_stroke(
        mask=seg.mask,
        plane="axial",
        slice_idx=15,
        curr_pt=(30.0, 40.0),
        geometry=test_geometry
    )
    assert seg.mask[15, 40, 30] == 0
    # Outer voxels within original radius 3 still 1
    assert seg.mask[15, 40 + 2, 30] == 1


def test_brush_sagittal_mapping(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("SagSeg")

    brush = BrushTool(radius=2, mode="paint")
    nz = test_geometry.num_z  # 32

    # In sagittal slice at x=20:
    # col is Y, row is inverted Z (row = (nz - 1) - z)
    # Paint at col=35 (Y=35), row=10 -> z = 31 - 10 = 21
    brush.apply_stroke(
        mask=seg.mask,
        plane="sagittal",
        slice_idx=20,
        curr_pt=(35.0, 10.0),
        geometry=test_geometry
    )

    # Verify center voxel in 3D volume
    expected_z = (nz - 1) - 10  # 21
    expected_y = 35
    expected_x = 20

    assert seg.mask[expected_z, expected_y, expected_x] == 1
    # Verify it was only written on slice x=20
    assert np.all(seg.mask[:, :, 19] == 0)
    assert np.all(seg.mask[:, :, 21] == 0)


def test_brush_coronal_mapping(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("CorSeg")

    brush = BrushTool(radius=2, mode="paint")
    nz = test_geometry.num_z  # 32

    # In coronal slice at y=25:
    # col is X, row is inverted Z (row = (nz - 1) - z)
    # Paint at col=45 (X=45), row=7 -> z = 31 - 7 = 24
    brush.apply_stroke(
        mask=seg.mask,
        plane="coronal",
        slice_idx=25,
        curr_pt=(45.0, 7.0),
        geometry=test_geometry
    )

    expected_z = (nz - 1) - 7  # 24
    expected_y = 25
    expected_x = 45

    assert seg.mask[expected_z, expected_y, expected_x] == 1
    # Verify written on slice y=25
    assert np.all(seg.mask[:, 24, :] == 0)
    assert np.all(seg.mask[:, 26, :] == 0)


def test_contour_tool_closed_polygon_fill(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("ContourSeg")

    contour = ContourTool()
    assert not contour.is_active()

    # Draw a 10x10 square contour on axial slice z=10
    contour.start_contour("axial", 10, (20.0, 20.0))
    contour.add_point(30.0, 20.0)
    contour.add_point(30.0, 30.0)
    contour.add_point(20.0, 30.0)
    assert contour.is_active()

    # Close and rasterize
    success = contour.close_and_fill(seg.mask, mode="paint")
    assert success is True
    assert not contour.is_active()

    # Center of square (col=25, row=25) at z=10 must be filled
    assert seg.mask[10, 25, 25] == 1
    # Outside square must be 0
    assert seg.mask[10, 5, 5] == 0
    assert seg.mask[10, 45, 45] == 0
    # Other slices must remain 0
    assert np.all(seg.mask[9, :, :] == 0)
    assert np.all(seg.mask[11, :, :] == 0)
