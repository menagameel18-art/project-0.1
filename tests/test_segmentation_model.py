"""
Unit tests for Segmentation Model, Segment metadata, and Undo/Redo command stack.
"""
import pytest
import numpy as np
from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import Segment, SegmentationModel
from segmentation.undo_redo import UndoRedoStack


@pytest.fixture
def test_geometry():
    return VolumeGeometry(
        shape=(32, 64, 64),
        spacing=(2.5, 1.0, 1.0),
        origin=(-32.0, -32.0, -40.0)
    )


def test_segment_creation_and_attributes(test_geometry):
    model = SegmentationModel(test_geometry)
    assert len(model.segments) == 0

    seg1 = model.add_segment("Liver", color=(255, 0, 0, 200), opacity=0.8)
    assert seg1.id == 1
    assert seg1.name == "Liver"
    assert seg1.color == (255, 0, 0, 200)
    assert seg1.opacity == 0.8
    assert seg1.visible is True
    assert seg1.mask.shape == (32, 64, 64)
    assert seg1.voxel_count == 0

    seg2 = model.add_segment("Tumor")
    assert seg2.id == 2
    assert seg2.name == "Tumor"
    assert len(model.segments) == 2


def test_segment_voxel_count_and_volume(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("Organ")

    # Fill a 10x10x10 block = 1000 voxels
    seg.mask[10:20, 10:20, 10:20] = 1
    assert seg.voxel_count == 1000

    # Spacing is 2.5 x 1.0 x 1.0 mm³ = 2.5 mm³ per voxel
    # 1000 voxels = 2500 mm³ = 2.5 cm³
    vol_cm3 = seg.calculate_volume_cm3(test_geometry)
    assert pytest.approx(vol_cm3, 0.001) == 2.5


def test_multi_segment_labelmap_generation(test_geometry):
    model = SegmentationModel(test_geometry)
    seg1 = model.add_segment("Seg1")
    seg2 = model.add_segment("Seg2")

    seg1.mask[5, 10, 10] = 1
    seg2.mask[5, 20, 20] = 1

    labelmap = model.get_labelmap()
    assert labelmap.shape == (32, 64, 64)
    assert labelmap[5, 10, 10] == seg1.id
    assert labelmap[5, 20, 20] == seg2.id
    assert labelmap[0, 0, 0] == 0  # Background


def test_segment_visibility_and_deletion(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("Organ")
    seg.visible = False
    assert not seg.visible

    seg.visible = True
    assert seg.visible

    assert model.remove_segment(seg.id) is True
    assert model.get_segment(seg.id) is None
    assert len(model.segments) == 0


def test_undo_redo_stack(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("Structure")
    stack = UndoRedoStack(max_depth=10)

    assert not stack.can_undo()
    assert not stack.can_redo()

    # Stroke 1: paint at [10, 15:20, 15:20]
    old_mask1 = seg.mask.copy()
    seg.mask[10, 15:20, 15:20] = 1
    stack.record_diff(seg.id, "Stroke 1", old_mask1, seg.mask)

    assert stack.can_undo()
    assert not stack.can_redo()
    assert seg.voxel_count == 25

    # Stroke 2: paint at [10, 20:25, 20:25]
    old_mask2 = seg.mask.copy()
    seg.mask[10, 20:25, 20:25] = 1
    stack.record_diff(seg.id, "Stroke 2", old_mask2, seg.mask)
    assert seg.voxel_count == 50

    # Undo Stroke 2
    cmd = stack.undo(model)
    assert cmd is not None
    assert seg.voxel_count == 25
    assert stack.can_redo()

    # Undo Stroke 1
    cmd = stack.undo(model)
    assert cmd is not None
    assert seg.voxel_count == 0
    assert not stack.can_undo()
    assert stack.can_redo()

    # Redo Stroke 1
    cmd = stack.redo(model)
    assert cmd is not None
    assert seg.voxel_count == 25

    # Redo Stroke 2
    cmd = stack.redo(model)
    assert cmd is not None
    assert seg.voxel_count == 50
    assert not stack.can_redo()
