"""
Unit tests for Semi-Automatic Segmentation algorithms (Thresholding and Region Growing).
Verifies intensity limits, CT HU values vs MRI values, and 2D/3D connectivity.
"""
import pytest
import numpy as np
from imaging.coordinate_manager import VolumeGeometry
from segmentation.manager import SegmentationManager
from segmentation.algorithms.threshold import ThresholdSegmentation
from segmentation.algorithms.region_growing import RegionGrowingSegmentation


@pytest.fixture
def test_geometry():
    return VolumeGeometry(shape=(20, 40, 40), spacing=(2.0, 1.0, 1.0))


@pytest.fixture
def synthetic_volume(test_geometry):
    # Base background: -1000 HU (Air)
    vol = np.full(test_geometry.shape, -1000.0, dtype=np.float32)
    # Sphere of Soft Tissue: 40 HU around center (10, 20, 20)
    zz, yy, xx = np.ogrid[:20, :40, :40]
    r_sq = (zz - 10)**2 + (yy - 20)**2 + (xx - 20)**2
    vol[r_sq <= 36] = 40.0
    # Bone shell: 800 HU at radius 7 to 9
    vol[(r_sq > 36) & (r_sq <= 81)] = 800.0
    return vol


def test_threshold_bone_segmentation(test_geometry, synthetic_volume):
    algo = ThresholdSegmentation()
    # Threshold for bone: 200 HU to 1500 HU
    mask = algo.execute(
        volume=synthetic_volume,
        geometry=test_geometry,
        parameters={"min_val": 200.0, "max_val": 1500.0, "slice_only": False}
    )

    assert mask.shape == synthetic_volume.shape
    # Soft tissue (40 HU) must be 0
    assert mask[10, 20, 20] == 0
    # Air (-1000 HU) must be 0
    assert mask[0, 0, 0] == 0
    # Bone region (800 HU) must be 1
    # radius ~7 voxels along X: x=27
    assert mask[10, 20, 27] == 1


def test_threshold_slice_only(test_geometry, synthetic_volume):
    algo = ThresholdSegmentation()
    # Segment soft tissue only on axial slice z=10
    mask = algo.execute(
        volume=synthetic_volume,
        geometry=test_geometry,
        parameters={
            "min_val": 20.0,
            "max_val": 60.0,
            "slice_only": True,
            "plane": "axial",
            "slice_idx": 10
        }
    )

    assert mask[10, 20, 20] == 1
    # Other slices along Z must remain completely 0
    assert np.all(mask[9, :, :] == 0)
    assert np.all(mask[11, :, :] == 0)


def test_region_growing_from_seed(test_geometry, synthetic_volume):
    algo = RegionGrowingSegmentation()
    # Seed inside soft tissue: (10, 20, 20) with value 40.0 HU
    mask = algo.execute(
        volume=synthetic_volume,
        geometry=test_geometry,
        parameters={
            "seed_point": (10, 20, 20),
            "tolerance": 25.0,  # 40 +/- 25 -> [15, 65]
            "connectivity": 6,
            "slice_only": False
        }
    )

    # Center soft tissue must be segmented
    assert mask[10, 20, 20] == 1
    assert mask[10, 22, 22] == 1
    # Bone (800 HU) must NOT be segmented
    assert mask[10, 20, 27] == 0
    # Air (-1000 HU) must NOT be segmented
    assert mask[0, 0, 0] == 0


def test_manager_preview_and_apply_workflow(test_geometry, synthetic_volume):
    manager = SegmentationManager()
    manager.set_geometry(test_geometry)

    seg = manager.get_active_segment()
    assert seg is not None
    assert seg.voxel_count == 0

    # Execute threshold synchronously via provider
    mask = manager.threshold_algo.execute(
        volume=synthetic_volume,
        geometry=test_geometry,
        parameters={"min_val": 30.0, "max_val": 50.0, "slice_only": False}
    )

    # Set as preview
    manager.preview_mask = mask
    assert manager.preview_mask is not None
    assert np.count_nonzero(manager.preview_mask) > 0

    # Apply preview to active segment
    applied = manager.apply_preview(combine_mode="union")
    assert applied is True
    assert manager.preview_mask is None  # Cleared after apply
    assert seg.voxel_count > 0
    assert manager.undo_stack.can_undo()

    # Undo application
    manager.undo()
    assert seg.voxel_count == 0
