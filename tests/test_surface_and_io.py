"""
Unit tests for 3D Surface Generation and Segmentation I/O Persistence.
Verifies physical bounds alignment, file saving/reloading, and geometry validation.
"""
import os
import tempfile
import pytest
import numpy as np

from imaging.coordinate_manager import VolumeGeometry
from segmentation.model import SegmentationModel
from segmentation.surface import SegmentSurfaceExtractor
from segmentation.io import SegmentationIO


@pytest.fixture
def test_geometry():
    return VolumeGeometry(
        shape=(24, 48, 48),
        spacing=(3.0, 1.5, 1.5),
        origin=(-36.0, -36.0, -36.0)
    )


def test_vtk_3d_surface_extraction_and_geometry_bounds(test_geometry):
    model = SegmentationModel(test_geometry)
    seg = model.add_segment("Cortex", color=(0, 200, 255, 200), opacity=0.75)

    # Empty segment produces None
    actor_empty = SegmentSurfaceExtractor.create_vtk_actor_from_segment(seg, test_geometry)
    assert actor_empty is None

    # Fill an ellipsoid / cube in center
    # z: [8, 16], y: [16, 32], x: [16, 32]
    seg.mask[8:16, 16:32, 16:32] = 1
    assert seg.voxel_count > 0

    actor = SegmentSurfaceExtractor.create_vtk_actor_from_segment(seg, test_geometry)
    assert actor is not None
    assert actor.GetVisibility() == 1

    # Check color
    col = actor.GetProperty().GetColor()
    assert pytest.approx(col[0], 0.01) == 0.0
    assert pytest.approx(col[1], 0.01) == 200 / 255.0
    assert pytest.approx(col[2], 0.01) == 255 / 255.0

    # Check opacity
    assert pytest.approx(actor.GetProperty().GetOpacity(), 0.01) == 0.75

    # Check physical coordinates bounds
    bounds = actor.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    # Origin is (-36.0, -36.0, -36.0), spacing is (3.0, 1.5, 1.5)
    # x bounds should be around: -36 + 16 * 1.5 = -12 to -36 + 32 * 1.5 = +12
    assert bounds[0] < bounds[1]
    assert bounds[2] < bounds[3]
    assert bounds[4] < bounds[5]
    # Verify bounds fall strictly within volume physical envelope
    total_x_len = 48 * 1.5
    total_y_len = 48 * 1.5
    total_z_len = 24 * 3.0
    assert bounds[0] >= -36.0 - 1.0
    assert bounds[1] <= -36.0 + total_x_len + 1.0
    assert bounds[4] >= -36.0 - 1.0
    assert bounds[5] <= -36.0 + total_z_len + 1.0


def test_segmentation_save_and_reload(test_geometry):
    model = SegmentationModel(test_geometry)
    seg1 = model.add_segment("Liver", color=(255, 50, 50, 200), opacity=0.8)
    seg2 = model.add_segment("Ventricles", color=(50, 150, 255, 200), opacity=0.6)

    # Populate some voxels
    seg1.mask[5:15, 10:25, 10:25] = 1
    seg2.mask[12:18, 20:30, 20:30] = 1

    count1_orig = seg1.voxel_count
    count2_orig = seg2.voxel_count

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_study.seg.npz")

        # Save
        saved = SegmentationIO.save_to_file(
            model=model,
            file_path=save_path,
            patient_id="TEST-PATIENT-1",
            series_uid="1.2.3.4.5"
        )
        assert saved is True
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 100

        # Reload
        reloaded_model, meta = SegmentationIO.load_from_file(save_path, test_geometry)

        assert meta["patient_id"] == "TEST-PATIENT-1"
        assert meta["series_uid"] == "1.2.3.4.5"
        assert len(reloaded_model.segments) == 2

        r_seg1 = reloaded_model.get_segment(seg1.id)
        assert r_seg1 is not None
        assert r_seg1.name == "Liver"
        assert r_seg1.color == (255, 50, 50, 200)
        assert pytest.approx(r_seg1.opacity, 0.01) == 0.8
        assert r_seg1.voxel_count == count1_orig
        assert np.array_equal(r_seg1.mask, seg1.mask)

        r_seg2 = reloaded_model.get_segment(seg2.id)
        assert r_seg2 is not None
        assert r_seg2.name == "Ventricles"
        assert r_seg2.color == (50, 150, 255, 200)
        assert pytest.approx(r_seg2.opacity, 0.01) == 0.6
        assert r_seg2.voxel_count == count2_orig
        assert np.array_equal(r_seg2.mask, seg2.mask)


def test_segmentation_load_geometry_mismatch_validation(test_geometry):
    model = SegmentationModel(test_geometry)
    model.add_segment("TestSeg")

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "mismatch_test.seg.npz")
        SegmentationIO.save_to_file(model, save_path)

        # Incompatible geometry: different shape (10, 20, 20) vs (24, 48, 48)
        incompatible_geom = VolumeGeometry(shape=(10, 20, 20), spacing=(1.0, 1.0, 1.0))

        with pytest.raises(ValueError, match="Geometry mismatch"):
            SegmentationIO.load_from_file(save_path, incompatible_geom)
