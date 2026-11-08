"""
Tests for SampleDataManager and Built-in Medical Case Library.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from dicom.sample_data_manager import SampleDataManager, BUILTIN_CASES


def test_builtin_catalog_structure():
    mgr = SampleDataManager()
    cases = mgr.get_cases()
    assert len(cases) >= 5

    case_ids = [c.id for c in cases]
    assert "demo_head_ct" in case_ids
    assert "demo_chest_ct" in case_ids
    assert "slicer_chest_ct" in case_ids
    assert "mri_brain_t1" in case_ids

    # Modality and anatomical regions must be properly populated
    for c in cases:
        assert c.modality in ["CT", "MR"]
        assert len(c.anatomical_region) > 0
        assert len(c.description) > 0
        assert len(c.license_info) > 0


def test_load_demo_cases():
    mgr = SampleDataManager()

    # Test Head CT demo
    head_proc = mgr.load_case_processor("demo_head_ct")
    assert head_proc.has_data()
    assert head_proc.geometry.num_z == 64
    assert head_proc.metadata.modality == "CT"

    # Test Chest CT demo
    chest_proc = mgr.load_case_processor("demo_chest_ct")
    assert chest_proc.has_data()
    assert chest_proc.geometry.num_z == 64
    assert chest_proc.metadata.modality == "CT"
