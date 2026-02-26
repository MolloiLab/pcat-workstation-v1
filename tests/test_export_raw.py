"""
test_export_raw.py
Tests for export_raw.py module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.export_raw import export_voi_raw, export_combined_voi_raw, load_voi_raw, export_voi_nifti, export_combined_voi_nifti
from pipeline.pcat_segment import build_tubular_voi

# Import simple_voi_mask from test_fixtures
from tests.test_fixtures import simple_voi_mask

# Define local fixtures to avoid pytest resolution issues
@pytest.fixture
def simple_centerline():
    return np.array([[z, 32, 32] for z in range(2, 18)])

@pytest.fixture
def spacing_mm():
    """Simple voxel spacing: [z, y, x] in mm."""
    return [1.0, 0.5, 0.5]




def test_export_voi_raw_creates_files(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that both .raw and .json files exist after export."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    assert raw_path.exists()
    assert json_path.exists()


def test_export_voi_raw_file_size(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that raw file size == Z*Y*X*2 bytes (int16)."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    expected_size = np.prod(small_volume.shape) * 2  # 2 bytes per int16
    assert raw_path.stat().st_size == expected_size


def test_export_voi_raw_loadable(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that can round-trip via load_voi_raw and get same shape."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    assert loaded_volume.shape == small_volume.shape


def test_export_voi_raw_sentinel_outside_voi(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that non-VOI voxels read back as 0 (sentinel)."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    # Check that voxels outside VOI are sentinel value (0)
    outside_voi = ~simple_voi_mask
    assert np.all(loaded_volume[outside_voi] == 0)


def test_export_voi_raw_voi_values_preserved(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that known HU value inside VOI survives round-trip."""
    # Set a known value inside VOI
    test_volume = small_volume.copy()
    test_volume[simple_voi_mask] = -100.0  # fat HU
    
    raw_path, json_path = export_voi_raw(
        test_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    # Check that VOI values are preserved (approximately due to int16 conversion)
    assert np.allclose(loaded_volume[simple_voi_mask], -100.0, atol=1.0)


def test_export_voi_raw_metadata_json_valid(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that JSON is valid and has all required keys."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    # Load and validate JSON
    with open(json_path) as f:
        meta = json.load(f)
    
    required_keys = [
        "shape_zyx", "dtype", "spacing_mm_zyx", "sentinel_hu", "n_voi_voxels",
        "origin_mm_xyz", "orientation", "prefix", "raw_file"
    ]
    
    for key in required_keys:
        assert key in meta
    
    # Check specific values
    assert meta["shape_zyx"] == list(small_volume.shape)
    assert meta["sentinel_hu"] == 0
    assert meta["n_voi_voxels"] == int(simple_voi_mask.sum())


def test_export_voi_raw_prefix_in_filename(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that prefix appears in output filenames."""
    prefix = "my_test_patient"
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix=prefix
    )
    
    assert prefix in raw_path.name
    assert prefix in json_path.name


def test_export_combined_voi_raw_union(small_volume, simple_centerline, small_meta, tmp_output_dir):
    """Test that combined mask is union of individual masks."""
    spacing_mm = [1.0, 0.5, 0.5]
    
    # Create two different VOIs
    radii1 = np.full(len(simple_centerline), 2.0)
    radii2 = np.full(len(simple_centerline), 1.5)
    
    voi1 = build_tubular_voi(small_volume.shape, simple_centerline, spacing_mm, radii1)
    
    # Shift centerline for second VOI
    centerline2 = simple_centerline + np.array([0, 5, 0])  # shift in y direction
    voi2 = build_tubular_voi(small_volume.shape, centerline2, spacing_mm, radii2)
    
    vessel_masks = {"vessel1": voi1, "vessel2": voi2}
    
    # Export combined
    raw_path, json_path = export_combined_voi_raw(
        small_volume, vessel_masks, small_meta, tmp_output_dir, prefix="combined"
    )
    
    # Load and check that it's the union
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    expected_voi = voi1 | voi2
    
    # Check that voxels in expected VOI are not sentinel (0)
    assert np.all(loaded_volume[expected_voi] != 0)


def test_load_voi_raw_returns_correct_dtype(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that loaded dtype matches what was saved."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, prefix="test"
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    # Default dtype is int16
    assert loaded_volume.dtype == np.int16


def test_export_voi_raw_creates_output_dir(small_volume, simple_voi_mask, small_meta, tmp_path):
    """Test that output dir is created if it doesn't exist."""
    nested_dir = tmp_path / "nested" / "output"
    
    # Directory should not exist initially
    assert not nested_dir.exists()
    
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, nested_dir, prefix="test"
    )
    
    # Directory should be created
    assert nested_dir.exists()
    assert raw_path.exists()
    assert json_path.exists()


def test_export_voi_raw_custom_sentinel(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that custom sentinel value works."""
    custom_sentinel = -999
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, 
        prefix="test", sentinel_hu=custom_sentinel
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    # Check that voxels outside VOI have custom sentinel
    outside_voi = ~simple_voi_mask
    assert np.all(loaded_volume[outside_voi] == custom_sentinel)
    
    # Check metadata
    with open(json_path) as f:
        meta = json.load(f)
    
    assert meta["sentinel_hu"] == custom_sentinel


def test_export_voi_raw_custom_dtype(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that custom dtype works."""
    raw_path, json_path = export_voi_raw(
        small_volume, simple_voi_mask, small_meta, tmp_output_dir, 
        prefix="test", dtype=np.int32
    )
    
    loaded_volume, loaded_meta = load_voi_raw(raw_path, json_path)
    
    # Check that loaded dtype matches what was saved
    assert loaded_volume.dtype == np.int32
    
    # Check metadata
    with open(json_path) as f:
        meta = json.load(f)
    
    assert meta["dtype"] == "int32"


# ─────────────────────────────────────────────
# NIfTI export tests
# ─────────────────────────────────────────────

def test_export_voi_nifti_creates_file(simple_voi_mask, tmp_output_dir):
    """Test that export_voi_nifti creates a .nii.gz file."""
    pytest.importorskip("nibabel")
    spacing_mm = [1.0, 0.5, 0.5]
    path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    assert path.exists()


def test_export_voi_nifti_returns_path(simple_voi_mask, tmp_output_dir):
    """Test that returned path is a Path ending with .nii.gz."""
    pytest.importorskip("nibabel")
    spacing_mm = [1.0, 0.5, 0.5]
    path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    assert isinstance(path, Path)
    assert path.name.endswith(".nii.gz")


def test_export_voi_nifti_creates_output_dir(simple_voi_mask, tmp_path):
    """Test that output directory is created if it does not exist."""
    pytest.importorskip("nibabel")
    nested_dir = tmp_path / "nested" / "output"
    assert not nested_dir.exists()
    spacing_mm = [1.0, 0.5, 0.5]
    path = export_voi_nifti(simple_voi_mask, spacing_mm, nested_dir, prefix="test")
    assert nested_dir.exists()
    assert path.exists()


def test_export_voi_nifti_loadable(simple_voi_mask, tmp_output_dir):
    """Test that saved .nii.gz can be loaded back with nibabel with matching shape."""
    nib = pytest.importorskip("nibabel")
    spacing_mm = [1.0, 0.5, 0.5]
    path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    img = nib.load(str(path))
    assert img.shape == simple_voi_mask.shape


def test_export_voi_nifti_voxel_count(simple_voi_mask, tmp_output_dir):
    """Test that the number of positive voxels is preserved after round-trip."""
    nib = pytest.importorskip("nibabel")
    spacing_mm = [1.0, 0.5, 0.5]
    path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    img = nib.load(str(path))
    data = img.get_fdata()
    assert int((data > 0).sum()) == int(simple_voi_mask.sum())


def test_export_combined_voi_nifti_union(small_volume, simple_centerline, tmp_output_dir):
    """Test that combined NIfTI mask is the union of individual VOI masks."""
    nib = pytest.importorskip("nibabel")
    spacing_mm = [1.0, 0.5, 0.5]
    radii1 = np.full(len(simple_centerline), 2.0)
    radii2 = np.full(len(simple_centerline), 1.5)
    voi1 = build_tubular_voi(small_volume.shape, simple_centerline, spacing_mm, radii1)
    centerline2 = simple_centerline + np.array([0, 5, 0])
    voi2 = build_tubular_voi(small_volume.shape, centerline2, spacing_mm, radii2)
    vessel_masks = {"LAD": voi1, "LCX": voi2}
    path = export_combined_voi_nifti(vessel_masks, spacing_mm, tmp_output_dir, prefix="combined")
    assert path.exists()
    assert path.name.endswith(".nii.gz")
    img = nib.load(str(path))
    combined_data = img.get_fdata() > 0
    expected = voi1 | voi2
    assert int(combined_data.sum()) == int(expected.sum())


def test_export_voi_nifti_missing_nibabel(simple_voi_mask, tmp_output_dir, monkeypatch):
    """Test that ImportError is raised with helpful message when nibabel is missing."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "nibabel":
            raise ImportError("No module named 'nibabel'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    spacing_mm = [1.0, 0.5, 0.5]
    with pytest.raises(ImportError, match="nibabel"):
        export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")


def test_export_voi_nifti_creates_file(simple_voi_mask, spacing_mm, tmp_output_dir):
    """Test that .nii.gz file exists after call."""
    nifti_path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    assert nifti_path.exists()
    assert nifti_path.name.endswith(".nii.gz")
    assert "test_voi.nii.gz" in nifti_path.name


def test_export_voi_nifti_returns_path(simple_voi_mask, spacing_mm, tmp_output_dir):
    """Test that return value is a Path and filename ends with .nii.gz."""
    nifti_path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    assert isinstance(nifti_path, Path)
    assert nifti_path.name.endswith(".nii.gz")


def test_export_voi_nifti_creates_output_dir(simple_voi_mask, spacing_mm, tmp_path):
    """Test that output dir is created if it doesn't exist."""
    nested_dir = tmp_path / "nested" / "output"
    assert not nested_dir.exists()
    nifti_path = export_voi_nifti(simple_voi_mask, spacing_mm, nested_dir, prefix="test")
    assert nested_dir.exists()
    assert nifti_path.exists()


def test_export_voi_nifti_loadable(small_volume, simple_voi_mask, small_meta, tmp_output_dir):
    """Test that can round-trip via nibabel and get same shape and correct voxel counts."""
    import nibabel as nib
    nifti_path = export_voi_nifti(simple_voi_mask, small_meta["spacing_mm"], tmp_output_dir, prefix="test")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    # Shape must match the original mask shape (export preserves Z,Y,X order)
    assert data.shape == simple_voi_mask.shape
    assert np.sum(data == 1) == simple_voi_mask.sum()
    assert np.sum(data == 0) == np.prod(simple_voi_mask.shape) - simple_voi_mask.sum()


def test_export_voi_nifti_voxel_count(simple_voi_mask, spacing_mm, tmp_output_dir):
    """Test that voxel count in saved NIfTI equals mask.sum()."""
    import nibabel as nib
    nifti_path = export_voi_nifti(simple_voi_mask, spacing_mm, tmp_output_dir, prefix="test")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    assert int(np.sum(data == 1)) == int(simple_voi_mask.sum())


def test_export_combined_voi_nifti_union(small_volume, simple_centerline, small_meta, tmp_output_dir):
    """Test that combined NIfTI mask is the union of two individual vessel masks."""
    import nibabel as nib
    spacing_mm = small_meta["spacing_mm"]
    radii1 = np.full(len(simple_centerline), 2.0)
    radii2 = np.full(len(simple_centerline), 1.5)
    voi1 = build_tubular_voi(small_volume.shape, simple_centerline, spacing_mm, radii1)
    centerline2 = simple_centerline + np.array([0, 5, 0])
    voi2 = build_tubular_voi(small_volume.shape, centerline2, spacing_mm, radii2)
    vessel_masks = {"vessel1": voi1, "vessel2": voi2}
    nifti_path = export_combined_voi_nifti(vessel_masks, spacing_mm, tmp_output_dir, prefix="combined")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    expected_voi = voi1 | voi2
    assert int(np.sum(data == 1)) == int(expected_voi.sum())
    assert int(np.sum(data == 0)) == int(np.prod(small_volume.shape) - expected_voi.sum())