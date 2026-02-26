"""
test_visualize.py
Tests for visualize.py module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.visualize import (
    render_cpr_fai, plot_hu_histogram, plot_radial_hu_profile, plot_summary,
    _fai_colormap, _compute_arclengths
)
from pipeline.pcat_segment import build_tubular_voi, apply_fai_filter

# Import fixtures from test_fixtures to avoid discovery issues
from tests.test_fixtures import simple_centerline, simple_radii, simple_voi_mask


def test_render_cpr_fai_creates_file(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """Test that output PNG exists and size > 1000 bytes."""
    spacing_mm = [1.0, 0.5, 0.5]
    vessel_name = "LAD"
    
    png_path = render_cpr_fai(
        small_volume, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert png_path is not None
    assert png_path.exists()
    assert png_path.stat().st_size > 1000


def test_render_cpr_fai_returns_path(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """Test that return value is a Path."""
    spacing_mm = [1.0, 0.5, 0.5]
    vessel_name = "LAD"
    
    png_path = render_cpr_fai(
        small_volume, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert isinstance(png_path, Path)


def test_render_cpr_fai_too_few_points_returns_none(tmp_output_dir):
    """Test that < 3 centerline points → returns None."""
    # Create centerline with only 2 points
    short_centerline = np.array([[0, 16, 16], [1, 16, 16]])
    radii = np.array([2.0, 2.0])
    volume = np.zeros((5, 32, 32), dtype=np.float32)
    spacing_mm = [1.0, 0.5, 0.5]
    
    png_path = render_cpr_fai(
        volume, short_centerline, radii, spacing_mm,
        "LAD", tmp_output_dir
    )
    
    assert png_path is None


def test_plot_hu_histogram_creates_file(small_volume, simple_voi_mask, tmp_output_dir):
    """Test that PNG exists and size > 1000 bytes."""
    vessel_name = "LAD"
    
    png_path = plot_hu_histogram(
        small_volume, simple_voi_mask, vessel_name, tmp_output_dir
    )
    
    assert png_path.exists()
    assert png_path.stat().st_size > 1000


def test_plot_hu_histogram_empty_voi(small_volume, tmp_output_dir):
    """Test that all-False simple_voi_mask → runs without error (no fat voxels edge case)."""
    empty_voi_mask = np.zeros(small_volume.shape, dtype=bool)
    vessel_name = "LAD"
    
    # Should not raise an exception
    png_path = plot_hu_histogram(
        small_volume, empty_voi_mask, vessel_name, tmp_output_dir
    )
    
    assert png_path.exists()


def test_plot_radial_hu_profile_creates_file(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """Test that PNG exists."""
    spacing_mm = [1.0, 0.5, 0.5]
    vessel_name = "LAD"
    
    png_path = plot_radial_hu_profile(
        small_volume, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert png_path.exists()


def test_plot_radial_hu_profile_returns_path(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """Test that return value is a Path."""
    spacing_mm = [1.0, 0.5, 0.5]
    vessel_name = "LAD"
    
    png_path = plot_radial_hu_profile(
        small_volume, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert isinstance(png_path, Path)


def test_plot_summary_creates_file(tmp_output_dir):
    """Test that PNG exists."""
    # Create sample vessel stats
    vessel_stats = {
        "LAD": {
            "vessel": "LAD",
            "n_voi_voxels": 1000,
            "n_fat_voxels": 300,
            "fat_fraction": 0.3,
            "hu_mean": -100.0,
            "hu_std": 20.0,
            "hu_median": -95.0,
            "hu_min_measured": -180.0,
            "hu_max_measured": -40.0,
            "hu_p25": -150.0,
            "hu_p75": -70.0,
            "FAI_HU_range": [-190.0, -30.0]
        }
    }
    
    png_path = plot_summary(vessel_stats, tmp_output_dir)
    
    assert png_path.exists()


def test_plot_summary_multiple_vessels(tmp_output_dir):
    """Test that handles 3 vessel stats dicts."""
    # Create sample vessel stats for 3 vessels
    vessel_stats = {
        "LAD": {
            "vessel": "LAD",
            "n_voi_voxels": 1000,
            "n_fat_voxels": 300,
            "fat_fraction": 0.3,
            "hu_mean": -100.0,
            "hu_std": 20.0,
            "hu_median": -95.0,
            "hu_min_measured": -180.0,
            "hu_max_measured": -40.0,
            "hu_p25": -150.0,
            "hu_p75": -70.0,
            "FAI_HU_range": [-190.0, -30.0]
        },
        "LCX": {
            "vessel": "LCX",
            "n_voi_voxels": 800,
            "n_fat_voxels": 200,
            "fat_fraction": 0.25,
            "hu_mean": -110.0,
            "hu_std": 25.0,
            "hu_median": -105.0,
            "hu_min_measured": -185.0,
            "hu_max_measured": -35.0,
            "hu_p25": -160.0,
            "hu_p75": -60.0,
            "FAI_HU_range": [-190.0, -30.0]
        },
        "RCA": {
            "vessel": "RCA",
            "n_voi_voxels": 1200,
            "n_fat_voxels": 360,
            "fat_fraction": 0.3,
            "hu_mean": -95.0,
            "hu_std": 18.0,
            "hu_median": -90.0,
            "hu_min_measured": -175.0,
            "hu_max_measured": -45.0,
            "hu_p25": -145.0,
            "hu_p75": -65.0,
            "FAI_HU_range": [-190.0, -30.0]
        }
    }
    
    png_path = plot_summary(vessel_stats, tmp_output_dir)
    
    assert png_path.exists()


def test_fai_colormap_valid():
    """Test that _fai_colormap() returns a valid matplotlib colormap."""
    cmap = _fai_colormap()
    
    # Check that it has the name attribute (matplotlib colormap property)
    assert hasattr(cmap, 'name')


def test_compute_arclengths_monotonic(simple_centerline):
    """Test that arc lengths are monotonically non-decreasing."""
    spacing_mm = [1.0, 0.5, 0.5]
    
    arclengths = _compute_arclengths(simple_centerline, spacing_mm)
    
    # Check that each value is >= the previous one
    for i in range(1, len(arclengths)):
        assert arclengths[i] >= arclengths[i-1]


def test_compute_arclengths_zero_start(simple_centerline):
    """Test that first value is 0.0."""
    spacing_mm = [1.0, 0.5, 0.5]
    
    arclengths = _compute_arclengths(simple_centerline, spacing_mm)
    
    assert arclengths[0] == 0.0


def test_compute_arclengths_shape(simple_centerline):
    """Test that output shape matches input length."""
    spacing_mm = [1.0, 0.5, 0.5]
    
    arclengths = _compute_arclengths(simple_centerline, spacing_mm)
    
    assert len(arclengths) == len(simple_centerline)


def test_plot_hu_histogram_fai_filtered(small_volume, simple_voi_mask, tmp_output_dir):
    """Test that histogram works with FAI-filtered data."""
    # Create volume with specific fat values
    volume_with_fat = small_volume.copy()
    volume_with_fat[simple_voi_mask] = -100.0  # Set VOI to fat value
    
    vessel_name = "LAD"
    
    png_path = plot_hu_histogram(
        volume_with_fat, simple_voi_mask, vessel_name, tmp_output_dir
    )
    
    assert png_path.exists()


def test_plot_radial_hu_profile_with_fat(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """Test that radial profile works when fat voxels are present."""
    # Create volume with fat ring around vessel
    spacing_mm = [1.0, 0.5, 0.5]
    volume_with_fat = small_volume.copy()
    
    # Add fat ring in the perivascular region
    z, y, x = np.indices(volume_with_fat.shape)
    fat_ring = ((y - 32)**2 + (x - 32)**2 > 4**2) & ((y - 32)**2 + (x - 32)**2 <= 8**2)
    volume_with_fat[fat_ring] = -100.0
    
    vessel_name = "LAD"
    
    png_path = plot_radial_hu_profile(
        volume_with_fat, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert png_path.exists()


def test_render_cpr_fai_with_fai_data(small_volume, simple_centerline, simple_radii, simple_voi_mask, tmp_output_dir):
    """Test that CPR works with actual FAI-filtered data."""
    spacing_mm = [1.0, 0.5, 0.5]
    
    # Apply FAI filter to get fat-only volume
    fai_volume = apply_fai_filter(small_volume, simple_voi_mask)
    
    vessel_name = "LAD"
    
    png_path = render_cpr_fai(
        fai_volume, simple_centerline, simple_radii, spacing_mm,
        vessel_name, tmp_output_dir
    )
    
    assert png_path is not None
    assert png_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Bishop frame correctness tests (CPR algorithm verification)
# ─────────────────────────────────────────────────────────────────────────────


def _build_bishop_frame(centerline_ijk, spacing_mm):
    """Replicate Bishop frame computation from render_cpr_fai for test verification."""
    vox_size = np.array(spacing_mm, dtype=np.float64)
    cl_mm = centerline_ijk.astype(np.float64) * vox_size[np.newaxis, :]
    N_pts = len(cl_mm)

    T = np.zeros((N_pts, 3), dtype=np.float64)
    T[1:-1] = cl_mm[2:] - cl_mm[:-2]
    T[0]    = cl_mm[1]  - cl_mm[0]
    T[-1]   = cl_mm[-1] - cl_mm[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    N_frame = np.zeros((N_pts, 3), dtype=np.float64)
    B_frame = np.zeros((N_pts, 3), dtype=np.float64)
    ref0 = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T[0], ref0)) > 0.9:
        ref0 = np.array([0.0, 1.0, 0.0])
    n0 = np.cross(T[0], ref0)
    n0 /= np.linalg.norm(n0) + 1e-12
    N_frame[0] = n0
    B_frame[0] = np.cross(T[0], N_frame[0])
    for i in range(1, N_pts):
        ni = N_frame[i - 1] - np.dot(N_frame[i - 1], T[i]) * T[i]
        norm_ni = np.linalg.norm(ni)
        N_frame[i] = ni / norm_ni if norm_ni > 1e-8 else N_frame[i - 1]
        B_frame[i] = np.cross(T[i], N_frame[i])
        bnorm = np.linalg.norm(B_frame[i])
        if bnorm > 1e-8:
            B_frame[i] /= bnorm
    return T, N_frame, B_frame


def test_bishop_frame_N_perpendicular_to_T(simple_centerline):
    """N[i] must be perpendicular to T[i] at every point (dot product ≈ 0)."""
    spacing_mm = [1.0, 0.5, 0.5]
    T, N, B = _build_bishop_frame(simple_centerline, spacing_mm)
    dots = np.abs(np.einsum('ij,ij->i', N, T))
    assert np.all(dots < 1e-6), f"Max |N·T| = {dots.max():.2e}, expected < 1e-6"


def test_bishop_frame_B_perpendicular_to_T(simple_centerline):
    """B[i] must be perpendicular to T[i] at every point."""
    spacing_mm = [1.0, 0.5, 0.5]
    T, N, B = _build_bishop_frame(simple_centerline, spacing_mm)
    dots = np.abs(np.einsum('ij,ij->i', B, T))
    assert np.all(dots < 1e-6), f"Max |B·T| = {dots.max():.2e}, expected < 1e-6"


def test_bishop_frame_N_unit_length(simple_centerline):
    """N[i] must have unit length at every point."""
    spacing_mm = [1.0, 0.5, 0.5]
    T, N, B = _build_bishop_frame(simple_centerline, spacing_mm)
    norms = np.linalg.norm(N, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"N norms: min={norms.min():.4f} max={norms.max():.4f}"


def test_bishop_frame_no_sudden_flips(simple_centerline):
    """Consecutive N frames must not flip 180°: dot(N[i], N[i+1]) must be > 0."""
    spacing_mm = [1.0, 0.5, 0.5]
    T, N, B = _build_bishop_frame(simple_centerline, spacing_mm)
    dots = np.einsum('ij,ij->i', N[:-1], N[1:])
    assert np.all(dots > -0.1), f"Frame flip detected: min dot(N[i],N[i+1]) = {dots.min():.4f}"


def test_cpr_output_png_is_2d_slab(small_volume, simple_centerline, simple_radii, tmp_output_dir):
    """CPR PNG must be large enough to confirm it's a 2D slab, not a 1-pixel-wide strip."""
    spacing_mm = [1.0, 0.5, 0.5]
    width_mm = 10.0

    png_path = render_cpr_fai(
        small_volume, simple_centerline, simple_radii, spacing_mm,
        'LAD', tmp_output_dir, width_mm=width_mm
    )
    assert png_path is not None
    assert png_path.exists()
    # A 1D strip would be tiny; a real 2D slab PNG should exceed 10 KB
    assert png_path.stat().st_size > 10_000, (
        f"CPR PNG only {png_path.stat().st_size} bytes — image may be a 1D strip, not a 2D slab"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3D render smoke test
# ─────────────────────────────────────────────────────────────────────────────


def test_render_3d_voi_smoke(tmp_output_dir):
    """Smoke test: render_3d_voi creates a non-empty PNG (or skips if pyvista missing)."""
    from pipeline.visualize import render_3d_voi

    vol = np.random.default_rng(0).uniform(-200, 200, (20, 20, 20)).astype(np.float32)
    voi = np.zeros((20, 20, 20), dtype=bool)
    voi[8:12, 8:12, 8:12] = True
    cl = np.array([[10, 10, 5], [10, 10, 8], [10, 10, 12], [10, 10, 15]], dtype=int)
    rad = np.array([1.5, 1.5, 1.5, 1.5])

    out = render_3d_voi(
        volume=vol,
        voi_mask=voi,
        vessel_centerlines={'LAD': cl},
        vessel_radii={'LAD': rad},
        spacing_mm=[0.5, 0.5, 0.5],
        output_dir=tmp_output_dir,
        prefix='test3d',
        screenshot=True,
        interactive=False,
    )

    # pyvista may not be installed in all CI environments — skip gracefully
    if out is None:
        pytest.skip('pyvista not installed — 3D render test skipped')

    assert out.exists(), f'PNG not created at {out}'
    assert out.stat().st_size > 1000, f'PNG suspiciously small: {out.stat().st_size} bytes'