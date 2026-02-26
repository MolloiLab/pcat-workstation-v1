"""
test_voi_editor.py
Tests for voi_editor.py — VOIEditor class and launch_voi_editor() helper.

These tests exercise the editor in headless mode (matplotlib Agg backend),
so no display is required.  plt.show() is monkey-patched to a no-op so that
the editor does not block waiting for a window to close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — MUST be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.voi_editor import VOIEditor, launch_voi_editor

# Import simple_voi_mask fixture so pytest can discover it
from tests.test_fixtures import simple_voi_mask  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_editor(tmp_path, volume=None, voi_mask=None):
    """Return a VOIEditor with a minimal synthetic volume."""
    if volume is None:
        volume = np.zeros((20, 32, 32), dtype=np.float32)
        volume[5:15, 12:20, 12:20] = -80.0   # simulated fat ring
    if voi_mask is None:
        voi_mask = np.zeros(volume.shape, dtype=bool)
        voi_mask[5:15, 13:19, 13:19] = True
    spacing_mm = [1.0, 0.5, 0.5]
    output_path = tmp_path / "test_voi.npy"
    return VOIEditor(
        volume=volume,
        voi_mask=voi_mask,
        spacing_mm=spacing_mm,
        vessel_name="LAD",
        output_path=output_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_voi_editor_instantiates(tmp_path):
    """VOIEditor should construct without errors."""
    editor = _make_editor(tmp_path)
    assert editor is not None
    plt.close("all")


def test_voi_editor_shape_matches_volume(tmp_path, small_volume, simple_voi_mask):
    """Editor shape attribute must match input volume shape."""
    editor = VOIEditor(
        volume=small_volume,
        voi_mask=simple_voi_mask,
        spacing_mm=[1.0, 0.5, 0.5],
        vessel_name="RCA",
        output_path=tmp_path / "out.npy",
    )
    assert editor.shape == small_volume.shape
    plt.close("all")


def test_voi_editor_initial_mask_is_bool(tmp_path):
    """Mask stored in editor should always be bool dtype."""
    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.ones((10, 20, 20), dtype=np.uint8)   # pass uint8, expect bool
    editor = VOIEditor(
        volume=volume,
        voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0],
        vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    assert editor.voi_mask.dtype == bool
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Slice extraction (orientation fix)
# ─────────────────────────────────────────────────────────────────────────────

def test_coronal_slice_flipped(tmp_path, small_volume, simple_voi_mask):
    """Coronal slice must be the vertically-flipped (Z,X) view — head up."""
    editor = VOIEditor(
        volume=small_volume,
        voi_mask=simple_voi_mask,
        spacing_mm=[1.0, 0.5, 0.5],
        vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    raw = small_volume[:, editor.y_slice, :]          # (Z, X)
    flipped = editor._coronal_slice()
    np.testing.assert_array_equal(flipped, np.flipud(raw))
    plt.close("all")


def test_sagittal_slice_flipped(tmp_path, small_volume, simple_voi_mask):
    """Sagittal slice must be the vertically-flipped (Z,Y) view — head up."""
    editor = VOIEditor(
        volume=small_volume,
        voi_mask=simple_voi_mask,
        spacing_mm=[1.0, 0.5, 0.5],
        vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    raw = small_volume[:, :, editor.x_slice]          # (Z, Y)
    flipped = editor._sagittal_slice()
    np.testing.assert_array_equal(flipped, np.flipud(raw))
    plt.close("all")


def test_axial_slice_not_flipped(tmp_path, small_volume, simple_voi_mask):
    """Axial slice must NOT be flipped."""
    editor = VOIEditor(
        volume=small_volume,
        voi_mask=simple_voi_mask,
        spacing_mm=[1.0, 0.5, 0.5],
        vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    expected = small_volume[editor.z_slice, :, :]
    np.testing.assert_array_equal(editor._axial_slice(), expected)
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Paint operations
# ─────────────────────────────────────────────────────────────────────────────

def test_paint_add_sets_voxels(tmp_path):
    """Painting in add-mode should turn voxels True."""
    volume = np.zeros((20, 32, 32), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    editor.paint_mode = "add"
    editor.brush_radius = 1
    editor._paint_voxels(10, 16, 16)
    assert editor.voi_mask[10, 16, 16] is np.bool_(True)
    assert editor.voi_mask.sum() > 0
    plt.close("all")


def test_paint_remove_clears_voxels(tmp_path):
    """Painting in remove-mode should turn voxels False."""
    volume = np.zeros((20, 32, 32), dtype=np.float32)
    voi_mask = np.ones(volume.shape, dtype=bool)   # all True to start
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    editor.paint_mode = "remove"
    editor.brush_radius = 0
    editor._paint_voxels(10, 16, 16)
    assert editor.voi_mask[10, 16, 16] is np.bool_(False)
    plt.close("all")


def test_paint_brush_radius_respected(tmp_path):
    """Paint with radius=0 should only affect the single target voxel."""
    volume = np.zeros((20, 32, 32), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    editor.paint_mode = "add"
    editor.brush_radius = 0
    editor._paint_voxels(5, 8, 8)
    assert editor.voi_mask[5, 8, 8] is np.bool_(True)
    assert editor.voi_mask.sum() == 1
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Undo stack
# ─────────────────────────────────────────────────────────────────────────────

def test_undo_restores_previous_state(tmp_path):
    """After one paint + undo, mask should match the pre-paint state."""
    volume = np.zeros((20, 32, 32), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    original = editor.voi_mask.copy()
    editor._save_to_undo_stack()
    editor.paint_mode = "add"
    editor.brush_radius = 2
    editor._paint_voxels(10, 16, 16)
    assert editor.voi_mask.sum() > 0          # paint happened
    editor._undo()
    np.testing.assert_array_equal(editor.voi_mask, original)
    plt.close("all")


def test_undo_stack_max_depth(tmp_path):
    """Undo stack should cap at 20 states (deque maxlen)."""
    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    for _ in range(30):
        editor._save_to_undo_stack()
    assert len(editor.undo_stack) == 20
    plt.close("all")


def test_undo_does_nothing_on_empty_stack(tmp_path):
    """Undo on empty stack should not raise — silently a no-op."""
    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=tmp_path / "out.npy",
    )
    original = editor.voi_mask.copy()
    editor._undo()   # should not raise
    np.testing.assert_array_equal(editor.voi_mask, original)
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def test_save_creates_npy_file(tmp_path):
    """_save() must write a .npy file at output_path."""
    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    voi_mask[3:7, 8:12, 8:12] = True
    out = tmp_path / "saved.npy"
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=out,
    )
    editor._save()
    assert out.exists()
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded, voi_mask)
    plt.close("all")


def test_save_creates_nifti_file(tmp_path):
    """_save() must also write a .nii.gz file when nibabel is available."""
    nibabel = pytest.importorskip("nibabel")
    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    voi_mask[3:7, 8:12, 8:12] = True
    out = tmp_path / "saved.npy"
    editor = VOIEditor(
        volume=volume, voi_mask=voi_mask,
        spacing_mm=[1.0, 1.0, 1.0], vessel_name="LAD",
        output_path=out,
    )
    editor._save()
    nii_path = out.with_suffix(".nii.gz")
    assert nii_path.exists()
    img = nibabel.load(str(nii_path))
    assert img.shape == volume.shape
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# launch_voi_editor (headless / no display required)
# ─────────────────────────────────────────────────────────────────────────────

def test_launch_voi_editor_returns_mask(tmp_path, monkeypatch):
    """launch_voi_editor must return the VOI mask array even when the GUI is skipped."""
    # Monkeypatch plt.show to skip the blocking call
    monkeypatch.setattr(plt, "show", lambda: None)

    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    voi_mask[3:7, 8:12, 8:12] = True

    result = launch_voi_editor(
        volume=volume,
        voi_mask=voi_mask,
        vessel_name="LAD",
        output_path=tmp_path / "out.npy",
        spacing_mm=[1.0, 1.0, 1.0],
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == volume.shape
    assert result.dtype == bool
    plt.close("all")


def test_launch_voi_editor_preserves_mask_without_edits(tmp_path, monkeypatch):
    """If the user makes no edits before closing, the returned mask equals input."""
    monkeypatch.setattr(plt, "show", lambda: None)

    volume = np.zeros((10, 20, 20), dtype=np.float32)
    voi_mask = np.zeros(volume.shape, dtype=bool)
    voi_mask[3:7, 8:12, 8:12] = True
    original = voi_mask.copy()

    result = launch_voi_editor(
        volume=volume,
        voi_mask=voi_mask,
        vessel_name="LCX",
        output_path=tmp_path / "out.npy",
        spacing_mm=[1.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(result, original)
    plt.close("all")
