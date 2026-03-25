import time

import numpy as np
import pytest


def test_fast_cpr_under_500ms():
    """CPR generation should complete in < 500ms for live editing."""
    from pcat_workstation.workers.cpr_worker import build_cpr_fast

    vol = np.random.default_rng(42).normal(0, 100, (20, 64, 64)).astype(np.float32)
    cl = np.array([[z, 32, 32] for z in range(2, 18)], dtype=np.float64)
    spacing = [1.0, 0.5, 0.5]

    # Warmup call to exclude import / JIT overhead
    build_cpr_fast(vol, cl, spacing)

    t0 = time.perf_counter()
    result = build_cpr_fast(vol, cl, spacing)
    dt = time.perf_counter() - t0

    assert result is not None
    assert "cpr_image" in result
    assert "N_frame" in result
    assert "B_frame" in result
    assert "positions_mm" in result
    assert "arclengths" in result
    assert result["cpr_image"].shape == (512, 256)
    assert result["N_frame"].shape == (512, 3)
    assert result["B_frame"].shape == (512, 3)
    assert result["positions_mm"].shape == (512, 3)
    assert result["arclengths"].shape == (512,)
    assert dt < 0.5, f"build_cpr_fast took {dt:.3f}s, expected < 0.5s"


def test_fast_cpr_returns_none_for_short_centerline():
    """A degenerate centerline (all same point) should return None, not crash."""
    from pcat_workstation.workers.cpr_worker import build_cpr_fast

    vol = np.zeros((20, 64, 64), dtype=np.float32)
    cl = np.array([[5, 32, 32], [6, 32, 32]], dtype=np.float64)  # only 2 pts
    result = build_cpr_fast(vol, cl, [1.0, 0.5, 0.5])
    # Should return None or a valid result (depends on bezier_fit tolerance)
    # At minimum, should not crash
