"""
Tests for pose normalization (features/normalize.py).

The autoencoder is supposed to learn "normal swimming *shape*", not "normal pixel
location". That only works if normalize_pose is invariant to where the swimmer is
in the frame (translation) and how big they appear (scale). These tests pin that
down, plus the confidence-gating behavior.
"""

from __future__ import annotations

import numpy as np

from hydro_knight.features.normalize import (
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    normalize_pose,
)


def _good_pose() -> np.ndarray:
    """
    A synthetic (17, 3) keypoint array with confident reference joints.

    Hips sit at y=100, shoulders at y=60 (40px torso); all confidences = 1.0 so
    the pose passes the reference-joint gate.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 1.0  # all keypoints confidently detected
    kpts[L_HIP] = [90.0, 100.0, 1.0]
    kpts[R_HIP] = [110.0, 100.0, 1.0]
    kpts[L_SHOULDER] = [90.0, 60.0, 1.0]
    kpts[R_SHOULDER] = [110.0, 60.0, 1.0]
    return kpts


def test_good_pose_returns_34_vector():
    v = normalize_pose(_good_pose())
    assert v is not None
    assert v.shape == (34,)
    assert v.dtype == np.float32


def test_hip_center_maps_to_origin():
    # By construction the hip center is the origin of the normalized frame,
    # so the average of the two hip keypoints must land at (0, 0).
    v = normalize_pose(_good_pose()).reshape(17, 2)
    hip_center = (v[L_HIP] + v[R_HIP]) / 2.0
    assert np.allclose(hip_center, [0.0, 0.0], atol=1e-5)


def test_translation_invariance():
    # Shifting the whole body by a constant offset must not change the output:
    # normalization removes WHERE the swimmer is.
    base = _good_pose()
    shifted = base.copy()
    shifted[:, :2] += np.array([37.0, -12.0])  # move everyone, keep confidences
    assert np.allclose(normalize_pose(base), normalize_pose(shifted), atol=1e-5)


def test_scale_invariance():
    # Doubling the swimmer's size (a closer/larger appearance) must not change
    # the output: normalization removes apparent size/distance.
    base = _good_pose()
    bigger = base.copy()
    bigger[:, :2] *= 2.0
    assert np.allclose(normalize_pose(base), normalize_pose(bigger), atol=1e-5)


def test_low_reference_confidence_drops_pose():
    # CURRENT behavior (documented intentionally): if ANY of the four reference
    # joints is below min_ref_conf, the entire pose is discarded (returns None).
    # This is exactly the "partial-pose retention" limitation tracked in PLAN.md
    # Open questions — when that fix lands, this test should be updated to assert
    # the high-confidence keypoints are kept instead.
    pose = _good_pose()
    pose[R_HIP, 2] = 0.1  # one weak reference joint
    assert normalize_pose(pose, min_ref_conf=0.3) is None


def test_degenerate_torso_drops_pose():
    # If shoulders and hips collapse onto each other, torso length ~ 0 and the
    # scale step would blow up, so the pose is rejected.
    pose = _good_pose()
    pose[L_SHOULDER, :2] = pose[L_HIP, :2]
    pose[R_SHOULDER, :2] = pose[R_HIP, :2]
    assert normalize_pose(pose) is None
