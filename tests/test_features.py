from __future__ import annotations

import numpy as np

from vision_landmark_robustness_lab.features import extract_features, feature_names
from vision_landmark_robustness_lab.synthetic import (
    BASE_LANDMARKS,
    apply_condition,
    generate_synthetic_landmarks,
)


def test_normalized_features_are_rotation_and_scale_invariant() -> None:
    original = BASE_LANDMARKS[None, :, :]
    rotated = apply_condition(original, "rotation_25deg", seed=1)
    scaled = apply_condition(original, "scale_1_5x", seed=1)
    assert np.allclose(extract_features(original), extract_features(rotated), atol=1e-5)
    assert np.allclose(extract_features(original), extract_features(scaled), atol=1e-5)


def test_missing_landmark_features_are_finite_and_flagged() -> None:
    landmarks, _ = generate_synthetic_landmarks(12, seed=2)
    changed = apply_condition(landmarks, "one_missing_point", seed=3)
    features = extract_features(changed)
    assert features.shape == (12, len(feature_names()))
    assert np.isfinite(features).all()
    assert (features[:, -12:].sum(axis=1) == 1).all()

