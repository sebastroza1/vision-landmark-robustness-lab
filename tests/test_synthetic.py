from __future__ import annotations

import numpy as np

from vision_landmark_robustness_lab.synthetic import (
    apply_condition,
    generate_synthetic_landmarks,
)


def test_generation_is_deterministic_and_balanced() -> None:
    first_x, first_y = generate_synthetic_landmarks(100, seed=7)
    second_x, second_y = generate_synthetic_landmarks(100, seed=7)
    assert np.allclose(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert np.bincount(first_y).tolist() == [50, 50]


def test_missing_condition_marks_one_non_anchor_point() -> None:
    landmarks, _ = generate_synthetic_landmarks(20, seed=8)
    changed = apply_condition(landmarks, "one_missing_point", seed=9)
    assert np.isnan(changed).any(axis=2).sum() == 20
    assert not np.isnan(changed[:, :2]).any()

