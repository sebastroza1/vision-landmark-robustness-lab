from __future__ import annotations

import numpy as np

POINT_NAMES = (
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "left_cheek",
    "right_cheek",
    "left_mouth",
    "right_mouth",
    "left_jaw",
    "right_jaw",
    "center_nose",
    "center_chin",
)
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
RIGHT_SIDE = np.array([1, 3, 5, 7, 9])
BASE_LANDMARKS = np.array(
    [
        [-0.35, 0.30],
        [0.35, 0.30],
        [-0.35, 0.48],
        [0.35, 0.48],
        [-0.45, 0.00],
        [0.45, 0.00],
        [-0.28, -0.35],
        [0.28, -0.35],
        [-0.35, -0.65],
        [0.35, -0.65],
        [0.00, 0.00],
        [0.00, -0.85],
    ],
    dtype=np.float32,
)


def _global_transform(
    landmarks: np.ndarray,
    angles_degrees: np.ndarray,
    scales: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    transformed = np.empty_like(landmarks)
    for index, (points, angle, scale, translation) in enumerate(
        zip(landmarks, angles_degrees, scales, translations, strict=True)
    ):
        radians = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        transformed[index] = (points @ rotation.T) * scale + translation
    return transformed


def generate_synthetic_landmarks(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.concatenate(
        (np.zeros(rows // 2, dtype=np.int64), np.ones(rows - rows // 2, dtype=np.int64))
    )
    generator.shuffle(labels)
    landmarks = np.repeat(BASE_LANDMARKS[None, :, :], rows, axis=0)
    landmarks += generator.normal(0.0, 0.012, size=landmarks.shape)
    for index in np.flatnonzero(labels):
        point_count = int(generator.integers(1, 3))
        points = generator.choice(RIGHT_SIDE, size=point_count, replace=False)
        directions = generator.normal(size=(point_count, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnitudes = generator.uniform(0.10, 0.22, size=(point_count, 1))
        landmarks[index, points] += directions * magnitudes
    return (
        _global_transform(
            landmarks,
            generator.uniform(-8.0, 8.0, rows),
            generator.uniform(0.90, 1.10, rows),
            generator.uniform(-0.20, 0.20, size=(rows, 2)),
        ).astype(np.float32),
        labels,
    )


def apply_condition(landmarks: np.ndarray, condition: str, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    changed = landmarks.copy()
    rows = len(changed)
    if condition == "clean":
        return changed
    if condition == "rotation_25deg":
        return _global_transform(changed, np.full(rows, 25.0), np.ones(rows), np.zeros((rows, 2)))
    if condition == "scale_1_5x":
        return _global_transform(changed, np.zeros(rows), np.full(rows, 1.5), np.zeros((rows, 2)))
    if condition == "gaussian_noise":
        changed += generator.normal(0.0, 0.05, size=changed.shape)
        return changed
    if condition == "single_point_shift":
        changed[:, 7] += np.array([0.12, -0.15])
        return changed
    if condition == "random_perturbation":
        for index in range(rows):
            points = generator.choice(len(POINT_NAMES), size=3, replace=False)
            changed[index, points] += generator.normal(0.0, 0.08, size=(3, 2))
        return changed
    if condition == "one_missing_point":
        candidates = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        missing = generator.choice(candidates, size=rows)
        changed[np.arange(rows), missing] = np.nan
        return changed
    raise ValueError(f"Unknown robustness condition: {condition}")

