from __future__ import annotations

import numpy as np

from .synthetic import PAIRS, POINT_NAMES


def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.empty_like(landmarks, dtype=np.float64)
    missing = np.isnan(landmarks).any(axis=2)
    for index, points in enumerate(landmarks):
        center = np.nanmean(points, axis=0)
        centered = points - center
        if not missing[index, 0] and not missing[index, 1]:
            eye_vector = centered[1] - centered[0]
            scale = max(float(np.linalg.norm(eye_vector)), 1e-8)
            angle = np.arctan2(eye_vector[1], eye_vector[0])
        else:
            scale = max(float(np.sqrt(np.nanmean(centered**2))), 1e-8)
            angle = 0.0
        rotation = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        normalized[index] = (centered @ rotation.T) / scale
    return np.nan_to_num(normalized, nan=0.0), missing.astype(np.float64)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    normalized, missing = normalize_landmarks(landmarks)
    engineered = []
    for left, right in PAIRS:
        left_points = normalized[:, left]
        right_points = normalized[:, right]
        engineered.extend(
            (
                np.abs(left_points[:, 0] + right_points[:, 0]),
                np.abs(left_points[:, 1] - right_points[:, 1]),
                np.abs(np.linalg.norm(left_points, axis=1) - np.linalg.norm(right_points, axis=1)),
                np.abs(right_points[:, 0] - left_points[:, 0]),
            )
        )
    central_axis = np.abs(normalized[:, [10, 11], 0])
    return np.column_stack(
        (
            normalized.reshape(len(normalized), -1),
            np.column_stack(engineered),
            central_axis,
            missing,
        )
    ).astype(np.float32)


def feature_names() -> list[str]:
    names = [f"normalized_{point}_{axis}" for point in POINT_NAMES for axis in ("x", "y")]
    for left, right in PAIRS:
        pair = f"{POINT_NAMES[left]}__{POINT_NAMES[right]}"
        names.extend(
            (
                f"{pair}_mirror_x_error",
                f"{pair}_vertical_gap",
                f"{pair}_radial_gap",
                f"{pair}_horizontal_span",
            )
        )
    names.extend(("center_nose_axis_error", "center_chin_axis_error"))
    names.extend(f"missing_{point}" for point in POINT_NAMES)
    return names

