from __future__ import annotations

import numpy as np

from vision_landmark_robustness_lab.features import extract_features
from vision_landmark_robustness_lab.modeling import predict_mlp, train_mlp
from vision_landmark_robustness_lab.synthetic import generate_synthetic_landmarks


def test_mlp_training_smoke_test() -> None:
    train_landmarks, train_y = generate_synthetic_landmarks(160, seed=10)
    validation_landmarks, validation_y = generate_synthetic_landmarks(80, seed=11)
    train_x = extract_features(train_landmarks)
    validation_x = extract_features(validation_landmarks)
    model, scaler, history = train_mlp(train_x, train_y, validation_x, validation_y, 12, 2)
    probabilities = predict_mlp(model, scaler, validation_x)
    assert len(history) == 2
    assert probabilities.shape == (80,)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()

