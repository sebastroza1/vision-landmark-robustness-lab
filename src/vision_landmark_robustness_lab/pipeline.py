from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features, feature_names
from .modeling import classification_metrics, predict_mlp, train_mlp
from .synthetic import apply_condition, generate_synthetic_landmarks

CONDITIONS = (
    "clean",
    "rotation_25deg",
    "scale_1_5x",
    "gaussian_noise",
    "single_point_shift",
    "random_perturbation",
    "one_missing_point",
)
MODES = {
    "quick": {"train": 2400, "validation": 600, "test": 1000, "epochs": 35},
    "full": {"train": 20000, "validation": 4000, "test": 5000, "epochs": 80},
}


def run(output_dir: str | Path, mode: str = "quick", seed: int = 42) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    configuration = MODES[mode]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_landmarks, train_y = generate_synthetic_landmarks(configuration["train"], seed)
    validation_landmarks, validation_y = generate_synthetic_landmarks(
        configuration["validation"], seed + 1
    )
    test_landmarks, test_y = generate_synthetic_landmarks(configuration["test"], seed + 2)
    train_x = extract_features(train_landmarks)
    validation_x = extract_features(validation_landmarks)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    baseline.fit(train_x, train_y)
    mlp, mlp_scaler, history = train_mlp(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed,
        configuration["epochs"],
    )
    records = []
    for condition_index, condition in enumerate(CONDITIONS):
        condition_landmarks = apply_condition(
            test_landmarks, condition, seed + 100 + condition_index
        )
        condition_features = extract_features(condition_landmarks)
        probabilities = {
            "linear_baseline": baseline.predict_proba(condition_features)[:, 1],
            "pytorch_mlp": predict_mlp(mlp, mlp_scaler, condition_features),
        }
        for model_name, values in probabilities.items():
            records.append(
                {
                    "condition": condition,
                    "model": model_name,
                    **classification_metrics(test_y, values),
                }
            )
    results = pd.DataFrame(records)
    results.to_csv(output / "robustness_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "input_features": train_x.shape[1],
            "feature_names": feature_names(),
            "class_names": ["balanced_pattern", "perturbed_pattern"],
            "synthetic_only": True,
            "seed": seed,
        },
        output / "landmark_mlp.pt",
    )
    clean = results[results["condition"] == "clean"].set_index("model")
    payload = {
        "status": "completed",
        "mode": mode,
        "seed": seed,
        "synthetic_only": True,
        "not_a_clinical_model": True,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "rows": {
            "train": configuration["train"],
            "validation": configuration["validation"],
            "test": configuration["test"],
        },
        "features": train_x.shape[1],
        "training_epochs_completed": len(history),
        "clean_test": {
            model: {
                "accuracy": float(clean.loc[model, "accuracy"]),
                "macro_f1": float(clean.loc[model, "macro_f1"]),
            }
            for model in clean.index
        },
        "conditions": list(CONDITIONS),
    }
    (output / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_robustness(results, output / "robustness_macro_f1.png")
    return payload


def _plot_robustness(results: pd.DataFrame, path: Path) -> None:
    pivot = results.pivot(index="condition", columns="model", values="macro_f1").loc[
        list(CONDITIONS)
    ]
    axis = pivot.plot(kind="bar", figsize=(11, 6), ylim=(0.0, 1.0))
    axis.set(title="Synthetic landmark robustness", xlabel="Condition", ylabel="Macro-F1")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Model")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)

