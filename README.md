# Synthetic Landmark Robustness Lab

This repository studies how a linear baseline and a small PyTorch MLP classify two synthetic geometric-landmark patterns, then measures their sensitivity to rotation, scale, noise, point shifts, random perturbations, and missing points.

> Research engineering demonstration only. This is not a stroke detector, medical device, diagnostic model, or clinical validation. It contains no photographs, patient data, private NeuroFAST code, or thesis performance claims.

## What this demonstrates

- Deterministic generation of 12 two-dimensional geometric landmarks and explicitly synthetic labels.
- Centering, eye-axis alignment, scale normalization, symmetry features, normalized coordinates, and missing-point indicators.
- A standardized logistic-regression baseline and a small PyTorch MLP with validation-based early stopping.
- Independent train, validation, and test generations using different random seeds.
- Per-condition accuracy and macro-F1 plus a committed robustness plot and model checkpoint.
- Quick CPU mode for verification and a larger full mode that is available but not claimed as executed.

## Run

```powershell
& 'C:\Users\sebas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
.\.venv\Scripts\python.exe -m vision_landmark_robustness_lab.cli --mode quick
```

See [DATASET_CARD.md](DATASET_CARD.md), [FEATURES.md](FEATURES.md), [MODEL_CARD.md](MODEL_CARD.md), [REPRODUCE.md](REPRODUCE.md), and [RESULTS.md](RESULTS.md).

In the verified quick run, the MLP reached clean macro-F1 0.9930 and remained at 0.9930 under the tested global rotation and scale, but fell to 0.3333 with Gaussian noise or one missing point. That failure is central evidence: strong clean synthetic performance is not robustness.

## How can neural networks be used to analyse visual features, and how would you evaluate their robustness?

Landmarks convert visual geometry into structured coordinates. After a documented normalization step, a neural network can learn nonlinear combinations of coordinates, pairwise symmetry errors, and missingness indicators. Evaluation needs an untouched test set and controlled perturbation suites, not only clean accuracy: apply transformations independently, measure every condition, compare with a simpler baseline, inspect failure modes, and preserve the exact data generator and checkpoint. Synthetic robustness can validate engineering behavior but cannot establish clinical safety or real-world validity.

## Clinical boundary

A real clinical investigation would require an approved protocol, representative and consented data, expert-defined outcomes, device and site diversity, bias and subgroup analysis, prospective external validation, uncertainty and calibration analysis, privacy/security controls, and regulatory review. None of that evidence exists in this repository.

