# Synthetic Landmark Results

Executed locally on 2026-10-29 in quick CPU mode with Python 3.12.14, PyTorch 2.9.1+cpu, and seed 42. The independently generated splits contained 2,400 training, 600 validation, and 1,000 test samples. Early stopping retained the best MLP state during 20 completed epochs.

## Accuracy / macro-F1 by condition

| Condition | Linear baseline | PyTorch MLP |
|---|---:|---:|
| Clean | 0.942 / 0.9419 | **0.993 / 0.9930** |
| Rotation 25° | 0.945 / 0.9449 | **0.993 / 0.9930** |
| Scale 1.5× | 0.946 / 0.9459 | **0.993 / 0.9930** |
| Gaussian coordinate noise | **0.506 / 0.3499** | 0.500 / 0.3333 |
| Fixed single-point shift | 0.491 / 0.3293 | **0.494 / 0.3307** |
| Random three-point perturbation | **0.558 / 0.4633** | 0.537 / 0.4134 |
| One missing point | 0.500 / 0.3333 | 0.500 / 0.3333 |

Geometric normalization made both models stable under the tested global rotation and scale. Clean MLP performance did not translate to local-corruption robustness: noise, shifts, and missing points reduced both methods toward majority-like behavior. The linear baseline exceeded the MLP under random perturbation, a useful negative result showing that the nonlinear model is not uniformly safer.

## Evidence boundary

The checkpoint loaded successfully with PyTorch's `weights_only=True`. All values come from generator-defined synthetic patterns. They do not measure facial behavior, disease detection, patient outcomes, demographic performance, image-extraction error, or clinical validity. Full mode was not executed.

