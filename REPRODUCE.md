# Reproduction Guide

Use Python 3.12. Install `requirements-lock.txt`, which selects PyTorch 2.9.1 CPU from the official PyTorch index, then install the local package with `--no-deps`. Run ruff and pytest before the quick experiment.

Quick mode uses seed 42, 2,400/600/1,000 independently generated train/validation/test rows, and at most 35 epochs. Full mode uses 20,000/4,000/5,000 rows and at most 80 epochs; it is available via `--mode full` but is not part of the verified evidence unless separately executed.

Artifacts under `reports/quick-cpu/` include the JSON run summary, per-condition CSV, training history, PyTorch state-dict checkpoint, and macro-F1 plot.

