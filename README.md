# Zero-Path Machine Learning

This repository investigates whether machine-learning models can predict
zero-weight path percolation in the two-dimensional random-bond Ising model.

The exact labels are generated using a zero-weight path detection algorithm.
Machine-learning models are then trained to predict whether a disorder
realization contains a system-spanning zero-weight path.

## Planned workflow

1. Generate labeled random-bond configurations.
2. Train baseline classification models.
3. Evaluate performance near the percolation threshold.
4. Compare machine-learning predictions with the exact algorithm.
5. Explore neural-network models.

## Repository structure

- `src/`: Python source code
- `notebooks/`: exploratory analysis
- `data/`: generated datasets
- `models/`: trained models
- `figures/`: generated plots
- `tests/`: automated tests

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate