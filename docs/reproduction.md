# Reproducing the experiments

## Public inference demo

Use Python 3.13 and install `requirements-cnn.txt`, then run from the repo root:

```bash
python -m src.demo
python -m unittest discover -s tests -v
```

The included portable snapshot reproduces final CNN accuracy, balanced
accuracy, ROC-AUC and curve MAE without the research repository. Reference-
solver integration tests skip when its source file is absent. The recorded
baseline and timing comparisons are in `artifacts/multisize_v1/results.json`;
the demo does not repeat those experiments. The complete experiment report
explains their methodology and limits.

You can also predict from a compatible NPZ containing `X` and scalar JSON
`metadata` with integer `L` (and optionally `p_sample`):

```bash
python -m src.predict_multisize \
  --model artifacts/multisize_v1/model.pt \
  --dataset artifacts/multisize_v1/L36_test.npz
```

`X` has shape (N,L*(2*L-1)) and ±1 entries: first all right bonds, then all
down bonds in row order. This command prints per-p mean probabilities. Its
optional `--output` writes probabilities to a new NPZ and refuses overwrite.

## Optional reference solver and full training

Generating new labels and reproducing training require the author's separate
`zero-path/test_GPT_tot.py` and the larger development datasets. That research
source is not bundled here. A public clone alone supports inference and model
tests, not full regeneration of solver labels.

Place the research repository beside this one, so the default solver path is
`../zero-path/test_GPT_tot.py`. Install both optional stacks:

```bash
python -m pip install -r requirements-cnn.txt -r requirements-dataset.txt
```

The adapter imports functions without running the script's main experiment.
Python/Numba caches are directed into this ML repository. It does not modify
the research source or research datasets. The selected solver source SHA-256 is
`469c79483ab8c86715b87c6e19d958173bc30b158c1e5dd866fb575c6c8d1708`.
Labels implement the solver's bounded search, including its iteration and
path-length limits; negative results do not prove path nonexistence.

In a fresh output directory, the experiment sequence is:

```bash
# 12,000 L=16 examples: 500 per p; then the original L=16 model
python -m src.generate_training --per-p 500 --seed 20261015
python -m src.train_lattice

# 4,800 examples per size; frozen L=16 transfer evaluation
python -m src.generate_transfer --L 20 24 --per-p 200 --seed 20261020
python -m src.evaluate_transfer

# 1,200 examples per size; multi-size training and frozen final evaluation
python -m src.generate_multisize --L 28 30 32 36 --per-p 50 --seed 20261028
python -m src.train_multisize
python -m src.evaluate_multisize
```

These commands can take substantial time. Existing generation checkpoints
resume only with matching settings; training scripts refuse existing run
directories. Do not remove existing results to rerun: use a separate checkout.
The published portable model is an inference export, not a replacement for all
intermediate training artifacts. Full training produces its own saved split,
history, baseline and model files under ignored `models/`.

The p grid contains 24 points from 0.0795128205 through 0.1384871795. See the
protocol documents for seeds, splits, budgets, selection and calibration.
Numerical retraining results may vary across hardware/software even with fixed
seeds. The public demo checks the frozen experiment rather than asserting that
every fresh training run will reproduce the same metrics bit for bit.

`requirements.txt` contains core scientific packages; `requirements-cnn.txt`
adds PyTorch; `requirements-dataset.txt` adds the reference solver's Numba stack.
`requirements-notebooks.txt` adds Jupyter. `requirements-environment.txt`
preserves the original notebook environment listing for provenance and is not
a portable runtime installation recommendation (it includes macOS packages).
