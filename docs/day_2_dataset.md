# Day 2: From realizations to examples

We are learning supervised binary classification: predict a 0/1 answer from
the bond configuration. The research solver supplies the training labels.
No model has been trained yet.

## 1. The research code we reuse

`../zero-path/test_GPT_tot.py` is the selected reference. Its
`run_single_trial_no_pbc` samples an `(L, L, 4)` array of ±1 values with
`P(-1)=p`, calls `enforce_edge_symmetry`, and tries boundary starting points
in both directions. `find_reachable_points_no_pbc` returns the total
percolation flag and path length. There is no middle cross-window requirement.

Despite the function name, boundaries are periodic horizontally and open
vertically. Direction indices are left, right, down, up. The algorithm tracks
accumulated weights from -L to L, performs four grid sweeps per frontier point,
and joins zero-weight segments. It rejects reused edges (including reverse
pairs), limits combined path length to `L**2 // 2`, and limits frontier
iterations to L. It retains selected paths rather than every possible path.
These labels reproduce that solver; we have not established that its negative
results prove absence of every spanning zero-weight path. Vertex revisits are
not explicitly forbidden by the edge-reuse checks.

## 2. One row of X is one physical configuration

The solver stores both directions of each bond and unused boundary entries.
For ML, we keep each physical bond once:

1. All right bonds, row by row, including the horizontal wrap: L² values.
2. All down bonds except those leaving the bottom boundary: L(L−1) values.

Thus one example has `L(2L−1)` features. For L=4, this is 28 features.
The order is fixed across examples, so a column always means the same edge.
`unpack_bonds` reverses the encoding to check labels after saving.
We use L≥3 because at L=2 the solver's coordinate representation aliases
left and right neighbors.

## 3. Pair X with y before generating another realization

`generate_bonds` → `pack_bonds` → `total_percolates`.

The adapter accepts existing bonds instead of calling the trial function,
which would sample a fresh realization. This prevents accidentally pairing
one configuration with another configuration's label.

The file stores `X` (int8, N×features), `y` (int8, N), `p_sample` (N), and
JSON metadata including lattice size, seed, boundary conditions, feature
order, and a SHA-256 fingerprint of the solver. `p_sample` is experimental
metadata, not an extra input feature. Path length is not included in X.

## 4. Generate a small demonstration dataset

From the zero-path-ml folder, using its virtual environment:

```bash
python -m src.dataset --samples 100 --L 4 --p 0.1 0.3 0.5 --seed 42
python -m unittest discover -s tests -p 'test_dataset.py'
```

The solver needs Numba as well as the existing NumPy and pandas dependencies.
Install the combined dependencies with `python -m pip install -r requirements-dataset.txt`.
The p values here are a demonstration range, not an estimate of the threshold.
They are cycled through with nearly equal sample counts. Existing output files
are never overwritten. Use `--output data/raw/another_name.npz` for another run.

```python
import json
import numpy as np

with np.load("data/raw/dataset_L4_demo.npz", allow_pickle=False) as dataset:
    X = dataset["X"]
    y = dataset["y"]
    metadata = json.loads(dataset["metadata"].item())
    print(X.shape, y.shape)
    print("Fraction labeled positive:", y.mean())
```

For 100 samples at L=4, expect shapes `(100, 28)` and `(100,)`.
The seed makes sampling repeatable in the recorded NumPy environment.

## 5. Before training

Inspect class counts and label reliability. When splitting data later, group
identical bond configurations so duplicates cannot appear in both training
and test sets. Shuffle rather than use the p-cycling row order as a split.
Choose larger lattice sizes and a scientifically useful p range only after
checking generation time and the intended research question.
