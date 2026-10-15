# Frozen L=16 to L=20 and L=24 evaluation

## Design fixed before viewing transfer predictions

Use the existing L16_cnn_v1/cnn.pt checkpoint, including its L=16 validation
temperature. Freeze both weights and temperature. Compare with the saved
L=16 density-only logistic baseline, also frozen. Do not fit, calibrate, or
select checkpoints using these transfer realizations.

Generate 200 independent realizations at each of the 24 research p-grid values
for L=20 and L=24: 4,800 per size, 9,600 total. All are evaluation examples.
This is an initial transfer test, not the eventual 5,000+ realizations per
(L,p) FSS precision study. Each sample uses SeedSequence([20261020,L,index]);
the grid is cycled in index order. These seeds are separate from training and
pilot generation. Checkpoint settings include original solver and CNN hashes.

## Preserve path targets

The new ML adapter total_percolates_with_length uses the original starting
point order and early-return rule. The previous binary adapter now returns
the first element of this pair, preserving its interface. No research source
is edited. New transfer files contain path_length as an int32 array. Zero
means no path detected; a positive value is the first spanning path returned
by the reference solver, not a proven shortest path.

Check binary-label agreement, length agreement with original trials, even
positive lengths (sum of ±1 is zero), and reference path-length bound. Record
conditional mean length plus the original unconditional mean/L² and variance/L²
at each p. No path-length predictor is fitted in this experiment.

## Measurements

For each size report per-realization accuracy, balanced accuracy, ROC-AUC,
log loss, Brier score, and confusion matrix. Accuracy Wilson intervals are
approximate pooled summaries across a fixed p grid. At each p report solver
fractions with Wilson intervals, mean model probabilities, and paired-bootstrap
intervals for mean prediction-minus-label residuals (3,000 bootstrap samples).
Report mean absolute and signed curve discrepancies separately. Intervals
condition on a single frozen model; they are not certified extrapolation bounds.

Report full-ensemble CNN inference time, including lattice encoding, with four
CPU threads after warmup (three repeats). Separately compare the original
solver's labels and lengths with the new data on 96 examples (four per p),
measuring a serial solver pass and one-thread batched CNN inference on those
same examples. Exclude warmup, model loading, random sampling, and file I/O.
The benchmark is local measured inference throughput, not a universal speedup.

This does not fit p_c or nu: two new sizes and this Monte Carlo budget are a
transfer diagnostic, not a reproduction of the paper's FSS. Do not interpret
the earlier L=16 accuracy as measured on the same population or sample count.

## Commands

```bash
python -m src.generate_transfer
python -m src.evaluate_transfer
```

Generation resumes checkpoints and refuses settings/hash mismatches. Results
go to models/L16_transfer_v1, with a figure in figures/L16_transfer_L20_L24.png.
The original L=16 checkpoint is not overwritten. Source and protocol files can
be reviewed in Git; generated data and models remain ignored.
