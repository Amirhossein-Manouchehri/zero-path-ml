# Frozen L=16 model tested at L=20 and L=24

## Main finding

The model remains a useful per-realization classifier on these two larger
sizes, but its probability estimates overpredict percolation, especially
around the transition. The inference speed advantage is substantial in the
measured batch benchmark. The present model is not yet validated for replacing
solver curves in a precision finite-size-scaling analysis.

No retraining, transfer calibration, or checkpoint selection was performed.
Both the CNN and density logistic model were trained at L=16 only. Original
solver source and CNN checkpoint hashes match the fingerprints recorded before
the experiment. No research code or research data were edited.

## Data

200 independent realizations per p at each of 24 p values, from 0.0795128205
to 0.1384871795: 4,800 examples at L=20 and 4,800 at L=24. All 9,600 were used
only for evaluation. No class balancing or outcome-based filtering was used.
All configurations within each size are distinct. Seed 20261020; samples use
SeedSequence([seed,L,index]). Generation used six CPU workers and took about
188 seconds at L=20 and 512 seconds at L=24. Positive labels: 3,523 and 3,544,
respectively. The feature counts are 780 and 1,128 physical bonds.

## Classification and probability measurements

| Size | Model | Accuracy | Balanced accuracy | ROC-AUC | Brier score | Log loss | Curve MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| 20 | Frozen CNN | 89.13% | 83.68% | 0.9577 | 0.07653 | 0.24147 | 3.76 percentage points |
| 20 | Frozen density logistic | 87.00% | 81.68% | 0.9382 | 0.08904 | 0.27861 | 1.62 points |
| 24 | Frozen CNN | 89.88% | 83.92% | 0.9616 | 0.07430 | 0.23503 | 4.41 points |
| 24 | Frozen density logistic | 87.83% | 82.20% | 0.9449 | 0.08456 | 0.26703 | 2.61 points |

Curve MAE averages absolute differences between mean predicted probability
and solver fraction at each p on the same realizations. It measures agreement
with a finite solver ensemble, not error against a precisely known population
probability. The 200-realization solver fractions still have Monte Carlo noise.

The CNN's signed bias averaged over this p grid was +3.71 percentage points
at L=20 (stratified paired-bootstrap 95% interval: +2.91 to +4.47) and +4.39
points at L=24 (interval: +3.66 to +5.19). These intervals condition on a single
frozen model. They do not account for training-seed variability or certify
reliability at untested sizes. Per-p 95% intervals are not adjusted for multiple
comparisons.

The largest observed CNN discrepancy was +9.64 points at p=0.0872051 for
L=20 and +13.66 points at p=0.0974615 for L=24. A small overall curve error can
therefore hide larger transition-region errors. No p_c or nu was fitted here.

## Speed measurements

Benchmark: the same 96 realizations at each size, four at every p. One serial
solver pass, Numba configured for one thread, compared with one-thread batched
CNN inference (median of three repeats). CNN timing includes bond-to-channel
conversion. Both exclude warmup, model loading, random-bond generation and
file I/O. Solver input reconstruction is outside its timer. Models and data
were already loaded; generation workers had finished.

| Size | Serial solver, 96 realizations | One-thread CNN, 96 realizations | Measured ratio |
|---|---:|---:|---:|
| 20 | 11.414 s | 0.03580 s | 319× |
| 24 | 24.487 s | 0.04454 s | 550× |

The full 4,800-example CNN ensemble took a median 1.191 seconds at L=20 and
1.404 seconds at L=24 with four CPU threads (three repeats, encoding included).
These are local batched inference measurements, not guaranteed end-to-end
speedups or estimates of performance at L=64. A cold start or a single-example
call has different overheads. Large inference speedups do not remove model bias.

## Path lengths

Each new dataset stores a path_length array returned by the original search
order. Nonpercolating labels have length zero. Positive lengths are even and
within the solver's bound. On the 96 benchmark realizations at each size,
rerunning the original functions reproduced both labels and lengths exactly.

For L=20, positive first-found lengths ranged from 22 to 98 edges, with median
36. The evaluation reports include conditional mean length and the original
unconditional mean/L² and variance/L² at each p. These are future regression
targets, not CNN inputs; a length-prediction head has not been trained.

All ten pipeline/model tests passed, including preservation of the original
trial labels and lengths by the new adapter. Complete ensembles were checked
for correct shape, label values, unique configurations, counts per p, source
and model fingerprints, length parity and bounds. The one-thread and four-thread
CNN predictions agreed numerically on benchmark rows.

## Next decision

Use the evidence to develop a model trained across sizes, with separate held-out
sizes for model selection and final transfer tests. These L=20/24 results can
now inform development, so they should not later be advertised as untouched
final tests of an improved model. Retain larger independent sizes, such as
L=36 and 40, for final evaluation. Keep the density baseline and consider a
stronger nonlinear density-only comparator before attributing gains to learned
connectivity. For precision FSS, collect much larger independent evaluation
ensembles and propagate model discrepancy as well as sampling uncertainty.

## Outputs

- data/raw/transfer_L20_n200_seed20261020.npz
- data/raw/transfer_L24_n200_seed20261020.npz
- models/L16_transfer_v1/L20_metrics.json and L24_metrics.json
- models/L16_transfer_v1/L20_predictions.npz and L24_predictions.npz
- figures/L16_transfer_L20_L24.png

Reproduction commands and the fixed design are in transfer_protocol.md.
