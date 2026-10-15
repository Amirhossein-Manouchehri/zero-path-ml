# Multi-size training and final L=36 test

## Outcome

A model trained across L=16,20,24,28,32 and validated using L=30 improved the
unseen L=36 probability curve compared with the original L=16-only model.
The remaining errors and limited Monte Carlo sample do not establish precision
finite-size-scaling validity. Neither p_c nor nu was estimated in this run.
No retraining or calibration followed inspection of the final L=36 results.

## Data and roles

Reuse the original L=16 data/split and the existing L=20/24 data, now explicitly
development data because the earlier transfer results had been inspected.
New data at L=28,30,32,36 contain 50 independent realizations per p on the 24-point
research grid (1,200 per size), seed 20261028. All new examples retain the solver's
first-found path length. Timing pilots use separate seed 20261027 and were
excluded from training, calibration and final testing.

| Size | Gradient-training pool | Validation | Diagnostic held-out rows | Role |
|---|---:|---:|---:|---|
| 16 | 8,400 | 1,800 | 1,800 | Original split retained |
| 20 | 3,360 | 720 | 720 | Development size |
| 24 | 3,360 | 720 | 720 | Development size |
| 28 | 840 | 168 | 192 | New training size |
| 32 | 840 | 168 | 192 | New training size |
| 30 | 0 | 1,200 | 0 | Size validation only |
| 36 | 0 | 0 | 1,200 | Final unseen-size test only |

There were 16,800 eligible gradient-training rows across five sizes. Each
epoch sampled 840 per size (4,200 total) to give equal size weight. Larger pools
were subsampled across epochs, not all consumed in every epoch. Warm-start
weights had previously been trained on the original 8,400 L=16 training rows.
No sampling p, path lengths, solver statistics or critical exponents were inputs.

Approximate six-worker generation times for the new ensembles: L=28 212 s,
L=32 421 s, L=30 287 s, L=36 1,254 s. Some realizations were much more expensive
than others; none were censored on runtime or outcome. Original research code
and research data were not edited.

## Training and calibration

The new model adds log(L/16) to the final head of the existing CNN. Its new
weights start at zero, preserving original predictions before fine-tuning.
All weights were fine-tuned. There are 13,489 trainable parameters. Training
stopped after epoch eight and selected epoch three, using mean validation
cross-entropy with equal weight per validation size, including L=30. This
score fell from 0.24043 initially to 0.21661 at selection. Training plus
calibration took approximately 171.6 seconds, overlapping test-data generation.

Validation-only calibration applies logit/T + b0 + bL log(L/16):
T=0.98973, b0=0.09368, bL=-0.50339 for the CNN. The stronger baseline uses
quadratic logistic regression on actual negative-bond fraction and log(L/16),
fit to a fixed equal-size sample of 4,200 training rows. It received the same
three-parameter validation-calibration opportunity; its T=0.94767,
b0=-0.14539, bL=0.19864. See multisize_protocol.md for the fixed design.

## Final unseen L=36 results

All 1,200 test rows, 50 per p, were excluded from fitting, selection and
calibration. There were 882 positive and 318 negative solver labels.

| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier | Log loss | Curve MAE |
|---|---:|---:|---:|---:|---:|---:|
| Original L=16 CNN | 88.33% | 78.99% | 0.9667 | 0.08541 | 0.28164 | 9.21 percentage points |
| Multi-size CNN, raw | 90.25% | 85.12% | 0.9670 | 0.06734 | 0.21328 | 3.91 points |
| Multi-size CNN, calibrated | 90.75% | 87.07% | 0.9670 | 0.06505 | 0.20520 | 2.51 points |
| Density + size, calibrated | 89.67% | 85.53% | 0.9631 | 0.07026 | 0.21743 | 2.24 points |
| Density + size, raw | 89.67% | 85.53% | 0.9631 | 0.06996 | 0.21719 | 2.13 points |

Curve MAE averages absolute prediction-minus-solver differences across the
24 p values on the same test rows. It compares with a finite solver ensemble,
not a precisely known population curve. The CNN got 1,089/1,200 labels right:
252 true negatives, 66 false positives, 45 false negatives, 837 true positives.

The calibrated CNN's average signed bias was +1.58 percentage points, with a
stratified paired-bootstrap 95% interval of +0.19 to +3.00 points. Original
L=16-only bias was +9.21 points (interval +7.75 to +10.69). Raw multi-size bias
was +3.60 points (interval +2.22 to +5.04). The new calibrated density-size
baseline bias was +1.90 points (interval +0.41 to +3.35).

The worst CNN point was p=0.0923333: solver 13/50=26%, model 45.98%, a +19.98
point discrepancy. Its paired-bootstrap interval was approximately +8.03 to
+31.63 points. The baseline also missed this point by about +21.43 points.
This calls for a larger independent audit around that p, not an assumption
that the average discrepancy describes every point or that the error is only
sampling noise. Per-p intervals are not adjusted for multiple comparisons.

Both multi-size training/size conditioning and explicit calibration improved
probability agreement. Ranking ROC-AUC changed very little compared with the
old model, so much of the practical gain concerns confidence calibration and
decision thresholds rather than a demonstrated new ability to recognize paths.
The simple baseline still matches the averaged curve slightly more closely.
These results do not prove the CNN learned connectivity or critical scaling.

## Inference speed

On the same 48 test realizations (two per p), after warmup and with one CPU
thread, a serial solver pass took 138.713 seconds. CNN batched times were
0.11472, 0.15074 and 0.14736 seconds; the median ratio was approximately 941×.
The model includes lattice encoding in its timer; solver input reconstruction
is excluded. Both exclude model loading, random-bond generation, warmup and
file I/O. There was one solver repeat and three CNN repeats. This is a local
batch benchmark, not a universal end-to-end speedup or evidence for L=64.

All 1,200 CNN predictions took about 1.73 seconds with one CPU thread, encoding
included, median of three repeats. Generation of solver labels and first-found
lengths took about 21 minutes with six workers, including checkpoint overhead.

## Verification and limitations

All twelve pipeline/model checks passed, including padding, split isolation,
label/length preservation, warm-start prediction agreement at several sizes,
and calibration behavior on synthetic validation data. New model/source data
fingerprints and exact split indices were saved. Reloading the standalone
prediction command reproduced all 1,200 saved final-test probabilities. The
benchmark sample reproduced original solver labels and lengths exactly.

Intervals condition on one frozen model and one training run. They omit
training-seed variability and do not certify error at larger sizes. Fifty
realizations per p yield wide ensemble intervals. Paper-size FSS requires larger
independent ensembles, localized discrepancy checks, the same fitting windows
and size cuts as the solver analysis, and propagation of model discrepancy
alongside Monte Carlo uncertainty. L=64, path-length prediction, p_c and nu
remain untested. New datasets store length targets; no regression head was fit.

## Use and outputs

```bash
python -m src.predict_multisize \
  --dataset data/raw/multisize_L36_n50_seed20261028.npz
```

Primary checkpoint: models/multisize_cnn_v1/cnn.pt. This contains the trained
weights, calibration, training sizes, warm-start hash and data manifest.
The same directory contains split_indices.npz, history.json,
training_summary.json, data_manifest.json and density_size_baseline.joblib.

Evaluation: models/multisize_eval_v1/L36_metrics.json and L36_predictions.npz;
summary.json also records development-size diagnostics and L=30 validation,
which are not independent final tests. Figure: figures/multisize_L36_evaluation.png.
Old checkpoints remain unchanged. Generated datasets/models are ignored by Git;
the portable public snapshot in artifacts/multisize_v1 contains the final CNN weights, test ensemble, metrics and provenance hashes. Run python -m src.demo to reproduce the CNN metrics without the research solver.
