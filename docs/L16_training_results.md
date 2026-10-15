# First trained L=16 model: results

Completed October 14, 2026. This run follows L16_training_protocol.md.

## Data and split

12,000 independent realizations, 500 at each of the research script's 24 p
values, from 0.0795128205 to 0.1384871795. Each row has 496 physical bond
features. All 12,000 configurations are distinct. Seed: 20261015; pilot data
were excluded. Labels: 8,746 positive, 3,254 negative.

| Portion | Realizations | Per p | Negative | Positive |
|---|---:|---:|---:|---:|
| Training | 8,400 | 350 | 2,277 | 6,123 |
| Validation | 1,800 | 75 | 485 | 1,315 |
| Test | 1,800 | 75 | 492 | 1,308 |

The original solver source fingerprint matches the pilot and L=4 reference:
469c79483ab8c86715b87c6e19d958173bc30b158c1e5dd866fb575c6c8d1708.
Parallel generation completed in approximately 492.5 elapsed seconds with
four workers. Six worker-generated examples were checked against seeded
configurations and the unmodified original trial function; all matched.

## CNN training

The architecture in src/lattice_cnn.py has 13,457 trainable parameters.
It uses bond planes and boundary masks, periodic horizontal padding,
open vertical padding, dilated convolutions, global pooling, and a scalar
negative-bond-density feature. The model receives no sampling p value.

Seed 42, CPU, ordinary unweighted binary cross-entropy, AdamW, batch size 128.
Training stopped at epoch 33; epoch 25 was selected by validation loss.
Training and final evaluation took approximately 332 seconds. Validation-only
temperature calibration gave T=1.00276, so it barely changed probabilities.
Test results were examined only after fitting and checkpoint selection.

## Fixed test-set measurements

| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier score | Log loss | Mean absolute curve discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| Constant training prevalence | 72.67% | 50.00% | 0.5000 | 0.19863 | 0.58656 | 20.78 percentage points |
| Negative-bond-fraction logistic | 85.72% | 80.60% | 0.9253 | 0.09899 | 0.30441 | 1.58 percentage points |
| Individual-bond logistic | 83.78% | 79.01% | 0.9134 | 0.10887 | 0.33663 | 1.94 percentage points |
| CNN, calibrated | 88.67% | 85.16% | 0.9565 | 0.07595 | 0.23550 | 1.62 percentage points |

Curve discrepancy is the average over p of the absolute difference between
mean predicted probability and mean solver label on the same test rows.
It is not an error against a precisely known population probability.
The CNN made 204 classification mistakes: 111 false positives and 93 false
negatives; 381 true negatives and 1,215 true positives. Calibration leaves
classification at threshold 0.5 unchanged; raw CNN log loss was 0.23547.

The CNN improved per-realization prediction over the baselines on this split,
but the density-only model matched the ensemble curve just as closely. This
does not establish that the CNN learned path connectivity: a stronger nonlinear
density-only baseline and spatial ablations would help isolate that question.
No test-driven architecture retuning was performed in this run.

## Physics interpretation and limitations

An atypical realization is still a valid example: its solver label depends on
its own bonds. The probability curve averages over realizations at fixed p,L.
More realizations improve that Monte Carlo estimate and provide more training
coverage of rare configurations. These are related but distinct benefits.

500 realizations per p were generated, with only 75 per p held out for testing.
Thus this first curve is not comparable in precision to 5,000+ independent
solver realizations per p. Wilson and paired-bootstrap intervals in the plot
show approximate test sampling uncertainty, conditional on a fixed model;
they do not capture variability across training seeds. A larger independent
evaluation ensemble is needed for high-precision curve claims.

Labels reproduce the research solver, including its search limits. Negative
labels have not been independently certified as mathematical nonexistence.
The original algorithms and research data were not modified. Model accuracy
at larger L, or p values outside the training range, has not been measured.

## Saved outputs

- data/raw/training_L16_n500_seed20261015.npz: bonds, labels, p and metadata.
- models/L16_cnn_v1/cnn.pt: trained CNN state, temperature and metadata.
- models/L16_cnn_v1/baselines.joblib: both logistic baselines.
- models/L16_cnn_v1/split.npz: exact split indices and dataset fingerprint.
- models/L16_cnn_v1/history.json: training and validation loss by epoch.
- models/L16_cnn_v1/test_predictions.npz: test indices, labels and predictions.
- models/L16_cnn_v1/metrics.json: complete measurements and per-p intervals.
- figures/L16_cnn_v1_evaluation.png: curves, residuals, calibration and losses.

All nine pipeline/model tests passed. Dataset shapes, ±1 bonds, class labels,
unique configurations, per-p counts, disjoint splits, and source fingerprint
were checked. Reloading cnn.pt reproduced all 1,800 saved test probabilities.

Source and documents are ready for Git review; generated model/data/figure
files remain ignored. Training refuses to overwrite the existing run folder.
