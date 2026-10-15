# Multi-size model: design before final L=36 evaluation

## Scientific question

Does training across L=16,20,24,28,32 improve the probability curve on an unseen
larger size? L=30 is excluded from gradient training and used for size validation.
L=36 is read for evaluation only after saving a frozen checkpoint. Do not fit
p_c or nu from this small development experiment.

Timing pilots use two realizations per p at L=28,30,32,36, seed 20261027. They
are excluded from model fitting, calibration and final evaluation. Only pilot
timings determine budgets, not their label fractions or model predictions.

Generate 50 realizations per p on the 24-point research grid for each new size
(L=28,30,32,36), seed 20261028. Each size has 1,200 examples, all with solver
labels and first-found path lengths. This is an initial budget reflecting the
cost of larger-size labels; it is not a high-precision FSS ensemble.

Reuse the original L=16 dataset and exact saved split. L=20/24 datasets were
previously inspected and now become development data, not pristine final tests.
For L=20,24,28,32, split within p 70/15/15 with seed 52, stratifying labels
where possible. L=28/32 then have 35 training, 7 validation and 8 diagnostic
test examples per p. All L=30 examples are used for size validation. Do not
feed sampling p, path lengths or solver-derived observables to the classifier.

## Model

Start from the existing L=16 CNN weights. Add one scalar, log(L/16), to its
final head, initializing the new weights at zero so initial predictions match
the original model. Keep the same bond planes, boundary masks, periodic
horizontal/open vertical padding and convolutional layers. Fine-tune all weights.

Each epoch draws 840 training examples per size without replacement within
that epoch. Rotate through the larger training pools across epochs. This gives
equal size weight, with natural class proportions, 4,200 examples per epoch.
Use CPU with two PyTorch threads, seed 42, AdamW lr=0.0003, weight_decay=0.0001,
batch size 64, maximum 20 epochs and patience five (improvement >=0.0001).
Select the lowest mean validation cross-entropy with equal weight per size,
including L=30. Allow the warm-start epoch-zero model to win if fine-tuning
never improves validation. Save all split indices and source fingerprints.

The stronger comparison is quadratic logistic regression on actual negative-
bond fraction and log(L/16). Fit it to a fixed equal-size training sample.
This controls for nonlinear density and size effects without spatial features.

For both new models, fit the same three-parameter calibration on validation
only: logit/T + b0 + bL log(L/16), with equal weight per validation size.
Bounds: log(T) in [-2,2], b0 in [-3,3], bL in [-5,5]. Primary evaluation is
calibrated; also report raw predictions. Never choose calibration or model
based on final L=36 results. This is an empirical extrapolation test, not an
assumption that calibrated probabilities remain reliable at all larger sizes.

## Final evaluation

Use 50 independent realizations per p at L=36, all 1,200 only for testing.
Compare the new calibrated and raw CNN, new calibrated and raw size-density
baseline, and frozen original L=16 CNN. Report per-realization accuracy,
balanced accuracy, ROC-AUC, Brier and log loss; per-p curve means, absolute and
signed discrepancy; paired-bootstrap intervals for model-minus-solver means.
Retain first-found path lengths for later regression; no length head is fitted.

Intervals condition on one frozen model, not training/architecture uncertainty.
Fifty realizations per p have wide Monte Carlo intervals. Do not claim precise
critical parameters or L=64 validity. L=36 remains unseen by gradient fitting,
model selection and calibration; known L=20/24 results are development evidence.

New artifacts go under models/multisize_cnn_v1 and models/multisize_eval_v1.
Old checkpoints and original research scripts/data remain unchanged.
