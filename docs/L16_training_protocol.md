# First L=16 CNN experiment

## Fixed before test evaluation

Generate 500 realizations at each of the 24 p values from the research script
(12,000 total). Use seed 20261015, separate from the exploratory pilot. Solver
workers import the existing functions without changing research code or data.
All stored configurations must be distinct. Each physical bond is stored once.

Within each p, split 350 training / 75 validation / 75 test realizations. Keep
approximately the original class proportions, stratifying by label where both
classes have at least two examples. Save exact row assignments and dataset
fingerprint before fitting. Reject duplicate configurations rather than allow
them across splits. Never feed the sampling p or solver-derived quantities to
the models. Evaluation is only at previously represented p values and L=16.

## Models

- Constant: the positive fraction in training data (a probability baseline).
- Density logistic: only the measured fraction of negative physical bonds.
- Bond logistic: one feature per physical bond; standardization fit on training.
- CNN: horizontal and vertical bond planes plus valid-vertical-bond, top, and
  bottom masks. Six 3×3 convolutional layers with 16 channels and dilations
  1,2,4,8,4,1; ReLU activations; circular horizontal padding and constant
  vertical padding. Concatenate global mean/max features and negative-bond
  fraction, then a 32-unit hidden layer and one output logit.

The density baseline is deliberate: predicting from negative-bond content alone
may reproduce much of the probability curve without learning connectivity.
Compare it fairly with the CNN instead of assuming the spatial model is better.

Train the CNN on CPU, fixed seed 42, AdamW lr=0.001 and weight_decay=0.0001,
batch size 128, at most 40 epochs. Select the lowest validation binary
cross-entropy, with patience eight and minimum improvement 0.0001. Use ordinary
unweighted loss to preserve the natural class frequencies. Fit a single
temperature parameter on validation predictions for calibration. Report raw
and calibrated CNN predictions; do not pick between them using the test set.

## Measurements

On the untouched test rows report accuracy at threshold 0.5, balanced accuracy,
ROC-AUC, log loss, Brier score, and confusion matrix. For each p compare mean
predicted probability against mean solver label on the same 75 realizations.
Report mean absolute curve discrepancy, solver Wilson intervals, and paired
bootstrap intervals for prediction-minus-label residuals. Bootstrap intervals
condition on a fixed model and are approximate with a label-stratified split;
they do not include uncertainty from training seeds or architecture choice.

More realizations reduce Monte Carlo uncertainty in P(p,L). An atypical
realization remains a valid classification example, because its solver label
is computed from its own bonds. Finite training data can still miss rare
patterns. The trained probability is a learned uncertainty estimate, not
intrinsic randomness in the deterministic solver label.

There are only 75 test realizations per p in this first experiment. Matching a
high-precision physics curve requires a larger independent evaluation ensemble,
e.g. 5,000+ per p. Training on L=16 does not establish transfer to larger L.

## Commands

```bash
python -m pip install -r requirements-cnn.txt -r requirements-dataset.txt
python -m src.generate_training
python -m src.train_lattice
python -m src.plot_training
```

Training creates models/L16_cnn_v1 and refuses to overwrite an existing run.
It saves baseline models, CNN weights/temperature, exact split indices,
dataset fingerprint, epoch history, test probabilities, and metrics. Generated
datasets and model files are ignored by Git; source and this protocol can be
committed. Model predictions can be made with src/predict_lattice.py.
