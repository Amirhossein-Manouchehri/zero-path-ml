# Day 3: The first classifier

The first model is logistic regression. It reads the flattened ±1 bond vector
and returns a probability for total zero-weight percolation. It is useful as a
baseline because it is fast and interpretable: it can learn one weight per
bond, but it cannot naturally represent complicated spatial interactions.

Before splitting, exact duplicate configurations are reduced to one copy. This
prevents the same lattice from appearing in both training and testing. The
remaining examples are split 80/20, stratified so both portions have similar
positive/negative proportions. `p_sample` is not given to the classifier.

We fit two models:

1. A dummy classifier that always predicts the most common training label.
2. Logistic regression using the 28 bond values for L=4.

The dummy result is the minimum comparison that a useful classifier should
beat. Metrics include ordinary accuracy, balanced accuracy (average recall for
the two classes), ROC-AUC (ranking positives above negatives), and a confusion
matrix in the order `[[true negatives, false positives], [false negatives,
true positives]]`.

Run the demonstration with:

```bash
python -m src.train_baseline
```

The 100-example dataset is only a pipeline demonstration. A single 20-example
test set gives an unstable estimate, and model selection must not be based on
it. The next scientific dataset needs many more examples, emphasis near the
transition, and a separate validation set for choosing models.
