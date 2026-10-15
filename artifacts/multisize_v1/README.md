# Portable frozen experiment

This snapshot makes the final CNN result reproducible without access to the
separate research repository. It contains the trained CNN, all 1,200 final
L=36 test realizations (50 per p), their reference labels, first-found path
lengths, and the complete recorded comparison metrics. These are ML experiment
outputs; no original research scripts are distributed here.

Run `python -m src.demo` from the repository root. No training or solver call
occurs. `model.pt` contains tensors and basic metadata and is loaded with
PyTorch's `weights_only=True`. Its numerical weights and calibration match the
original frozen checkpoint. Local paths were removed; timing columns were
omitted from the exported test data. The manifest records export checksums and
the original source fingerprints. The demo verifies the export checksums.

`results.json` includes the original experiment's baseline comparisons,
pointwise intervals and local batch benchmark. The demo recomputes the CNN
metrics and majority-class baseline; it does not rerun baseline fitting or the
solver timing benchmark. Retraining requires the separate datasets and solver
described in ../../docs/reproduction.md. This now-public test set should not be
reused as an untouched test after further tuning.
