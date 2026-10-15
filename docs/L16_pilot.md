# L=16 timing and class-balance pilot

Completed October 14, 2026 using the unchanged `zero-path/test_GPT_tot.py`.

## Reproduce or resume

```bash
python -m src.pilot_dataset
python -m src.analyze_pilot
```

The first command reuses a completed checkpoint, or resumes an interrupted
one. It checks the solver fingerprint and sampling settings before resuming.
Each completed realization is saved atomically in the ML repository. The
solver's compiled-function cache is also kept in the ML repository.

## Dataset

- L=16; horizontal periodic and vertical open boundaries.
- The exact 24-point p grid from the selected research script: linspace over
  0.059 to 0.159 with 40 points, filtered to 0.079 <= p <= 0.14.
- Actual endpoints: 0.0795128205 and 0.1384871795.
- 20 independent realizations per p, 480 total, all distinct configurations.
- X shape (480, 496); 340 positive and 140 negative solver labels.
- Seed 20261014. Each sample uses SeedSequence([seed, index]) so restarting
  does not change subsequent samples.
- Output: data/raw/pilot_L16_n20_seed20261014.npz.

## Timing

After solver warmup, total measured solver time was 57.76 seconds.
Mean per realization: 0.1203 seconds; median: 0.0447 seconds; maximum: 2.1210
seconds. These are elapsed times on this machine under its current load;
compilation and saving checkpoints are excluded.

A linear projection gives about 24 minutes of solver time for 12,000
realizations at this L and p distribution. This is a planning estimate,
not a guarantee, and does not predict costs at larger L.

## Probability observations

At the lowest two p values, the positive counts were 5/20 and 4/20.
At p=0.09490 and 0.09746, they were both 10/20. At the highest three p
values, all 20 examples were positive at each p. The grid therefore covers
a mixed-label region and a high-probability region. A lower-p extension would
be useful if the objective includes the near-zero tail of the whole curve.

Only 20 samples per p were used. The plotted Wilson 95% intervals show the
large sampling uncertainty; 20/20 does not establish a population probability
of exactly one. Non-monotonic sample fractions should not be interpreted as
resolved physical structure. These are solver results, not ML predictions.

## Validation and next experiment

All 480 rows were checked for binary labels, ±1 bonds, correct feature count,
finite nonnegative timings, and equal counts on the intended p grid. Six
examples were regenerated from their seeds and matched the saved features;
reconstructed solver inputs reproduced their labels. The solver SHA-256
fingerprint matches the one recorded in the original L=4 demonstration.

Next: use a separate seed for the larger L=16 dataset. Keep this timing pilot
as exploratory data, rather than treating its label summaries as a held-out
test. Fix train/validation/test assignments before training. Evaluate both
per-configuration predictions and mean predicted probabilities at each p.
The CNN for transfer to larger L is not implemented by this pilot.
