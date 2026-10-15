"""Validate pilot data and plot solver probability and timing by p."""

import argparse
import json
import os
from pathlib import Path

from src.dataset import PROJECT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pilot_dataset import research_p_grid, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/raw/pilot_L16_n20_seed20261014.npz")
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        X, y, p, seconds = (data[key] for key in ("X", "y", "p_sample", "solver_seconds"))
        metadata = json.loads(data["metadata"].item())
    L = metadata["L"]
    assert len(y) == metadata["target_samples"] == metadata["samples"]
    assert X.shape == (len(y), L*(2*L-1))
    assert np.isin(X, [-1, 1]).all() and np.isin(y, [0, 1]).all()
    assert np.isfinite(seconds).all() and (seconds >= 0).all()
    assert np.array_equal(p, np.tile(research_p_grid(), metadata["samples_per_p"]))
    unique = len(np.unique(X, axis=0))
    report = summarize(y, p, seconds, research_p_grid())
    report["unique_configurations"] = unique
    report["L"] = L
    report["dataset"] = str(args.dataset)
    args.dataset.with_suffix(".summary.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = report["per_p"]
    ps = np.array([r["p"] for r in rows])
    probabilities = np.array([r["probability"] for r in rows])
    lower = np.array([r["wilson_95_low"] for r in rows])
    upper = np.array([r["wilson_95_high"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    axes[0].errorbar(ps, probabilities, yerr=[probabilities-lower, upper-probabilities],
                     fmt="o", capsize=3, color="#225ea8", markersize=4)
    axes[0].set(xlabel="p = probability of a negative bond", ylabel="Solver percolation fraction",
                ylim=(-0.03, 1.03), title=f"L={L}: {metadata['samples_per_p']} realizations per p")
    axes[0].text(0.03, 0.96, "95% Wilson intervals", transform=axes[0].transAxes, va="top", fontsize=9)
    axes[1].plot(ps, [r["mean_solver_seconds"] for r in rows], "o-", color="#238b45", markersize=4)
    axes[1].set(xlabel="p", ylabel="Mean solver time per realization (seconds)", title="Compilation and file I/O excluded")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure = PROJECT / "figures" / f"{args.dataset.stem}.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    print(f"Verified {len(y)} examples, {unique} unique configurations; X={X.shape}")
    for row in rows:
        print(f"p={row['p']:.5f}: {row['positive']}/{row['samples']} positive; {row['mean_solver_seconds']:.3f} s/sample")
    print(f"Saved {figure}")


if __name__ == "__main__":
    main()
