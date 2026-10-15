"""Time a resumable L=16 pilot on the research p grid.

Run: python -m src.pilot_dataset
Each completed realization is checkpointed. Repeating the command resumes it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.dataset import DEFAULT_SOLVER, PROJECT, generate_bonds, load_solver, pack_bonds, total_percolates


def research_p_grid():
    p = np.linspace(0.109 - 0.05, 0.109 + 0.05, 40)
    return p[(p >= 0.079) & (p <= 0.14)]


def save_checkpoint(output, X, y, p_sample, seconds, metadata):
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, X=X, y=y, p_sample=p_sample,
            solver_seconds=seconds, metadata=json.dumps(metadata),
        )
    temporary.replace(output)


def summarize(y, p_sample, seconds, p_grid):
    rows = []
    for p in p_grid:
        mask = p_sample == p
        n = int(mask.sum())
        if not n:
            continue
        positives = int(y[mask].sum())
        # Wilson 95% interval; unlike normal intervals, remains meaningful at 0/20 or 20/20.
        q, z = positives / n, 1.959963984540054
        denominator = 1 + z*z/n
        center = (q + z*z/(2*n)) / denominator
        half = z*np.sqrt(q*(1-q)/n + z*z/(4*n*n)) / denominator
        rows.append({
            "p": float(p), "samples": n, "positive": positives,
            "probability": q, "wilson_95_low": center-half,
            "wilson_95_high": center+half,
            "mean_solver_seconds": float(seconds[mask].mean()),
        })
    return {
        "completed": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
        "solver_seconds_total": float(seconds.sum()),
        "solver_seconds_mean": float(seconds.mean()) if len(seconds) else None,
        "solver_seconds_median": float(np.median(seconds)) if len(seconds) else None,
        "solver_seconds_max": float(seconds.max()) if len(seconds) else None,
        "estimated_seconds_for_12000_samples": float(seconds.mean()*12000) if len(seconds) else None,
        "estimate_note": "Linear projection at this L and p grid; excludes compilation and I/O; timings vary by realization.",
        "per_p": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--per-p", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.L < 3 or args.per_p < 1 or args.seed < 0:
        parser.error("Require L >= 3, per-p >= 1, seed >= 0")
    output = args.output or PROJECT / f"data/raw/pilot_L{args.L}_n{args.per_p}_seed{args.seed}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    p_grid = research_p_grid()
    scheduled_p = np.tile(p_grid, args.per_p)
    target = len(scheduled_p)
    source_hash = hashlib.sha256(DEFAULT_SOLVER.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1, "L": args.L, "seed": args.seed, "target_samples": target,
        "samples_per_p": args.per_p, "p_values": p_grid.tolist(),
        "p_definition": "P(bond=-1)",
        "boundary_conditions": "horizontal periodic, vertical open",
        "feature_order": "right[L,L] row-major, then down[L-1,L] row-major",
        "solver_path": str(DEFAULT_SOLVER), "solver_sha256": source_hash,
        "numpy_version": np.__version__,
        "sampling": "default_rng(SeedSequence([seed, sample_index])); p cycles through grid",
        "label": "1: reference detects total spanning zero-weight percolation; 0: not detected",
        "limitations": "Reference search limits preserved; negative labels are not certified nonexistence.",
    }
    X = np.empty((target, args.L*(2*args.L-1)), dtype=np.int8)
    y = np.empty(target, dtype=np.int8)
    seconds = np.empty(target, dtype=float)
    completed = 0
    if output.exists():
        with np.load(output, allow_pickle=False) as old:
            previous = json.loads(old["metadata"].item())
            for key, value in metadata.items():
                if previous.get(key) != value:
                    raise ValueError(f"Cannot resume: metadata mismatch for {key}")
            completed = len(old["y"])
            if completed > target or not np.array_equal(old["p_sample"], scheduled_p[:completed]):
                raise ValueError("Invalid checkpoint sample schedule")
            X[:completed], y[:completed], seconds[:completed] = old["X"], old["y"], old["solver_seconds"]
    print(f"L={args.L}, {len(p_grid)} p values, {target} samples; resuming at {completed}", flush=True)
    if completed < target:
        solver = load_solver(DEFAULT_SOLVER)
        started = time.perf_counter()
        # Compile with the actual lattice size and representation, outside measured trials.
        warmup = np.ones((args.L, args.L, 4), dtype=np.int8)
        total_percolates(warmup, solver)
        print(f"Solver loaded/warmed in {time.perf_counter()-started:.2f}s", flush=True)
        for index in range(completed, target):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
            edges = generate_bonds(args.L, float(scheduled_p[index]), rng, solver)
            X[index] = pack_bonds(edges)
            started = time.perf_counter()
            y[index] = total_percolates(edges, solver)
            seconds[index] = time.perf_counter()-started
            completed = index+1
            metadata["samples"] = completed
            save_checkpoint(output, X[:completed], y[:completed], scheduled_p[:completed], seconds[:completed], metadata)
            print(f"{completed}/{target}: p={scheduled_p[index]:.5f}, y={y[index]}, solver={seconds[index]:.3f}s", flush=True)
    summary = summarize(y, scheduled_p, seconds, p_grid)
    summary["L"] = args.L
    summary["dataset"] = str(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_p"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
