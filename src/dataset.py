"""Generate total-percolation labels using the existing research solver.

Run from zero-path-ml: python -m src.dataset --samples 100 --L 4 --seed 42
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVER = PROJECT.parent / "zero-path" / "test_GPT_tot.py"


def load_solver(path):
    """Import the research functions without running their main experiment."""
    path = Path(path).resolve()
    # Keep Python/Numba caches out of the reference research repository.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT / ".cache" / "numba-import-v1"))
    name = "zero_path_total_" + hashlib.sha256(path.read_bytes()).hexdigest()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Numba's persistent cache must be able to find the module by name.
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


def generate_bonds(L, p, rng, solver):
    """Sample ±1 bonds with P(-1)=p, then apply reference reciprocity."""
    if L < 3:
        raise ValueError("Use L >= 3; L=2 aliases left and right neighbors.")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")
    edges = rng.choice([1, -1], size=(L, L, 4), p=[1-p, p]).astype(np.int8)
    solver.enforce_edge_symmetry(L, edges)
    return edges


def pack_bonds(edges):
    """Right bonds (including wrap), then down bonds, both row-major."""
    return np.concatenate((edges[:, :, 1].ravel(), edges[:-1, :, 2].ravel()))


def unpack_bonds(features, L):
    """Reconstruct solver input. Unused exterior vertical entries are zero."""
    features = np.asarray(features)
    if features.shape != (L * (2 * L - 1),) or not np.isin(features, [-1, 1]).all():
        raise ValueError("Expected one ±1 value for each physical bond.")
    edges = np.zeros((L, L, 4), dtype=np.int8)
    edges[:, :, 1] = features[:L*L].reshape(L, L)
    edges[:, :, 0] = np.roll(edges[:, :, 1], 1, axis=1)
    edges[:-1, :, 2] = features[L*L:].reshape(L-1, L)
    edges[1:, :, 3] = edges[:-1, :, 2]
    return edges


def total_percolates(edges, solver):
    """Same starting points and early return as run_single_trial_no_pbc.

    Unlike that function, accept an existing configuration so X and y refer
    to the same realization. No cross-window criterion is applied.
    """
    L = edges.shape[0]
    for start_row, target_row in ((L-1, 0), (0, L-1)):
        for idx in range(L):
            start_col = L // 2 + (-1)**idx * ((idx+1)//2)
            percolated, _ = solver.find_reachable_points_no_pbc(
                L, edges, start_row=start_row, start_col=start_col, target_row=target_row
            )
            if percolated == 1:
                return 1
    return 0


def build_dataset(samples, L, p_values, seed, solver_path):
    if samples < 1 or L < 3 or seed < 0:
        raise ValueError("Require samples >= 1, L >= 3, and seed >= 0.")
    if not p_values or any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("Provide at least one p in [0, 1].")
    solver_path = Path(solver_path).resolve()
    source_hash = hashlib.sha256(solver_path.read_bytes()).hexdigest()
    solver = load_solver(solver_path)
    rng = np.random.default_rng(seed)
    X = np.empty((samples, L*(2*L-1)), dtype=np.int8)
    y = np.empty(samples, dtype=np.int8)
    # Cycle through p values so their sample counts differ by at most one.
    p_sample = np.resize(np.asarray(p_values, dtype=float), samples)
    for index, p in enumerate(p_sample):
        edges = generate_bonds(L, p, rng, solver)
        X[index] = pack_bonds(edges)
        y[index] = total_percolates(edges, solver)
        if (index+1) % 10 == 0 or index+1 == samples:
            print(f"Labeled {index+1}/{samples}", flush=True)
    metadata = {
        "schema_version": 1, "L": L, "samples": samples, "seed": seed,
        "p_values": list(p_values), "p_definition": "P(bond=-1)",
        "boundary_conditions": "horizontal periodic, vertical open",
        "feature_order": "right[L,L] row-major, then down[L-1,L] row-major",
        "label": "1: reference solver detects total spanning zero-weight percolation; 0: not detected",
        "solver_path": str(solver_path), "solver_sha256": source_hash,
        "numpy_version": np.__version__,
        "limitations": "Reference search limits and path selection are preserved; negative labels are not certified nonexistence.",
    }
    return X, y, p_sample, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--p", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = PROJECT / f"data/raw/dataset_L{args.L}_demo.npz"
    if args.output.suffix != ".npz":
        parser.error("Output must end in .npz")
    if args.output.exists():
        parser.error("Output exists; choose a new filename.")
    X, y, p_sample, metadata = build_dataset(args.samples, args.L, args.p, args.seed, args.solver)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, X=X, y=y, p_sample=p_sample, metadata=json.dumps(metadata))
    print(f"Saved {args.output}: X={X.shape}, y={y.shape}; {int(y.sum())} positive labels")


if __name__ == "__main__":
    main()
