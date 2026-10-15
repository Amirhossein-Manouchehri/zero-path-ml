"""Generate the training ensemble with four independent solver workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import time

import numpy as np
from src.dataset import PROJECT, DEFAULT_SOLVER, load_solver, generate_bonds, pack_bonds, total_percolates
from src.pilot_dataset import research_p_grid, save_checkpoint

SOLVER = None

def initialize():
    global SOLVER
    SOLVER = load_solver(DEFAULT_SOLVER)

def realization(task):
    index, L, p, seed = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    edges = generate_bonds(L, p, rng, SOLVER)
    start = time.perf_counter()
    label = total_percolates(edges, SOLVER)
    return pack_bonds(edges), label, time.perf_counter()-start

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-p', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.per_p, args.workers) < 1 or args.seed < 0:
        parser.error('Require positive per-p/workers and nonnegative seed')
    L = 16
    ps = np.tile(research_p_grid(), args.per_p)
    N = len(ps)
    output = PROJECT / f'data/raw/training_L16_n{args.per_p}_seed{args.seed}.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(L=L, seed=args.seed, target_samples=N, samples_per_p=args.per_p,
        p_values=research_p_grid().tolist(),
        solver_sha256=hashlib.sha256(DEFAULT_SOLVER.read_bytes()).hexdigest(),
        solver_path=str(DEFAULT_SOLVER), numpy_version=np.__version__,
        sampling='default_rng(SeedSequence([seed, sample_index])); p cycles through grid',
        feature_order='right[L,L] row-major, then down[L-1,L] row-major',
        boundary_conditions='horizontal periodic, vertical open',
        label='reference total-percolation detection; negative labels are not certified nonexistence')
    X = np.empty((N, L*(2*L-1)), dtype=np.int8)
    y = np.empty(N, dtype=np.int8)
    seconds = np.empty(N)
    completed = 0
    if output.exists():
        with np.load(output, allow_pickle=False) as d:
            previous = json.loads(d['metadata'].item())
            if any(previous.get(k) != v for k,v in meta.items()):
                raise ValueError('Checkpoint settings or solver changed')
            completed = len(d['y'])
            assert completed <= N and np.array_equal(d['p_sample'], ps[:completed])
            X[:completed], y[:completed], seconds[:completed] = d['X'], d['y'], d['solver_seconds']
    start = time.perf_counter()
    if completed < N:
        initialize()
        total_percolates(np.ones((L,L,4), dtype=np.int8), SOLVER)
        tasks = ((i,L,float(ps[i]),args.seed) for i in range(completed,N))
        print(f'Starting {completed}/{N} with {args.workers} workers', flush=True)
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            for row, label, elapsed in pool.map(realization, tasks, chunksize=8):
                X[completed], y[completed], seconds[completed] = row, label, elapsed
                completed += 1
                if completed % 96 == 0 or completed == N:
                    meta['samples'] = completed
                    save_checkpoint(output,X[:completed],y[:completed],ps[:completed],seconds[:completed],meta)
                    print(f'{completed}/{N}, {time.perf_counter()-start:.1f}s this run',flush=True)
    print(f'Complete: {output}, positive={int(y.sum())}, negative={N-int(y.sum())}',flush=True)

if __name__ == '__main__':
    main()
