"""Independent frozen-model transfer ensembles, retaining first-found path lengths."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import time
import numpy as np
from src.dataset import DEFAULT_SOLVER,PROJECT,load_solver,generate_bonds,pack_bonds,total_percolates_with_length
from src.pilot_dataset import research_p_grid

SOLVER=None
def initialize():
    global SOLVER
    SOLVER=load_solver(DEFAULT_SOLVER)

def realization(task):
    L,index,p,seed=task
    rng=np.random.default_rng(np.random.SeedSequence([seed,L,index]))
    edges=generate_bonds(L,p,rng,SOLVER)
    start=time.perf_counter()
    label,length=total_percolates_with_length(edges,SOLVER)
    return pack_bonds(edges),label,length,time.perf_counter()-start

def generate(L,per_p,seed,workers):
    p=np.tile(research_p_grid(),per_p);N=len(p)
    path=PROJECT/f'data/raw/transfer_L{L}_n{per_p}_seed{seed}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(schema_version=2,L=L,seed=seed,target_samples=N,samples_per_p=per_p,
        p_values=research_p_grid().tolist(),
        solver_path=str(DEFAULT_SOLVER),solver_sha256=hashlib.sha256(DEFAULT_SOLVER.read_bytes()).hexdigest(),
        cnn_sha256=hashlib.sha256((PROJECT/'models/L16_cnn_v1/cnn.pt').read_bytes()).hexdigest(),
        sampling='SeedSequence([seed,L,index]); p cycles through research grid',
        feature_order='right[L,L] row-major, then down[L-1,L] row-major',
        boundary_conditions='horizontal periodic, vertical open',
        label='reference solver total-percolation detection',
        path_length='First spanning path returned by reference; zero if not detected; not shortest-path guarantee',
        purpose='Independent transfer evaluation; no model fitting or calibration',numpy_version=np.__version__)
    X=np.empty((N,L*(2*L-1)),np.int8);y=np.empty(N,np.int8)
    lengths=np.empty(N,np.int32);seconds=np.empty(N)
    completed=0;previous_wall=0
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            prev=json.loads(old['metadata'].item())
            if any(prev.get(k)!=v for k,v in meta.items()): raise ValueError('Checkpoint settings changed')
            completed=len(old['y']);previous_wall=prev.get('generation_wall_seconds',0)
            assert completed<=N and np.array_equal(old['p_sample'],p[:completed])
            X[:completed],y[:completed],lengths[:completed],seconds[:completed]=old['X'],old['y'],old['path_length'],old['solver_seconds']
    if completed==N:
        print(f'L={L}: complete checkpoint already exists',flush=True);return
    start=time.perf_counter()
    initialize();total_percolates_with_length(np.ones((L,L,4),np.int8),SOLVER)
    tasks=((L,i,float(p[i]),seed) for i in range(completed,N))
    print(f'L={L}: starting {completed}/{N}, workers={workers}',flush=True)
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for row,label,length,elapsed in pool.map(realization,tasks,chunksize=4):
            X[completed],y[completed],lengths[completed],seconds[completed]=row,label,length,elapsed
            completed+=1
            if completed%96==0 or completed==N:
                meta['samples']=completed
                meta['generation_wall_seconds']=previous_wall+time.perf_counter()-start
                temp=path.with_suffix('.npz.tmp')
                with temp.open('wb') as handle:
                    np.savez_compressed(handle,X=X[:completed],y=y[:completed],path_length=lengths[:completed],
                        p_sample=p[:completed],solver_seconds=seconds[:completed],metadata=json.dumps(meta))
                temp.replace(path)
                print(f'L={L}: {completed}/{N}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    print(f'L={L} complete: positives={int(y.sum())}, saved {path}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--L',type=int,nargs='+',default=[20,24])
    parser.add_argument('--per-p',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if min(args.L)<3 or min(args.per_p,args.workers)<1 or args.seed<0: parser.error('Invalid generation settings')
    for L in args.L:generate(L,args.per_p,args.seed,args.workers)

if __name__=='__main__':main()
