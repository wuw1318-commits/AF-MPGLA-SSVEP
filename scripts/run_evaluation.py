"""Explicit-path evaluation driver for the unchanged mathematical primitives."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse
import csv
import itertools
import json
import numpy as np
from threadpoolctl import threadpool_limits
import core
import run_sweep as RS
import varlen_core as V
import beta_data as BD
import xr_data as XR

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['benchmark','beta','xr'])
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--windows',type=float,nargs='+',default=[.6])
    p.add_argument('--target-subjects',type=int,nargs='+')
    p.add_argument('--xr-descriptive-only',action='store_true')
    a=p.parse_args();threadpool_limits(limits=1)
    if not a.data_root.is_dir():p.error('data-root must be an existing directory')
    if any(d not in (.2,.4,.6,.8,1.) for d in a.windows):p.error('window must be one of .2 .4 .6 .8 1.0')
    if a.output.exists() and any(a.output.iterdir()):p.error('output must be empty to protect existing results')
    if a.dataset=='xr' and not a.xr_descriptive_only:p.error('XR identity is unresolved; this driver permits descriptive use only')
    if a.dataset=='xr' and any(d!=.6 for d in a.windows):p.error('this release retains the frozen 0.6 s XR loader; other XR lengths are not implemented here')
    if a.dataset=='benchmark':ids=list(range(1,36));core.DATA_DIR=str(a.data_root);loader=lambda u,d:core.preprocess(u,core.CH_IDX9,d)
    elif a.dataset=='beta':ids=list(range(1,71));BD.DATA_DIR=str(a.data_root);loader=lambda u,d:BD.preprocess_beta(u,BD.CH_IDX9_BETA,d)
    else:ids=[u for u in range(1,31) if u not in (9,10)];XR.SOURCE=str(a.data_root);XR.CACHE=str(a.data_root);loader=lambda u,d:XR.load_subject_epochs(u)
    targets=a.target_subjects or ids
    if len(set(targets))!=len(targets) or not set(targets).issubset(ids):p.error('invalid or duplicate target subjects')
    a.output.mkdir(parents=True,exist_ok=True)
    for d in a.windows:
        eps={u:loader(u,d) for u in ids}
        sources={u:RS.source_trcs(eps[u]) for u in ids}
        rows=[]
        for u in targets:
            ep=eps[u];nm,nf,nb,nc,ns=ep.shape
            source_ids=[v for v in ids if v!=u]
            Y=np.stack([sources[v] for v in source_ids],axis=1)
            calsets=list(itertools.combinations(range(nb),3));nrest=nb-3
            predictions=np.empty((len(calsets),nrest,4,nf),dtype=np.int16)
            heldout=[]
            for k,cal in enumerate(calsets):
                rest=[b for b in range(nb) if b not in cal];heldout.append(rest)
                V.assert_split_clean(cal,rest,n_block=nb)
                prep=V.prepare(ep,Y,cal,forbidden=rest)
                for bidx,b in enumerate(rest):
                    for j,c in enumerate(V.CELLS):predictions[k,bidx,j]=V.predict(prep,c,ep[:,:,b])
            counts=(predictions==np.arange(nf)[None,None,None,:]).sum(axis=(0,1,3))
            folder=a.output/('d%.1f'%d);folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/('subject_%02d.npz'%u),predictions=predictions,methods=V.CELLS,calibration=calsets,heldout_blocks=heldout,source_ids=source_ids)
            for j,c in enumerate(V.CELLS):rows.append(dict(subject=u,method=c,correct=int(counts[j]),n_predictions=len(calsets)*nrest*nf,accuracy_pct=float(counts[j]*100/(len(calsets)*nrest*nf))))
            print('completed',a.dataset,d,u,flush=True)
        with open(folder/'subject_accuracy.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if a.dataset=='beta':BD.clear_memo()
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(dataset=a.dataset,windows=a.windows,target_subjects=targets,source_cohort=ids,calibration_blocks=3,cells=list(V.CELLS),XR_identity_resolved=False if a.dataset=='xr' else None),indent=2)+'\n')

if __name__=='__main__':main()
