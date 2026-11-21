#!/usr/bin/env python3
"""Independent small-system density-matrix checks of the delivered kernels."""
from pathlib import Path
import itertools
import json
import os
import shutil
import sys
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.linalg import expm
import torch
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
sys.path.insert(0,str(CODE/'BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration'))
from hamiltonian_qrc import compute_suffix_banks,mix_suffix_bank
import hamiltonian_dtc_qrc as alg


def kron_all(parts):
    out=np.ones((1,1),complex)
    for part in parts:out=np.kron(out,part)
    return out


def main():
    torch.set_num_threads(1)
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.])
    rng=np.random.default_rng(819)
    vectors=np.r_[np.zeros((1,3)),rng.normal(size=(32,3))]
    exact=np.stack([expm(-.5j*(x*X+y*Y+z*Z)) for x,y,z in vectors])
    computed=alg.hamiltonian_modulation_gates(torch.tensor(vectors),torch.complex128).numpy()
    gate_error=float(np.abs(exact-computed).max());assert gate_error<1e-12
    records=[]
    for q in (2,3):
        tokens=rng.normal(scale=.2,size=(3,4,3*q)).astype(np.float32)
        seed=10;g=.88;params=alg.generate_dtc_params(q,seed,.5)
        signs=np.array(list(itertools.product([1.,-1.],repeat=q)))
        obs=np.vstack([signs[:,i] for i in range(q)]+[signs[:,i]*signs[:,j] for i in range(q) for j in range(i+1,q)])
        phase=np.diag(np.exp(-.5j*((signs[:,:-1]*signs[:,1:])@params['J']+np.pi*signs@params['h'])))
        F=phase@kron_all([expm(-.5j*g*np.pi*X)]*q)
        zero=np.zeros((2**q,2**q),complex);zero[0,0]=1
        banks,_=compute_suffix_banks(tokens,q=q,t_values=(4,),seed=seed,device='cpu',g=g,batch_size=3)
        for p in (0.,.1,.8):
            expected=[]
            for sequence in tokens:
                rho=zero.copy()
                for event in sequence:
                    U=kron_all([expm(-.5j*(x*X+y*Y+z*Z)) for x,y,z in event.reshape(q,3)])
                    U=np.linalg.matrix_power(F,3)@U
                    rho=(1-p)*(U@rho@U.conj().T)+p*zero
                expected.append(obs@np.diag(rho).real)
            error=float(np.max(np.abs(np.array(expected)-mix_suffix_bank(banks[4],p))))
            assert error<2e-5,(q,p,error)
            records.append(dict(q=q,p=p,check='graph_full_density_matrix',maximum_error=error))
        # CPU C++ hopping suffixes: all suffixes and all three endpoint samples.
        sys.path.insert(0,str(CODE/'BFS-QRC/hopping_gscan_fixed_p_20260908'))
        import backend
        work=ROOT/'runs/numeric_kernel';work.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CODE/'BFS-QRC/hopping_gscan_fixed_p_20260908/cpu_kernel.cpp',work/'cpu_kernel.cpp')
        backend.HERE=work
        bank,_=backend.suffixes(tokens,seed=seed,g=g,threads=1)
        expected=np.empty_like(bank,dtype=float)
        for s in range(4):
            for i,sequence in enumerate(tokens):
                rho=zero.copy()
                for event in sequence[s:]*np.float32(.5):
                    U=kron_all([expm(-.5j*(x*X+y*Y+z*Z)) for x,y,z in event.reshape(q,3)])
                    U=np.linalg.matrix_power(F,3)@U;rho=U@rho@U.conj().T
                blocks=[]
                for j in range(3):
                    blocks.append(obs@np.diag(rho).real);rho=F@rho@F.conj().T
                expected[s,i]=np.concatenate(blocks)
        error=float(np.abs(bank-expected).max());assert error<2e-5,(q,error)
        records.append(dict(q=q,check='hopping_CPU_all_suffixes_and_TM3',maximum_error=error))
    report=dict(passed=True,gate_exponential_maximum_error=gate_error,checks=records,
                scope='Independent SciPy dense-matrix calculations at Q=2,3; not a full application rerun.')
    (ROOT/'audit/numerical_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
