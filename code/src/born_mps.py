"""Born-MPS (sqrt p) truncation: bond dimension vs target error, and sequential-circuit resources."""
import sys, json
sys.path.insert(0, "/home/claude/mps/src")
import numpy as np
from mps_model import ChainPortfolio, full_tensor


def tt_svd(T, chi):
    d = T.ndim; cores = []; C = T.copy(); r = 1
    for k in range(d - 1):
        C = C.reshape(r * 10, -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        rk = min(chi, s.size)
        cores.append(U[:, :rk].reshape(r, 10, rk)); C = s[:rk, None] * Vt[:rk]; r = rk
    cores.append(C.reshape(r, 10, 1))
    X = cores[0]
    for G in cores[1:]:
        X = np.tensordot(X, G, axes=([-1], [0]))
    return X.reshape(T.shape)


def targets(T, P):
    n = T.ndim; idx = np.array(np.unravel_index(np.arange(T.size), T.shape)).T; f = T.ravel() / T.sum()
    L = P.lossU[idx].sum(1)
    return dict(EL=(f * L).sum(), all_pay=f[(idx > 0).all(1)].sum(), exceed=f[L > P.T_U].sum(),
                stop=(f * np.maximum(L - P.T_U, 0)).sum(), all_exh=f[(idx == 9).all(1)].sum())

out = {}
for n in (4, 5):
    P = ChainPortfolio(n); T = full_tensor(P); ex = targets(T, P); S = np.sqrt(np.clip(T, 0, None))
    rows = []
    for chi in (1, 2, 3, 4, 6, 8, 10, 13, 16):
        A = tt_svd(S, chi); Tq = A ** 2
        tg = targets(Tq, P)
        err = {k: abs(tg[k] - ex[k]) / ex[k] for k in ex}
        fid = (S.ravel() @ A.ravel()) ** 2 / ((S.ravel() @ S.ravel()) * (A.ravel() @ A.ravel()))
        q_bond = int(np.ceil(np.log2(chi))) if chi > 1 else 0
        cnot_site = int(np.ceil(23 / 48 * 4 ** (q_bond + 4)))        # generic (q_bond+4)-qubit unitary
        rows.append(dict(chi=chi, fidelity=float(fid), max_rel_err=max(err.values()), errs=err, bond_qubits=q_bond,
                         qubits=4 * n + q_bond, cnot_per_site=cnot_site, cnot_loader=cnot_site * n))
        print(f"n={n} chi={chi:2d} infid={1-fid:.2e} maxrel={max(err.values()):.2e} (exh {err['all_exh']:.1e}, exc {err['exceed']:.1e}) "
              f"qubits={4*n+q_bond} CNOT/site~{cnot_site:,}", flush=True)
    out[n] = rows
json.dump(out, open("/home/claude/mps/results/born_mps.json", "w"), indent=1, default=float)
