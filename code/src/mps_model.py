"""Tensor-network (MPS) joint-loss model for chains of countries.

Dependence: shared earthquake events on each LINK (i, i+1) of a chain along a fault system arrive at rate c
and hit both neighbours (sizes independent per country, as in the ADB brief); one global shared
precipitation shock (rate c_r) hits all countries. Each country keeps its total earthquake rate Lambda_i,
so its idiosyncratic rate is Lambda_i - deg_i * c. For n = 2 this is exactly the ADB benchmark.

The 3-year capped term distribution is an exact MPS: the bond on link i carries the shared-event counts
(k^1, k^2, k^3) of that link in the three years (64 values with K <= 3 per year); the global rain shock
is summed outside (8 ordered year patterns).
"""
import itertools, sys
sys.path.insert(0, "/home/claude/mps/src")
import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson
from model import cp_rates, TAILS

KMAX = 3
UNIT = 2.5e6


class ChainPortfolio:
    def __init__(self, n, m=1.0, rain_mult=1.0, rain_share=0.25, eq_limit=70e6, pr_limit=10e6, c_link=None):
        self.n = n
        names = [("KGZ", "TJK")[i % 2] for i in range(n)]
        R = [cp_rates(TAILS[c]) for c in names]
        self.Lam = np.array([r[1] for r in R]); self.w = np.array([r[0] / r[1] for r in R])
        # ADB Eq. (4) for every link (identical for alternating KGZ/TJK)
        self.c = m * 0.5 * (0.075 * self.Lam[0] + 0.074 * self.Lam[1 % n]) if c_link is None else c_link
        self.deg = np.array([(i > 0) + (i < n - 1) for i in range(n)])
        mu = -np.log(1 - 0.02) * rain_mult
        self.mu = mu; self.cr = rain_share * mu
        self.eqU = int(round(eq_limit / 4 / UNIT)); self.prU = int(round(pr_limit / UNIT))
        self.principal = eq_limit + pr_limit
        self.lossU = np.array([self.eqU * (s // 2) + self.prU * (s % 2) for s in range(10)])
        self.T_U = int(round(self.principal * n / 2 / UNIT))
        self.pk = poisson.pmf(np.arange(KMAX + 1), self.c)          # truncated, not renormalised
        self.trunc = 1 - self.pk.sum()
        self._annual = [self._annual_tables(i) for i in range(n)]
        cap = np.zeros((10, 10), dtype=int)
        for s in range(10):
            for x in range(10):
                cap[s, x] = 2 * min(s // 2 + x // 2, 4) + min(s % 2 + x % 2, 1)
        self.cap = cap
        self._term_cache = {}

    def _annual_tables(self, i):
        """annual[k_tot, J] -> pmf over local state 2a+r, k_tot = shared EQ events hitting country i."""
        rate = self.Lam[i] - self.deg[i] * self.c
        G = np.zeros((5, 5))
        for a in range(5):
            for k in range(4):
                G[a, min(a + k + 1, 4)] += rate * self.w[i, k]
            G[a, a] -= G[a].sum()
        idio = expm(G)[0]
        out = np.zeros((2 * KMAX + 1, 2, 10))
        for kt in range(2 * KMAX + 1):
            p = idio.copy()
            for _ in range(kt):
                q = np.zeros(5)
                for a in range(5):
                    for k in range(4):
                        q[min(a + k + 1, 4)] += p[a] * self.w[i, k]
                p = q
            pri = 1 - np.exp(-(self.mu - self.cr))
            for J in (0, 1):
                pr = np.array([1 - pri, pri]) if J == 0 else np.array([0.0, 1.0])
                out[kt, J] = np.outer(p, pr).ravel()
        return out

    def _capconv(self, p, q):
        out = np.zeros(10)
        for s in range(10):
            if p[s]:
                np.add.at(out, self.cap[s], p[s] * q)
        return out

    def cores(self, Jt):
        """MPS cores for one ordered rain pattern Jt (3 bits). Bond index = (k1,k2,k3) in base KMAX+1."""
        B = (KMAX + 1) ** 3
        trip = np.array(list(itertools.product(range(KMAX + 1), repeat=3)))   # B x 3
        wlink = self.pk[trip].prod(1)                                        # B
        cores = []
        for i in range(self.n):
            Dl = 1 if i == 0 else B; Dr = 1 if i == self.n - 1 else B
            C = np.zeros((Dl, 10, Dr))
            A = self._annual[i]
            for al in range(Dl):
                kl = trip[al] if i > 0 else np.zeros(3, int)
                for br in range(Dr):
                    kr = trip[br] if i < self.n - 1 else np.zeros(3, int)
                    t = np.zeros(10); t[0] = 1.0
                    for y in range(3):
                        t = self._capconv(t, A[kl[y] + kr[y], Jt[y]])
                    C[al, :, br] = t * (wlink[br] if i < self.n - 1 else 1.0)
            cores.append(C)
        return cores

    def rain_patterns(self):
        pJ1 = 1 - np.exp(-self.cr)
        for Jt in itertools.product((0, 1), repeat=3):
            yield Jt, np.prod([pJ1 if j else 1 - pJ1 for j in Jt])


def contract_targets(P: ChainPortfolio):
    """Exact targets by MPS contraction, summing the 8 rain patterns."""
    n = P.n
    pay = (np.arange(10) > 0).astype(float); exh = (np.arange(10) == 9).astype(float)
    Lmax = int(P.lossU.max()) * n
    out = dict(norm=0.0, EL=0.0, all_pay=0.0, all_exh=0.0, exceed=0.0, stop=0.0)
    for Jt, wJ in P.rain_patterns():
        cores = P.cores(Jt)
        def chain(vecs):
            env = np.ones(1)
            for C, v in zip(cores, vecs):
                env = env @ np.tensordot(C, v, axes=([1], [0]))
            return float(env.sum())
        ones = [np.ones(10)] * n
        out["norm"] += wJ * chain(ones)
        out["all_pay"] += wJ * chain([pay] * n)
        out["all_exh"] += wJ * chain([exh] * n)
        # expected loss: one linear sweep with (marginal, first-moment) pair
        env0 = np.ones(1); env1 = np.zeros(1)
        for C in cores:
            M0 = C.sum(1); M1 = np.tensordot(C, P.lossU.astype(float), axes=([1], [0]))
            env0, env1 = env0 @ M0, env1 @ M0 + env0 @ M1
        out["EL"] += wJ * float(env1.sum()) * UNIT
        # loss distribution: environment over (bond, loss units)
        E = np.zeros((1, Lmax + 1)); E[0, 0] = 1.0
        for C in cores:
            En = np.zeros((C.shape[2], Lmax + 1))
            for x in range(10):
                u = int(P.lossU[x]); T = E.T @ C[:, x, :]                      # (L, Dr)
                En[:, u:] += T[: Lmax + 1 - u].T
            E = En
        dist = E.sum(0); k = np.arange(Lmax + 1)
        out["exceed"] += wJ * dist[k > P.T_U].sum()
        out["stop"] += wJ * (dist * np.maximum(k - P.T_U, 0)).sum() * UNIT
    return out


def full_tensor(P: ChainPortfolio):
    T = 0
    for Jt, wJ in P.rain_patterns():
        cores = P.cores(Jt); X = cores[0]
        for C in cores[1:]:
            X = np.tensordot(X, C, axes=([-1], [0]))
        T = T + wJ * X.reshape((10,) * P.n)
    return T


def tt_ranks(T, tol):
    """TT-SVD ranks at relative Frobenius tolerance tol (Oseledets 2011)."""
    d = T.ndim; ranks = []; C = T.copy(); r = 1
    delta = tol / np.sqrt(d - 1) * np.linalg.norm(T)
    for k in range(d - 1):
        C = C.reshape(r * 10, -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        rk = max(1, int(np.sum(tail > delta)))
        ranks.append(rk)
        C = s[:rk, None] * Vt[:rk]; r = rk
    return ranks
