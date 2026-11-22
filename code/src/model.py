"""ADB joint-loss benchmark model (Challenge Brief, Sec. 3b), generalised to n countries.

Annual state per country: (a_i in 0..4 earthquake quarter steps, r_i in 0..1 precipitation trigger).
Term: 3 independent years, running totals capped at (4, 1) per country.
Dependence: shared earthquake events (rate c) hit every country; shared precipitation rate c_r hits every country.
For n = 2 this is exactly the brief's model (CTMC on 25 states, Eqs. 2-7).
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

TAILS = {"KGZ": [0.03883, 0.02331, 0.01790, 0.01184],
         "TJK": [0.03925, 0.01743, 0.01332, 0.01043]}
COND = {"KGZ": 0.075, "TJK": 0.074}   # P(other triggered | this triggered), brief step 1


def cp_rates(t):
    """Compound-Poisson rates from tail probabilities, brief Eqs. (2)-(3)."""
    p0 = 1 - t[0]; p1 = t[0] - t[1]; p2 = t[1] - t[2]; p3 = t[2] - t[3]
    L = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1 ** 2 / 2
    l3 = p3 / p0 - l1 * l2 - l1 ** 3 / 6
    return np.array([l1, l2, l3, L - l1 - l2 - l3]), L


SCENARIOS = {
    "adb-base":         dict(),
    "adb-independent":  dict(m=0.0, rain_share=0.0),
    "adb-common-shock": dict(m=2.0),
    "adb-rain-stress":  dict(rain_mult=2.0),
    "adb-rain-cover":   dict(eq_limit=60e6, pr_limit=20e6),
}


class Portfolio:
    def __init__(self, countries=("KGZ", "TJK"), m=1.0, rain_q=0.02, rain_mult=1.0,
                 rain_share=0.25, eq_limit=70e6, pr_limit=10e6, years=3, cond=None):
        self.countries = list(countries)
        self.n = len(self.countries)
        self.years = years
        self.eq_limit, self.pr_limit = eq_limit, pr_limit
        self.principal = eq_limit + pr_limit
        R = [cp_rates(TAILS[c]) for c in self.countries]
        self.lam = np.array([r[0] for r in R])      # n x 4
        self.Lam = np.array([r[1] for r in R])      # n
        self.w = self.lam / self.Lam[:, None]       # size distribution per event
        cond = cond or COND
        # shared EQ rate: brief Eq. (4) for n = 2; mean of P(other|this)*Lambda for n > 2
        self.c = m * np.mean([cond[c] * L for c, L in zip(self.countries, self.Lam)])
        mu = -np.log(1 - rain_q) * rain_mult
        self.mu = np.full(self.n, mu)
        self.cr = rain_share * self.mu.min()

    # ---------- per-country building blocks ----------
    def _eq_idio_pmf(self, i):
        """Annual capped step distribution (0..4) from idiosyncratic events of country i."""
        rate = self.Lam[i] - self.c
        G = np.zeros((5, 5))
        for a in range(5):
            for k in range(4):
                G[a, min(a + k + 1, 4)] += rate * self.w[i, k]
            G[a, a] -= G[a].sum()
        return expm(G)[0]

    def _shared_step_pmf(self, i, K):
        """Distribution of capped steps that K shared events add to country i (sizes iid w_i)."""
        p = np.zeros(5); p[0] = 1.0
        for _ in range(K):
            q = np.zeros(5)
            for a in range(5):
                for k in range(4):
                    q[min(a + k + 1, 4)] += p[a] * self.w[i, k]
            p = q
        return p

    @staticmethod
    def _capsum(p, q, cap):
        out = np.zeros(cap + 1)
        for a, pa in enumerate(p):
            for b, qb in enumerate(q):
                out[min(a + b, cap)] += pa * qb
        return out

    def annual_factor_model(self, kmax=6):
        """Exact conditional-independence (single common-shock) representation of one year.

        Returns list of (weight, [per-country pmf over 10 local states (a,r) -> 2a+r ... index a*2+r])."""
        comps = []
        idio = [self._eq_idio_pmf(i) for i in range(self.n)]
        for K in range(kmax + 1):
            wK = poisson.pmf(K, self.c)
            for J in (0, 1):     # common precipitation shock (only "at least one" matters, cap 1)
                wJ = np.exp(-self.cr) if J == 0 else 1 - np.exp(-self.cr)
                locs = []
                for i in range(self.n):
                    eq = self._capsum(idio[i], self._shared_step_pmf(i, K), 4)
                    p_r_idio = 1 - np.exp(-(self.mu[i] - self.cr))
                    pr = np.array([1 - p_r_idio, p_r_idio]) if J == 0 else np.array([0.0, 1.0])
                    locs.append(np.outer(eq, pr).ravel())   # local index = 2*a + r
                comps.append((wK * wJ, locs))
        tot = sum(w for w, _ in comps)
        return [(w / tot, l) for w, l in comps], 1 - tot

    # ---------- exact joint distributions (enumeration) ----------
    def annual_joint(self):
        """Full annual joint pmf over 10^n states (index = sum over countries of local*10^i)."""
        comps, _ = self.annual_factor_model()
        P = np.zeros((10,) * self.n)
        for w, locs in comps:
            T = locs[0]
            for l in locs[1:]:
                T = np.multiply.outer(T, l)
            P += w * T
        return P.ravel(order="C")

    def annual_joint_ctmc(self):
        """Brief's literal CTMC construction (n = 2 only), used to validate annual_joint()."""
        assert self.n == 2
        (l1, L1), (l2, L2) = (self.lam[0], self.Lam[0]), (self.lam[1], self.Lam[1])
        w1, w2, c = l1 / L1, l2 / L2, self.c
        G = np.zeros((25, 25)); ix = lambda a, b: 5 * a + b
        for a in range(5):
            for b in range(5):
                s = ix(a, b)
                for k in range(4):
                    G[s, ix(min(a + k + 1, 4), b)] += (L1 - c) * w1[k]
                    G[s, ix(a, min(b + k + 1, 4))] += (L2 - c) * w2[k]
                    for j in range(4):
                        G[s, ix(min(a + k + 1, 4), min(b + j + 1, 4))] += c * w1[k] * w2[j]
                G[s, s] -= G[s].sum()
        eq = expm(G)[0].reshape(5, 5)
        mu1, mu2, cr = self.mu[0], self.mu[1], self.cr
        P00 = np.exp(-(mu1 + mu2 - cr)); P10 = np.exp(-mu2) - P00; P01 = np.exp(-mu1) - P00
        pr = np.array([[P00, P01], [P10, 1 - P00 - P10 - P01]])
        A = np.einsum("ab,rs->arbs", eq, pr)          # (a, r1, b, r2) -> local idx 2a+r
        return A.reshape(10, 10).ravel()

    # ---------- local-state helpers ----------
    @staticmethod
    def local_ar(idx):
        return idx // 2, idx % 2

    def capadd_table(self):
        """10x10 table: capped sum of two local states (a<=4, r<=1)."""
        T = np.zeros((10, 10), dtype=np.int64)
        for s in range(10):
            for x in range(10):
                a = min(s // 2 + x // 2, 4); r = min(s % 2 + x % 2, 1)
                T[s, x] = 2 * a + r
        return T

    def local_loss(self, idx):
        a, r = idx // 2, idx % 2
        return self.eq_limit * a / 4 + self.pr_limit * r

    def term_joint(self):
        """Exact 3-year term pmf over 10^n capped states by direct convolution (brief step 7)."""
        n, S = self.n, 10 ** self.n
        P = self.annual_joint()
        T = self.capadd_table()
        digits = np.array(np.unravel_index(np.arange(S), (10,) * n)).T   # S x n
        term = np.zeros(S); term[0] = 1.0
        nz = np.nonzero(P)[0]
        for _ in range(self.years):
            new = np.zeros(S)
            for s in np.nonzero(term > 0)[0]:
                tgt = T[digits[s][None, :], digits[nz]]            # |nz| x n
                flat = np.ravel_multi_index(tgt.T, (10,) * n)
                np.add.at(new, flat, term[s] * P[nz])
            term = new
        return term, digits

    # ---------- targets ----------
    def target_values(self, digits):
        """Per-state value of the five focus targets (brief Table 3), generalised to n countries."""
        Li = np.stack([self.local_loss(digits[:, i]) for i in range(self.n)], 1)
        L = Li.sum(1)
        T = self.principal * self.n / 2          # = USD 80m for the 2-country benchmark
        return {
            "expected_principal_loss_usd": L,
            "joint_attachment_probability": (Li > 0).all(1).astype(float),
            "portfolio_loss_exceedance_probability": (L > T + 1e-6).astype(float),
            "portfolio_stop_loss_usd": np.maximum(L - T, 0.0),
            "all_bonds_exhausted_probability": (Li >= self.principal - 1e-6).all(1).astype(float),
        }, T

    def exact_targets(self):
        term, digits = self.term_joint()
        vals, _ = self.target_values(digits)
        return {k: float(term @ v) for k, v in vals.items()}


TARGETS = ["expected_principal_loss_usd", "joint_attachment_probability",
           "portfolio_loss_exceedance_probability", "portfolio_stop_loss_usd",
           "all_bonds_exhausted_probability"]

BRIEF_TABLE5 = {
    "adb-base":         [10.06e6, 0.04414, 0.01205, 282355, 0.000032],
    "adb-independent":  [10.06e6, 0.02716, 0.00719, 171266, 0.000004],
    "adb-common-shock": [10.06e6, 0.05053, 0.01607, 385043, 0.000044],
    "adb-rain-stress":  [11.17e6, 0.07046, 0.01322, 306286, 0.000069],
    "adb-rain-cover":   [9.97e6,  0.04414, 0.00700, 170927, 0.000032],
}
