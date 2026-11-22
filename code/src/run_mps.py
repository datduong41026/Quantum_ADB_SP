import sys, json, time
sys.path.insert(0, "/home/claude/mps/src")
import numpy as np
from mps_model import ChainPortfolio, contract_targets, full_tensor, tt_ranks
from model import Portfolio, SCENARIOS

res = {}
# 1) n = 2 chain == ADB benchmark, all scenarios
print("== n=2 vs brief")
res["adb"] = {}
for sc, kw in SCENARIOS.items():
    kw2 = dict(kw); kw2.pop("rain_q", None)
    t0 = time.perf_counter(); o = contract_targets(ChainPortfolio(2, **kw2)); dt = time.perf_counter() - t0
    ex = Portfolio(**kw).exact_targets()
    pairs = dict(EL="expected_principal_loss_usd", all_pay="joint_attachment_probability", exceed="portfolio_loss_exceedance_probability",
                 stop="portfolio_stop_loss_usd", all_exh="all_bonds_exhausted_probability")
    rel = max(abs(o[k] - ex[v]) / ex[v] for k, v in pairs.items())
    res["adb"][sc] = dict(o, max_rel_vs_exact=rel, seconds=dt)
    print(f"{sc:17s} norm={o['norm']:.12f} maxrel={rel:.1e} EL={o['EL']/1e6:.4f}m pay={o['all_pay']*100:.4f}% exc={o['exceed']*100:.4f}% stop={o['stop']:.0f} exh={o['all_exh']*100:.5f}% ({dt:.2f}s)")

# 2) brute-force validation of the chain model, n = 3, 4 (full tensor vs targets)
print("== brute-force check of contraction")
res["bruteforce"] = {}
for n in (3, 4):
    P = ChainPortfolio(n); T = full_tensor(P)
    idx = np.array(np.unravel_index(np.arange(10 ** n), (10,) * n)).T
    L = P.lossU[idx].sum(1); flat = T.ravel()
    bf = dict(all_pay=flat[(idx > 0).all(1)].sum(), all_exh=flat[(idx == 9).all(1)].sum(), exceed=flat[L > P.T_U].sum(),
              stop=(flat * np.maximum(L - P.T_U, 0)).sum() * 2.5e6, EL=(flat * L).sum() * 2.5e6)
    o = contract_targets(P)
    rel = max(abs(o[k] - bf[k]) / abs(bf[k]) for k in bf)
    res["bruteforce"][n] = rel
    print(f"n={n}: max rel diff contraction vs full tensor {rel:.1e}; norm {flat.sum():.12f}")

# 3) TT ranks discovered by TT-SVD without using the model structure (p and sqrt p)
print("== TT-SVD ranks")
res["ranks"] = {}
for n in (3, 4, 5):
    T = full_tensor(ChainPortfolio(n))
    for name, X in (("p", T), ("sqrt_p", np.sqrt(np.clip(T, 0, None)))):
        for tol in (1e-4, 1e-8):
            r = tt_ranks(X, tol)
            res["ranks"][f"{n}_{name}_{tol:g}"] = r
            print(f"n={n} {name:6s} tol={tol:g}: ranks {r}")

# 4) scaling of exact contraction
print("== scaling")
res["scaling"] = []
for n in (2, 5, 10, 20, 30, 40):
    P = ChainPortfolio(n)
    t0 = time.perf_counter(); o = contract_targets(P); dt = time.perf_counter() - t0
    p = o["all_exh"]
    row = dict(n=n, seconds=dt, norm=o["norm"], all_pay=o["all_pay"], exceed=o["exceed"], stop=o["stop"], EL=o["EL"], all_exh=p,
               mc_samples_exceed_1pct=(1 - o["exceed"]) / (o["exceed"] * 1e-4), trunc_per_link_year=P.trunc)
    res["scaling"].append(row)
    print(f"n={n:2d} t={dt:6.2f}s norm={o['norm']:.10f} P(all pay)={o['all_pay']:.3e} P(L>T)={o['exceed']:.3e} stop={o['stop']:.4g} P(all exh)={p:.3e}", flush=True)
json.dump(res, open("/home/claude/mps/results/mps_results.json", "w"), indent=1, default=float)
