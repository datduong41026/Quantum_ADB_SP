import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
B = json.load(open("/home/claude/mps/results/born_mps.json"))["5"]
R = json.load(open("/home/claude/mps/results/mps_results.json"))
G = json.load(open("/home/claude/adbq/results/scaling.json"))["rows"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURF)
for a in ax: a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6)
chi = [r["chi"] for r in B]
names = dict(EL="E[L]", all_pay="P(all pay)", exceed="P(L > T)", stop="Stop-loss", all_exh="P(all exhausted)")
mk = ["o", "s", "^", "D", "v"]
for j, (k, lab) in enumerate(names.items()):
    ax[0].plot(chi, [max(r["errs"][k], 1e-12) for r in B], color=C[j], marker=mk[j], lw=2, ms=4, label=lab)
ax[0].plot(chi, [max(1 - r["fidelity"], 1e-16) for r in B], color=INK2, ls="--", lw=1.5, label="State infidelity")
ax[0].set_yscale("log"); ax[0].set_xlabel("Born-MPS bond dimension χ (n = 5)"); ax[0].set_ylabel("Relative error / infidelity")
ax[0].set_title("Truncated √p MPS: tail error vs fidelity", loc="left", fontsize=10); ax[0].legend(frameon=False, fontsize=7)
S = R["scaling"]; n = [r["n"] for r in S]
ax[1].plot(n, [r["seconds"] for r in S], color=C[0], marker="o", lw=2)
ax[1].set_yscale("log"); ax[1].set_xlabel("Number of countries n (10^n annual states)"); ax[1].set_ylabel("Exact contraction time (s)")
ax[1].set_title("Exact MPS evaluation, chain dependence", loc="left", fontsize=10)
gn = [r["n"] for r in G if r["n"] <= 40]; gp = [r["all_pay"] for r in G if r["n"] <= 40]
ax[2].plot(n, [r["all_pay"] for r in S], color=C[2], marker="o", lw=2, label="Chain (neighbour) shocks — MPS")
ax[2].plot(gn, gp, color=C[1], marker="s", lw=2, ls="--", label="Single global shock — paper 1 model")
ax[2].set_xlabel("Number of countries n"); ax[2].set_ylabel("P(all countries pay in term)")
ax[2].set_title("Dependence structure changes the joint tail", loc="left", fontsize=10); ax[2].legend(frameon=False, fontsize=7)
ax[2].set_yscale("log")
fig.tight_layout()
fig.savefig("/home/claude/mps/figures/fig_mps.png", dpi=200, facecolor=SURF)
print("ok")
