# Paper 2 — Tensor-network (MPS) joint-loss model, ADB Quantum Joint-Loss Challenge

```bash
python3 src/run_mps.py    # ADB reproduction (n=2), brute-force check (n=3,4), TT-SVD ranks, scaling to n=40  (~1 min)
python3 src/born_mps.py   # Born-MPS truncation: bond dimension vs target error and loader resources            (~20 s)
python3 src/figures.py    # figures/fig_mps.png (needs ../adbq/results/scaling.json from paper 1 for the right panel)
```

* `src/model.py` — ADB benchmark model (shared with paper 1), used as ground truth for n = 2.
* `src/mps_model.py` — chain model with neighbour-coupled shared earthquakes + global shared rain; exact MPS cores
  (bond 64 = shared-event counts of a link over 3 years, K <= 3 per link-year); exact contraction of all targets.
* `src/born_mps.py` — TT-SVD of sqrt(p), truncation study, sequential-preparation resource estimates
  (qubits 4n + ceil(log2 chi); CNOT/site from the generic 23/48·4^q bound, not transpiled).

Environment: Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11; 2-core CPU, no GPU, no quantum SDK.

AI use: code, analysis and drafting were assisted by an AI model (Claude, Anthropic). Edit to describe your team's use.
