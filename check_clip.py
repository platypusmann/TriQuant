#!/usr/bin/env python
"""Diagnose position-limit utilisation and book neutrality at a given config,
so we understand WHY gross helps and how close we run to the clipping cliff."""
import numpy as np
import harness as H
from strat import make_strategy

prcAll = H.load_prices()
nInst, nt = prcAll.shape
commRate, dlrPosLimit = H._limits(nInst)

def diagnose(cfg, label, start, end):
    strat = make_strategy(cfg)
    strat.reset()
    clip_frac = []      # fraction of names hitting the dollar limit each day
    net_dollar = []     # net dollar (should be ~0 if neutral)
    algo_frac = []      # ALGO usage vs its 100k limit
    gross_used = []
    for t in range(start, end):
        prc = prcAll[:, :t]
        px = prc[:, -1]
        raw = np.asarray(strat(prc))
        posLimits = (dlrPosLimit / px).astype(int)
        clipped = np.clip(raw, -posLimits, posLimits).astype(int)
        dollars = clipped * px
        n_at_limit = np.sum(np.abs(clipped[1:]) >= posLimits[1:] - 0)  # constituents
        clip_frac.append(n_at_limit / 50.0)
        net_dollar.append(dollars.sum())
        algo_frac.append(abs(dollars[0]) / 100_000.0)
        gross_used.append(np.abs(dollars[1:]).sum())
    print(f"{label:34s} clip%={np.mean(clip_frac):6.1%}  "
          f"maxclip%={np.max(clip_frac):6.1%}  "
          f"|net$|avg={np.mean(np.abs(net_dollar)):8.0f}  "
          f"algoUse={np.mean(algo_frac):5.1%}(max{np.max(algo_frac):4.0%})  "
          f"gross${np.mean(gross_used):8.0f}")

# quiet window is the binding one
for g in [600_000, 1_000_000, 1_600_000, 2_500_000]:
    cfg = {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": float(g)}
    diagnose(cfg, f"e1.4 s0.35 gross{g//1000}k [quiet 250-375]", 250, 375)
print()
for g in [600_000, 1_000_000, 1_600_000, 2_500_000]:
    cfg = {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": float(g)}
    diagnose(cfg, f"e1.4 s0.35 gross{g//1000}k [recent 375-500]", 375, 500)
