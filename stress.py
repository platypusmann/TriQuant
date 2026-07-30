#!/usr/bin/env python
"""Stress-test finalist configs against MANY window slicings to guard against
overfitting to the 3x125 split: different window lengths, offsets, and an
anchored expanding walk-forward.  Report worst-window score for each slicing."""
import numpy as np
import harness as H
from strat import make_strategy

prcAll = H.load_prices()
nInst, nt = prcAll.shape

def worst_over(cfg, windows):
    scores = []
    for (s, e) in windows:
        strat = make_strategy(cfg)
        r = H.run_window(prcAll, strat, s, e)
        scores.append(r["score"])
    return min(scores), np.mean(scores), scores

def slicing_sets():
    sets = {}
    # 3 x 125 (the primary set)
    sets["3x125"] = H.sub_windows(nt, n=3, test_len=125)
    # 5 x 100 non-overlapping
    sets["5x100"] = H.sub_windows(nt, n=5, test_len=100)
    # 4 x 100 offset by 50 (overlapping-ish coverage, distinct starts)
    w = []
    s = 100
    while s + 100 <= nt:
        w.append((s, s + 100)); s += 100
    sets["seq100"] = w
    # rolling 100-day windows every 50 days (anchored-ish, overlapping)
    w = []
    s = 150
    while s + 100 <= nt:
        w.append((s, s + 100)); s += 50
    sets["roll100/50"] = w
    # rolling 75-day windows every 75 days -- DROP windows starting inside the
    # warmup (t<~140) where every strategy is flat (mu=sigma=0 -> score 0).
    sets["75-postwarm"] = [w for w in H.sub_windows(nt, n=6, test_len=75)
                            if w[0] >= 140]
    # anchored walk-forward, 125-day test, step 62
    w = []
    s = 250
    while s + 125 <= nt:
        w.append((s, s + 125)); s += 62
    if not w or w[-1][1] != nt:
        w.append((nt - 125, nt))
    sets["wf125/62"] = w
    return sets

FINALISTS = {
    "REFERENCE":            {},
    "e1.4 s0.35 g900k":     {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 900_000.0},
    "e1.4 s0.35 g1.0M":     {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 1_000_000.0},
    "e1.4 s0.35 g1.1M":     {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 1_100_000.0},
    "e1.4 s0.35 g1.3M":     {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 1_300_000.0},
    "e1.3 s0.35 g1.0M":     {"ENTRY": 1.3, "SMOOTH": 0.35, "GROSS": 1_000_000.0},
    "e1.5 s0.35 g1.0M":     {"ENTRY": 1.5, "SMOOTH": 0.35, "GROSS": 1_000_000.0},
    "e1.4 s0.35 g1.9M":     {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 1_900_000.0},
}

sets = slicing_sets()
print(f"nt={nt}. Slicing sets:")
for k, v in sets.items():
    print(f"  {k:12s} ({len(v)} wdw): {v}")

print("\n" + "=" * 88)
header = f"{'CONFIG':22s}" + "".join(f"{k:>13s}" for k in sets)
print(header)
print("-" * 88)
for label, cfg in FINALISTS.items():
    row = f"{label:22s}"
    worsts = []
    for k, wins in sets.items():
        wmin, wmean, _ = worst_over(cfg, wins)
        row += f"{wmin:>13.1f}"
        worsts.append(wmin)
    row += f"   | min-of-worsts={min(worsts):.1f}"
    print(row)
print("\n(each cell = WORST-window score over that slicing; higher is more robust)")
