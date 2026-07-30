#!/usr/bin/env python
"""Stress finalists across MANY window slicings of the full 1000-day file.

Same philosophy as stress.py: a config must survive the worst window of every
slicing, not just the one slicing it was tuned on.  Rebuilt for nt=1000 so the
windows span the whole history (including the loss regime around days 625-750)
rather than only the recent tail.
"""
import numpy as np
import harness as H
from strat import make_strategy

prcAll = H.load_prices()
nInst, nt = prcAll.shape


def slicing_sets():
    sets = {}
    for lab, ln in [("x125", 125), ("x100", 100), ("x200", 200)]:
        w, s = [], 125
        while s + ln <= nt:
            w.append((s, s + ln)); s += ln
        sets[lab] = w
    # offset packs: same lengths, shifted start, so window boundaries differ
    w, s = [], 187
    while s + 125 <= nt:
        w.append((s, s + 125)); s += 125
    sets["x125off"] = w
    # rolling, overlapping
    w, s = [], 150
    while s + 125 <= nt:
        w.append((s, s + 125)); s += 62
    sets["roll125"] = w
    # short windows, post-warmup only
    w, s = [], 150
    while s + 75 <= nt:
        w.append((s, s + 75)); s += 75
    sets["x75"] = w
    return sets


FINALISTS = {
    "SHIPPED e1.4 g1.9M nofix": {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 1_900_000.0},
    "e1.4 g0.7M fix":  {"ENTRY": 1.4, "SMOOTH": 0.35, "GROSS": 700_000.0, "HEDGE_CLIPPED": True},
    "e1.6 g0.7M fix":  {"ENTRY": 1.6, "SMOOTH": 0.35, "GROSS": 700_000.0, "HEDGE_CLIPPED": True},
    "e1.8 g0.7M fix":  {"ENTRY": 1.8, "SMOOTH": 0.35, "GROSS": 700_000.0, "HEDGE_CLIPPED": True},
    "e2.0 g0.7M fix":  {"ENTRY": 2.0, "SMOOTH": 0.35, "GROSS": 700_000.0, "HEDGE_CLIPPED": True},
    "e1.8 g0.5M fix":  {"ENTRY": 1.8, "SMOOTH": 0.35, "GROSS": 500_000.0, "HEDGE_CLIPPED": True},
    "e1.8 g0.9M fix":  {"ENTRY": 1.8, "SMOOTH": 0.35, "GROSS": 900_000.0, "HEDGE_CLIPPED": True},
    "e2.0 g0.9M fix":  {"ENTRY": 2.0, "SMOOTH": 0.35, "GROSS": 900_000.0, "HEDGE_CLIPPED": True},
}

if __name__ == "__main__":
    sets = slicing_sets()
    print(f"nt={nt}. Slicings:")
    for k, v in sets.items():
        print(f"  {k:9s} ({len(v)} wdw)")
    print("\n" + "=" * 104)
    print(f"{'CONFIG':26s}" + "".join(f"{k:>11s}" for k in sets) + f"{'min-of-worsts':>15s}")
    print("-" * 104)
    rows = []
    for label, cfg in FINALISTS.items():
        row, worsts = f"{label:26s}", []
        for k, wins in sets.items():
            wmin = min(H.run_window(prcAll, make_strategy(cfg), s, e)["score"]
                       for (s, e) in wins)
            row += f"{wmin:>11.1f}"
            worsts.append(wmin)
        mow = min(worsts)
        rows.append((mow, label))
        print(row + f"{mow:>15.1f}")
    print("\n(each cell = WORST window score in that slicing; higher = more robust)")
    print("ranked by min-of-worsts:")
    for mow, label in sorted(rows, reverse=True):
        print(f"  {mow:>8.1f}  {label}")
