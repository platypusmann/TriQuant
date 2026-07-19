#!/usr/bin/env python
"""Sweep strategy configs on the walk-forward harness, ranked by WORST-window
score across 3 non-overlapping sub-windows (+ report full-250 for reference)."""
import numpy as np
import harness as H
from strat import make_strategy, REF_CFG

prcAll = H.load_prices()
nInst, nt = prcAll.shape

SUBS = H.sub_windows(nt, n=3, test_len=125)     # [(125,250),(250,375),(375,500)]
FULL = (nt - 250, nt)

def eval_cfg(cfg, label):
    per = []
    for (s, e) in SUBS:
        strat = make_strategy(cfg)
        r = H.run_window(prcAll, strat, s, e)
        r["window"] = (s, e)
        per.append(r)
    # full-250
    strat = make_strategy(cfg)
    rf = H.run_window(prcAll, strat, *FULL)
    worst = min(r["score"] for r in per)
    return dict(label=label, per=per, full=rf, worst=worst,
                mean=float(np.mean([r["score"] for r in per])))

def show(res):
    print(f"\n=== {res['label']} ===")
    print(f"{'window':>14} {'score':>8} {'sharpe':>7} {'mu':>8} {'sigma':>9} {'turn/day':>10}")
    for r in res["per"]:
        s, e = r["window"]
        print(f"{f'({s},{e}]':>14} {r['score']:>8.2f} {r['sharpe']:>7.2f} "
              f"{r['mu']:>8.1f} {r['sigma']:>9.1f} {r['mean_turnover']:>10.0f}")
    rf = res["full"]
    print(f"{'FULL(250,500]':>14} {rf['score']:>8.2f} {rf['sharpe']:>7.2f} "
          f"{rf['mu']:>8.1f} {rf['sigma']:>9.1f} {rf['mean_turnover']:>10.0f}")
    print(f"  >>> WORST sub-window: {res['worst']:.2f}   mean sub: {res['mean']:.2f}   "
          f"full-250: {rf['score']:.2f}")

if __name__ == "__main__":
    import sys
    # baseline via parameterised strat (must match reference exactly)
    results = []
    results.append(eval_cfg({}, "BASELINE (params=reference)"))

    # define candidate configs to test (each a delta from reference)
    candidates = eval(open(sys.argv[1]).read()) if len(sys.argv) > 1 else []
    for label, cfg in candidates:
        results.append(eval_cfg(cfg, label))

    for res in results:
        show(res)

    # ranking table
    print("\n\n" + "=" * 78)
    print(f"{'CONFIG':40s} {'worst':>8} {'meanSub':>8} {'full':>8}")
    print("-" * 78)
    base_worst = results[0]["worst"]
    for res in sorted(results, key=lambda r: -r["worst"]):
        flag = ""
        if res["worst"] > base_worst + 1e-6:
            flag = "  <== beats baseline worst"
        print(f"{res['label'][:40]:40s} {res['worst']:>8.2f} "
              f"{res['mean']:>8.2f} {res['full']['score']:>8.2f}{flag}")
    print(f"\nBaseline worst-window to beat: {base_worst:.2f}")
