#!/usr/bin/env python
"""Establish the reference strategy's robustness profile across sub-windows."""
import importlib
import numpy as np
import harness as H

prcAll = H.load_prices()
nInst, nt = prcAll.shape
print(f"Loaded {nInst} instruments x {nt} days")

# The reference module is stateful (global _prev). Build a *fresh* callable per
# window by reloading the module so smoothing state never leaks across windows.
import teamName_reference as ref

def make_ref():
    importlib.reload(ref)
    return ref.getMyPosition

# --- window sets ---
subs = H.sub_windows(nt, n=3, test_len=125)      # 3 non-overlapping recent windows
full = [(nt - 250, nt)]                           # eval.py's exact window
wf   = H.anchored_walkforward(nt, first_test_start=250, step=125, test_len=125)

print(f"\nsub-windows (3x125): {subs}")
print(f"full-250 window    : {full}")
print(f"anchored walk-fwd  : {wf}")

res_sub  = H.evaluate_config(prcAll, make_ref, subs,  label="REFERENCE  sub-windows")
res_full = H.evaluate_config(prcAll, make_ref, full,  label="REFERENCE  full-250")
res_wf   = H.evaluate_config(prcAll, make_ref, wf,    label="REFERENCE  walk-forward")

H.print_config_table(res_sub)
H.print_config_table(res_full)
H.print_config_table(res_wf)

print("\n\nBASELINE SUMMARY")
print(f"  full-250 score      : {res_full['per'][0]['score']:.2f}")
print(f"  worst sub-window    : {res_sub['worst_score']:.2f}")
print(f"  worst walk-fwd wdw  : {res_wf['worst_score']:.2f}")
