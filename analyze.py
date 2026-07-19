#!/usr/bin/env python
"""Diagnostics on the price file to inform robust improvements (no fitting to
specific days -- just understanding the structure)."""
import numpy as np
import harness as H

prc = H.load_prices()
nInst, nt = prc.shape
logp = np.log(prc)
rets = np.diff(logp, axis=1)          # 51 x 499
algo = rets[0]
names = np.arange(1, nInst)
cons = rets[names]                     # 50 x 499

# 1. ALGO vs equal-weight basket R^2 (confirm ALGO IS the index)
basket = cons.mean(axis=0)
b = np.polyfit(basket, algo, 1)
pred = np.polyval(b, basket)
r2 = 1 - np.var(algo - pred) / np.var(algo)
print(f"ALGO vs EW-basket R^2 = {r2:.4f}, slope={b[0]:.3f}")
print(f"ALGO daily vol={algo.std():.5f}  basket vol={basket.std():.5f}  "
      f"mean constituent vol={cons.std(axis=1).mean():.5f}")

# 2. How much variance does ALGO explain vs a 5-factor PCA on constituents?
def var_explained_by_algo(block):
    # regress each constituent on algo over the block, residual variance frac
    a = block[0]
    c = block[names]
    A = np.vstack([a, np.ones_like(a)]).T
    tot, res = 0.0, 0.0
    for i in range(c.shape[0]):
        coef, *_ = np.linalg.lstsq(A, c[i], rcond=None)
        r = c[i] - A @ coef
        tot += c[i].var(); res += r.var()
    return 1 - res / tot

print(f"\nFrac of constituent var explained by ALGO (full): "
      f"{var_explained_by_algo(rets):.3f}")

# 3. Residual short-term reversal strength by regime.
# Build ALGO-neutral residuals, cumulate, measure lag-1 autocorr of the
# residual spread changes (reversal => negative autocorr of residual returns).
def resid_algo(block):
    a = block[0]; c = block[names]
    A = np.vstack([a, np.ones_like(a)]).T
    R = np.zeros_like(c)
    for i in range(c.shape[0]):
        coef, *_ = np.linalg.lstsq(A, c[i], rcond=None)
        R[i] = c[i] - A @ coef
    return R  # 50 x T

def reversal_autocorr(R):
    # average lag-1 autocorrelation of residual returns across names
    acs = []
    for i in range(R.shape[0]):
        x = R[i] - R[i].mean()
        if x[:-1].std() > 0:
            acs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return np.mean(acs)

# regime blocks: split the 499 returns into thirds mirroring the day windows
for (lo, hi, tag) in [(0, 249, "days~1-250"),
                      (124, 374, "days~125-375 (quiet mid)"),
                      (249, 499, "days~250-500 (recent)"),
                      (124, 249, "sub (125,250]"),
                      (249, 374, "sub (250,375]"),
                      (374, 499, "sub (375,500]")]:
    block = rets[:, lo:hi]
    R = resid_algo(block)
    ac = reversal_autocorr(R)
    print(f"  {tag:26s} lag1 residual autocorr = {ac:+.3f}  "
          f"(more negative = stronger reversal)")

# 4. compare PCA-5 neutralization residual reversal vs ALGO regression
def resid_pca(block, nfact=5):
    c = block[names]
    Rc = c - c.mean(axis=1, keepdims=True)
    C = np.cov(Rc)
    wv, V = np.linalg.eigh(C)
    B = V[:, np.argsort(wv)[::-1][:nfact]]
    return (Rc) - B @ (B.T @ Rc)

print("\nReversal autocorr: PCA-5 vs ALGO-regression neutralization")
for (lo, hi, tag) in [(124, 249, "(125,250]"),
                      (249, 374, "(250,375]"),
                      (374, 499, "(375,500]")]:
    block = rets[:, lo:hi]
    print(f"  {tag}:  PCA5={reversal_autocorr(resid_pca(block)):+.3f}   "
          f"ALGO={reversal_autocorr(resid_algo(block)):+.3f}")
