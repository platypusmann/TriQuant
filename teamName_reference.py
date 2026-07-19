import numpy as np

# ============================================================================
# Factor-neutral short-term reversal (stat-arb), hedged with ALGO (instrument 0)
#
# Core idea:
#   - Instrument 0 (ALGO) is the market index: ~99.5 percent R-squared vs the
#     equal-weight basket of the other 50 names, tiny idiosyncratic vol, plus a
#     10x position limit and 1/5 commission. So we do not seek alpha in ALGO.
#     We use it as a cheap, high-capacity instrument to hedge the book's beta.
#   - The 50 constituents carry a weak but persistent idiosyncratic reversal:
#     strip out the common factors, cumulate the residual into a mean-reverting
#     spread, z-score it, and fade the deviation (buy losers, sell winners).
#   - Score = mean_daily_PnL * Sharpe^2 / (Sharpe^2 + 1). Sharpe is invariant to
#     book scale, so we size up to the position limits while keeping the book
#     factor-neutral to hold Sharpe high, and we smooth positions to avoid
#     trading noise (commission and variance drag).
# ============================================================================

nInst = 51

# --- hyperparameters (tune with walk-forward, do NOT overfit the last 250) ---
NFACT  = 5          # PCA factors removed from constituent returns
ZWIN   = 15         # lookback (days) for the residual-spread z-score
VOLWIN = 60         # lookback (days) for inverse-vol sizing
GROSS  = 600_000.0  # target gross dollar exposure across the constituent book
SMOOTH = 0.45       # fraction of the gap to target we close each day (inertia)
ZCLIP  = 3.0        # winsorise z-scores to tame outliers
BETAWIN = 120       # lookback for factor loadings and ALGO betas
# ----------------------------------------------------------------------------

_prev = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global _prev
    nins, t = prcSoFar.shape
    if t < ZWIN + BETAWIN + 5:
        return np.zeros(nins).astype(int)

    logp = np.log(prcSoFar)
    rets = np.diff(logp, axis=1)               # nins x (t-1) log returns
    names = np.arange(1, nins)                  # the 50 non-ALGO constituents
    algo_r = rets[0]

    # 1. estimate common factors from the constituent covariance and remove them
    win = rets[:, -BETAWIN:]
    Rc = win - win.mean(axis=1, keepdims=True)
    C = np.cov(Rc[names])
    wv, V = np.linalg.eigh(C)
    B = V[:, np.argsort(wv)[::-1][:NFACT]]       # top-NFACT loadings

    sub = rets[names]
    mu = sub.mean(axis=1, keepdims=True)
    resid = (sub - mu) - B @ (B.T @ (sub - mu))  # idiosyncratic returns

    # 2. cumulate residuals into a mean-reverting spread and z-score it
    spread = np.cumsum(resid, axis=1)
    s = spread[:, -ZWIN:]
    z = (spread[:, -1] - s.mean(axis=1)) / (s.std(axis=1) + 1e-9)
    z = np.clip(z, -ZCLIP, ZCLIP)
    signal = -z                                  # contrarian: fade the deviation

    # 3. size: inverse idiosyncratic vol, dollar-neutral, scaled to GROSS
    vol = resid[:, -VOLWIN:].std(axis=1) + 1e-9
    raw = signal / vol
    raw = raw - raw.mean()                        # net-zero dollar tilt
    denom = np.abs(raw).sum() + 1e-9
    dollars = raw / denom * GROSS

    px = prcSoFar[:, -1]
    target = np.zeros(nins)
    target[names] = dollars / px[names]

    # 4. hedge net market beta with ALGO (cheap, 10x limit)
    algo_var = algo_r[-BETAWIN:].var() + 1e-12
    betas = np.array([np.cov(rets[i, -BETAWIN:], algo_r[-BETAWIN:])[0, 1] / algo_var
                      for i in names])
    net_beta_dollars = np.sum(betas * dollars)
    target[0] = -net_beta_dollars / px[0]

    # 5. move partway to target (inertia cuts turnover and PnL variance)
    newpos = _prev + SMOOTH * (target - _prev)
    _prev = newpos.copy()
    return newpos.astype(int)
