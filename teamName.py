import numpy as np

# ============================================================================
# Factor-neutral short-term reversal (stat-arb), hedged with ALGO (instrument 0)
#
# CORE IDEA
#   - ALGO (instrument 0) is the market index: ~99% R^2 vs the equal-weight
#     basket of the other 50 names, near-zero idiosyncratic vol, a 10x position
#     limit (100k vs 10k) and 1/5 the commission (0.2bp vs 1bp). We never seek
#     alpha in ALGO; we use it purely as a cheap beta hedge.
#   - The 50 constituents carry a weak, REGIME-DEPENDENT factor-neutral
#     reversal. Strip the common factors out with PCA, cumulate the residual
#     into a mean-reverting spread, z-score it, and fade the deviation.
#   - Score = mean_daily_PnL * SR^2/(SR^2+1). Sharpe is scale-invariant, so the
#     score rewards a bigger book as long as Sharpe holds -> size up to, but not
#     past, the point where position limits and the ALGO hedge saturate.
#
# HOW THE PARAMETERS WERE CHOSEN
#   Residual autocorrelation is strongly negative in some sub-windows and
#   near-zero in others, so a config that looks good on the full 250 days can be
#   pure regime luck. Every number below was picked by its WORST window score
#   across SIX different slicings (3x125, 5x100, seq100, roll100/50,
#   75-postwarm, wf125/62) -- "min-of-worsts" -- never on the full-250 alone.
#
#   ENTRY   = 1.4   z deadband. Only trade names whose deviation is meaningfully
#                   large; kills noise churn where there is no edge. Commission
#                   drag in the quiet window falls to ~8% (reference: ~21%).
#   GROSS   = 1.9M  A dense grid (ENTRY {1.3,1.4,1.5} x GROSS 900k-3.0M, all six
#                   slicings) shows a single smooth peak here: min-of-worsts 39.0
#                   vs 27.8 for the previous best (ENTRY 1.3 / GROSS 1.0M) and
#                   0.7 for the un-tuned reference. Above ~2.2M it falls off a
#                   cliff as the ALGO hedge saturates its 100k cap. The book runs
#                   55-65% clipped at per-name limits, so much of the nominal
#                   target is deliberately inert -- that is understood, not a bug.
#   SMOOTH  = 0.35  Inertia. With the deadband cutting turnover, lighter inertia
#                   tracks the cleaner target without adding churn.
#   NFACT   = 5     Kept from the reference; the score is genuinely sensitive to
#   BETAWIN = 120   both, so these are the fragile knobs -- we do not push them
#                   and we do not add leverage that amplifies their cliff.
#   ZWIN=15, VOLWIN=60, ZCLIP=3.0 also inherited from the reference.
#
#   Final numbers: full-250 score 131.35 (Sharpe 1.97), min-of-worsts 39.0,
#   3x125 breakdown 94.8 / 57.4 / 184.4.
#
# TRIED AND REJECTED (do not reintroduce)
#   - GATE ('autocorr' / 'hitrate') edge gating: collapses min-of-worsts to ~2.
#   - VOLTGT (vol targeting on realised daily PnL): collapses it to ~0-1.
#   Both shrink gross exactly during the quiet/choppy regime that the ENTRY
#   deadband was already built to survive, so they de-risk at the worst moment
#   and double-count a fragility this config already handles.
#
# SUBMISSION HARDENING (guards only -- the strategy math is untouched, and
# output is bit-identical to the tuned version on clean data)
#   - Prices are sanitised: non-finite / non-positive values are forward-filled,
#     and a name with no usable price is held flat. One bad tick used to turn
#     the whole book to NaN, which .astype(int) rendered as INT_MIN and the
#     grader's clip turned into every name pinned at max short -- permanently,
#     since the state stayed poisoned. That path is now unreachable.
#   - Positions are finite-clamped before the integer cast, so a non-finite
#     value can never reach the grader as INT_MIN garbage.
#   - The model is wrapped: on any unexpected failure we HOLD yesterday's book
#     (no turnover, no commission) instead of raising.
#   - All dimensions derive from prcSoFar.shape; nothing is hardcoded to 51
#     instruments. Degenerate universes return a correctly-sized flat book.
#   - State is validated on every call: it re-initialises on the first call, on a
#     universe-size change, or if the call sequence is not a contiguous forward
#     walk -- so a restarted or replayed pass cannot inherit a stale book. A
#     normal eval.py run is strictly contiguous, so this never fires there.
# ============================================================================

# --- hyperparameters (worst-window robustness, not full-250 peak) ------------
NFACT   = 5           # PCA factors removed from constituent returns
ZWIN    = 15          # lookback (days) for the residual-spread z-score
VOLWIN  = 60          # lookback (days) for inverse-idiosyncratic-vol sizing
GROSS   = 1_900_000.0 # target gross dollar exposure across the constituent book
SMOOTH  = 0.35        # fraction of the gap to target we close each day (inertia)
ZCLIP   = 3.0         # winsorise z-scores to tame outliers
BETAWIN = 120         # lookback for factor loadings and ALGO betas
ENTRY   = 1.4         # z deadband: ignore |z| < ENTRY (turnover / commission cut)
WARMUP  = ZWIN + BETAWIN + 5   # 140 days of history before the first live trade
# -----------------------------------------------------------------------------

_MAXPOS = 1e12        # finite clamp so the integer cast can never emit INT_MIN

# --- state, lazily sized to the observed universe and validated every call ---
_prev = None          # yesterday's book in (fractional) shares
_last_t = None        # day count seen on the previous call, for the contiguity check


def _sanitise(prc):
    """Replace non-finite / non-positive prices with the last good value.

    Returns (clean_prices, dead_mask); a name with no usable price anywhere is
    flagged dead and held flat by the caller. Strict no-op on clean data.
    """
    bad = ~np.isfinite(prc) | (prc <= 0.0)
    if not bad.any():
        return prc, np.zeros(prc.shape[0], dtype=bool)

    t = prc.shape[1]
    dead = bad.all(axis=1)
    # forward-fill: carry the index of the most recent good observation
    idx = np.where(~bad, np.arange(t)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    # leading bad values have no earlier good price -> back-fill from the first
    first_good = np.argmax(~bad, axis=1)
    idx = np.where(idx < 0, first_good[:, None], idx)
    prc = np.take_along_axis(prc, idx, axis=1)
    prc[dead] = 1.0        # harmless placeholder; these names are forced flat
    return prc, dead


def _target_book(prc, dead):
    """Desired book in shares. Identical math to the tuned config."""
    nins = prc.shape[0]
    names = np.arange(1, nins)

    logp = np.log(prc)
    rets = np.diff(logp, axis=1)                 # nins x (t-1) log returns
    algo_r = rets[0]
    px = prc[:, -1]

    # 1. estimate the common factors from the constituent covariance, remove them
    win = rets[:, -BETAWIN:]
    Rc = win - win.mean(axis=1, keepdims=True)
    C = np.cov(Rc[names])
    wv, V = np.linalg.eigh(C)
    nf = int(min(NFACT, len(names) - 1))          # clamp for tiny universes
    B = V[:, np.argsort(wv)[::-1][:nf]]           # top-nf loadings

    sub = rets[names]
    mu = sub.mean(axis=1, keepdims=True)
    resid = (sub - mu) - B @ (B.T @ (sub - mu))   # idiosyncratic returns

    # 2. cumulate residuals into a mean-reverting spread and z-score it
    spread = np.cumsum(resid, axis=1)
    s = spread[:, -ZWIN:]
    z = (spread[:, -1] - s.mean(axis=1)) / (s.std(axis=1) + 1e-9)
    z = np.clip(z, -ZCLIP, ZCLIP)

    # entry deadband: only act on names whose deviation is meaningfully large
    z = np.where(np.abs(z) < ENTRY, 0.0, z)
    signal = -z                                   # contrarian: fade the deviation

    # 3. size: inverse idiosyncratic vol, dollar-neutral, scaled to GROSS
    vol = resid[:, -VOLWIN:].std(axis=1) + 1e-9
    raw = signal / vol
    if dead.any():
        raw[dead[names]] = 0.0                    # never position a dead name
    raw = raw - raw.mean()                        # net-zero dollar tilt
    denom = np.abs(raw).sum() + 1e-9
    dollars = raw / denom * GROSS

    target = np.zeros(nins)
    target[names] = dollars / px[names]

    # 4. hedge net market beta with ALGO (cheap, 10x limit). On this file the
    #    PCA residual is already near-neutral so the hedge barely moves the
    #    in-sample score, but it is the only structural guard against a future
    #    regime where the book picks up market beta.
    algo_var = algo_r[-BETAWIN:].var() + 1e-12
    betas = np.array([np.cov(rets[i, -BETAWIN:], algo_r[-BETAWIN:])[0, 1] / algo_var
                      for i in names])
    net_beta_dollars = np.sum(betas * dollars)
    if not dead[0]:
        target[0] = -net_beta_dollars / px[0]
    return target


def getMyPosition(prcSoFar):
    global _prev, _last_t

    prc = np.asarray(prcSoFar, dtype=float)
    if prc.ndim != 2:
        return np.zeros(0, dtype=int)
    nins, t = prc.shape

    # --- state validation: first call, universe change, or a non-contiguous
    #     call sequence (restart / replay / gap) all start from a flat book, so
    #     a fresh pass can never silently inherit stale positions.
    if (_prev is None or _prev.shape[0] != nins
            or _last_t is None or t != _last_t + 1):
        _prev = np.zeros(nins)
    _last_t = t

    # --- warmup: stay flat until every estimator has enough history ----------
    if t < WARMUP or nins < 3:
        _prev = np.zeros(nins)
        return np.zeros(nins, dtype=int)

    prc, dead = _sanitise(prc)

    try:
        target = _target_book(prc, dead)
    except Exception:
        target = None

    # unusable target -> hold yesterday's book rather than trade on garbage
    if target is None or target.shape != (nins,) or not np.all(np.isfinite(target)):
        target = _prev.copy()

    # --- move partway to target (inertia cuts turnover and PnL variance) -----
    newpos = _prev + SMOOTH * (target - _prev)
    if not np.all(np.isfinite(newpos)):
        newpos = np.where(np.isfinite(newpos), newpos, 0.0)
    np.clip(newpos, -_MAXPOS, _MAXPOS, out=newpos)
    _prev = newpos.copy()
    return newpos.astype(int)
