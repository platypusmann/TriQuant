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
#   - Score = mean_daily_PnL * SR^2/(SR^2+1), so score can never exceed mean
#     daily PnL. With every name capped at $10k the whole constituent book is
#     capped at $500k, which bounds how large a score is reachable at all.
#
# THE HEDGE MUST BE SIZED ON THE BOOK WE ACTUALLY HOLD
#   This is the single most important correctness point in this file. The
#   grader clips every constituent to POSLIM dollars. Sizing the ALGO hedge
#   against the *pre-clip* target therefore hedges positions we do not own:
#   measured over days 625-750, intended net beta exposure was +-$6.3k while
#   REALISED exposure was +-$56.4k, a 9x error, with a persistent net short of
#   $16k. That is why 4 of the 5 worst days in that window were days the market
#   rallied +1.5% to +3.0%, and why the strategy lost $237/day across a regime
#   in which the reversal edge itself was working fine (residual AC -0.021).
#   Clipping the target ourselves before computing the hedge cuts mean absolute
#   residual beta from $56,397 to $610 (92x) and unpins ALGO from its cap
#   (41% of days -> 3%). ALGO is set directly rather than eased into, because
#   its commission is 1/5 and tracking the hedge exactly is cheap.
#
# HOW THE PARAMETERS WERE CHOSEN
#   The edge is regime-dependent, so a config is judged on its WORST window
#   across many slicings of the full 1000-day history (see stress1000.py),
#   never on the full-250 headline. Chasing that headline is exactly how the
#   previous config went wrong: it scored 149.7 on the last 250 days while
#   making only $71/day at Sharpe 0.77 across all 500 genuinely unseen days.
#
#   ENTRY   = 1.8   z deadband. Higher than it looks like it should be: across
#                   six slicings ENTRY 1.8 gives min-of-worsts -84 vs -160 at
#                   1.4 and -190 at 1.6. (ENTRY 2.0 looks better still on one
#                   slicing at -7.6, but is -120.6 across all six -- a single-
#                   slicing artifact of exactly the kind that burned us before.)
#   GROSS   = 900k  Down from 1.9M. Once the hedge is correct, a smaller book
#                   is both safer and better: min-of-worsts -84.3, all-unseen
#                   score 85.1 at Sharpe 1.77, versus -388.5 / 26.2 / 0.77 for
#                   the old 1.9M config. Larger GROSS mostly buys more clipping,
#                   and clipping is what corrupts the hedge.
#   SMOOTH  = 0.35  Inertia on the constituent book (ALGO bypasses it).
#   NFACT   = 5     Inherited; the score is genuinely sensitive to both, so
#   BETAWIN = 120   these are left alone rather than fitted.
#   ZWIN=15, VOLWIN=60, ZCLIP=3.0 also inherited.
#
# HONEST STATE OF THE EVIDENCE
#   - The hedge fix is validated OUT OF SAMPLE: measured at the untouched old
#     parameters on 500 days the config had never seen, it moved the all-unseen
#     score from 26.2 to 39.8 before any retuning.
#   - The ENTRY/GROSS retune on top is fitted to all 1000 visible days and is
#     therefore NOT out-of-sample evidence. It is a reasonable bet, not a
#     measured result. Anything added from here should hold out days 750-1000.
#   - The worst window is still NEGATIVE (-84). This strategy has regimes in
#     which it loses money. The bleeding is ~5x smaller, not cured.
#
# TRIED AND REJECTED (do not reintroduce)
#   - GATE ('autocorr' / 'hitrate') and VOLTGT. Re-tested on the full 1000 days
#     after a real loss regime became visible, in case the earlier verdict was
#     an artifact of a too-easy sample: every variant was still worse on the
#     worst window (-77 to -91 vs -70 for plain). They shrink gross during the
#     quiet regime the ENTRY deadband already handles.
#
# SUBMISSION HARDENING (guards; no effect on clean data)
#   - Non-finite / non-positive prices are forward-filled; a name with no usable
#     price is held flat. Previously one bad tick turned the book to NaN, which
#     .astype(int) rendered as INT_MIN and the grader's clip turned into every
#     name pinned at max short, permanently.
#   - Positions are finite-clamped before the integer cast.
#   - On any unexpected failure we hold yesterday's book instead of raising.
#   - All dimensions derive from prcSoFar.shape; degenerate universes return a
#     correctly sized flat book.
#   - State re-initialises on the first call, a universe-size change, or a
#     non-contiguous call sequence, so a restarted pass cannot inherit a stale
#     book. A normal eval.py run is strictly contiguous, so this never fires.
#
# ASSUMPTION: POSLIM / POSLIM0 below mirror the position limits in the problem
# spec (eval.py). If a future stage changes those limits, update them here --
# the hedge is sized against them.
# ============================================================================

# --- hyperparameters (worst-window robustness, not full-250 peak) ------------
NFACT   = 5           # PCA factors removed from constituent returns
ZWIN    = 15          # lookback (days) for the residual-spread z-score
VOLWIN  = 60          # lookback (days) for inverse-idiosyncratic-vol sizing
GROSS   = 900_000.0   # target gross dollar exposure across the constituent book
SMOOTH  = 0.35        # fraction of the gap to target we close each day (inertia)
ZCLIP   = 3.0         # winsorise z-scores to tame outliers
BETAWIN = 120         # lookback for factor loadings and ALGO betas
ENTRY   = 1.8         # z deadband: ignore |z| < ENTRY (turnover / commission cut)
WARMUP  = ZWIN + BETAWIN + 5   # 140 days of history before the first live trade
POSLIM  = 10_000.0    # per-name dollar position limit (problem spec)
POSLIM0 = 100_000.0   # ALGO dollar position limit (problem spec)
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
    idx = np.where(~bad, np.arange(t)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)      # carry last good index
    first_good = np.argmax(~bad, axis=1)             # back-fill leading gaps
    idx = np.where(idx < 0, first_good[:, None], idx)
    prc = np.take_along_axis(prc, idx, axis=1)
    prc[dead] = 1.0        # harmless placeholder; these names are forced flat
    return prc, dead


def _target_book(prc, dead):
    """Desired CONSTITUENT book in shares, plus the ALGO betas.

    target[0] is deliberately left at zero: the hedge is applied later, against
    the post-clip book we will actually hold.
    """
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

    # 4. betas vs ALGO, used by the caller to hedge the realised book
    algo_var = algo_r[-BETAWIN:].var() + 1e-12
    betas = np.array([np.cov(rets[i, -BETAWIN:], algo_r[-BETAWIN:])[0, 1] / algo_var
                      for i in names])
    return target, betas


def getMyPosition(prcSoFar):
    global _prev, _last_t

    prc = np.asarray(prcSoFar, dtype=float)
    if prc.ndim != 2:
        return np.zeros(0, dtype=int)
    nins, t = prc.shape

    # --- state validation: first call, universe change, or a non-contiguous
    #     call sequence all start from a flat book, so a fresh pass can never
    #     silently inherit stale positions.
    if (_prev is None or _prev.shape[0] != nins
            or _last_t is None or t != _last_t + 1):
        _prev = np.zeros(nins)
    _last_t = t

    # --- warmup: stay flat until every estimator has enough history ----------
    if t < WARMUP or nins < 3:
        _prev = np.zeros(nins)
        return np.zeros(nins, dtype=int)

    prc, dead = _sanitise(prc)
    px = prc[:, -1]

    try:
        target, betas = _target_book(prc, dead)
    except Exception:
        target, betas = None, None

    # unusable target -> hold yesterday's book rather than trade on garbage
    if target is None or target.shape != (nins,) or not np.all(np.isfinite(target)):
        target, betas = _prev.copy(), None

    # --- inertia on the constituent book ------------------------------------
    newpos = _prev + SMOOTH * (target - _prev)

    # --- hedge the book we ACTUALLY hold ------------------------------------
    # Apply the grader's per-name clip ourselves, then size ALGO against what
    # survives it. ALGO is set directly (not smoothed): its commission is 1/5,
    # so tracking the hedge exactly is cheap, and lagging it re-introduces the
    # very beta drift this is here to remove.
    if betas is not None and not dead[0]:
        names = np.arange(1, nins)
        lim_sh = POSLIM / px[names]
        held = np.clip(newpos[names], -lim_sh, lim_sh)
        net_beta_dollars = np.sum(betas * (held * px[names]))
        algo_sh = -net_beta_dollars / px[0]
        lim0 = POSLIM0 / px[0]
        if np.isfinite(algo_sh) and np.isfinite(lim0):
            newpos[0] = float(np.clip(algo_sh, -lim0, lim0))

    if not np.all(np.isfinite(newpos)):
        newpos = np.where(np.isfinite(newpos), newpos, 0.0)
    np.clip(newpos, -_MAXPOS, _MAXPOS, out=newpos)
    _prev = newpos.copy()
    return newpos.astype(int)
