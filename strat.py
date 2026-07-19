#!/usr/bin/env python
"""Parameterised strategy factory so the harness can sweep configs.

make_strategy(cfg) returns a fresh stateful getMyPosition closure.  All ideas
from the brief are switchable via cfg so we can measure each on the walk-forward
harness before keeping it.  Defaults reproduce the reference exactly when
cfg == REF_CFG.
"""
import numpy as np

REF_CFG = dict(
    NFACT=5, ZWIN=15, VOLWIN=60, GROSS=600_000.0, SMOOTH=0.45,
    ZCLIP=3.0, BETAWIN=120,
    # --- new switches, all off by default (off == reference behaviour) ---
    NFACTS=None,         # list of factor counts to ensemble (avg z over them);
                         #   None -> single NFACT. Flattens NFACT-cliff overfit.
    ZWINS=None,          # list of z lookbacks to blend, e.g. [10,15,25]; None=single ZWIN
    ENTRY=0.0,           # deadband: zero the signal where |z|<ENTRY (cuts churn)
    EXIT=0.0,            # hysteresis floor (unused unless >0); kept for clarity
    GATE=None,           # None | 'hitrate' | 'autocorr' : scale gross by recent edge
    GATE_WIN=60,         # lookback for the edge gate
    GATE_FLOOR=0.30,     # min gross multiplier when edge is weakest
    GATE_CAP=1.0,        # max gross multiplier
    VOLTGT=None,         # None or target daily-PnL $ std; scales gross to hit it
    VOLTGT_WIN=40,       # lookback (days) for realised daily-PnL vol
    VOLTGT_CAP=1.6,      # max leverage multiplier from vol targeting
    BETA_ON_ALGO=True,   # hedge net beta with ALGO (True == reference)
)


def make_strategy(cfg=None):
    c = dict(REF_CFG)
    if cfg:
        c.update(cfg)
    NFACT, ZWIN, VOLWIN = c["NFACT"], c["ZWIN"], c["VOLWIN"]
    GROSS, SMOOTH, ZCLIP, BETAWIN = c["GROSS"], c["SMOOTH"], c["ZCLIP"], c["BETAWIN"]
    NFACTS = c["NFACTS"] if c["NFACTS"] else [NFACT]
    ZWINS = c["ZWINS"]
    ENTRY = c["ENTRY"]
    GATE, GATE_WIN, GATE_FLOOR, GATE_CAP = c["GATE"], c["GATE_WIN"], c["GATE_FLOOR"], c["GATE_CAP"]
    VOLTGT, VOLTGT_WIN, VOLTGT_CAP = c["VOLTGT"], c["VOLTGT_WIN"], c["VOLTGT_CAP"]
    BETA_ON_ALGO = c["BETA_ON_ALGO"]

    zwin_list = ZWINS if ZWINS else [ZWIN]
    max_zwin = max(zwin_list)
    warmup = max_zwin + BETAWIN + 5

    state = {"prev": None, "pnl_hist": [], "prev_target_dollars": None,
             "prev_px": None}

    def reset():
        state["prev"] = None
        state["pnl_hist"] = []
        state["prev_target_dollars"] = None
        state["prev_px"] = None

    def getMyPosition(prcSoFar):
        nins, t = prcSoFar.shape
        if state["prev"] is None:
            state["prev"] = np.zeros(nins)
        prev = state["prev"]

        if t < warmup:
            return np.zeros(nins).astype(int)

        logp = np.log(prcSoFar)
        rets = np.diff(logp, axis=1)
        names = np.arange(1, nins)
        algo_r = rets[0]
        px = prcSoFar[:, -1]

        # --- 1. PCA factor neutralisation on the constituents ---
        # eigendecompose once; slice the top-k loadings per NFACT in the ensemble.
        win = rets[:, -BETAWIN:]
        Rc = win - win.mean(axis=1, keepdims=True)
        C = np.cov(Rc[names])
        wv, V = np.linalg.eigh(C)
        order = np.argsort(wv)[::-1]
        Vsorted = V[:, order]

        sub = rets[names]
        mu_r = sub.mean(axis=1, keepdims=True)
        subc = sub - mu_r

        # --- 2. residual spread z-score, ensembled over NFACT and lookbacks ---
        # Averaging z across factor counts flattens the NFACT-cliff (no single
        # magic factor count), which is the main overfit risk in this signal.
        z_acc = np.zeros(len(names))
        n_terms = 0
        resid = None  # keep the max-NFACT residual for vol sizing (most neutral)
        for nf in NFACTS:
            B = Vsorted[:, :nf]
            resid_nf = subc - B @ (B.T @ subc)
            if resid is None or nf == max(NFACTS):
                resid = resid_nf
            spread = np.cumsum(resid_nf, axis=1)
            for w in zwin_list:
                s = spread[:, -w:]
                zz = (spread[:, -1] - s.mean(axis=1)) / (s.std(axis=1) + 1e-9)
                z_acc += np.clip(zz, -ZCLIP, ZCLIP)
                n_terms += 1
        z = z_acc / n_terms
        z = np.clip(z, -ZCLIP, ZCLIP)

        # entry deadband: ignore tiny deviations (turnover/commission control)
        if ENTRY > 0:
            z = np.where(np.abs(z) < ENTRY, 0.0, z)

        signal = -z

        # --- 3. inverse-vol sizing, dollar-neutral, scaled to GROSS ---
        vol = resid[:, -VOLWIN:].std(axis=1) + 1e-9
        raw = signal / vol
        raw = raw - raw.mean()
        denom = np.abs(raw).sum() + 1e-9

        gross_mult = 1.0

        # --- edge gate: shrink gross when the recent edge is weak ---
        if GATE == "autocorr":
            R = resid[:, -GATE_WIN:]
            acs = []
            for i in range(R.shape[0]):
                x = R[i] - R[i].mean()
                if x[:-1].std() > 1e-12:
                    acs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            ac = np.mean(acs) if acs else 0.0
            # reversal => ac<0.  map ac in [-0.05, 0] -> [CAP, FLOOR]
            frac = np.clip(-ac / 0.05, 0.0, 1.0)
            gross_mult = GATE_FLOOR + (GATE_CAP - GATE_FLOOR) * frac
        elif GATE == "hitrate":
            # did yesterday's signal predict today's residual reversal?  measure
            # rolling correlation between -z_{d-1} and resid_d over GATE_WIN.
            wlen = min(GATE_WIN, spread.shape[1] - max_zwin - 1)
            hits = []
            for d in range(spread.shape[1] - wlen, spread.shape[1]):
                if d - max_zwin < 1:
                    continue
                sp = spread[:, :d]
                s0 = sp[:, -ZWIN:]
                zd = (sp[:, -1] - s0.mean(axis=1)) / (s0.std(axis=1) + 1e-9)
                fwd = resid[:, d] if d < resid.shape[1] else None
                if fwd is None:
                    continue
                # signal = -zd should be positively correlated with fwd resid
                if np.std(zd) > 1e-9 and np.std(fwd) > 1e-12:
                    hits.append(np.corrcoef(-zd, fwd)[0, 1])
            hr = np.mean(hits) if hits else 0.0
            frac = np.clip(hr / 0.05, 0.0, 1.0)
            gross_mult = GATE_FLOOR + (GATE_CAP - GATE_FLOOR) * frac

        # --- vol targeting on realised daily PnL ---
        if VOLTGT is not None and len(state["pnl_hist"]) >= 5:
            recent = np.array(state["pnl_hist"][-VOLTGT_WIN:])
            realised = recent.std()
            if realised > 1e-6:
                vt = np.clip(VOLTGT / realised, 0.0, VOLTGT_CAP)
                gross_mult *= vt

        dollars = raw / denom * GROSS * gross_mult

        target = np.zeros(nins)
        target[names] = dollars / px[names]

        # --- 4. beta hedge with ALGO ---
        if BETA_ON_ALGO:
            algo_var = algo_r[-BETAWIN:].var() + 1e-12
            betas = np.array([
                np.cov(rets[i, -BETAWIN:], algo_r[-BETAWIN:])[0, 1] / algo_var
                for i in names])
            net_beta_dollars = np.sum(betas * dollars)
            target[0] = -net_beta_dollars / px[0]

        # --- track realised daily PnL for vol targeting (uses prev target) ---
        if state["prev_target_dollars"] is not None and state["prev_px"] is not None:
            # approx daily pnl of yesterday's book using today's prices
            shares = state["prev_target_dollars"] / (state["prev_px"] + 1e-12)
            pnl = float(np.sum(shares * (px - state["prev_px"])))
            state["pnl_hist"].append(pnl)
        state["prev_target_dollars"] = target * px  # dollar positions
        state["prev_px"] = px.copy()

        # --- 5. inertia: move partway to target ---
        newpos = prev + SMOOTH * (target - prev)
        state["prev"] = newpos.copy()
        return newpos.astype(int)

    getMyPosition.reset = reset
    return getMyPosition
