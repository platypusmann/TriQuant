# Algothon 2026 — Submission Notes

Submission file: **`teamName.py`** (rename to `<TeamName>.py` before zipping — see
"Packaging" below). Single entry point: `getMyPosition(prcSoFar) -> int shares`.

---

## What the strategy does

A factor-neutral short-term reversal (stat-arb) book on the 50 constituents,
beta-hedged with instrument 0.

1. **ALGO (instrument 0) is the market index**, not an alpha source. It has ~99%
   R² against the equal-weight basket of the other 50 names, near-zero
   idiosyncratic vol, a 10× position limit (100k vs 10k) and 1/5 the commission
   (0.2bp vs 1bp). We use it purely as a cheap beta hedge.
2. **Neutralise the common factors.** Take log returns over a 120-day window,
   eigendecompose the constituent covariance, and project out the top 5 factors,
   leaving idiosyncratic residuals.
3. **Build a mean-reverting spread.** Cumulate each name's residual, z-score it
   over a 15-day lookback, winsorise at ±3.
4. **Fade the deviation**, but only where it is meaningful: a z deadband at 1.4
   zeroes small signals so we don't churn commission on noise.
5. **Size** by inverse idiosyncratic vol, force the book dollar-neutral, and
   scale to a $1.9M nominal gross target.
6. **Hedge** the residual net market beta with ALGO.
7. **Move 35% of the way to target each day** (inertia), which cuts turnover and
   PnL variance.

## Final configuration

| Param | Value | Why (worst-window evidence) |
|---|---|---|
| `ENTRY` | 1.4 | z deadband; drops quiet-window commission drag to ~8% (reference ~21%) |
| `GROSS` | 1_900_000 | Peak of a dense grid; see below |
| `SMOOTH` | 0.35 | Lighter inertia tracks the deadband-cleaned target without adding churn |
| `NFACT` | 5 | Inherited; score is genuinely sensitive, so not pushed |
| `BETAWIN` | 120 | Inherited; same sensitivity caveat |
| `ZWIN` / `VOLWIN` / `ZCLIP` | 15 / 60 / 3.0 | Inherited from the reference |

## Results

Because the residual reversal is **regime-dependent** (residual autocorrelation is
strongly negative in some sub-windows, near-zero in others), every config was
judged on its **worst window across six different slicings** — 3×125, 5×100,
seq100, roll100/50, 75-postwarm, wf125/62 — never on the full-250 alone.

| Metric | Value |
|---|---|
| **Full-250 (official `eval.py`)** | **score 131.35**, mean PL 165.2, Sharpe 1.97 |
| **min-of-worsts (six slicings)** | **39.0** |
| 3×125 breakdown | 94.8 · **57.4 (binding)** · 184.4 |
| Forward test, days (375,500] | score 184.43, Sharpe 2.62 |

For comparison: previous best config (ENTRY 1.3 / GROSS 1.0M) scored 27.8
min-of-worsts; the un-tuned reference scored 0.7 (full-250 75.0 — i.e. it looked
fine in aggregate and was actually fragile).

## Tried and rejected — do not reintroduce

- **`GATE` ('autocorr' / 'hitrate')** — edge-strength gating that scales gross by
  recent signal quality. Collapses min-of-worsts from 39.0 to **1.8**.
- **`VOLTGT`** — vol targeting on realised daily PnL. Collapses it to **~0–1**.

Both fail the same way: they shrink gross exactly during the quiet/choppy regime
that the `ENTRY` deadband was already built to survive, so they de-risk at the
worst possible moment and double-count a fragility the config already handles.

## Known risks

- **Regime dependence.** The edge is real but weak and not stationary. The quiet
  window `(250,375]` scores 57.4 vs 184.4 in the recent window. A grading period
  that looks like the quiet regime will score toward the low end. This is the
  dominant risk and it is not removable by tuning.
- **Clipping saturation.** The book runs **55–65% of names pinned** at their
  per-name $10k limit, so a large share of the $1.9M nominal target is inert.
  This is deliberate — it buys diversified participation from names that would
  otherwise be under-sized — but it means the strategy is less sensitive to
  `GROSS` than the nominal number suggests, and it will not scale linearly if
  position limits change.
- **Hedge cliff above ~2.2M.** Past roughly $2.2M nominal gross the ALGO hedge
  saturates its 100k cap on large-signal days and can no longer neutralise net
  beta; min-of-worsts falls off sharply. 1.9M sits below that cliff on purpose.
- **Parameter sensitivity.** `NFACT` and `BETAWIN` are the fragile knobs; the
  score is genuinely sensitive to both. They were left at reference values rather
  than tuned, to avoid fitting noise.
- **Unseen-universe assumption.** The strategy assumes instrument 0 remains the
  index-like, cheap-to-trade, high-limit name. If a future stage changes that,
  the hedge leg is mis-specified (the book would still be factor-neutral, but the
  ALGO leg would no longer be a valid beta hedge).

## Packaging

**No `requirements.txt` is needed.** `teamName.py` imports **only numpy**, which is
in the grading sandbox's pinned set. Do **not** ship `requirements-dev.txt`.

Verified by **clean-room test**: `teamName.py` + `eval.py` + `prices.txt` alone in
an empty directory reproduces **score 131.35** exactly, with no imports from any
of our development modules (`harness.py`, `strat.py`, `sweep.py`, `stress.py`,
`check_clip.py`, `analyze.py`).

> **Action before submitting:** copy `teamName.py` → `<RegisteredTeamName>.py`.
> Note the existing `TriQuant.py` in this repo is still the *untouched starter
> baseline*, not our strategy — if the registered team name is "TriQuant", that
> file must be overwritten with our algorithm, so check this deliberately.

## Submission hardening applied

The strategy math and all hyperparameters are **unchanged**. Output is
**bit-identical** on `prices.txt` (all 250 scored days × 51 positions match the
pre-hardening snapshot exactly; `eval.py` score 131.35 before and after). The
changes are guards only:

| # | Fix | Why |
|---|---|---|
| 1 | Sanitise prices: forward-fill non-finite/non-positive values; a name with no usable price is held flat | **Critical.** One NaN/inf/zero/negative price turned the whole book to NaN; `.astype(int)` rendered that as `INT_MIN`, which the grader's `np.clip` turned into *every name pinned at max short* — and the state stayed poisoned, so the rest of the run never recovered |
| 2 | Finite-clamp positions before the integer cast | Makes the `INT_MIN` garbage path unreachable regardless of cause |
| 3 | Wrap the model; on failure **hold** yesterday's book | Never raises in the grader; holding avoids paying spread to exit on a transient data glitch |
| 4 | Derive all dimensions from `prcSoFar.shape`; drop the hardcoded `nInst=51`; clamp `NFACT` to the available names; return a correctly-sized flat book for degenerate universes | Previously any instrument count ≠ 51 raised `ValueError`, and a 1-instrument input silently returned a **length-51** vector |
| 5 | Validate state each call — re-initialise on first call, universe-size change, or a non-contiguous call sequence | A restarted/replayed pass in the same process silently inherited a stale book (up to 3,457 shares of drift). Cannot fire under `eval.py`, which is strictly contiguous |

Not changed: the per-name `np.cov` beta loop was left as-is — at **3.4 ms/day**
(0.84s for 250 days) it is nowhere near a bottleneck, so vectorising it would add
risk for no gain.
