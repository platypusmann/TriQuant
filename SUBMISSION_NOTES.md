# Algothon 2026 — Submission Notes

Submission file: **`teamName.py`** (rename to `<TeamName>.py` before zipping — see
"Packaging"). Single entry point: `getMyPosition(prcSoFar) -> int shares`.

---

## What the strategy does

A factor-neutral short-term reversal (stat-arb) book on the 50 constituents,
beta-hedged with instrument 0.

1. **ALGO (instrument 0) is the market index**, not an alpha source: ~99% R²
   against the equal-weight basket, near-zero idiosyncratic vol, a 10× position
   limit (100k vs 10k) and 1/5 the commission (0.2bp vs 1bp). Used purely as a
   cheap beta hedge.
2. **Neutralise the common factors** — 120-day window of log returns,
   eigendecompose the constituent covariance, project out the top 5 factors.
3. **Build a mean-reverting spread** — cumulate each name's residual, z-score
   over 15 days, winsorise at ±3.
4. **Fade the deviation, selectively** — a z deadband at 1.8 zeroes small
   signals so we don't churn commission on noise.
5. **Size** by inverse idiosyncratic vol, dollar-neutral, scaled to $900k gross.
6. **Move 35% of the way to target each day** (inertia) on the constituent book.
7. **Hedge the book we actually hold** — see below; this is the important part.

## The hedge must be sized on the post-clip book

The single most important correctness point in this strategy.

The grader clips every constituent to $10k. Sizing the ALGO hedge against the
**pre-clip target** therefore hedges positions we do not own. Measured over days
625–750: intended net beta exposure ±$6.3k, **realised ±$56.4k — a 9× error**,
with a persistent net short of $16k. That is why 4 of the 5 worst days in that
window were days the market rallied +1.5% to +3.0%, and why the strategy lost
**$237/day across a regime in which the reversal edge was working fine**
(residual autocorrelation −0.021, i.e. reversal, not momentum).

Clipping the target ourselves before computing the hedge:

| | Mean abs residual beta | Max | ALGO pinned at cap |
|---|---|---|---|
| Before | $56,397 | $174,881 | 41% of days |
| After | **$610** | $11,813 | 3% of days |

A 92× reduction. ALGO is set directly rather than eased into, because its
commission is 1/5 and lagging the hedge re-introduces the drift.

## Final configuration

| Param | Value | Why (worst-window evidence) |
|---|---|---|
| `ENTRY` | 1.8 | min-of-worsts −84 across six slicings, vs −160 at 1.4 and −190 at 1.6 |
| `GROSS` | 900_000 | Down from 1.9M; larger mostly buys clipping, and clipping corrupts the hedge |
| `SMOOTH` | 0.35 | Inertia on constituents (ALGO bypasses it) |
| `NFACT` | 5 | Inherited; score genuinely sensitive, so left alone |
| `BETAWIN` | 120 | Inherited; same caveat |
| `ZWIN`/`VOLWIN`/`ZCLIP` | 15 / 60 / 3.0 | Inherited |
| `POSLIM`/`POSLIM0` | 10k / 100k | Mirrors the problem spec; **update if limits change** |

`ENTRY=2.0` looks better on one slicing (−7.6) but is −120.6 across all six —
a single-slicing artifact of exactly the kind that produced the previous config.

## Results

| Metric | Previous (e1.4, 1.9M) | **Current (e1.8, 900k, hedge fix)** |
|---|---|---|
| `eval.py` last-250 score | 149.73 | **140.90** |
| `eval.py` Sharpe | 2.05 | **2.59** |
| All 500 unseen days — score | 26.2 | **85.1** |
| All 500 unseen days — Sharpe | 0.77 | **1.77** |
| All 500 unseen days — mean PL | $71/day | **$112/day** |
| min-of-worsts (6 slicings, 1000d) | −388.5 | **−84.3** |
| Worst 125d window (625,750] | −236.9 | **−33.1** |

**The official `eval.py` score went down 9 points and that is intentional.**
`eval.py` scores only the last 250 days, which sit entirely outside the strategy's
worst regime — it measures our best stretch. Across all 500 genuinely unseen days
the previous config made $71/day at Sharpe 0.77 because one 125-day stretch lost
$237/day. We traded 9 points of visible score for ~3× the out-of-sample
performance.

## Honest state of the evidence

- **The hedge fix is validated out-of-sample.** Measured at the *untouched* old
  parameters on 500 days the config had never seen, it moved the all-unseen score
  26.2 → 39.8 **before any retuning**. That result is real.
- **The ENTRY/GROSS retune is not.** It is fitted to all 1000 visible days — a
  reasonable bet, not a measured result. Anything added from here should hold out
  days 750–1000 and be confirmed once, rather than tuned against everything
  visible.
- **The worst window is still negative (−84).** This strategy has regimes in
  which it loses money. The bleeding is ~5× smaller, not cured.
- **Multi-slicing on a fixed file is weaker evidence than it feels.** The previous
  config survived six slicings of 500 days and still lost $237/day out of sample.

## Tried and rejected — do not reintroduce

**`GATE` ('autocorr'/'hitrate') and `VOLTGT`.** Re-tested on the full 1000 days
after a real loss regime became visible, in case the original verdict was an
artifact of a too-easy sample. Every variant was still worse on the worst window
(−77 to −91 vs −70 for plain). They shrink gross during the quiet regime that the
`ENTRY` deadband already handles.

## Known risks

- **Regime dependence.** The edge is weak and non-stationary. Windows range from
  −33 to +268. A grading period resembling the bad regime scores badly.
- **The score is capped by position limits.** Score ≤ mean daily PnL, and the
  constituent book is capped at 50 × $10k = $500k. Reaching a score of ~800 would
  need roughly **4× more alpha per dollar**, not more size — a signal problem, not
  a tuning problem.
- **`POSLIM`/`POSLIM0` are assumptions.** The hedge is sized against them. If a
  future stage changes position limits, they must be updated here.
- **`NFACT` and `BETAWIN` are the fragile knobs** and were deliberately not tuned.
- **Unseen-universe assumption.** Assumes instrument 0 stays the index-like,
  cheap, high-limit name.

## Packaging

**No `requirements.txt` is needed.** `teamName.py` imports **only numpy**, which is
in the grading sandbox's pinned set. Do **not** ship `requirements-dev.txt`.

Verified by clean-room test: `teamName.py` + `eval.py` + `prices.txt` alone in an
empty directory reproduces **140.90** exactly, with no imports from any dev module
(`harness.py`, `strat.py`, `sweep.py`, `stress.py`, `stress1000.py`,
`check_clip.py`, `analyze.py`).

> **Action before submitting:** copy `teamName.py` → `<RegisteredTeamName>.py`.
> `TriQuant.py` in this repo is still the **untouched starter baseline** — it
> scores **0.10**. If the registered team name is "TriQuant", that file must be
> overwritten with our algorithm, or the submission is the starter.

## Submission hardening

Guards only; no effect on clean data. Verified against 13 degenerate inputs
(NaN/inf/zero/negative/all-constant/all-NaN/singular covariance/leading gaps) and
8 universe shapes — all return correctly sized, sane integers.

| Fix | Why |
|---|---|
| Forward-fill non-finite/non-positive prices; hold unusable names flat | One bad tick turned the book to NaN → `.astype(int)` gave `INT_MIN` → grader clipped to *every name at max short*, **permanently** |
| Finite-clamp before the integer cast | Makes that path unreachable |
| Wrap the model; hold yesterday's book on failure | Never raises; no spread paid on a transient glitch |
| Dimensions from `prcSoFar.shape`; `NFACT` clamped; flat book for degenerate universes | Any instrument count ≠ 51 used to raise `ValueError`; 1 instrument silently returned a length-51 vector |
| State re-init on first call / universe change / non-contiguous sequence | A restarted pass inherited a stale book; cannot fire under `eval.py` |

Runtime 5.1 ms/day. The per-name `np.cov` beta loop is not a bottleneck and was
left un-vectorised deliberately.
