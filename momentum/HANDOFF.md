# Cappuccino Momentum System — Handoff Report
**Date:** 2026-03-25 (updated same session — fee model rebuild + capital testing)
**Session summary:** Full system overhaul — exchange migration, universe expansion, composite signal upgrade, factor research. Fee model rebuilt, hard stop active on v2 upgrade.

---

## WHO YOU ARE WORKING WITH
Solo developer, MS Economics graduating May 2026, Lubbock TX.
~$500 live capital on Coinbase Derivatives (futures), ~$431 current value (+0.62% from inception).
Extra $1,000 arriving imminently (confirmed). Total ~$1,500 expected soon.
Goal: accumulate to $1,000–1,500 to trade CME Micro ETH futures via Tradovate.

---

## SYSTEM MAP

### System 1 — Momentum Engine
**Location:** `~/cappuccino/momentum/`
**Exchange:** Coinbase Derivatives (futures, live — `paper_mode: false`)
**Strategy:** Cross-sectional momentum, top-2 assets by composite signal
**Signal:** Equal-weight z-score of `ret_24h` + `vol_accel` ✅ CONFIRMED LIVE in engine.py
**Universe:** 5 assets LIVE (BTC, ETH, DOGE, ADA, SOL) — 14-asset v2 blocked by hard stop (see below)
**Regime filter:** BTC 168h MA with ±0.5% hysteresis buffer
**Rebalance:** Weekly (168 hours)
**Fees modelled (live engine):** flat 0.20% RT taker via `fee_rate` in config.yaml
**Fees modelled (backtest v2):** ✅ Dynamic per-asset: (taker_per_side×2) + NFA_pct(asset, price) via fee_model.py
**Correct Sharpe benchmark for live system:** 5-asset 5-fold OOS = **+0.015** (Fold5=-3.303; 3-fold benchmark of +0.530 was misleading — masked current bad regime)

### System 2 — Coinalyze Data Pipeline
**Location:** `~/cappuccino/data_pipeline/`
**Source:** Coinalyze API (key in `collector_config.json`)
**DB:** `binance_features.db` (despite name — Coinalyze aggregated perp data)
**Symbols:** BTCUSDT_PERP.A, ETHUSDT_PERP.A, DOGEUSDT_PERP.A, ADAUSDT_PERP.A, SOLUSDT_PERP.A
**Tables:** funding_rates, open_interest, ls_ratio, liquidations
**History:** 60 days (2026-01-24 → present) with two ~7-day gaps in OI and L/S
**Gaps:** OI: Feb 21–28 and Mar 14–21 | L/S: Feb 7–14 and Mar 7–14 (collector was down)

### System 3 — XGBoost Signal
**Location:** `~/cappuccino/xgboost/`
**Status:** PARTIAL — combined OOS AUC 0.524, advisory only, not driving live trades
**Retrain readiness:** NOT YET. OI has only 4 days of clean recent data (post Mar 21 gap).
  Retrain when OI has 14+ consecutive clean days (~Mar 28–30 at current pace).

### System 4 — Job Application Bot
**Location:** `~/jobs/` — DO NOT TOUCH. Running independently.

### Cappuccino DRL
**Location:** `/opt/user-data/experiment/cappuccino/Version 0.0.1/`
**Status:** RETIRED. Do not create Optuna studies. Do not modify.

---

## CURRENT LIVE STATE (as of 2026-03-26 00:00 UTC)

| Field | Value |
|---|---|
| Engine PID | 3981023 (miniconda python3) |
| Collector PID | 3053310 (miniconda python3) |
| Mode | LIVE — Coinbase Derivatives |
| Portfolio value | ~$431.68 (as of 2026-03-26 00:00 UTC) |
| Peak value | $431.82 |
| Inception value | $429.03 |
| P&L since inception | **+0.62%** (recovered from -1.45% earlier) |
| Regime | BULL (BTC above 168h MA) |
| Positions | None (rebal #1 on 2026-03-23 had 0 trades — regime was BEAR at that time) |
| Last rebalance | 2026-03-23 19:00 UTC (rebal #1, 0 trades) |
| Next rebalance | **2026-03-30 19:00 UTC** — FIRST real trade expected |
| Circuit breaker | None |
| Current signal (v2) | Long DOGE + ADA (composite scores leading 5-asset universe) |

**Important:** The portfolio value fluctuation ($429→$422) with empty positions is NOT a bug.
The engine reads live Coinbase account balances directly. The account holds crypto that fluctuates.
`positions: {}` in state.json is an unused paper-mode tracking field.

---

## THE COMPOSITE SIGNAL (THIS SESSION'S MAIN CHANGE)

### What was changed
Previous signal: rank by 24h return only (`ret_24h`)
New signal: equal-weight z-score composite of two factors

**Factor 1 — `ret_24h`:** 24h price return. Classic cross-sectional momentum.

**Factor 2 — `vol_accel`:** Current 24h volume / prior 24h volume.
Is participation growing? Assets with accelerating volume behind their move are more likely
to continue. Assets with high return but shrinking volume are fading — downweighted.

Each factor is z-scored cross-sectionally at every hourly bar, then averaged.
Top-2 assets by composite score get 25% weight each (50% total deployed).

### Factor research results (this session)

Full 3-stage grid search across 6 factors on 14-asset, 28-month dataset:

| Combo | OOS Sharpe | CAGR | MaxDD | Folds |
|---|---|---|---|---|
| ret_24h only (old live) | +1.105 | +42.3% | -32.9% | [-0.450, +2.928, +0.168] |
| vol_surge only | +1.023 | +65.8% | -27.2% | [+0.228, +1.834, +0.107] |
| **ret_24h + vol_accel (NEW live)** | **+1.312** | **+55.4%** | **-30.1%** | **[+0.130, +2.941, +0.319]** |
| best 4-factor (ret0.6+vol0.4+ret3d+ret4h) | +1.379 | +74.4% | -23.6% | [+0.980, +2.401, -0.203] |

**Why `ret_24h + vol_accel` over the 4-factor winner:**
The 4-factor combo has fold 3 = -0.203 (Dec 2025→Mar 2026 = what we're trading into now).
`ret_24h + vol_accel` is positive in all three folds and simpler — less overfitting risk.

**What consistently HURTS performance:**
- `inv_vol` (inverse volatility) — crypto momentum rewards volatility, penalising it kills best assets
- `ret_7d` — 7-day lookback conflicts with 24h signal at weekly rebalance cadence
- `vol_surge` (level relative to 30d baseline) — raw level is noisier than acceleration

**What might still be missing (can't test — need more Coinalyze data):**
- Funding rate signal: negative funding + positive return = highest conviction long
- OI trend: rising open interest + rising price confirms conviction
Add these once Coinalyze DB has 14+ clean days for all 5 symbols.

---

## UNIVERSE EXPANSION (THIS SESSION)

Previous universe: 5 assets (BTC, ETH, DOGE, ADA, SOL)
New universe: 14 assets (added XRP, AVAX, LINK, LTC, BCH, XLM, SUI, DOT, HBAR)

Historical 1h OHLCV data fetched from OKX for all 10 new assets.
Files: `~/cappuccino/data/{ASSET}_1h.parquet` — 15 files total (14 live + SHIB on disk but not in universe).

**Why 14, not 15 (SHIB excluded):**
Contract economics: SHIB contract at 0.01 multiplier = tiny notional → NFA fee dominates.
Exact contract specs for all 15 assets still TBD — user will provide when available.
Once specs are in, recalculate per-asset effective fee and decide if any others should be excluded.

**Universe expansion backtest results:**
| Universe | OOS Sharpe | MaxDD |
|---|---|---|
| 5-asset baseline | +0.831 | -31.5% |
| 14-asset top-2 | +1.105 | -32.9% |
| 15-asset top-2 (with SHIB) | +1.312 | -40.8% |

Cross-sectional momentum improves with larger universe. More assets = better signal spread.

---

## EXCHANGE HISTORY & FEE REALITY

1. Bybit — geo-blocked (CloudFront 403)
2. Binance — geo-blocked (HTTP 451)
3. OKX — accessible server-side but blocked in Texas for the user
4. Kraken spot — accessible but 0.52% RT taker kills all alpha
5. Coinbase Advanced Trade (spot) — was briefly live, now superseded
6. **Coinbase Derivatives (futures) — CURRENT** — 0.10% taker, 0.095% maker + $0.15 NFA/contract
7. Deribit — blocked for US residents (acquired by Coinbase Aug 2025, US product TBD 2026)

**Fee model in backtest:** 0.20% RT taker (base only). NFA fee varies by contract size.
User to provide per-asset contract specs so NFA can be modelled properly.
At current BTC ~$71k: 0.01 BTC contract = $715 → NFA adds ~0.042% RT (negligible).
ETH, SOL, smaller alts have smaller contracts → higher NFA % → model carefully before trading.

**Leverage:** User mentioned 4.3× but this is not yet in the engine. Engine trades notional
equal to account value (1× effective). Leverage implementation is future work.

---

## PENDING WORK — PRIORITY ORDER

### 0. ❌ HARD STOP — v2 engine blocked (deeper than originally diagnosed)
Fold 1 (Dec 2023–Sep 2024 bull run) is negative across all tested configurations.
The problem is **regime-structural**: 24h cross-sectional momentum has near-zero discriminative
power when the entire market is trending up together. Every investigated fix has failed:

| Attempted fix | Fold 1 result | Verdict |
|---|---|---|
| Capital injection ($1K) | -0.871 (worse than $500) | ❌ |
| Lookback 72h | -0.918 | ❌ |
| Lookback 168h | -1.181 (OOS < 0.4 hard stop) | ❌ |
| Exclude AVAX | -1.094 (worse than baseline) | ❌ |
| Exclude AVAX+XLM | -0.982 | ❌ |

AVAX was 85% of Fold 1 loss in decomposition — but when AVAX is removed, the
strategy shifts into equally bad alternatives. Loss is systemic to the signal, not asset-specific.

**Pending decision from Big Claude:** See DECISION LOG at bottom of file for options.
**DO NOT deploy engine_v2.py until Fold 1 is resolved.**

### 1. ✅ Contract specs — RESOLVED
`fee_model.py` and `config_v2.yaml` complete with all 14 contract sizes and fee tiers.
**AVAX WARNING:** At current price ~$9.68, effective RT T1 fee ≈ 0.51% — in warning zone.
Monitor: if AVAX falls below $15 → fee >0.50%; below $10 → fee >0.60%. Reassess inclusion.

### 2. Leverage implementation
User specified 4.3× leverage. Engine currently trades 1×.
Need to: update `compute_target_weights` to scale notional by leverage factor,
add margin/liquidation monitoring, tighten circuit breakers appropriately.
**Do not implement without discussing liquidation risk at 4.3× first.**

### 3. Funding rate + OI signal (when data ready ~Mar 28–30)
When Coinalyze OI has 14+ consecutive days:
- Add `funding_rate` z-score and `oi_change` z-score to composite
- Retrain with 4-factor signal and compare to current 2-factor
- Hypothesis: negative funding + positive return = strongest signal tier

### 4. XGBoost retrain
Prerequisites: OI gap heals (~Mar 28–30), all _meta flags set (already done).
Then: `cd ~/cappuccino/xgboost && python3 train.py`
XGBoost is advisory only — does not override momentum signal yet.

### 5. Dynamic universe (future)
Currently static 14-asset list. Future: review monthly against top-50 spot volume,
add/remove assets as Coinbase Derivatives listings change.
Requires point-in-time universe construction logic to avoid lookahead bias in backtest.

### 6. Coinbase Derivatives CFTC-regulated options (watch)
Coinbase acquired Deribit ($2.9B, Aug 2025). Working toward US-regulated crypto options
under CFTC DCM/DCO license. No timeline confirmed. Would be massive for strategy optionality.

---

## FILE MAP

```
~/cappuccino/
├── momentum/
│   ├── config.yaml          ← PROTECTED — live engine config (5-asset, flat fee)
│   ├── engine.py            ← PROTECTED — live loop: composite signal, 5-asset
│   ├── backtest.py          ← PROTECTED — original backtest
│   ├── config_v2.yaml       ← v2 config: 14 assets, fee_model section, contract_specs ✅
│   ├── fee_model.py         ← Coinbase Derivatives fee calculator ✅ COMPLETE
│   ├── backtest_v2.py       ← 14-asset backtest, dynamic fees, --equity/--lookback/--decompose ✅
│   ├── engine_v2.py         ← DOES NOT EXIST — blocked by Fold 1 hard stop
│   ├── dashboard.py         ← rich TUI
│   ├── state.json           ← live portfolio state
│   ├── audit.db             ← immutable trade audit log
│   ├── logs/YYYY-MM-DD.jsonl
│   └── HANDOFF.md           ← this file
├── data/
│   └── {ASSET}_1h.parquet   ← 15 assets, 28 months (OKX source)
├── data_pipeline/
│   ├── collector.py         ← Coinalyze async daemon
│   ├── collector_config.json ← API key
│   ├── binance_features.db  ← funding/OI/LS/liq data (60 days)
│   └── export_features.py   ← parquet export for XGBoost
└── xgboost/
    ├── train.py / predict.py / retrain.py
    ├── config.yaml
    └── results/metrics.json ← AUC 0.524, PARTIAL verdict
```

---

## OPERATIONS QUICK REFERENCE

```bash
# Check engine alive
pgrep -fa engine.py

# Live log tail
tail -f ~/cappuccino/momentum/logs/$(date +%Y-%m-%d).jsonl

# Restart engine (after reboot or crash)
cd ~/cappuccino/momentum
nohup /opt/miniconda3/bin/python3 engine.py > /tmp/momentum_engine.out 2>&1 &

# Check collector
pgrep -fa collector.py

# Restart collector
cd ~/cappuccino/data_pipeline
nohup /opt/miniconda3/bin/python3 collector.py > /tmp/collector.out 2>&1 &

# Clear circuit breaker
# Edit state.json → set "circuit_breaker": null → restart engine
```

---

## STANDING RULES

1. Do NOT touch `~/jobs/` — job bot running independently
2. Do NOT create new Optuna studies — DRL is retired
3. Do NOT modify `/opt/user-data/experiment/cappuccino/Version 0.0.1/`
4. Always show diff before applying any file change
5. Do NOT implement leverage without discussing liquidation risk first
6. Historical OHLCV data source is OKX (server-accessible). Binance and Kraken have
   geo-block or depth issues. Coinbase AT requires API key for historical data.
7. The `_meta` flags in `binance_features.db` are benign one-time backfill guards — do not clear them

---

## DECISION LOG

### 2026-03-25 — AVAX Exclusion Test + Fold 1 Investigation (UNRESOLVED)

**Grounds for exclusion tested:**
- (1) RT fee at current price ($9.68) = 0.51% T1, above 0.500% warning threshold
- (2) 85% concentration of Fold 1 loss per per-asset decomposition

**Authorization:** Granted by CTSP controller 2026-03-25.

**Outcome: STOP — exclusion did not resolve Fold 1.**
AVAX exclusion worsened Fold 1 (-0.871 → -1.094). The strategy rotated into equally
bad replacements. Loss is systemic to signal regime, not asset-specific.
AVAX+XLM exclusion also failed (-0.982). No exclusion approach clears Fold 1.

**5-asset live benchmark computed (Task 2 complete — 3-fold):**
| Fold | Sharpe |
|------|--------|
| Fold 1 | -0.302 |
| Fold 2 | +1.309 |
| Fold 3 | +0.707 |
| Combined OOS | **+0.530** |
⚠️ **SUPERSEDED** by 5-fold benchmark in next session entry: OOS = +0.015 (Fold5 = -3.303).
The 3-fold split masked the current bad period entirely.

**All tested approaches and outcomes:**
| Approach | Fold 1 | OOS | Notes |
|---|---|---|---|
| 14-asset baseline | -0.871 | +1.045 | Reference |
| Capital injection $1K | -0.871 | +1.045 | No change to Fold 1 |
| Lookback 72h | -0.918 | +0.811 | Worse |
| Lookback 168h | -1.181 | +0.164 | OOS hard stop |
| Exclude AVAX | -1.094 | +1.207 | Fold 1 worse |
| Exclude AVAX+XLM | -0.982 | +1.045 | Fold 1 worse |

**Pending — awaiting Big Claude decision on paths forward:**

A. **Minimum cross-sectional spread filter** — only deploy if score gap between top-2
   and median asset exceeds threshold. Detects "everything ranked random" regimes.

B. **Correlation regime filter** — add a second filter: when 14-asset 24h return
   correlation exceeds ~0.85, go to cash regardless of BTC MA. High correlation means
   cross-sectional signal is noise.

C. **Accept Fold 1 as untraded history** — Fold 1 is Dec 2023–Sep 2024. We are now
   in Fold 3 analog conditions. The system has never traded through a Fold 1 equivalent
   market since going live (2026-03-23). Document the failure mode and deploy v2 with
   the awareness that a broad trending bull market will produce losses, relying on the
   regime filter and circuit breakers to limit drawdown.

D. **Short side (long/short)** — add bottom-2 shorts to the strategy. Generates alpha
   from both ends of the cross-section. May invert the Fold 1 loss into a gain if
   assets at the bottom of the cross-section during the bull run underperformed.

---

### 2026-03-25 — 5-Fold Decomposition + Dispersion Filter Sweep (HARD STOP REMAINS)

**Authorization:** Granted by CTSP controller 2026-03-25.

**Task 1 — 5-fold walk-forward baseline (13 assets, no AVAX, $1K equity, T1 fees)**

Switching from 3-fold to 5-fold reveals the bad period structure more clearly:

| Fold | Period (approx) | Sharpe | CAGR% | Notes |
|------|-----------------|--------|-------|-------|
| Fold 1 | Dec 2023–May 2024 | +0.673 | +41.2% | Acceptable — early bull not as bad as 3-fold suggested |
| Fold 2 | May–Nov 2024 | -1.278 | -46.9% | ❌ Primary bad period: mid-bull altcoin rotation |
| Fold 3 | Nov 2024–Mar 2025 | +1.020 | +68.4% | Strong trend following period |
| Fold 4 | Mar–Jul 2025 | +0.430 | +14.9% | Mild positive |
| Fold 5 | Oct 2025–Mar 2026 | **-3.061** | -73.4% | ❌ **Current live window** — severe |
| Combined OOS | Full 1.31 years | -0.043 | — | Hard stop |

**Critical finding:** 5-fold exposes Fold 5 = -3.061 Sharpe, which was **masked** in the 3-fold result.
Fold 5 covers Oct 2025–Mar 2026 — the window we are actively trading in right now.
The 3-fold OOS (+1.045) was driven largely by Fold 3 (Nov 2024–Mar 2025 bull).

**Task 2 — Cross-sectional dispersion filter sweep (threshold = xs_std of ret_24h)**

Logic: if `std(ret_24h across 13 assets) < threshold` at a rebalance bar → go flat.
Low dispersion = high cross-sectional correlation = signal has no discrimination power.

| Threshold | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | OOS Sharpe | Filtered% | Status |
|-----------|-------|-------|-------|-------|-------|------------|-----------|--------|
| 0.00 (baseline) | +0.673 | -1.278 | +1.020 | +0.430 | -3.061 | -0.043 | 0% | ❌ |
| 0.01 | +0.673 | -1.192 | +1.020 | +0.905 | -4.002 | +0.024 | ~10% | ❌ Fold5 worse |
| 0.02 | +0.608 | -0.430 | +1.976 | +1.465 | -1.391 | +0.799 | ~48% | ❌ Fold2 neg; filtered >40% |
| 0.03 | +0.283 | -2.645 | +2.613 | +1.746 | -0.130 | +0.865 | ~78% | ❌ Fold2 paradoxically worse; filtered 78% |
| 0.04 | -0.110 | -2.645 | +2.613 | 0.000 | -0.130 | +0.543 | ~85% | ❌ Fold1 turns negative; Fold4 flatlined |

**Gate conditions required:** all folds positive AND OOS ≥ 0.8 AND filtered_pct ≤ 40%.
**Outcome: NO threshold satisfies all three conditions simultaneously. STOP.**

**Root cause analysis of sweep paradox at 0.03:**
The 0.03 threshold filters ~78% of rebalance bars — it is too aggressive. It removes
profitable Fold 2 rebalances along with the unprofitable ones, producing a worse outcome.
Fold 2 (May–Nov 2024) and Fold 5 (Oct 2025–Mar 2026) fail for structurally different reasons
that cannot be captured by a single xs_std threshold.

**Task 3 — BLOCKED.** No candidate threshold found. `config_v2.yaml` universe_v2 list
was updated (AVAX excluded, fee-economic grounds only — authorized unconditionally).
The `signal:` dispersion filter section was NOT added per STOP condition.

**Task 4 — BLOCKED.** Depends on Task 3.

**engine_v2.py: still does not exist — hard stop ACTIVE.**

**Standing recommendation for Big Claude:**

The dispersion filter approach is structurally limited — it operates on a scalar summary
statistic that averages away the fold-specific regime information. Fold 2 and Fold 5 are
different market regimes that respond differently to the filter.

Recommended next investigation paths (in priority order):

**Path 1 — Correlation regime filter (Candidate B from prior log)**
Compute pairwise correlation matrix of 13-asset ret_24h over rolling 168h window.
If median pairwise correlation > 0.70 → go flat. More targeted than xs_std because
it directly measures the failure mode (cross-sectional signal collapses when assets are correlated).
Expected to be more effective than xs_std on Fold 5 without damaging Fold 2 as severely.

**Path 2 — Accept Fold 1/2/5 as regime-dependent failure modes (Candidate C)**
The strategy has never traded through a Fold 2-analog (mid-bull altcoin rotation) since going live.
Current live window (Fold 5) is already showing the structural failure. Consider deploying v2
with strict circuit breakers (8% drawdown halt) and the knowledge that cross-sectional momentum
on 1h bars at weekly rebalance underperforms in high-correlation trending markets.
The regime filter (BTC MA) provides partial protection — when BTC trends up uniformly, the
signal can still lose money despite regime = BULL.

**Path 3 — Short side addition (Candidate D)**
Requires exchange support for shorting all 13 assets simultaneously (verify Coinbase
Derivatives futures permit shorting for each symbol). May flip Fold 2 positive if
bottom-2 assets during mid-bull rotation underperformed significantly.

---

### 2026-03-25 — 5-Asset Benchmark + Correlation Filter Sweep (HARD STOP REMAINS)

**Authorization:** Granted by CTSP controller 2026-03-25.

---

#### Task 1 — 5-Asset 5-Fold Benchmark (URGENT — before March 30 trade)

Command: `python3 backtest_v2.py --equity 1000 --fee-tier 1 --folds 5 --assets BTC ETH DOGE ADA SOL`

| Fold | Period | Sharpe | CAGR | MaxDD | Notes |
|------|--------|--------|------|-------|-------|
| Fold 1 | 2023-12-01 → 2024-05-18 | +1.372 | +49.1% | -14.5% | Strong — ETH/DOGE lead |
| Fold 2 | 2024-05-18 → 2024-11-03 | **-1.990** | -40.7% | -20.3% | ❌ BTC 72% of loss |
| Fold 3 | 2024-11-03 → 2025-04-21 | +0.648 | +19.9% | -23.4% | Positive; ADA/DOGE lead |
| Fold 4 | 2025-04-21 → 2025-10-07 | +1.680 | +58.0% | -11.6% | Strong; ADA/ETH/BTC lead |
| Fold 5 | 2025-10-08 → 2026-03-23 | **-3.303** | -51.2% | -20.8% | ❌ **CURRENT LIVE WINDOW** |
| Combined OOS | 2023-12-01 → 2026-03-23 | +0.015 | -4.0% | -28.7% | Hard stop |

**Fold 5 deep-dive (Oct 2025–Mar 2026 = current live window):**
- Sharpe: -3.303 | MaxDD: -20.8% | Duration: 0.26 yrs
- ALL 5 assets have negative gross contributions — this is systemic signal failure, not fee drag
- DOGE worst: net -8.714%, avg hold return -0.0857% (337h held), 48% of total loss
- ADA: net -4.734%, avg hold return -0.0409% (450h held)
- BTC: net -1.964%, avg hold return -0.0515% (94h held)
- ETH: gross return -0.050% (near zero), but 1.46% fee drag → net -1.510%
- SOL: gross return -0.284%, fee drag -0.999% → net -1.283%
- **Interpretation:** The strategy is SELECTING THE WRONG ASSETS in the current period. The problem is not fees — the composite signal (ret_24h + vol_accel) is anti-predictive in the Oct 2025–Mar 2026 regime.

**5-asset combined OOS Sharpe = +0.015** (was +0.530 in 3-fold benchmark from prior session).
The 3-fold benchmark masked the severity of the current live window. The 5-fold result reveals
that the engine is entering its first real trade in a known-bad regime for the v1 system too.

---

#### Task 2 — Correlation Regime Filter Sweep (13 assets, no AVAX)

Logic: at each rebalance bar, compute rolling 168h pairwise return correlation matrix.
If median pairwise correlation > threshold → go flat.

| Threshold | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | OOS | Filtered% | Gate |
|-----------|-------|-------|-------|-------|-------|-----|-----------|------|
| 0.00 (baseline) | +0.673 | -1.278 | +1.020 | +0.430 | -3.061 | -0.043 | 0% | ❌ |
| 0.50 | +3.149 | -2.451 | +0.298 | 0.000 | 0.000 | +1.044 | 93% | ❌ F2 neg; F4/F5=100% flat |
| 0.60 | +2.109 | -1.135 | +2.556 | -0.251 | 0.000 | +1.330 | 84% | ❌ F2,F4 neg; F5=100% flat |
| 0.70 | +1.908 | -3.143 | +2.335 | -0.629 | 0.000 | +0.795 | 59% | ❌ F2,F4 neg; F5=100% flat |
| 0.75 | +1.908 | -3.168 | +2.683 | +0.715 | -1.920 | +0.986 | 49% | ❌ F2,F5 neg; >40% filtered |
| 0.80 | +0.889 | +0.557 | +1.248 | +0.303 | -2.256 | +0.477 | 27% | ❌ F5 neg; OOS 0.477 < 0.8 |

Gate conditions: all folds positive AND OOS ≥ 0.8 AND filtered ≤ 40%.
**STOP — no threshold satisfies all three gate conditions.**

**Key structural observations:**

1. **Fold 5 is the terminal barrier.** Median pairwise correlation in Oct 2025–Mar 2026 is:
   - Always > 0.70 (100% filtered at that threshold — the strategy NEVER trades in this window)
   - Above 0.80 for 50% of rebalance bars (and those remaining 50% of bars produce -2.256 Sharpe)
   - The current market is in a persistently high-correlation regime. Cross-sectional ranking
     is noise. The signal has no discrimination power in this period.

2. **The Fold 2 / Fold 5 crossover:**
   - Fold 2 (May–Nov 2024) turns positive between 0.75 and 0.80 (at 0.80: +0.557, filtered 21%)
   - Fold 5 (Oct 2025–Mar 2026) does NOT turn positive at any threshold ≤ 0.80 with ≤ 40% filtering
   - At 0.80: Folds 1-4 are all positive ✅, filtered 27.4% ✅, but Fold 5 = -2.256 ❌
   - The best achievable result is 0.80: fixes all historical bad periods but cannot fix the current one

3. **Threshold 0.80 is the "best bad" outcome:**
   - If the current Fold 5 regime eventually normalizes (correlations fall below 0.80 more often),
     threshold 0.80 would be viable. It produces excellent results for Folds 1-4.
   - The corr filter at 0.80 with the current regime would have the engine flat ~50% of the time
     in the current window — but the other 50% would still lose.

4. **Why the filter works better than dispersion for Folds 1-4:**
   - At 0.80: Fold 2 = +0.557 (was -1.278 unfiltered) — fixed ✅
   - At 0.80: Fold 4 = +0.303 (was +0.430 unfiltered) — slightly worse but positive ✅
   - Pairwise correlation is a more targeted measure than xs_std — it directly captures
     the mechanism by which cross-sectional momentum fails

**Task 3 — BLOCKED.** No candidate threshold. 5-asset corr filter benchmark not run.

**engine_v2.py: still does not exist — hard stop ACTIVE.**

---

#### Task 4 — Circuit Breaker Configuration (read-only diagnostic)

Source: `engine.py` lines 512–532 + `config.yaml`

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Portfolio drawdown from peak | -15.0% | `sys.exit(1)` — process terminates |
| Single-asset 1h drop | -8.0% | `sys.exit(1)` — process terminates |
| Consecutive ccxt network errors | 3 errors | `sys.exit(1)` — process terminates |

**Action on trigger:**
1. Writes reason string to `state["circuit_breaker"]`
2. Sends Telegram alert
3. Calls `sys.exit(1)` — **process dies immediately**
4. Does **NOT** flatten open positions — any live positions remain open until manually closed

**Reset procedure:**
1. `nano ~/cappuccino/momentum/state.json` → set `"circuit_breaker": null`
2. Restart engine: `nohup /opt/miniconda3/bin/python3 engine.py > /tmp/momentum_engine.out 2>&1 &`
3. On startup, engine re-checks state and halts again if circuit_breaker is still set

**Current state (as of 2026-03-26):**
- `circuit_breaker`: null ✅ (inactive)
- `peak_value`: $431.82
- Latest value: $431.54 → current DD = -0.065% (far from -15% trigger)
- `consecutive_errors`: 0

**Critical note for March 30 trade:**
If the circuit breaker fires AFTER the first real trade executes (e.g., a position is opened
and then the market drops -8% intraday), the engine will halt with the position STILL OPEN.
Someone must manually close the position and then reset state.json. Monitor closely on
the hours surrounding 2026-03-30 19:00 UTC.

---

#### Updated Live Benchmark Sharpe (March 30 context)

| Benchmark | Folds | OOS Sharpe | Fold 5 Sharpe | Notes |
|-----------|-------|-----------|---------------|-------|
| 5-asset 3-fold (prior session) | 3 | +0.530 | N/A (fold 3 of 3) | Was the stated benchmark |
| **5-asset 5-fold (this session)** | 5 | **+0.015** | **-3.303** | ❌ Current live window is Fold 5 |
| 13-asset 5-fold baseline | 5 | -0.043 | -3.061 | v2 blocked |
| 13-asset 5-fold corr=0.80 | 5 | +0.477 | -2.256 | Best corr filter result |

**The March 30 trade is entering a known-bad regime for the v1 engine (5-asset, Fold 5 = -3.303).**
The 3-fold benchmark (+0.530) was misleading — Fold 5 did not exist as a distinct fold in that split.

**Recommendation for March 30:**
- The regime filter (BTC 168h MA) is the PRIMARY protection. If BTC is below MA at 2026-03-30 19:00 UTC, the engine will go flat and no trade executes. Check `last_regime` in state.json before the rebalance.
- If regime = BULL and trade executes, maximum deployed capital is 50% (25% × 2 positions).
- Circuit breaker at -15% portfolio DD provides backstop.
- The v1 engine has no correlation filter. It will trade if regime is BULL regardless of cross-sectional correlation.

---

### 2026-03-26 — Engine Crash + Bug Fix Session

#### What happened
- **05:00 UTC**: BTC dipped to $69,546 vs 168h MA $70,106 → regime breach exit triggered
- **Bug 1** (`CreateOrderResponse.get()`): sell order for BTC 0.003705 submitted to Coinbase API, but response parsing crashed before LTC sell was attempted. Engine caught as `[unexpected_error]`, slept 60s, continued.
- **Bug 2** (USDC-only balance): at next hourly check, Coinbase reported USD cash from futures settlement separately from USDC. Engine only read USDC, saw $117 instead of ~$406. Circuit breaker fired: `"drawdown -72.8% exceeds -15.0%"`. Engine halted.
- **Real portfolio**: ~$406 (all cash, no open positions). No real loss from the incident beyond normal market moves.

#### Bugs fixed in engine.py (authorized by owner 2026-03-26)

**Bug 1 — `create_market_order` line 232** (`CoinbaseExchange.create_market_order`):
- Before: `order = resp.get("success_response", resp)` — crashed because `resp` is a `CreateOrderResponse` object, not a dict
- After: `sr = resp.success_response if hasattr(resp, "success_response") else resp` — uses attribute access, consistent with pattern used in `fetch_ohlcv` and `fetch_balance`

**Bug 2 — `fetch_balance` lines 179-184** (`CoinbaseExchange.fetch_balance`):
- Before: `if asset == "USDC": bal["USD"] = {...}` — only USDC spot position mapped to USD; separate USD cash entry from futures settlement was invisible or overwrote USDC
- After: `if asset in ("USDC", "USD"):` — merges both additively into `bal["USD"]`; neither overwrites the other regardless of ordering in spot_positions

#### Restart outcome
- Circuit breaker cleared: **yes** — `state.json` `circuit_breaker` set to null
- Engine PID: **1912802**
- Balance reading post-fix: **$406.53** (correct — USD + USDC now summed)
- Regime at restart: **BEAR** (`last_regime=false`, BTC $68,839 < MA $70,037)
- No orders placed on restart ✅

#### BTC regime status (23:00 UTC 2026-03-26)
- BTC: $68,839 | 168h MA: $70,037 | BULL threshold (MA×1.005): $70,387
- Gap to BULL: **+$1,548 (+2.2%)**
- **March 30 19:00 UTC trade: will NOT fire unless BTC crosses $70,387 before then**
- Monitor: if BTC recovers above $70,387, next hourly loop flips to BULL and the rebal scheduled for Mar 30 will execute

#### Hard stop status (after 2026-03-26 bug fix session)
- `engine_v2.py`: still does not exist
- Long/short backtest run this session — **rejected**: F1=-1.770, F4=-2.329, F5=-3.795, OOS=-0.535 (worse than long-only on every metric in bull/recovery folds)
- Best candidate still: corr=0.80 filter (F1-F4 all positive, OOS=+0.477, 27% filtered) — Fold 5 = -2.256 remains the blocker
- Research continues next session

---

### 2026-03-26 — Vol-Gated Filter + 7d Gate Research Session

#### Approaches tested

Two new filter modifications were implemented in `backtest_v2.py` and exhaustively tested. Neither passes the hard stop gate (all folds positive AND OOS ≥ 0.8).

**Approach A — Vol-gated correlation filter (`--vol-threshold`)**

Modified the corr filter to require BOTH `median_corr > 0.80 AND avg_168h_vol > vol_threshold` to go flat (AND logic). Rationale: trending-down together (high corr + high vol) is the failure mode; consolidating together (high corr + low vol) might still permit selection. Results:

| vol_threshold | F1 | F2 | F3 | F4 | F5 | OOS Combined | corr_filtered% |
|---|---|---|---|---|---|---|---|
| 0.00 (pure corr) | +0.889 | +0.557 | +1.248 | +0.303 | **-2.256** | +0.477 | 27.4% |
| 0.60 | +0.889 | +0.557 | +1.248 | +0.303 | **-2.256** | +0.477 | 27.4% |
| 0.80 | +0.889 | +0.557 | +1.248 | +0.438 | **-4.002** | +0.300 | 20.5% |
| 1.00 | +0.889 | +0.557 | +1.248 | +0.430 | **-3.757** | +0.265 | 13.7% |
| 1.20 | +0.889 | +0.129 | +1.248 | +0.430 | **-3.757** | +0.212 | 8.2% |

- vol=0.60: identical to baseline — market vol was always > 0.60 when corr triggered, so AND gate never engaged
- vol≥0.80: **actively harmful** — unblocks trades during low-vol selloffs that the corr filter correctly caught; F5 drops from -2.256 to -4.002
- Conclusion: the low-vol consolidation regime in F5 is not safe to trade; relaxing the filter there makes it worse

**Approach B — 7d return gate (`--ret7d-gate`)**

Selection-level filter: at each rebalance bar, only select assets with positive 7-day return; re-rank composite signal within eligible set. If 0 eligible → flat; if 1 → 1 position; ≥2 → normal 2 positions.

| Fold | Sharpe | 7d_flat% | 7d_partial% |
|------|--------|----------|-------------|
| F1 | +1.057 | 13% | 7% |
| F2 | **-1.552** | 20% | 13% |
| F3 | +1.963 | 13% | 20% |
| F4 | +2.004 | 13% | 13% |
| F5 | **-2.877** | 40% | 13% |
| Combined OOS | +0.550 | | |

- F2 goes negative (was -1.3 baseline): the 7d gate removed protective flat periods in F2
- F5 worsens from -2.256 (corr filter) to -2.877: the gate filtered eligible assets during the only positive 7-day windows, leaving trades in the worst assets
- Combined passes the OOS ≥ 0.4 threshold (+0.550) but fails the all-folds-positive gate
- 40% flat rate in F5 is at the 40% gate limit — filter working overtime but signal is still anti-predictive

**Approach C (Combination) — Skipped**
Neither Task A nor Task B showed partial improvement in F5 (both made it worse). Combination skipped per protocol.

#### Root cause diagnosis
F5 (Oct 2025 – Mar 2026) failure is structural:
- Prior session confirmed: LINK gross avg hold return = **-0.339%/hr** (anti-predictive, not fee-dominated)
- The entire universe trended down together in F5 — cross-sectional dispersion collapsed
- The corr filter at 0.80 already blocks 50% of F5 rebalances — near the maximum allowable (40% gate)
- No selection or timing filter can rescue signal quality when the signal itself is inverted

#### Hard stop status (current)
- **Hard stop remains active.** `engine_v2.py` does not exist and should not be created.
- Best result across all tested configurations: corr=0.80 filter, OOS=+0.477, F5=-2.256
- No configuration achieves all folds positive

#### Pending research (next session)
- **XGBoost retrain**: ~Mar 28-30 when OI data has 14+ clean days of history (15 days needed for OHLCV features). This is a separate ML overlay, not a v2 prerequisite.
- **What would unblock v2**: Wait for F5 to roll off the OOS window (fold boundary shifts ~May 2026) OR identify a filter that makes the F5 signal non-negative. No current candidate exists.
- **Current v1 status**: Running, PID 1912802, BEAR regime, BTC $68,839 vs MA $70,037. No trade until BTC > $70,387.
