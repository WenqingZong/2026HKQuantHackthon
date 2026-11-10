# Walk-forward PnL and position report: 27 parameter combinations

Research backtest, not live trading. Combos = 3 models (XGBoost, LightGBM, MLP) × 3 label horizons
(model trained to predict the 1-, 5- or 10-trading-day forward return) × 3 rebalance frequencies
(every 1, 5 or 10 trading days). For each model and label horizon the hyperparameters and factor set
are the best by 2023 validation ICIR. Each chart lists every hyperparameter.

- Period: 2024-01-02 to 2026-10-15. Models are retrained every January (expanding window from
  2019-04-30, early stopping on the previous year, purge of label_lag + horizon days).
- Portfolio: top 10 by model score, equal weight at entry, then weights drift. On each rebalance only
  names entering or leaving the top 10 trade. A new entrant gets at most 1/10. Execution is at the
  close one trading day after the signal date. Cost is 1.5 bp per side on traded notional.
- Benchmark: equal-weight buy-and-hold of the whole pool, no cost.
- Capital: USD 1,000,000 at the start. Dollar amounts and share counts assume this capital, fills at
  the unadjusted close, and fractional shares.
- Start day: with 5- and 10-day rebalancing the result depends on which day the cycle starts. The
  curves, trades and positions use the cycle starting on the first test day. The shaded band in each
  chart shows the range across all start days. `summary.csv` gives the average, min and max CAGR
  across start days. Compare combos with the average, not the single path.
- Returns use dividend- and split-adjusted total-return prices.

## Files

| File | Content |
|---|---|
| `charts/<model>_label<H>d_rebal<R>d.png` | NAV vs benchmark (with start-day band), drawdown, daily weight by asset class, and all hyperparameters and results |
| `charts/overview_all_combos.png` | All 27 NAV curves on one page (rows = model, columns = label horizon, lines = rebalance frequency) |
| `trades.csv` | One row per symbol per rebalance: buys, sells and unchanged holdings |
| `daily_positions.csv` | End-of-day holdings for every combo and day (after that day's trades) |
| `daily_nav.csv` | Daily net return, NAV, cost and traded notional per combo |
| `summary.csv` | One row per combo: hyperparameters and performance metrics |
| `make_report.py` | Script that generated this folder |

## `trades.csv` columns

| Column | Meaning |
|---|---|
| `combo`, `model`, `model_config`, `label_horizon_days`, `rebalance_days`, `factor_set` | Which parameter combination (full hyperparameters are in `summary.csv` and the charts) |
| `signal_date` | Date whose close-of-day factors produced the scores |
| `trade_date` | Date the trade executes, at the close |
| `symbol`, `asset_class` | ETF and its asset class |
| `action` | `BUY` (new position), `SELL` (full exit) or `HOLD` (kept, not traded) |
| `model_score`, `model_rank` | Model prediction on the signal date and its rank in the pool (1 = best) |
| `weight_before`, `weight_after`, `weight_traded` | Portfolio weight before and after the trade, and the change (+ buy, − sell) |
| `value_before_usd`, `value_after_usd` | Position value before and after the trade |
| `trade_notional_usd` | Dollar amount bought or sold (0 for HOLD) |
| `close_price_raw` | Unadjusted closing price on the trade date |
| `shares_traded`, `shares_after` | Shares bought (+) or sold (−), and shares held after the trade |
| `nav_before_trade_usd` | Portfolio NAV at the close, before trading costs |
| `cash_weight_after` | Weight left in cash after the rebalance |

## Caveats

- Over the period, all combos run beta of 1.19–1.47 to the benchmark. Alpha after beta is not
  statistically significant for any of them: the highest Newey-West t is 1.66. Most of the excess
  return over the benchmark is the high-beta tilt, mainly equity ETFs overweight versus the pool.
- Fills assume the full size trades at the close. There is no market impact, and 1.5 bp is
  optimistic for the least liquid ETFs.
