# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research pipeline for the 2026 HK Quant Hackathon (this repo is the modelling piece of a team effort): Databento daily OHLCV for a US ETF universe → Alpha101 factors → XGBoost / LightGBM / MLP → Rank IC + top-k equal-weight backtest. It is research/exploratory code; results are pipeline validation, not strategy conclusions. The README (in Chinese) is the authoritative design doc — read it before changing data, label, or backtest conventions.

## Commands

Python 3.14, managed with `uv`. Torch is pinned to the `pytorch-cu130` index in `pyproject.toml`.

```bash
uv sync                                              # install deps
uv run python main.py download                       # Databento -> data/raw/bars_daily.parquet (costs money, ~$0.2)
uv run python main.py features                       # -> data/processed/features.parquet (+ .meta.json)
uv run python main.py train [--models xgboost,mlp]   # -> artifacts/
uv run python main.py all                            # download + features + train
uv run python main.py --config path.yaml --log-level DEBUG train   # global flags go before the subcommand

uv run python -m pytest tests -q
uv run python -m pytest tests/test_pipeline.py::test_forward_return_respects_lag -q
uv run ruff check . && uv run ruff format --check .
uv run mypy                                          # config in pyproject; untyped defs are errors
```

`download` needs `DATABENTO_API_KEY` in the environment or in a git-ignored `.env` at the repo root (`hkq/data.py:api_key`). `data/` and `artifacts/` are git-ignored, so `features`/`train` fail on a fresh clone until `download` has run.

## Architecture

Everything is driven by `config/default.yaml`, parsed into frozen dataclasses in `hkq/config.py` (relative paths resolve against the repo root). Adding a config key means updating both the YAML and `Config.load`. `main.py` configures logging and dispatches to `hkq/pipeline.py`'s `run_download` / `run_features` / `run_train`; each stage communicates only through parquet files on disk. `pipeline.py` only orchestrates: IC statistics live in `hkq/ic.py`, portfolio simulation and cost accounting in `hkq/portfolio.py`, every artifact (JSON, per-model CSVs, `report.md`) is written by `hkq/report.py`, and dividend/split adjustment lives in `hkq/corporate_actions.py`. Modules log via `logging.getLogger(__name__)`, never `print`.

Two data shapes, and conversion between them is the main thing to keep straight:

- **Wide** (`date × symbol` DataFrame): `hkq.data.Panel` holds one per OHLCV field. Alpha functions (`(Panel) -> DataFrame`), `forward_return`, and `backtest_topk` all operate on wide frames.
- **Long** (`(date, symbol)` MultiIndex): `hkq.dataset.build_dataset` melts factors + `label` column into this. Models, `split_by_date`, and `daily_rank_ic` use long frames. Every column except `LABEL` is treated as a feature (`feature_columns`).

Key conventions (changing these changes results; README explains the reasoning):

- **Label** = return entering at close of t+`label_lag`, held `horizon` days (`forward_return`). `label_lag=1` exists to avoid lookahead because factors use day-t close. Label is cross-sectionally normalised (`label_norm`, default rank).
- **Features** are cross-sectionally ranked to `[-0.5, 0.5]` per date and NaNs filled with 0 (neutral); no scaler anywhere downstream. First `min_history` dates are dropped as warm-up.
- **Splits** are fixed date ranges in config, inclusive on both ends; there is no purge/embargo gap, so the last `label_lag + horizon` days of train have labels that overlap the start of valid. No walk-forward yet.
- **Prices are adjusted** for dividends and splits (`hkq/corporate_actions.py`, actions from yfinance): factors, labels and backtest all use total-return series. `download` writes `data/raw/corporate_actions.parquet`; `pipeline._adjusted_panel` is the single place that applies it. Unadjusted raw prices still live in `bars_daily.parquet`.
- **Backtest** (`hkq/portfolio.py`) is a daily mark-to-market simulation over the test window, using daily total returns or `rebalance_days` — not the dataset label. Execution happens `execution_lag = label_lag` trading days after the signal. Positions are equal-weighted at entry and then left to drift: on each rebalance only names entering/leaving trade. Traded notional is `Σ|Δw|` over securities (cash is not a trade), and cost is `(1+gross)×(1 − Σ|Δw|×cost_bps/1e4) − 1` in `portfolio.net_of_cost`, shared with `cost_sensitivity` (levels generated from the configured cost). New entrants are capped at `1/top_k`; leftover cash stays in cash. Benchmark is equal-weight buy-and-hold of the pool, no cost. Ties are shuffled with the seed before `nlargest`. Fills assume full execution at close.
- Training universe and trading universe are both downloaded (union); training currently uses the full downloaded set. The `*_universe.csv` files carry metadata (e.g. `overlap_family`) not yet used by the pipeline; ticker CSVs have a UTF-8 BOM.

Extension points:

- **New factor**: add `(Panel) -> DataFrame` in `hkq/alphas.py` with the paper formula as docstring, register it in `ALPHAS`, add the name to `dataset.alphas` in the config. Factors needing vwap, adv20, or industry classification are deliberately excluded (no true inputs at daily OHLCV granularity). Factors that rank price *levels* cross-sectionally degenerate on an ETF pool.
- **New model**: implement the `Regressor` protocol (`fit(X, y, X_valid, y_valid)` / `predict(X)` on numpy arrays) in `hkq/models.py`, add a branch in `build_model`, and a params block under `models:` in config. Optional `feature_importance_` attribute gets written to `feature_importance.csv`.
- **Other datasets**: `hkq/data.py:_consolidate` merges multi-venue rows (needed for `DBEQ.BASIC`); intraday schemas get converted to New York time before bucketing to dates. Default `EQUS.SUMMARY` only starts 2024-07-01.
- **New corporate action source**: `hkq/corporate_actions.py` expects a `date, symbol, dividend, split_ratio` frame; `adjust_panel` is the only consumer.

## Code style

Docstrings, comments, error messages and `report.md` are written in Chinese; log messages are mostly English with a `[stage]` prefix. Match this. pandas-stubs types numpy ufuncs on DataFrames as `ndarray`; the code uses `typing.cast(pd.DataFrame, ...)` for those. Ruff config ignores RUF001–003 (full-width punctuation) and E501 in `hkq/alphas.py` (formula docstrings copied verbatim from the paper).