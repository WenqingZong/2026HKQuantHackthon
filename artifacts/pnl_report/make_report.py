"""PnL curves + trade ledger for the 27 walk-forward combos (model x label horizon x rebalance).

Replays hkq.portfolio.backtest_topk step by step (same _select / _target_weights / RNG / cost
formula) so every trade can be written out, then checks the replay reproduces backtest_topk's
daily net returns exactly.

usage: uv run --with matplotlib python pnl_report.py <out_dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path("/home/wenqing/Projects/2026HKQuantHackthon")
sys.path.insert(0, str(ROOT))

from hkq.config import Config  # noqa: E402
from hkq.corporate_actions import adjust_panel, load_actions  # noqa: E402
from hkq.data import load_panel  # noqa: E402
from hkq.portfolio import (  # noqa: E402
    CASH,
    TRADING_DAYS_PER_YEAR,
    _benchmark_daily,
    _select,
    _target_weights,
    backtest_topk,
    net_of_cost,
    performance,
)

HORIZONS = (1, 5, 10)
REBALANCE = (1, 5, 10)
FAMILIES = ("xgboost", "lightgbm", "mlp")
MODEL_NAME = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}
INITIAL_CAPITAL = 1_000_000.0
END = pd.Timestamp("2026-10-15")  # common end of all 9 walk-forward prediction sets (10d label)

# reference palette (dataviz skill), light mode
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
STRATEGY, BAND = "#2a78d6", "#cde2fb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ASSET_ORDER = ["股票", "债券", "商品", "数字资产", "房地产", "外汇"]
ASSET_EN = {"股票": "Equity", "债券": "Bond", "商品": "Commodity", "数字资产": "Digital asset",
            "房地产": "Real estate", "外汇": "FX", CASH: "Cash"}


def replay(pred: pd.DataFrame, returns: pd.DataFrame, raw_close: pd.DataFrame, cfg: Config,
           rebalance: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same loop as backtest_topk, recording trades and end-of-day positions."""
    top_k, lag, bps = cfg.backtest.top_k, cfg.dataset.label_lag, cfg.backtest.cost_bps
    dates = pred.index[pred.index.isin(returns.index)]
    symbols = list(returns.columns)
    positions = [*symbols, CASH]
    weights = pd.Series(0.0, index=positions)
    weights[CASH] = 1.0
    rng = np.random.default_rng(cfg.train.seed)
    nav = INITIAL_CAPITAL
    days, trades, held = [], [], []
    for offset, date in enumerate(dates):
        step = pd.concat([returns.loc[date].reindex(symbols).fillna(0.0), pd.Series({CASH: 0.0})])
        gross = float((weights * step).sum())
        weights = weights * (1.0 + step) / (1.0 + gross)
        nav_pre = nav * (1.0 + gross)
        signal = offset - lag
        traded = 0.0
        if signal >= 0 and signal % rebalance == 0:
            signal_date = dates[signal]
            tradable = returns.loc[signal_date].dropna().index
            scores = pred.loc[signal_date]
            selected = _select(scores, tradable, top_k, rng)
            if len(selected) == top_k:
                target = _target_weights(weights, selected, positions, top_k)
                delta = target.drop(CASH) - weights.drop(CASH)
                traded = float(delta.abs().sum())
                rank = scores.dropna().rank(ascending=False, method="first")
                for symbol in sorted(set(weights.drop(CASH)[weights.drop(CASH) > 0].index) | set(selected)):
                    before, after = float(weights[symbol]), float(target[symbol])
                    action = "BUY" if before == 0 else "SELL" if after == 0 else "HOLD"
                    price = raw_close.at[date, symbol]
                    notional = abs(after - before) * nav_pre
                    trades.append({
                        "signal_date": signal_date.date(), "trade_date": date.date(), "symbol": symbol,
                        "action": action, "weight_before": before, "weight_after": after,
                        "weight_traded": after - before, "nav_before_trade_usd": nav_pre,
                        "value_before_usd": before * nav_pre, "value_after_usd": after * nav_pre,
                        "trade_notional_usd": notional if action != "HOLD" else 0.0,
                        "close_price_raw": price,
                        "shares_traded": (np.sign(after - before) * notional / price) if action != "HOLD" else 0.0,
                        "shares_after": after * nav_pre / price,
                        "model_score": float(scores.get(symbol, np.nan)),
                        "model_rank": float(rank.get(symbol, np.nan)),
                        "cash_weight_after": float(target[CASH]),
                    })
                weights = target
        net = (1.0 + gross) * (1.0 - traded * bps / 1e4) - 1.0
        nav = nav * (1.0 + net)
        days.append({"date": date, "gross_return": gross, "traded_notional": traded, "net_return": net,
                     "nav_usd": nav, "cost_usd": nav_pre * traded * bps / 1e4})
        for symbol, weight in weights[weights > 0].items():
            held.append({"date": date.date(), "symbol": symbol, "weight": float(weight), "value_usd": float(weight) * nav})
    daily = pd.DataFrame(days).set_index("date")
    daily["benchmark_return"] = _benchmark_daily(returns.reindex(dates))
    return daily, pd.DataFrame(trades), pd.DataFrame(held)


def ols_alpha(y: pd.Series, x: pd.Series, lags: int = 5) -> tuple[float, float, float]:
    frame = pd.concat([y, x], axis=1).dropna()
    yy, xx = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones_like(xx), xx])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ coef
    n = len(yy)
    xtx_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = (X[lag:] * resid[lag:, None]).T @ (X[:-lag] * resid[:-lag, None]) / n
        S += w * (G + G.T)
    cov = n * xtx_inv @ S @ xtx_inv
    return float(coef[0] * TRADING_DAYS_PER_YEAR), float(coef[1]), float(coef[0] / np.sqrt(cov[0, 0]))


def best_config(h: int, family: str) -> dict[str, object]:
    name = "sweep_mlp.csv" if family == "mlp" else "sweep_trees.csv"
    table = pd.read_csv(ROOT / f"artifacts/sweep/h{h}/{name}")
    table = table[table["family"] == family]
    row = table.loc[table["valid_icir"].idxmax()]
    return {"config": row["config"], "factor_set": row["factor_set"], "n_factors": int(row["n_factors"]),
            "params": json.loads(row["params"]), "valid_icir": float(row["valid_icir"])}


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def plot_combo(path: Path, title: str, daily: pd.DataFrame, band: pd.DataFrame, exposure: pd.DataFrame,
               notes: list[str]) -> None:
    fig = plt.figure(figsize=(14, 9.5), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], height_ratios=[2.2, 1, 1.4], wspace=0.08, hspace=0.32)
    ax_nav, ax_dd, ax_exp = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[2, 0])
    ax_txt = fig.add_subplot(grid[:, 1])
    for ax in (ax_nav, ax_dd, ax_exp):
        style(ax)

    nav = daily["nav_usd"] / 1e6
    bench = (1.0 + daily["benchmark_return"]).cumprod() * INITIAL_CAPITAL / 1e6
    if band is not None:
        ax_nav.fill_between(band.index, band["min"] / 1e6, band["max"] / 1e6, color=BAND, linewidth=0,
                            label="Range across rebalance start days")
    ax_nav.plot(bench.index, bench, color=MUTED, linewidth=1.5, label="Benchmark: equal-weight pool, buy & hold")
    ax_nav.plot(nav.index, nav, color=STRATEGY, linewidth=2, label="Strategy NAV (net of cost)")
    ax_nav.set_ylabel("NAV (USD mn, start = 1.0)", color=INK_2, fontsize=9)
    ax_nav.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    ax_nav.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")

    dd = nav / nav.cummax() - 1.0
    bdd = bench / bench.cummax() - 1.0
    ax_dd.plot(bdd.index, bdd, color=MUTED, linewidth=1.2, label="Benchmark")
    ax_dd.fill_between(dd.index, dd, 0, color=STRATEGY, alpha=0.25, linewidth=0)
    ax_dd.plot(dd.index, dd, color=STRATEGY, linewidth=1.2, label="Strategy")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dd.set_ylabel("Drawdown", color=INK_2, fontsize=9)
    ax_dd.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_2, ncol=2)

    cols = [c for c in [*ASSET_ORDER, CASH] if c in exposure.columns]
    colors = [SERIES[ASSET_ORDER.index(c)] if c in ASSET_ORDER else "#c3c2b7" for c in cols]
    ax_exp.stackplot(exposure.index, *[exposure[c] for c in cols], colors=colors,
                     labels=[ASSET_EN[c] for c in cols], linewidth=0)
    ax_exp.set_ylim(0, 1)
    ax_exp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_exp.set_ylabel("Weight by asset class", color=INK_2, fontsize=9)
    ax_exp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cols), fontsize=8, frameon=False,
                  labelcolor=INK_2)

    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(notes), va="top", ha="left", family="monospace", fontsize=7.6, color=INK,
                transform=ax_txt.transAxes)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    (out / "charts").mkdir(parents=True, exist_ok=True)
    cfg = Config.load(ROOT / "config/default.yaml")
    raw = load_panel(cfg.data.output)
    returns = adjust_panel(raw, load_actions(cfg.data.corporate_actions)).close.pct_change(fill_method=None)
    universe = pd.read_csv(cfg.universe.training_universe, encoding="utf-8-sig").set_index("ticker")
    trading = pd.read_csv(cfg.universe.trading_universe, encoding="utf-8-sig").set_index("ticker")
    asset_class = universe["asset_class"].combine_first(trading["asset_class"])

    all_trades, all_held, all_nav, summary = [], [], [], []
    overview: dict[tuple[int, str, int], pd.Series] = {}
    bench_curve = None
    for family in FAMILIES:
        for h in HORIZONS:
            best = best_config(h, family)
            pred = pd.read_parquet(ROOT / f"artifacts/walk_forward/h{h}_{family}/{family}/predictions.parquet")
            pred = pred[pred.index.get_level_values("date") <= END]["prediction"].unstack("symbol")
            for r in REBALANCE:
                combo = f"{family}_label{h}d_rebal{r}d"
                daily, trades, held = replay(pred, returns, raw.close, cfg, r)
                check = backtest_topk(pred, returns, top_k=cfg.backtest.top_k, rebalance_days=r,
                                      cost_bps=cfg.backtest.cost_bps, execution_lag=cfg.dataset.label_lag,
                                      seed=cfg.train.seed)
                diff = float((check.daily["net_return"] - daily["net_return"]).abs().max())
                assert diff < 1e-12, (combo, diff)

                band = None
                if r > 1:
                    paths = []
                    for phase in range(r):
                        res = daily if phase == 0 else backtest_topk(
                            pred.iloc[phase:], returns, top_k=cfg.backtest.top_k, rebalance_days=r,
                            cost_bps=cfg.backtest.cost_bps, execution_lag=cfg.dataset.label_lag,
                            seed=cfg.train.seed).daily
                        paths.append(((1 + res["net_return"]).cumprod() * INITIAL_CAPITAL).reindex(daily.index)
                                     .fillna(INITIAL_CAPITAL))
                    stack = pd.concat(paths, axis=1)
                    band = pd.DataFrame({"min": stack.min(axis=1), "max": stack.max(axis=1)})
                    phase_cagr = [(p.iloc[-1] / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / len(p)) - 1 for p in paths]
                else:
                    phase_cagr = [None]

                stats = performance(daily["net_return"], TRADING_DAYS_PER_YEAR)
                bstats = performance(daily["benchmark_return"], TRADING_DAYS_PER_YEAR)
                alpha, beta, alpha_t = ols_alpha(daily["net_return"], daily["benchmark_return"])
                years = len(daily) / TRADING_DAYS_PER_YEAR
                turnover = daily.loc[daily["traded_notional"] > 0, "traded_notional"].mean() / 2
                held_ac = held.assign(asset_class=held["symbol"].map(asset_class))
                exposure = held_ac.pivot_table(index="date", columns="asset_class", values="weight", aggfunc="sum",
                                               fill_value=0.0)
                exposure.index = pd.to_datetime(exposure.index)
                exposure = exposure.reindex(daily.index, fill_value=0.0)
                exposure[CASH] = (1.0 - exposure.sum(axis=1)).clip(lower=0.0)
                avg_exp = exposure.mean()

                p = best["params"]
                notes = [
                    "MODEL & HYPERPARAMETERS",
                    f"  model            {MODEL_NAME[family]} ({best['config']})",
                    f"  label horizon    {h} trading day(s) fwd return,",
                    "                   cross-sectional rank",
                    f"  rebalance every  {r} trading day(s)",
                    f"  factor set       {best['factor_set']} ({best['n_factors']} factors)",
                    *[f"  {k:<16} {v}" for k, v in p.items()],
                    f"  early stopping   {cfg.train.early_stopping_rounds} rounds" if family != "mlp"
                    else "  early stopping   see patience/stop_on",
                    *(["  note             subsample inactive",
                       "                   (no subsample_freq)"] if family == "lightgbm" else []),
                    f"  seed             {cfg.train.seed}",
                    f"  selection        best valid ICIR (2023)",
                    f"                   = {best['valid_icir']:.3f} in label-{h}d sweep",
                    "",
                    "TRAINING (walk-forward)",
                    "  expanding window from 2019-04-30;",
                    "  each test year Y: train ..Y-2,",
                    "  early-stop on Y-1, predict Y",
                    f"  purge {cfg.dataset.label_lag + h} days at train/valid ends",
                    "  features: Alpha101 + Alpha158,",
                    "  daily cross-sectional rank",
                    "",
                    "PORTFOLIO",
                    f"  top_k            {cfg.backtest.top_k}, equal weight at entry",
                    "  then drift; only entries/exits trade",
                    f"  execution        close of t+{cfg.dataset.label_lag} after signal",
                    f"  cost             {cfg.backtest.cost_bps} bp per side",
                    "  universe         182 US ETFs (seasoned)",
                    f"  capital          USD {INITIAL_CAPITAL:,.0f}",
                    "",
                    f"RESULTS {daily.index[0].date()} .. {daily.index[-1].date()}",
                    f"  CAGR             {pct(stats['cagr'])}  (bench {pct(bstats['cagr'])})",
                    *([f"  CAGR all starts  avg {pct(float(np.mean(phase_cagr)))}",
                       f"                   range {pct(min(phase_cagr))} .. {pct(max(phase_cagr))}",
                       "  (curve & metrics below: cycle",
                       "   starting on first test day)"] if r > 1 else []),
                    f"  Sharpe           {stats['sharpe']:.2f}  (bench {bstats['sharpe']:.2f})",
                    f"  max drawdown     {pct(stats['max_drawdown'])}  (bench {pct(bstats['max_drawdown'])})",
                    f"  ann. vol         {stats['ann_vol']:.1%}",
                    f"  beta vs bench    {beta:.2f}",
                    f"  alpha (ann.)     {pct(alpha)}  (NW t {alpha_t:.1f})",
                    f"  turnover/rebal   {turnover:.0%} one-way",
                    f"  cost drag        {daily['cost_usd'].sum() / INITIAL_CAPITAL / years:.2%}/yr of start cap",
                    "",
                    "AVG WEIGHT BY ASSET CLASS",
                    *[f"  {ASSET_EN[c]:<16} {avg_exp[c]:.0%}" for c in [*ASSET_ORDER, CASH] if c in avg_exp],
                ]
                title = f"{MODEL_NAME[family]} | label {h}d | rebalance every {r}d | walk-forward, net of cost"
                plot_combo(out / "charts" / f"{combo}.png", title, daily, band, exposure, notes)

                meta = {"combo": combo, "model": family, "model_config": best["config"], "label_horizon_days": h,
                        "rebalance_days": r, "factor_set": best["factor_set"]}
                trades = trades.assign(asset_class=trades["symbol"].map(asset_class).map(ASSET_EN))
                all_trades.append(trades.assign(**meta))
                all_held.append(held.assign(asset_class=held["symbol"].map(asset_class).map(ASSET_EN), combo=combo))
                all_nav.append(daily[["net_return", "nav_usd", "cost_usd", "traded_notional"]]
                               .assign(combo=combo).reset_index())
                summary.append({**meta, "n_factors": best["n_factors"], "params": json.dumps(p),
                                "start": daily.index[0].date(), "end": daily.index[-1].date(),
                                "cagr": stats["cagr"], "cagr_start_day_avg": float(np.mean(phase_cagr)) if r > 1 else stats["cagr"],
                                "cagr_start_day_min": min(phase_cagr) if r > 1 else stats["cagr"],
                                "cagr_start_day_max": max(phase_cagr) if r > 1 else stats["cagr"],
                                "sharpe": stats["sharpe"], "max_drawdown": stats["max_drawdown"],
                                "ann_vol": stats["ann_vol"], "beta": beta, "alpha_ann": alpha, "alpha_t_nw": alpha_t,
                                "bench_cagr": bstats["cagr"], "bench_max_drawdown": bstats["max_drawdown"],
                                "turnover_per_rebalance": turnover,
                                "n_trades": int((trades["action"] != "HOLD").sum())})
                overview[(h, family, r)] = daily["nav_usd"]
                bench_curve = (1 + daily["benchmark_return"]).cumprod() * INITIAL_CAPITAL
                print(f"{combo:32s} CAGR {pct(stats['cagr'])} MDD {pct(stats['max_drawdown'])} "
                      f"trades {summary[-1]['n_trades']} (replay diff {diff:.1e})", flush=True)

    # overview: rows = model, cols = label horizon, lines = rebalance frequency
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True, facecolor=SURFACE)
    for i, family in enumerate(FAMILIES):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            style(ax)
            ax.plot(bench_curve.index, bench_curve / 1e6, color=MUTED, linewidth=1.3, label="Benchmark (EW pool)")
            for k, r in enumerate(REBALANCE):
                s = overview[(h, family, r)]
                ax.plot(s.index, s / 1e6, color=SERIES[k], linewidth=1.6, label=f"Rebalance every {r}d")
                ax.annotate(f"{r}d", (s.index[-1], s.iloc[-1] / 1e6), xytext=(3, 0), textcoords="offset points",
                            fontsize=7, color=INK_2, va="center")
            ax.set_title(f"{MODEL_NAME[family]} | label {h}d", loc="left", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("NAV (USD mn)", color=INK_2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle("Walk-forward NAV, net of cost, USD 1mn start, 2024-01 .. 2026-10-15 "
                 "(start-day phase 0; per-combo hyperparameters in individual charts)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "charts" / "overview_all_combos.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    trade_cols = ["combo", "model", "model_config", "label_horizon_days", "rebalance_days", "factor_set",
                  "signal_date", "trade_date", "symbol", "asset_class", "action", "model_score", "model_rank",
                  "weight_before", "weight_after", "weight_traded", "value_before_usd", "value_after_usd",
                  "trade_notional_usd", "close_price_raw", "shares_traded", "shares_after",
                  "nav_before_trade_usd", "cash_weight_after"]
    pd.concat(all_trades)[trade_cols].to_csv(out / "trades.csv", index=False, float_format="%.6g")
    pd.concat(all_held)[["combo", "date", "symbol", "asset_class", "weight", "value_usd"]].to_csv(
        out / "daily_positions.csv", index=False, float_format="%.6g")
    pd.concat(all_nav)[["combo", "date", "net_return", "nav_usd", "cost_usd", "traded_notional"]].to_csv(
        out / "daily_nav.csv", index=False, float_format="%.8g")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, float_format="%.6g")
    shutil.copy(__file__, out / "make_report.py")


if __name__ == "__main__":
    main()
