"""评估：逐日 IC、按日分组统计、top-k 等权组合回测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def daily_rank_ic(prediction: pd.Series, label: pd.Series) -> pd.Series:
    """按日计算预测值与未来收益的 Spearman 相关系数（Rank IC）。"""
    frame = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1).dropna()
    ic = frame.groupby(level="date").apply(
        lambda day: day["prediction"].corr(day["label"], method="spearman")
    )
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict:
    """把逐日 IC 序列压成几个常用指标。"""
    ic = ic.dropna()
    n_days = len(ic)
    if n_days < 2:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_win_rate": np.nan,
            "ic_t": np.nan,
            "n_days": n_days,
        }

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_win_rate": float((ic > 0).mean()),
        "ic_t": icir * np.sqrt(n_days) if std > 0 else np.nan,
        "n_days": n_days,
    }


def performance(returns: pd.Series, periods_per_year: float) -> dict:
    """由每个调仓期的收益率算出年化绩效。"""
    if returns.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }

    equity = (1.0 + returns).cumprod()
    years = len(returns) / periods_per_year
    std = float(returns.std(ddof=1))
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "win_rate": float((returns > 0).mean()),
    }


@dataclass(frozen=True)
class BacktestResult:
    """回测结果：逐期明细、持仓、组合与基准绩效。"""

    period: pd.DataFrame
    holdings: pd.DataFrame
    strategy: dict
    benchmark: dict


def backtest_topk(
    predictions: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    top_k: int,
    rebalance_days: int,
    cost_bps: float,
    seed: int = 0,
) -> BacktestResult:
    """每个调仓日等权买入预测值最高的 top_k 只 ETF，持有 rebalance_days 个交易日。

    predictions / forward_returns 都是 日期 × 代码 的宽表，两者按日期对齐。
    换手按 0.5 * sum(|w_new - w_old|) 计算（现金也算一个仓位，所以建仓那期是 100%），
    再乘单边成本。基准是全池等权、不计成本。
    同分股票会先随机打散再取前 k，避免默认按代码字母序挑选。
    """
    dates = predictions.index[predictions.index.isin(forward_returns.index)]
    rebalance_dates = dates[::rebalance_days]
    symbols = predictions.columns

    rng = np.random.default_rng(seed)
    positions = [*symbols, "CASH"]
    previous_weights = pd.Series(0.0, index=positions)
    previous_weights["CASH"] = 1.0  # 回测起点是空仓
    records, holdings = [], {}

    for date in rebalance_dates:
        scores = predictions.loc[date].dropna()
        realized = forward_returns.loc[date].dropna()
        scores = scores[scores.index.isin(realized.index)]
        scores = scores.iloc[rng.permutation(len(scores))]
        if len(scores) < top_k:
            continue

        selected = scores.nlargest(top_k).index
        weights = pd.Series(0.0, index=positions)
        weights[selected] = 1.0 / top_k
        turnover = float(0.5 * (weights - previous_weights).abs().sum())

        records.append(
            {
                "date": date,
                "gross_return": float((weights[selected] * realized[selected]).sum()),
                "benchmark_return": float(realized.mean()),
                "turnover": turnover,
            }
        )
        holdings[date] = list(selected)
        previous_weights = weights

    period = pd.DataFrame(records).set_index("date")
    if period.empty:
        period = pd.DataFrame(columns=["gross_return", "benchmark_return", "turnover"])
    else:
        period["net_return"] = period["gross_return"] - period["turnover"] * cost_bps / 1e4

    periods_per_year = 252.0 / rebalance_days
    net = period.get("net_return", pd.Series(dtype=float))
    benchmark = period.get("benchmark_return", pd.Series(dtype=float))
    return BacktestResult(
        period=period,
        holdings=pd.DataFrame(holdings).T,
        strategy=performance(net, periods_per_year),
        benchmark=performance(benchmark, periods_per_year),
    )
