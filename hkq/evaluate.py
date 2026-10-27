"""评估：逐日 IC、按日分组统计、top-k 等权组合回测、成本敏感度。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hkq.dataset import LABEL

TRADING_DAYS_PER_YEAR = 252.0
COST_LEVELS_BPS = (0, 5, 10, 20, 30)


def daily_rank_ic(prediction: pd.Series, label: pd.Series) -> pd.Series:
    """按日计算预测值与未来收益的 Spearman 相关系数（Rank IC）。"""
    frame = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1).dropna()
    ic = frame.groupby(level="date").apply(
        lambda day: day["prediction"].corr(day["label"], method="spearman")
    )
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
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


def alpha_ic_table(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """逐因子计算 Rank IC 统计，作为后续因子筛选的基线。"""
    rows = {name: ic_summary(daily_rank_ic(data[name], data[LABEL])) for name in features}
    table = pd.DataFrame(rows).T
    table["n_days"] = table["n_days"].astype(int)
    table.index.name = "factor"
    return table.sort_values("ic_mean", ascending=False)


def periods_per_year(rebalance_days: int) -> float:
    return TRADING_DAYS_PER_YEAR / rebalance_days


def net_of_cost(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """按每期换手计提成本：net = gross − turnover × cost_bps / 1e4。

    回测和成本敏感度都走这一个函数，改成本口径只改这里。
    """
    return gross - turnover * cost_bps / 1e4


def performance(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
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
    strategy: dict[str, float]
    benchmark: dict[str, float]


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
    records: list[dict[str, object]] = []
    holdings: dict[pd.Timestamp, list[str]] = {}

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
        period["net_return"] = net_of_cost(period["gross_return"], period["turnover"], cost_bps)

    per_year = periods_per_year(rebalance_days)
    net = period.get("net_return", pd.Series(dtype=float))
    benchmark = period.get("benchmark_return", pd.Series(dtype=float))
    return BacktestResult(
        period=period,
        holdings=pd.DataFrame(holdings).T,
        strategy=performance(net, per_year),
        benchmark=performance(benchmark, per_year),
    )


def cost_sensitivity(
    period: pd.DataFrame, rebalance_days: int, levels_bps: tuple[float, ...] = COST_LEVELS_BPS
) -> pd.Series:
    """不同成本假设下的年化收益，用来看策略吃不吃交易成本。

    回测已经把每期的毛收益和换手存下来了，这里只是换个数重算一遍，不用重新训练。
    """
    per_year = periods_per_year(rebalance_days)
    return pd.Series(
        {
            bps: performance(
                net_of_cost(period["gross_return"], period["turnover"], bps), per_year
            )["cagr"]
            for bps in levels_bps
        },
        name="cagr",
    )
