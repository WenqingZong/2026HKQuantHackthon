"""组合回测：日频盯市模拟、成本计提、绩效与成本敏感度。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
CASH = "CASH"


def cost_levels(baseline_bps: float) -> tuple[float, ...]:
    """成本敏感度的档位，以默认成本为基准按倍数外推，避免和 config 脱钩。"""
    return tuple(round(baseline_bps * multiple, 4) for multiple in (0.0, 1.0, 2.0, 4.0, 8.0))


def net_of_cost(gross: pd.Series, traded_notional: pd.Series, cost_bps: float) -> pd.Series:
    """把成交成本从毛收益里扣掉：net = (1+gross) × (1 − Σ|Δw| × bps) − 1。

    traded_notional 是当期成交的名义金额 Σ|Δw|，买入和卖出两边都在里面，
    所以乘的是一个「单边」费率。回测和成本敏感度都走这个函数。
    """
    return (1.0 + gross) * (1.0 - traded_notional * cost_bps / 1e4) - 1.0


def performance(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    """由逐日（或逐期）收益率算出年化绩效。"""
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
    """回测结果：逐日明细、调仓后的持仓权重、组合与基准绩效。"""

    daily: pd.DataFrame
    holdings: pd.DataFrame
    strategy: dict[str, float]
    benchmark: dict[str, float]

    @property
    def avg_turnover(self) -> float:
        """每次调仓的单边换手率 0.5 × Σ|Δw|，只在调仓日统计。"""
        traded = self.daily.loc[self.daily["traded_notional"] > 0, "traded_notional"]
        return float(traded.mean() / 2.0) if len(traded) else float("nan")


def backtest_topk(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    top_k: int,
    rebalance_days: int,
    cost_bps: float,
    execution_lag: int = 1,
    seed: int = 0,
) -> BacktestResult:
    """按预测值买入前 top_k 只 ETF，逐日盯市，持仓权重随涨跌自然漂移。

    规则：

    - 信号日 t 出预测，第 t + execution_lag 个交易日收盘成交（和标签的 label_lag 对齐，
      不会用到当天收盘后才有信息去成交）；
    - 建仓时等权；之后每 rebalance_days 个交易日调一次仓，只有进出的标的才交易，
      留在组合里的标的权重不重置，省掉「为了恢复等权」的无效换手；
    - 单只标的的权重不会因为「只有它一只新进场」而被灌成重仓，多出来的钱留在现金里；
    - 成本按 Σ|Δw| × cost_bps 计提（只统计证券，现金不算成交，买卖两边都收费）；
    - 基准是期初等权买入全池、之后不再调仓，不计成本。

    predictions / returns 都是 日期 × 代码 的宽表，returns 必须是含分红的总收益日收益。
    """
    dates = predictions.index[predictions.index.isin(returns.index)]
    if len(dates) <= execution_lag:
        raise ValueError(f"回测区间只有 {len(dates)} 天，不够一个 execution_lag")

    symbols = list(returns.columns)
    positions = [*symbols, CASH]
    weights = pd.Series(0.0, index=positions)
    weights[CASH] = 1.0  # 起点空仓

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    holdings: list[dict[str, object]] = []

    for offset, date in enumerate(dates):
        asset_return = returns.loc[date].reindex(symbols).fillna(0.0)
        step = pd.concat([asset_return, pd.Series({CASH: 0.0})])
        gross = float((weights * step).sum())
        # 赚了钱的标的权重自然变大，这是「不重置等权」的核心
        weights = weights * (1.0 + step) / (1.0 + gross)

        signal = offset - execution_lag
        traded = 0.0
        if signal >= 0 and signal % rebalance_days == 0:
            tradable = returns.loc[dates[signal]].dropna().index
            selected = _select(predictions.loc[dates[signal]], tradable, top_k, rng)
            if len(selected) == top_k:
                target = _target_weights(weights, selected, positions, top_k)
                # 现金不算「成交」：从现金买入只是一个买入方向，不能把现金变化也计一遍
                traded = float((target.drop(CASH) - weights.drop(CASH)).abs().sum())
                weights = target
                holdings.extend(
                    {"date": date, "symbol": symbol, "weight": float(weights[symbol])}
                    for symbol in selected
                )

        records.append({"date": date, "gross_return": gross, "traded_notional": traded})

    daily = pd.DataFrame(records).set_index("date")
    daily["net_return"] = net_of_cost(daily["gross_return"], daily["traded_notional"], cost_bps)
    daily["benchmark_return"] = _benchmark_daily(returns.reindex(dates))

    return BacktestResult(
        daily=daily,
        holdings=pd.DataFrame(holdings, columns=["date", "symbol", "weight"]),
        strategy=performance(daily["net_return"], TRADING_DAYS_PER_YEAR),
        benchmark=performance(daily["benchmark_return"], TRADING_DAYS_PER_YEAR),
    )


def _select(
    scores: pd.Series, tradable: pd.Index, top_k: int, rng: np.random.Generator
) -> pd.Index:
    """预测值最高的前 top_k 只，只在当天有行情的标的里选；同分先打散再取。"""
    scores = scores.dropna()
    scores = scores[scores.index.isin(tradable)]
    scores = scores.iloc[rng.permutation(len(scores))]
    return scores.nlargest(top_k).index


def _target_weights(
    weights: pd.Series, selected: pd.Index, positions: list[str], top_k: int
) -> pd.Series:
    """留在组合里的标的权重原样保留，卖掉腾出的钱分给新进场的标的。

    新进场的标的最多拿等权目标 1/top_k：如果某一期只进来一只（其余全是卖出），
    剩下的钱留在现金里，而不是把腾出来的 90% 全压在它身上。
    """
    held = weights.drop(CASH)
    held = held[held > 0]
    keep = held.index.intersection(selected)
    entering = selected.difference(held.index)
    freed = float(weights[CASH] + held.drop(keep).sum())

    target = pd.Series(0.0, index=positions)
    target[keep] = held[keep]
    per_name = min(freed / len(entering), 1.0 / top_k) if len(entering) else 0.0
    if len(entering):
        target[entering] = per_name
    target[CASH] = freed - per_name * len(entering)
    return target


def _benchmark_daily(returns: pd.DataFrame) -> pd.Series:
    """期初等权买入全池、之后不再调仓的日收益（不计成本）。"""
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    equity = growth.mean(axis=1)
    return equity.pct_change().fillna(float(equity.iloc[0] - 1.0))


def cost_sensitivity(
    daily: pd.DataFrame, baseline_bps: float, levels: tuple[float, ...] | None = None
) -> pd.Series:
    """不同成本假设下的年化收益，用来看策略吃不吃交易成本。

    档位默认按默认成本的倍数生成，报告里的「默认那一档」永远等于 config 的 cost_bps。
    回测已经把逐日毛收益和成交名义金额存下来了，这里只是换个费率重算一遍。
    """
    ladder = levels or cost_levels(baseline_bps)
    return pd.Series(
        {
            bps: performance(
                net_of_cost(daily["gross_return"], daily["traded_notional"], bps),
                TRADING_DAYS_PER_YEAR,
            )["cagr"]
            for bps in ladder
        },
        name="cagr",
    )
