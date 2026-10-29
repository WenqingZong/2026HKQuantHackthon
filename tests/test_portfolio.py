"""组合回测：成本双边计提、权重漂移、新进场封顶、成本敏感度。"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from hkq.portfolio import (
    CASH,
    TRADING_DAYS_PER_YEAR,
    _target_weights,
    backtest_topk,
    cost_levels,
    cost_sensitivity,
    net_of_cost,
    performance,
)

WideFactory = Callable[..., pd.DataFrame]


def test_net_of_cost_charges_both_sides() -> None:
    gross = pd.Series([0.01])
    # 卖出一半再买入一半 = 成交名义金额 1.0，买卖各收一次 10bp
    result = net_of_cost(gross, pd.Series([1.0]), cost_bps=10)

    assert result.iloc[0] == pytest.approx(1.01 * (1.0 - 0.001) - 1.0)


def test_first_buy_only_charges_the_buy_side() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    predictions = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])

    result = backtest_topk(predictions, returns, top_k=2, rebalance_days=5, cost_bps=10)

    # 空仓建仓只成交一次 100%，现金变化不算交易
    assert result.daily["traded_notional"].max() == pytest.approx(1.0)
    assert result.daily["net_return"].iloc[1] == pytest.approx(-0.001)


def test_first_trade_happens_one_day_after_the_signal() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    predictions = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])

    result = backtest_topk(
        predictions, returns, top_k=2, rebalance_days=5, cost_bps=0, execution_lag=1
    )

    # 信号日当天不成交，避免「用收盘价算信号、又按同一个收盘价成交」
    assert result.daily["traded_notional"].iloc[0] == 0.0
    assert result.daily["traded_notional"].iloc[1] == pytest.approx(1.0)


def test_full_rotation_charges_both_sides() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    predictions = pd.DataFrame(
        {
            "A": [2.0] * 4 + [0.0] * 4,
            "B": [1.0] * 4 + [0.0] * 4,
            "C": [0.0] * 4 + [1.0] * 4,
            "D": [0.0] * 4 + [2.0] * 4,
        },
        index=dates,
    )
    returns = pd.DataFrame(0.0, index=dates, columns=list("ABCD"))

    result = backtest_topk(predictions, returns, top_k=2, rebalance_days=4, cost_bps=10)

    # 卖 A、B（100%）+ 买 C、D（100%）= 200%
    assert result.daily["traded_notional"].max() == pytest.approx(2.0)


def test_kept_holdings_drift_instead_of_resetting() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    predictions = pd.DataFrame({"A": 2.0, "B": 1.0, "C": 0.0}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C"])
    returns["A"] = 0.02

    result = backtest_topk(predictions, returns, top_k=2, rebalance_days=2, cost_bps=0)

    weights = result.holdings.loc[result.holdings["symbol"] == "A", "weight"]
    assert weights.is_monotonic_increasing
    assert weights.iloc[-1] > weights.iloc[0] + 0.02
    # 一直在手的标的不会被重新等权，所以除了建仓那一次，再没有成交
    assert (result.daily["traded_notional"] > 0).sum() == 1


def test_single_entrant_is_capped_at_equal_weight() -> None:
    positions = ["A", "B", "C", CASH]
    weights = pd.Series({"A": 0.8, "B": 0.2, "C": 0.0, CASH: 0.0})

    target = _target_weights(weights, pd.Index(["B", "C"]), positions, top_k=2)

    assert target["B"] == pytest.approx(0.2)  # 留在组合里的不动
    assert target["C"] == pytest.approx(0.5)  # 新进场的顶格 1/top_k
    assert target[CASH] == pytest.approx(0.3)  # 剩下的钱留在现金，不会全压一只
    assert float(target.sum()) == pytest.approx(1.0)


def test_cost_sensitivity_is_anchored_on_the_configured_cost(
    wide_factory: WideFactory,
) -> None:
    returns = wide_factory(seed=1)
    predictions = wide_factory(seed=2)

    result = backtest_topk(predictions, returns, top_k=3, rebalance_days=5, cost_bps=1.5)
    sensitivity = cost_sensitivity(result.daily, baseline_bps=1.5)

    assert list(sensitivity.index) == list(cost_levels(1.5))
    assert 1.5 in sensitivity.index
    assert sensitivity[1.5] == pytest.approx(result.strategy["cagr"])
    gross_cagr = performance(result.daily["gross_return"], TRADING_DAYS_PER_YEAR)["cagr"]
    assert sensitivity[0.0] == pytest.approx(gross_cagr)
    assert sensitivity.is_monotonic_decreasing


def test_benchmark_has_one_return_per_day(wide_factory: WideFactory) -> None:
    returns = wide_factory(n_days=20, seed=3)
    predictions = wide_factory(n_days=20, seed=4)

    result = backtest_topk(predictions, returns, top_k=3, rebalance_days=5, cost_bps=0)

    assert len(result.daily) == len(returns)
    assert result.daily["benchmark_return"].notna().all()
