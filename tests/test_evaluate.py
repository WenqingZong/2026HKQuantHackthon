"""评估模块：成本计提、成本敏感度与回测口径一致、逐因子 IC 表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkq.dataset import LABEL
from hkq.evaluate import (
    COST_LEVELS_BPS,
    alpha_ic_table,
    backtest_topk,
    cost_sensitivity,
    net_of_cost,
    performance,
    periods_per_year,
)


def make_wide(n_days: int = 60, n_symbols: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_symbols)),
        index=pd.bdate_range("2024-01-01", periods=n_days),
        columns=[f"S{i}" for i in range(n_symbols)],
    )


def test_net_of_cost_deducts_turnover_times_bps() -> None:
    gross = pd.Series([0.01, -0.02])
    turnover = pd.Series([1.0, 0.5])

    result = net_of_cost(gross, turnover, cost_bps=10)

    pd.testing.assert_series_equal(result, pd.Series([0.01 - 0.001, -0.02 - 0.0005]))


def test_cost_sensitivity_matches_backtest_at_configured_cost() -> None:
    forward_returns = make_wide(seed=1)
    result = backtest_topk(
        make_wide(seed=2), forward_returns, top_k=3, rebalance_days=5, cost_bps=5
    )

    sensitivity = cost_sensitivity(result.period, rebalance_days=5)

    assert list(sensitivity.index) == list(COST_LEVELS_BPS)
    assert sensitivity[5] == pytest.approx(result.strategy["cagr"])
    gross_cagr = performance(result.period["gross_return"], periods_per_year(5))["cagr"]
    assert sensitivity[0] == pytest.approx(gross_cagr)
    assert sensitivity.is_monotonic_decreasing


def test_alpha_ic_table_sorts_by_ic_mean() -> None:
    label = make_wide(n_days=10, seed=3).stack()
    label.index.names = ["date", "symbol"]
    data = pd.DataFrame({LABEL: label, "good": label, "bad": -label})

    table = alpha_ic_table(data, ["bad", "good"])

    assert list(table.index) == ["good", "bad"]
    assert table.index.name == "factor"
    assert table.loc["good", "ic_mean"] == pytest.approx(1.0)
    assert table.loc["bad", "ic_mean"] == pytest.approx(-1.0)
    assert table["n_days"].dtype.kind == "i"
