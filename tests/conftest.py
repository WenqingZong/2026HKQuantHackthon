"""测试共用的随机数据构造器，避免每个测试文件各写一份。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from hkq.data import Panel

PanelFactory = Callable[..., Panel]
WideFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def panel_factory() -> PanelFactory:
    """构造一个可复现的 OHLCV 面板，字段齐全但数值随机。"""

    def build(n_days: int = 40, seed: int = 0) -> Panel:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        close = pd.DataFrame(
            100 + rng.normal(0, 1, (n_days, len(symbols))).cumsum(axis=0),
            index=dates,
            columns=symbols,
        )
        open_ = close + rng.normal(0, 0.5, close.shape)
        high = open_.where(open_ > close, close) + 0.5
        low = open_.where(open_ < close, close) - 0.5
        volume = pd.DataFrame(
            rng.integers(1_000, 10_000, close.shape).astype(float), index=dates, columns=symbols
        )
        return Panel(open=open_, high=high, low=low, close=close, volume=volume)

    return build


@pytest.fixture
def wide_factory() -> WideFactory:
    """构造一个随机日收益/预测的宽表（日期 × 代码）。"""

    def build(n_days: int = 60, n_symbols: int = 8, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.normal(0, 0.01, (n_days, n_symbols)),
            index=pd.bdate_range("2024-01-01", periods=n_days),
            columns=[f"S{i}" for i in range(n_symbols)],
        )

    return build
