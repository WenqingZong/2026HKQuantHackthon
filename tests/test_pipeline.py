"""因子与数据集的基础校验：公式正确、标签对齐、切分不越界。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkq.alphas import alpha012, alpha033, alpha101, compute_alphas
from hkq.data import Panel
from hkq.dataset import LABEL, build_dataset, feature_columns, forward_return, split_by_date


def make_panel(n_days: int = 40, seed: int = 0) -> Panel:
    """构造一个可复现的小面板，字段齐全但数值随机。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    close = pd.DataFrame(
        100 + rng.normal(0, 1, (n_days, len(symbols))).cumsum(axis=0),
        index=dates,
        columns=symbols,
    )
    open_ = close + rng.normal(0, 0.5, close.shape)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = pd.DataFrame(
        rng.integers(1_000, 10_000, close.shape).astype(float), index=dates, columns=symbols
    )
    return Panel(open=open_, high=high, low=low, close=close, volume=volume)


def test_alpha101_matches_formula():
    panel = make_panel()
    expected = (panel.close - panel.open) / (panel.high - panel.low + 0.001)
    pd.testing.assert_frame_equal(alpha101(panel), expected)


def test_alpha012_follows_volume_direction():
    panel = make_panel()
    expected = np.sign(panel.volume.diff()) * (-panel.close.diff())
    pd.testing.assert_frame_equal(alpha012(panel), expected)


def test_alpha033_is_cross_sectional_rank():
    panel = make_panel()
    expected = (panel.open / panel.close - 1.0).rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(alpha033(panel), expected)


def test_compute_alphas_rejects_unknown_factor():
    with pytest.raises(KeyError):
        compute_alphas(make_panel(), ["alpha999"])


def test_forward_return_respects_lag():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"AAA": [1.0, 2.0, 4.0, 8.0]}, index=dates)
    result = forward_return(close, horizon=1, lag=1)

    assert result["AAA"].iloc[0] == pytest.approx(4.0 / 2.0 - 1.0)
    assert np.isnan(result["AAA"].iloc[-1])


def test_build_dataset_is_nan_free():
    data = build_dataset(make_panel(), alphas=["alpha101", "alpha012"], horizon=2, min_history=5)

    assert data.index.names == ["date", "symbol"]
    assert not data[feature_columns(data)].isna().any().any()
    assert data[LABEL].notna().all()


def test_split_by_date_stays_inside_windows():
    data = build_dataset(make_panel(), alphas=["alpha101"], horizon=2, min_history=5)
    dates = sorted(data.index.get_level_values("date").unique())
    start, middle, end = (str(dates[i].date()) for i in (0, 20, -1))

    splits = split_by_date(data, {"first": (start, middle), "second": (middle, end)})

    first_dates = splits["first"].index.get_level_values("date")
    second_dates = splits["second"].index.get_level_values("date")
    assert first_dates.max() <= pd.Timestamp(middle)
    assert second_dates.min() >= pd.Timestamp(middle)
    assert len(splits["first"]) + len(splits["second"]) >= len(data)
