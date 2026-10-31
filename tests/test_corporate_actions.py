"""复权口径：分红/拆股不能变成假亏损，也不能提前知道未来的公司行为。"""

from __future__ import annotations

import pandas as pd
import pytest

from hkq.corporate_actions import adjust_panel, raw_dividends
from hkq.data import Panel


def make_actions(
    dates: list[pd.Timestamp],
    *,
    dividends: list[float] | None = None,
    splits: list[float] | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": symbols or ["AAA"] * len(dates),
            "dividend": dividends or [0.0] * len(dates),
            "split_ratio": splits or [1.0] * len(dates),
        }
    )


def make_panel(close_values: list[float], volume_values: list[float] | None = None) -> Panel:
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    close = pd.DataFrame({"AAA": close_values}, index=dates)
    volume = pd.DataFrame({"AAA": volume_values or [100.0] * len(close_values)}, index=dates)
    return Panel(open=close, high=close, low=close, close=close, volume=volume)


def empty_actions() -> pd.DataFrame:
    return make_actions([])


def test_dividend_is_added_back_to_return() -> None:
    # 除息日价格从 100 掉到 99，同时派 1 元，真实收益应该是 0
    panel = make_panel([100.0, 100.0, 99.0, 99.0])
    actions = make_actions([panel.dates[2]], dividends=[1.0])

    returns = adjust_panel(panel, actions).close["AAA"].pct_change()

    assert returns.iloc[2] == pytest.approx(0.0)
    assert returns.iloc[1] == pytest.approx(0.0)


def test_split_does_not_create_a_fake_crash() -> None:
    # 3:1 拆股，价格从 300 变成 100，真实收益应该是 0
    panel = make_panel([300.0, 300.0, 100.0, 100.0])
    actions = make_actions([panel.dates[2]], splits=[3.0])

    adjusted = adjust_panel(panel, actions)

    assert adjusted.close["AAA"].pct_change().iloc[2] == pytest.approx(0.0)
    # 因果复权后价格序列连续：四天都是 300
    assert adjusted.close["AAA"].tolist() == pytest.approx([300.0, 300.0, 300.0, 300.0])


def test_volume_is_split_adjusted() -> None:
    panel = make_panel([300.0, 300.0, 100.0, 100.0], volume_values=[100.0, 100.0, 300.0, 300.0])
    actions = make_actions([panel.dates[2]], splits=[3.0])

    adjusted = adjust_panel(panel, actions)

    # 拆股前 100 股等价于拆股后的 300 股，成交量应该落在同一个口径
    assert adjusted.volume["AAA"].tolist() == pytest.approx([100.0, 100.0, 100.0, 100.0])


def test_reverse_split_is_handled() -> None:
    panel = make_panel([30.0, 30.0, 90.0, 90.0])
    actions = make_actions([panel.dates[2]], splits=[1.0 / 3.0])

    adjusted = adjust_panel(panel, actions)

    assert adjusted.close["AAA"].pct_change().iloc[2] == pytest.approx(0.0)


def test_adjust_panel_is_causal() -> None:
    """加一个「未来」的拆股，历史价格不应该变。"""
    panel = make_panel([100.0, 101.0, 102.0, 103.0, 104.0])
    future_split = make_actions([panel.dates[3]], splits=[3.0])

    with_split = adjust_panel(panel, future_split).close["AAA"]
    without_split = adjust_panel(panel, empty_actions()).close["AAA"]

    assert with_split.iloc[:3].tolist() == pytest.approx(without_split.iloc[:3].tolist())


def test_dividends_are_restored_to_pre_split_amounts() -> None:
    # yfinance 把拆股前的分红按之后的拆股除过：3:1 之前派 0.75，它给 0.25
    actions = make_actions(
        [pd.Timestamp("2024-09-25"), pd.Timestamp("2024-10-11"), pd.Timestamp("2024-12-11")],
        dividends=[0.25, 0.0, 0.265],
        splits=[1.0, 3.0, 1.0],
    )

    restored = raw_dividends(actions)

    assert restored["dividend"].tolist() == pytest.approx([0.75, 0.0, 0.265])


def test_reverse_split_and_other_symbols_are_handled() -> None:
    actions = make_actions(
        [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-03"), pd.Timestamp("2020-01-02")],
        dividends=[0.4, 0.0, 0.1],
        splits=[1.0, 0.25, 1.0],
        symbols=["AAA", "AAA", "BBB"],
    )

    restored = raw_dividends(actions)

    # 1:4 反向拆股之前每股派 0.1，yfinance 按新股数折算成 0.4；BBB 没拆股不受影响
    assert restored["dividend"].tolist() == pytest.approx([0.1, 0.0, 0.1])
