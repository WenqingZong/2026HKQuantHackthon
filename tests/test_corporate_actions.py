"""复权口径校验：分红与拆股不能变成假亏损。"""

from __future__ import annotations

import pandas as pd
import pytest

from hkq.corporate_actions import adjust_panel, adjustment_factors
from hkq.data import Panel


def make_actions(dates: list[pd.Timestamp], **kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": kwargs.get("symbols", ["AAA"] * len(dates)),
            "dividend": kwargs.get("dividends", [0.0] * len(dates)),
            "split_ratio": kwargs.get("splits", [1.0] * len(dates)),
        }
    )


def make_panel(close_values: list[float], volume_values: list[float] | None = None) -> Panel:
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    close = pd.DataFrame({"AAA": close_values}, index=dates)
    volume = pd.DataFrame({"AAA": volume_values or [100.0] * len(close_values)}, index=dates)
    return Panel(open=close, high=close, low=close, close=close, volume=volume)


def test_dividend_is_added_back_to_return() -> None:
    # 除息日价格从 100 掉到 99，同时派 1 元，真实收益应该是 0
    panel = make_panel([100.0, 100.0, 99.0, 99.0])
    actions = make_actions([panel.dates[2]], dividends=[1.0])

    adjusted = adjust_panel(panel, actions)
    returns = adjusted.close["AAA"].pct_change()

    assert returns.iloc[2] == pytest.approx(0.0)
    assert returns.iloc[1] == pytest.approx(0.0)


def test_split_does_not_create_a_fake_crash() -> None:
    # 3:1 拆股，价格从 300 变成 100，真实收益应该是 0
    panel = make_panel([300.0, 300.0, 100.0, 100.0])
    actions = make_actions([panel.dates[2]], splits=[3.0])

    adjusted = adjust_panel(panel, actions)
    returns = adjusted.close["AAA"].pct_change()

    assert returns.iloc[2] == pytest.approx(0.0)
    assert adjusted.close["AAA"].iloc[0] == pytest.approx(100.0)


def test_volume_is_split_adjusted() -> None:
    panel = make_panel([300.0, 300.0, 100.0, 100.0], volume_values=[100.0, 100.0, 300.0, 300.0])
    actions = make_actions([panel.dates[2]], splits=[3.0])

    adjusted = adjust_panel(panel, actions)

    # 拆股前 100 股等价于拆股后的 300 股，成交量应该被拉到同一个口径
    assert adjusted.volume["AAA"].tolist() == pytest.approx([300.0, 300.0, 300.0, 300.0])


def test_reverse_split_is_handled() -> None:
    # 1:3 反向拆股，价格从 30 涨到 90，真实收益是 0
    panel = make_panel([30.0, 30.0, 90.0, 90.0])
    actions = make_actions([panel.dates[2]], splits=[1.0 / 3.0])

    adjusted = adjust_panel(panel, actions)

    assert adjusted.close["AAA"].pct_change().iloc[2] == pytest.approx(0.0)


def test_factor_is_one_on_the_last_day() -> None:
    panel = make_panel([100.0, 100.0, 99.0, 99.0])
    actions = make_actions([panel.dates[2]], dividends=[1.0])

    factor = adjustment_factors(panel.close, actions)

    assert factor["AAA"].iloc[-1] == pytest.approx(1.0)
