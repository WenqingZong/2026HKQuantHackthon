"""防未来函数的回归测试：因子看不到未来、标签只用持有窗口、切分不交叉。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from hkq.alphas import ALPHAS, compute_alphas
from hkq.data import Panel
from hkq.dataset import build_dataset, feature_columns, forward_return, split_by_date

PanelFactory = Callable[..., Panel]


def test_factors_never_look_at_future_bars(panel_factory: PanelFactory) -> None:
    baseline = panel_factory(seed=1)
    future = panel_factory(seed=99)
    cut = 40

    # 把 cut 之后的行情换成另一组随机数，cut 及之前的因子值必须一模一样
    mixed = {
        field: pd.concat(
            [
                getattr(baseline, field).iloc[: cut + 1],
                getattr(future, field).iloc[cut + 1 :],
            ]
        )
        for field in ("open", "high", "low", "close", "volume")
    }
    patched = Panel(**mixed)

    for name in ALPHAS:
        expected = compute_alphas(baseline, [name])[name].iloc[: cut + 1]
        actual = compute_alphas(patched, [name])[name].iloc[: cut + 1]
        pd.testing.assert_frame_equal(actual, expected)


def test_label_only_uses_the_holding_window() -> None:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = pd.DataFrame({"AAA": 100 + rng.normal(0, 1, 30).cumsum()}, index=dates)
    horizon, lag, day = 5, 1, 10

    perturbed = close.copy()
    perturbed.iloc[: day + lag] *= 7.0  # 进场日之前
    perturbed.iloc[day + lag + horizon + 1 :] *= 7.0  # 持有期之后

    expected = forward_return(close, horizon=horizon, lag=lag)["AAA"].iloc[day]
    actual = forward_return(perturbed, horizon=horizon, lag=lag)["AAA"].iloc[day]

    assert actual == expected


def test_splits_are_chronological_and_disjoint(
    panel_factory: PanelFactory, no_actions: pd.DataFrame
) -> None:
    data = build_dataset(
        panel_factory(n_days=60), no_actions, alphas=["alpha101"], horizon=2, min_history=5
    )
    dates = sorted(data.index.get_level_values("date").unique())

    splits = split_by_date(
        data,
        {
            "train": (str(dates[0].date()), str(dates[19].date())),
            "valid": (str(dates[20].date()), str(dates[39].date())),
            "test": (str(dates[40].date()), str(dates[-1].date())),
        },
    )

    train_dates = set(splits["train"].index.get_level_values("date"))
    valid_dates = set(splits["valid"].index.get_level_values("date"))
    test_dates = set(splits["test"].index.get_level_values("date"))

    assert max(train_dates) < min(valid_dates) < max(valid_dates) < min(test_dates)
    assert not train_dates & valid_dates
    assert not valid_dates & test_dates


def test_dataset_features_come_from_the_configured_alphas(
    panel_factory: PanelFactory, no_actions: pd.DataFrame
) -> None:
    data = build_dataset(
        panel_factory(), no_actions, alphas=["alpha101", "alpha012"], horizon=2, min_history=5
    )

    assert set(feature_columns(data)) == {"alpha101", "alpha012"}


def test_dataset_ignores_corporate_actions_after_the_cut(
    panel_factory: PanelFactory, no_actions: pd.DataFrame
) -> None:
    """在 cut 之后补一次拆股，cut 及之前的特征和标签都不该变。"""
    panel = panel_factory(n_days=60)
    cut = 40
    later_split = pd.DataFrame(
        {
            "date": [panel.dates[50]],
            "symbol": ["AAA"],
            "dividend": [0.0],
            "split_ratio": [3.0],
        }
    )

    # alpha009 用的是价格差，对复权尺度敏感，正好能测出「未来拆股」有没有渗进来
    baseline = build_dataset(panel, no_actions, alphas=["alpha009"], horizon=2, min_history=20)
    with_split = build_dataset(panel, later_split, alphas=["alpha009"], horizon=2, min_history=20)

    cutoff = panel.dates[cut]
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], with_split.loc[:cutoff])
