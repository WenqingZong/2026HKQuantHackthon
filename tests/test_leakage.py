"""防未来函数的回归测试：因子看不到未来、标签只用持有窗口、切分不交叉。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from hkq.alpha158 import ALPHA158_NAMES
from hkq.alphas import ALPHAS, compute_alphas
from hkq.data import Panel
from hkq.dataset import build_dataset, feature_columns, forward_return, split_by_date

PanelFactory = Callable[..., Panel]


def test_factors_never_look_at_future_bars(panel_factory: PanelFactory) -> None:
    # 面板必须比最长回看（250 天）还长，否则 cut 之前全是 NaN，测试会空跑
    n_days, cut = 320, 290
    baseline = panel_factory(n_days=n_days, seed=1)
    future = panel_factory(n_days=n_days, seed=99)

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
    cutoff = baseline.dates[cut]

    names = [*ALPHAS, *ALPHA158_NAMES]
    expected_frames = compute_alphas(baseline, names)
    actual_frames = compute_alphas(patched, names)
    for name in names:
        # 按日期切：Alpha158 的滑窗算子返回的表从第 window-1 行才开始，按位置切会错位
        expected = expected_frames[name].loc[:cutoff]
        actual = actual_frames[name].loc[:cutoff]
        assert expected.notna().to_numpy().any(), f"{name} 在 cut 之前没有任何值，测试是空跑"
        pd.testing.assert_frame_equal(actual, expected)


def test_conditional_factors_stay_missing_until_inputs_exist(panel_factory: PanelFactory) -> None:
    """未上市的日子不能有因子值：NaN 比较为 False，曾让三元因子在缺失处输出固定的 ±1。"""
    panel = panel_factory(n_days=320, seed=4)
    listed = 270
    late = Panel(
        **{
            field: getattr(panel, field).assign(
                AAA=getattr(panel, field)["AAA"].where(
                    pd.Series(range(320), index=panel.dates) >= listed
                )
            )
            for field in ("open", "high", "low", "close", "volume")
        }
    )

    names = [*ALPHAS, *ALPHA158_NAMES]
    frames = compute_alphas(late, names)
    leaked = [
        name
        for name in names
        if frames[name].reindex(panel.dates)["AAA"].iloc[:listed].notna().any()
    ]
    assert leaked == []


def test_build_dataset_rejects_warmup_shorter_than_lookback(
    panel_factory: PanelFactory, no_actions: pd.DataFrame
) -> None:
    # alpha019 要 250 天的 sum(returns)，预热期只给 20 天必须直接报错，而不是悄悄填 0
    with pytest.raises(ValueError, match="alpha019"):
        build_dataset(
            panel_factory(n_days=320), no_actions, alphas=["alpha019"], horizon=2, min_history=20
        )


def test_build_dataset_rejects_factor_without_any_value(
    panel_factory: PanelFactory, no_actions: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def all_missing(panel: Panel) -> pd.DataFrame:
        return panel.close * np.nan

    monkeypatch.setitem(ALPHAS, "alpha101", all_missing)
    with pytest.raises(ValueError, match="都算不出值"):
        build_dataset(panel_factory(), no_actions, alphas=["alpha101"], horizon=2, min_history=5)


def test_late_listing_enters_after_its_own_warmup(
    panel_factory: PanelFactory, no_actions: pd.DataFrame
) -> None:
    panel = panel_factory(n_days=80, seed=2)
    listed, min_history = 30, 10
    alive = pd.Series(range(80), index=panel.dates) >= listed
    late = Panel(
        **{
            field: getattr(panel, field).assign(AAA=getattr(panel, field)["AAA"].where(alive))
            for field in ("open", "high", "low", "close", "volume")
        }
    )

    data = build_dataset(late, no_actions, alphas=["STD5"], horizon=2, min_history=min_history)
    first = data.xs("AAA", level="symbol").index.min()
    others = data.drop(index="AAA", level="symbol").index.get_level_values("date").min()

    assert first == panel.dates[listed + min_history]
    assert others == panel.dates[min_history]


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
