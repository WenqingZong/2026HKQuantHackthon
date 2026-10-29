"""IC 统计：逐因子表按 IC 均值排序。"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from hkq.dataset import LABEL
from hkq.ic import alpha_ic_table, daily_rank_ic, ic_summary

WideFactory = Callable[..., pd.DataFrame]


def test_alpha_ic_table_sorts_by_ic_mean(wide_factory: WideFactory) -> None:
    label = cast(pd.Series, wide_factory(n_days=10, seed=5).stack())
    label.index.names = ["date", "symbol"]
    data = pd.DataFrame({LABEL: label, "good": label, "bad": -label})

    table = alpha_ic_table(data, ["bad", "good"])

    assert list(table.index) == ["good", "bad"]
    assert table.index.name == "factor"
    assert table.loc["good", "ic_mean"] == pytest.approx(1.0)
    assert table.loc["bad", "ic_mean"] == pytest.approx(-1.0)
    assert table["n_days"].dtype.kind == "i"


def test_ic_summary_of_perfect_signal(wide_factory: WideFactory) -> None:
    label = cast(pd.Series, wide_factory(n_days=10, seed=1).stack())
    label.index.names = ["date", "symbol"]

    summary = ic_summary(daily_rank_ic(label, label))

    assert summary["ic_mean"] == pytest.approx(1.0)
    assert summary["ic_win_rate"] == pytest.approx(1.0)
    assert summary["n_days"] == 10


def test_ic_summary_handles_too_few_days() -> None:
    summary = ic_summary(pd.Series([0.1]))

    assert summary["n_days"] == 1
    assert np.isnan(summary["ic_mean"])
