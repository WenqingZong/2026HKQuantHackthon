"""IC 统计：逐日 Rank IC、汇总指标、逐因子 IC 表。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hkq.dataset import LABEL


def daily_rank_ic(prediction: pd.Series, label: pd.Series) -> pd.Series:
    """按日计算预测值与未来收益的 Spearman 相关系数（Rank IC）。"""
    frame = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1).dropna()
    ic = frame.groupby(level="date").apply(
        lambda day: day["prediction"].corr(day["label"], method="spearman")
    )
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """把逐日 IC 序列压成几个常用指标。"""
    ic = ic.dropna()
    n_days = len(ic)
    if n_days < 2:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_win_rate": np.nan,
            "ic_t": np.nan,
            "n_days": n_days,
        }

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_win_rate": float((ic > 0).mean()),
        "ic_t": icir * np.sqrt(n_days) if std > 0 else np.nan,
        "n_days": n_days,
    }


def alpha_ic_table(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """逐因子计算 Rank IC 统计，作为后续因子筛选的基线。"""
    rows = {name: ic_summary(daily_rank_ic(data[name], data[LABEL])) for name in features}
    table = pd.DataFrame(rows).T
    table["n_days"] = table["n_days"].astype(int)
    table.index.name = "factor"
    return table.sort_values("ic_mean", ascending=False)
