"""因子筛选：先在训练集和验证集上都看到信号，再按相关性去重。

只看训练集 IC 挑因子等于在挑噪声：24 个候选里有一半在验证集上符号翻转。
所以这里要求两段同号、且各自的 ICIR 都过门槛，再按相关性去重。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

# 两段各自的门槛。训练集 226 天、验证集 148 天，样本都不长，只能是启发式
MIN_ABS_ICIR_TRAIN = 0.08
MIN_ABS_ICIR_VALID = 0.05
# 两只因子日均相关系数超过这个数就当成同一个信号，只留分数高的那只
MAX_ABS_RHO = 0.80


def factor_correlation(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """逐日横截面相关再按日平均（数据集里的特征已按日 rank，等价于日均 Spearman）。"""
    wide = {name: data[name].unstack("symbol") for name in features}
    total = np.zeros((len(features), len(features)))
    dates = wide[features[0]].index
    for date in dates:
        day = pd.DataFrame({name: wide[name].loc[date] for name in features})
        total += day.corr().to_numpy()
    return pd.DataFrame(total / len(dates), index=features, columns=features)


def select_factors(
    ic_train: pd.DataFrame,
    ic_valid: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    min_icir_train: float = MIN_ABS_ICIR_TRAIN,
    min_icir_valid: float = MIN_ABS_ICIR_VALID,
    max_abs_rho: float = MAX_ABS_RHO,
) -> pd.DataFrame:
    """贪心保留：两段同号 + 两段 ICIR 都过关 + 不和已保留的因子撞车。

    打分用两段里较弱的那个 |ICIR|（保守），分数高的先留。
    返回每个因子一行，status 说明留下还是被谁顶掉。
    """
    table = pd.DataFrame(
        {
            "ic_train": ic_train["ic_mean"],
            "icir_train": ic_train["icir"],
            "ic_valid": ic_valid["ic_mean"],
            "icir_valid": ic_valid["icir"],
            "n_days_train": ic_train["n_days"],
            "n_days_valid": ic_valid["n_days"],
        }
    ).dropna()
    table["same_sign"] = table["ic_train"] * table["ic_valid"] > 0
    table["score"] = table[["icir_train", "icir_valid"]].abs().min(axis=1)

    kept: list[str] = []
    rows: list[dict[str, object]] = []
    for label, row in table.sort_values("score", ascending=False).iterrows():
        name = str(label)
        if not bool(row["same_sign"]):
            reason = "dropped_sign_flips"
        elif abs(float(row["icir_train"])) < min_icir_train:
            reason = "dropped_weak_train"
        elif abs(float(row["icir_valid"])) < min_icir_valid:
            reason = "dropped_weak_valid"
        else:
            clashes = [
                other for other in kept if abs(cast(float, corr.loc[name, other])) >= max_abs_rho
            ]
            reason = f"dropped_correlated_with_{clashes[0]}" if clashes else "kept"
            if not clashes:
                kept.append(name)

        rows.append(
            {
                "factor": name,
                "ic_train": float(row["ic_train"]),
                "icir_train": float(row["icir_train"]),
                "ic_valid": float(row["ic_valid"]),
                "icir_valid": float(row["icir_valid"]),
                "score": float(row["score"]),
                "status": reason,
            }
        )

    return pd.DataFrame(rows).set_index("factor")
