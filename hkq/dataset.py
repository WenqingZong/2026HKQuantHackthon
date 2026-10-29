"""把行情面板加工成 (date, symbol) 长表特征数据集。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hkq.alphas import compute_alphas
from hkq.corporate_actions import cumulative_price_factor, point_in_time_panels
from hkq.data import Panel

LABEL = "label"
# PIT 滚动窗比预热期多留几天，保证窗口头部的滚动算子还有足够样本
_WINDOW_MARGIN = 5


def build_dataset(
    panel: Panel,
    actions: pd.DataFrame,
    *,
    alphas: tuple[str, ...] | list[str],
    horizon: int,
    normalize: str = "rank",
    min_history: int = 20,
    label_lag: int = 1,
    label_norm: str = "rank",
) -> pd.DataFrame:
    """计算因子和前瞻收益，返回以 (date, symbol) 为索引的长表。

    panel 是**原始**行情，复权在这里做：因子走逐日的 point-in-time 窗口（只用当日及之前
    的分红/拆股），标签走因果复权后的收益率。这样 t 日的特征不会沾到 t 之后的公司行为。

    参数
    ----
    actions: 分红/拆股长表，字段 date / symbol / dividend / split_ratio。
    horizon: 标签是未来多少个交易日的收益，应与调仓周期一致。
    normalize: 横截面标准化的方式，rank 为百分位排名（默认），zscore 为标准化。
    min_history: 因子最长回看天数；PIT 滚动窗取 min_history + 5，所以实际预热期比它多几天。
    label_lag: 因子用当日收盘价算出来，最早只能下一个交易日成交，标签因此后移的天数。
    label_norm: 标签的横截面标准化方式。选股只关心同一天的相对强弱，
        去掉市场整体涨跌（beta）之后模型更容易学到真正有用的排序信号。
    """
    if normalize not in {"rank", "zscore"}:
        raise ValueError(f"不支持的 normalize 方式：{normalize}")

    window = min_history + _WINDOW_MARGIN
    if window >= len(panel.dates):
        raise ValueError(f"min_history={min_history} 超过了数据长度 {len(panel.dates)}")

    frames = _point_in_time_alphas(panel, actions, alphas=alphas, window=window)
    # 收益率对复权常数不敏感，用因果复权后的收盘价即可
    growth_close = panel.close * cumulative_price_factor(panel.close, actions)
    frames[LABEL] = _normalize_label(
        forward_return(growth_close, horizon=horizon, lag=label_lag), label_norm
    )

    data = pd.concat([_to_long(frame, name) for name, frame in frames.items()], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).sort_index()

    # PIT 窗口头几天凑不满回看窗口，这些行直接裁掉（不留「全 0 = 中性」的假样本）
    data = data[data.index.get_level_values("date") >= panel.dates[window - 1]]

    features = feature_columns(data)
    grouped = data.groupby(level="date")[features]
    if normalize == "rank":
        data[features] = grouped.rank(pct=True) - 0.5
    else:
        data[features] = grouped.transform(lambda column: (column - column.mean()) / column.std())

    # 缺失（历史不足、停牌）统一给中性值 0，树模型和 MLP 都能直接吃
    data[features] = data[features].fillna(0.0)
    data = data.dropna(subset=[LABEL])
    return data.sort_index()


def _point_in_time_alphas(
    panel: Panel, actions: pd.DataFrame, *, alphas: tuple[str, ...] | list[str], window: int
) -> dict[str, pd.DataFrame]:
    """逐日算因子，每天只用当天及之前的数据，价格按「以当天为止」的事件复权。

    这样处理的代价是要跑 len(dates) 次滚动窗口，换来的是因子不会知道未来的分红和拆股。
    """
    last_rows: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in alphas}
    for date, window_panel in point_in_time_panels(panel, actions, window=window):
        for name, frame in compute_alphas(window_panel, alphas).items():
            last_rows[name][date] = frame.iloc[-1]
    return {name: pd.DataFrame(rows).T.sort_index() for name, rows in last_rows.items()}


def _normalize_label(label: pd.DataFrame, method: str) -> pd.DataFrame:
    """按日对标签做横截面标准化，none 表示保留原始收益。"""
    if method == "none":
        return label
    if method == "rank":
        return label.rank(axis=1, pct=True) - 0.5
    if method == "zscore":
        return label.sub(label.mean(axis=1), axis=0).div(label.std(axis=1), axis=0)
    raise ValueError(f"不支持的 label_norm 方式：{method}")


def forward_return(close: pd.DataFrame, *, horizon: int, lag: int = 0) -> pd.DataFrame:
    """前瞻收益：lag 个交易日后的收盘价进场，持有 horizon 个交易日。"""
    return close.shift(-(lag + horizon)) / close.shift(-lag) - 1.0


def _to_long(frame: pd.DataFrame, name: str) -> pd.Series:
    """宽表转长表，返回以 (date, symbol) 为索引的一列。"""
    wide = frame.rename_axis(index="date", columns="symbol").reset_index()
    long = wide.melt(id_vars="date", var_name="symbol", value_name=name)
    return long.set_index(["date", "symbol"])[name]


def feature_columns(data: pd.DataFrame) -> list[str]:
    """数据集里除标签以外的列都是因子特征。"""
    return [column for column in data.columns if column != LABEL]


def split_by_date(
    data: pd.DataFrame, ranges: dict[str, tuple[str, str]]
) -> dict[str, pd.DataFrame]:
    """按日期区间切分训练 / 验证 / 测试集。"""
    dates = data.index.get_level_values("date")
    splits = {}
    for name, (start, end) in ranges.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        splits[name] = data[mask]
    return splits
