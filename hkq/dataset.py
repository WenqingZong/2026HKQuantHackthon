"""把行情面板加工成 (date, symbol) 长表特征数据集。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hkq.alphas import compute_alphas
from hkq.corporate_actions import adjust_panel
from hkq.data import Panel

LABEL = "label"


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

    panel 是**原始**行情，复权在这里做：因果复权（只用当日及之前的分红/拆股）之后，
    因子和标签都从同一份复权价上算，t 日的特征不会沾到 t 之后才发生的事件。

    参数
    ----
    actions: 分红/拆股长表，字段 date / symbol / dividend / split_ratio。
    horizon: 标签是未来多少个交易日的收益，应与调仓周期一致。
    normalize: 横截面标准化的方式，rank 为百分位排名（默认），zscore 为标准化。
    min_history: 预热期，必须不小于所有因子里最长的回看天数（否则会直接报错）。
        每只标的从**自己**的第 min_history 根有效 K 线起才进入数据集，晚上市的 ETF 也一样，
        这样不会出现「一半特征还没算出来、被填成 0」的样本。
    label_lag: 因子用当日收盘价算出来，最早只能下一个交易日成交，标签因此后移的天数。
    label_norm: 标签的横截面标准化方式。选股只关心同一天的相对强弱，
        去掉市场整体涨跌（beta）之后模型更容易学到真正有用的排序信号。
    """
    if normalize not in {"rank", "zscore"}:
        raise ValueError(f"不支持的 normalize 方式：{normalize}")
    if min_history >= len(panel.dates):
        raise ValueError(f"min_history={min_history} 超过了数据长度 {len(panel.dates)}")

    adjusted = adjust_panel(panel, actions)
    trading = adjusted.close.notna()
    # 当天没有行情的标的（未上市、已退市、缺数据）不允许带着因子值参与横截面排名：
    # 有些因子在输入缺失时仍会吐出数值，会把别的标的的 rank 挤歪
    frames = {
        name: frame.reindex(index=trading.index, columns=trading.columns).where(trading)
        for name, frame in compute_alphas(adjusted, alphas).items()
    }
    _check_warmup(frames, trading, min_history)
    frames[LABEL] = _normalize_label(
        forward_return(adjusted.close, horizon=horizon, lag=label_lag), label_norm
    )

    data = pd.concat([_to_long(frame, name) for name, frame in frames.items()], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).sort_index()

    # 预热期按标的各自计算：第 min_history 根有效 K 线（从 0 数）之前的样本直接裁掉
    seasoned = _to_long(trading.cumsum().where(trading) > min_history, "seasoned")
    data = data[seasoned.reindex(data.index, fill_value=False).to_numpy()]

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


def factor_warmup(frame: pd.DataFrame, trading: pd.DataFrame) -> int | None:
    """因子从标的上市到第一次有值需要的交易日数（取各标的里最短的那个）。

    用「各标的里最短」而不是日历上的第一天，是为了不被晚上市的 ETF 拉长；全程没有值返回 None。
    """
    has_value = frame.notna().to_numpy()
    listed = trading.to_numpy()
    covered = has_value.any(axis=0) & listed.any(axis=0)
    if not covered.any():
        return None
    first_value = has_value.argmax(axis=0)[covered]
    first_listed = listed.argmax(axis=0)[covered]
    return int((first_value - first_listed).min())


def _check_warmup(frames: dict[str, pd.DataFrame], trading: pd.DataFrame, min_history: int) -> None:
    """预热期比因子回看短，训练集开头的样本就会是「半截因子 + 填 0」，和测试集分布不一致。"""
    too_long: dict[str, int] = {}
    empty: list[str] = []
    for name, frame in frames.items():
        warmup = factor_warmup(frame, trading)
        if warmup is None:
            empty.append(name)
        elif warmup > min_history:
            too_long[name] = warmup
    if empty:
        raise ValueError(f"这些因子在整个样本里都算不出值：{empty}")
    if too_long:
        raise ValueError(
            f"min_history={min_history} 短于这些因子的预热期 {too_long}；"
            f"把 dataset.min_history 调到 {max(too_long.values())} 以上，或者去掉这些因子"
        )


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
