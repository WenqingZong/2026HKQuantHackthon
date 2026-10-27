"""Alpha101 里只依赖日频 OHLCV 的那部分因子。

原文：Kakushadze, "101 Formulaic Alphas", 2015 (https://arxiv.org/abs/1601.00991)。
约定与原文一致：

* rank 是横截面百分位排名（每天在全部 ETF 之间排一次）；
* delta / ts_* 是按代码沿时间轴的算子；
* 返回值越大，代表该因子看多这只 ETF 的力度越强。

需要 vwap、行业分类或 adv20 的因子没有实现——日频 OHLCV 给不出这些输入，
硬凑一个近似值只会给后面的因子评估添噪声。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from hkq.data import Panel

# --- 基础算子 ---------------------------------------------------------------


def _cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """横截面百分位排名（原文的 rank），按日计算，缺失值保持缺失。"""
    return frame.rank(axis=1, pct=True)


def _delta(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    return frame - frame.shift(periods)


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).sum()


def _ts_std(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).std()


def _ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).max()


def _ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).min()


def _ts_corr(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=window).corr(right)


def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """窗口内最后一个值在序列中的百分位排名。"""

    def last_value_rank(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float((values <= values[-1]).mean())

    return frame.rolling(window, min_periods=window).apply(last_value_rank, raw=True)


def _daily_return(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(1) - 1.0


def _log_volume(volume: pd.DataFrame) -> pd.DataFrame:
    """成交量为 0 的停牌/无成交日按缺失处理，避免 log(0) 得到 -inf。"""
    return np.log(volume.where(volume > 0))


# --- 因子 -------------------------------------------------------------------


def alpha002(panel: Panel) -> pd.DataFrame:
    """-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)"""
    volume_change = _cs_rank(_delta(_log_volume(panel.volume), 2))
    intraday_return = _cs_rank((panel.close - panel.open) / panel.open)
    return -_ts_corr(volume_change, intraday_return, 6)


def alpha008(panel: Panel) -> pd.DataFrame:
    """-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))"""
    flow = _ts_sum(panel.open, 5) * _ts_sum(_daily_return(panel.close), 5)
    return -_cs_rank(flow - flow.shift(10))


def alpha014(panel: Panel) -> pd.DataFrame:
    """rank(delta(returns, 3)) * correlation(open, volume, 10)"""
    return _cs_rank(_delta(_daily_return(panel.close), 3)) * _ts_corr(panel.open, panel.volume, 10)


def alpha020(panel: Panel) -> pd.DataFrame:
    """((-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1))) * rank(open - delay(low, 1))"""
    gap_to_high = _cs_rank(panel.open - panel.high.shift(1))
    gap_to_close = _cs_rank(panel.open - panel.close.shift(1))
    gap_to_low = _cs_rank(panel.open - panel.low.shift(1))
    return (-gap_to_high) * gap_to_close * gap_to_low


def alpha022(panel: Panel) -> pd.DataFrame:
    """-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))"""
    price_volume_corr = _ts_corr(panel.high, panel.volume, 5)
    return -1.0 * _delta(price_volume_corr, 5) * _cs_rank(_ts_std(panel.close, 20))


def alpha026(panel: Panel) -> pd.DataFrame:
    """-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""
    corr = _ts_corr(_ts_rank(panel.volume, 5), _ts_rank(panel.high, 5), 5)
    return -_ts_max(corr, 3)


def alpha044(panel: Panel) -> pd.DataFrame:
    """-1 * correlation(high, rank(volume), 5)"""
    return -_ts_corr(panel.high, _cs_rank(panel.volume), 5)


def alpha006(panel: Panel) -> pd.DataFrame:
    """-1 * correlation(open, volume, 10)"""
    return -_ts_corr(panel.open, panel.volume, 10)


def alpha009(panel: Panel) -> pd.DataFrame:
    """(0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -delta(close, 1))"""
    change = _delta(panel.close, 1)
    all_up = _ts_min(change, 5) > 0
    all_down = _ts_max(change, 5) < 0
    return (-change).where(~all_up, change).where(~all_down, change)


def alpha012(panel: Panel) -> pd.DataFrame:
    """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
    return np.sign(_delta(panel.volume, 1)) * (-_delta(panel.close, 1))


def alpha023(panel: Panel) -> pd.DataFrame:
    """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    above_average = _ts_sum(panel.high, 20) / 20 < panel.high
    return (-_delta(panel.high, 2)).where(above_average, 0.0)


def alpha033(panel: Panel) -> pd.DataFrame:
    """rank((-1 * ((1 - (open / close)) ^ 1)))"""
    return _cs_rank(-(1.0 - panel.open / panel.close))


def alpha034(panel: Panel) -> pd.DataFrame:
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    returns = _daily_return(panel.close)
    volatility_ratio = _ts_std(returns, 2) / _ts_std(returns, 5)
    return _cs_rank((1 - _cs_rank(volatility_ratio)) + (1 - _cs_rank(_delta(panel.close, 1))))


def alpha038(panel: Panel) -> pd.DataFrame:
    """((-1 * rank(ts_rank(close, 10))) * rank((close / open)))"""
    return (-_cs_rank(_ts_rank(panel.close, 10))) * _cs_rank(panel.close / panel.open)


def alpha040(panel: Panel) -> pd.DataFrame:
    """((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    return (-_cs_rank(_ts_std(panel.high, 10))) * _ts_corr(panel.high, panel.volume, 10)


def alpha053(panel: Panel) -> pd.DataFrame:
    """-1 * delta((((close - low) - (high - close)) / (close - low)), 9)"""
    close_location = ((panel.close - panel.low) - (panel.high - panel.close)) / (
        panel.close - panel.low
    )
    return -_delta(close_location, 9)


def alpha101(panel: Panel) -> pd.DataFrame:
    """(close - open) / ((high - low) + 0.001)"""
    return (panel.close - panel.open) / (panel.high - panel.low + 0.001)


AlphaFunc = Callable[[Panel], pd.DataFrame]

ALPHAS: dict[str, AlphaFunc] = {
    "alpha002": alpha002,
    "alpha006": alpha006,
    "alpha008": alpha008,
    "alpha009": alpha009,
    "alpha012": alpha012,
    "alpha014": alpha014,
    "alpha020": alpha020,
    "alpha022": alpha022,
    "alpha023": alpha023,
    "alpha026": alpha026,
    "alpha033": alpha033,
    "alpha034": alpha034,
    "alpha038": alpha038,
    "alpha040": alpha040,
    "alpha044": alpha044,
    "alpha053": alpha053,
    "alpha101": alpha101,
}


def compute_alphas(panel: Panel, names: tuple[str, ...] | list[str]) -> dict[str, pd.DataFrame]:
    """按名称批量计算因子，返回 因子名 -> 日期 × 代码 的宽表。"""
    unknown = sorted(set(names) - set(ALPHAS))
    if unknown:
        raise KeyError(f"未实现的因子：{unknown}，可选：{sorted(ALPHAS)}")
    return {name: ALPHAS[name](panel) for name in names}
