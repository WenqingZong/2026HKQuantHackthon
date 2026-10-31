"""Alpha101 因子（78 个，全部只依赖日频 OHLCV）。

原文：Kakushadze, "101 Formulaic Alphas", 2015 (https://arxiv.org/abs/1601.00991)。
约定与原文一致：

* rank 是横截面百分位排名（每天在全部 ETF 之间排一次）；
* delta / ts_* 是按代码沿时间轴的算子；
* 返回值越大，代表该因子看多这只 ETF 的力度越强。

* 日频没有真实 vwap，用 HLC3 代理；advN 用 N 日平均美元成交额；原文的分数窗口取整。
  口径和 psq 的相关性分享包一致。需要行业分类的因子没有实现。
* 条件分支（x < y ? a : b、min/max）在任一输入缺失时结果也缺失：NaN 比较为 False，
  不处理的话预热期和未上市的日子会输出一个确定的 ±1，看起来像信号。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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


def _ts_argmax(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """最大值在窗口内的位置（1 表示窗口最后一个值最大）。"""

    def last_argmax(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.argmax(values) + 1)

    return frame.rolling(window, min_periods=window).apply(last_argmax, raw=True)


def _ts_argmin(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """最小值在窗口内的位置（1 表示窗口最后一个值最小）。"""

    def last_argmin(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.argmin(values) + 1)

    return frame.rolling(window, min_periods=window).apply(last_argmin, raw=True)


def _scale(frame: pd.DataFrame) -> pd.DataFrame:
    """按日归一化，使绝对值之和为 1（原文的 scale）。"""
    return frame.div(frame.abs().sum(axis=1), axis=0)


def _vwap(panel: Panel) -> pd.DataFrame:
    """日频没有真实成交额加权价，用 HLC3 代理（和 psq 相关性包的口径一致）。"""
    return (panel.high + panel.low + panel.close) / 3.0


def _adv(panel: Panel, window: int = 20) -> pd.DataFrame:
    """window 日平均成交额（美元），当作原文的 advN 代理。"""
    return (panel.close * panel.volume).rolling(window, min_periods=window).mean()


def _decay_linear(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权平均，权重 1..window（原文的 decay_linear）。"""
    weights = np.arange(1.0, window + 1.0)
    return frame.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()), raw=True
    )


def _signed_power(frame: pd.DataFrame, power: float) -> pd.DataFrame:
    """保留符号的幂（原文的 signedpower）。"""
    return cast(pd.DataFrame, np.sign(frame) * frame.abs() ** power)


def _ts_product(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).apply(
        lambda values: float(np.prod(values)), raw=True
    )


def _daily_return(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(1) - 1.0


def _log_volume(volume: pd.DataFrame) -> pd.DataFrame:
    """成交量为 0 的停牌/无成交日按缺失处理，避免 log(0) 得到 -inf。"""
    return cast(pd.DataFrame, np.log(volume.where(volume > 0)))


def _valid(*frames: pd.DataFrame) -> pd.DataFrame:
    """所有输入都不缺失的位置。条件分支要靠它把「输入还没算出来」留成缺失。"""
    mask = frames[0].notna()
    for frame in frames[1:]:
        mask &= frame.notna()
    return mask


def _if_less(
    left: pd.DataFrame, right: pd.DataFrame, then: float, otherwise: float
) -> pd.DataFrame:
    """left < right ? then : otherwise。

    NaN 参与比较的结果是 False，直接 np.where 会把预热期/未上市的缺失值变成 otherwise
    这个确定的信号，所以任一边缺失时结果也必须缺失。
    """
    chosen = pd.DataFrame(
        np.where(left < right, then, otherwise), index=left.index, columns=left.columns
    )
    return chosen.where(_valid(left, right))


def _min(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """逐元素 min，任一边缺失则缺失（DataFrame.where 在 first 缺失时会静默取 second）。"""
    return first.where(first < second, second).where(_valid(first, second))


def _max(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """逐元素 max，任一边缺失则缺失。"""
    return first.where(first > second, second).where(_valid(first, second))


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
    window_min, window_max = _ts_min(change, 5), _ts_max(change, 5)
    signed = (-change).where(~(window_min > 0), change).where(~(window_max < 0), change)
    return signed.where(_valid(window_min, window_max))


def alpha012(panel: Panel) -> pd.DataFrame:
    """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
    return cast(pd.DataFrame, np.sign(_delta(panel.volume, 1)) * (-_delta(panel.close, 1)))


def alpha023(panel: Panel) -> pd.DataFrame:
    """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    average = _ts_sum(panel.high, 20) / 20
    return (-_delta(panel.high, 2)).where(average < panel.high, 0.0).where(_valid(average))


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


def alpha010(panel: Panel) -> pd.DataFrame:
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    change = _delta(panel.close, 1)
    window_min, window_max = _ts_min(change, 4), _ts_max(change, 4)
    signed = (-change).where(~(window_min > 0), change).where(~(window_max < 0), change)
    return _cs_rank(signed.where(_valid(window_min, window_max)))


def alpha018(panel: Panel) -> pd.DataFrame:
    """-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))"""
    body = panel.close - panel.open
    shape = _ts_std(body.abs(), 5) + body + _ts_corr(panel.close, panel.open, 10)
    return -_cs_rank(shape)


def alpha046(panel: Panel) -> pd.DataFrame:
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? -1 : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : -1 * (close - delay(close, 1))))"""
    trend = (panel.close.shift(20) - panel.close.shift(10)) / 10 - (
        panel.close.shift(10) - panel.close
    ) / 10
    change = -(panel.close - panel.close.shift(1))
    # trend > 0.25 -> -1；trend < 0 -> +1；其余用当日涨跌
    return change.where(trend >= 0, 1.0).where(trend <= 0.25, -1.0).where(_valid(trend))


def alpha049(panel: Panel) -> pd.DataFrame:
    """((((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.1) ? 1 : -1 * (close - delay(close, 1))))"""
    trend = (panel.close.shift(20) - panel.close.shift(10)) / 10 - (
        panel.close.shift(10) - panel.close
    ) / 10
    change = -(panel.close - panel.close.shift(1))
    return change.where(trend >= -0.1, 1.0).where(_valid(trend))


def alpha051(panel: Panel) -> pd.DataFrame:
    """((((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.05) ? 1 : -1 * (close - delay(close, 1))))"""
    trend = (panel.close.shift(20) - panel.close.shift(10)) / 10 - (
        panel.close.shift(10) - panel.close
    ) / 10
    change = -(panel.close - panel.close.shift(1))
    return change.where(trend >= -0.05, 1.0).where(_valid(trend))


def alpha054(panel: Panel) -> pd.DataFrame:
    """-1 * ((low - close) * (open ^ 5)) / ((low - high) * (close ^ 5))"""
    return -((panel.low - panel.close) * panel.open**5) / (
        (panel.low - panel.high) * panel.close**5
    )


def alpha060(panel: Panel) -> pd.DataFrame:
    """0 - (1 * ((2 * scale(rank((((close - low) - (high - close)) / (high - low)) * volume))) - scale(rank(ts_argmax(close, 10)))))"""
    money_flow = _cs_rank(
        (((panel.close - panel.low) - (panel.high - panel.close)) / (panel.high - panel.low))
        * panel.volume
    )
    peak_age = _cs_rank(_ts_argmax(panel.close, 10))
    return -1.0 * (2.0 * _scale(money_flow) - _scale(peak_age))


def alpha001(panel: Panel) -> pd.DataFrame:
    """rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5"""
    returns = _daily_return(panel.close)
    volatility = _ts_std(returns, 20)
    base = panel.close.where(returns >= 0, volatility).where(_valid(volatility))
    return _cs_rank(_ts_argmax(_signed_power(base, 2), 5)) - 0.5


def alpha003(panel: Panel) -> pd.DataFrame:
    """-1 * correlation(rank(open), rank(volume), 10)"""
    return -_ts_corr(_cs_rank(panel.open), _cs_rank(panel.volume), 10)


def alpha004(panel: Panel) -> pd.DataFrame:
    """-1 * ts_rank(rank(low), 9)"""
    return -_ts_rank(_cs_rank(panel.low), 9)


def alpha005(panel: Panel) -> pd.DataFrame:
    """rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))"""
    vwap = _vwap(panel)
    return _cs_rank(panel.open - _ts_sum(vwap, 10) / 10) * (-_cs_rank(panel.close - vwap).abs())


def alpha011(panel: Panel) -> pd.DataFrame:
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    gap = _vwap(panel) - panel.close
    return (_cs_rank(_ts_max(gap, 3)) + _cs_rank(_ts_min(gap, 3))) * _cs_rank(
        _delta(panel.volume, 3)
    )


def alpha013(panel: Panel) -> pd.DataFrame:
    """-1 * rank(covariance(rank(close), rank(volume), 5))"""
    covariance = _cs_rank(panel.close).rolling(5, min_periods=5).cov(_cs_rank(panel.volume))
    return -_cs_rank(covariance)


def alpha015(panel: Panel) -> pd.DataFrame:
    """-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    corr = _ts_corr(_cs_rank(panel.high), _cs_rank(panel.volume), 3)
    return -_ts_sum(_cs_rank(corr), 3)


def alpha016(panel: Panel) -> pd.DataFrame:
    """-1 * rank(covariance(rank(high), rank(volume), 5))"""
    covariance = _cs_rank(panel.high).rolling(5, min_periods=5).cov(_cs_rank(panel.volume))
    return -_cs_rank(covariance)


def alpha017(panel: Panel) -> pd.DataFrame:
    """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    return (
        (-_cs_rank(_ts_rank(panel.close, 10)))
        * _cs_rank(_delta(_delta(panel.close, 1), 1))
        * _cs_rank(_ts_rank(panel.volume / _adv(panel, 20), 5))
    )


def alpha019(panel: Panel) -> pd.DataFrame:
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    returns = _daily_return(panel.close)
    trend = (panel.close - panel.close.shift(7)) + _delta(panel.close, 7)
    return -cast(pd.DataFrame, np.sign(trend)) * (1 + _cs_rank(1 + _ts_sum(returns, 250)))


def alpha021(panel: Panel) -> pd.DataFrame:
    """连板/回归条件：均值与标准差的相对位置 + 成交量条件"""
    close = panel.close
    ratio = panel.volume / _adv(panel, 20)
    mean8, std8 = _ts_sum(close, 8) / 8, _ts_std(close, 8)
    mean2 = _ts_sum(close, 2) / 2
    chosen = pd.DataFrame(
        np.where(
            mean8 + std8 < mean2,
            -1.0,
            np.where(mean2 < mean8 - std8, 1.0, np.where(ratio >= 1, 1.0, -1.0)),
        ),
        index=close.index,
        columns=close.columns,
    )
    return chosen.where(_valid(mean8, std8, mean2, ratio))


def alpha024(panel: Panel) -> pd.DataFrame:
    """((((delta(sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))"""
    trend = (_delta(_ts_sum(panel.close, 100) / 100, 100)) / panel.close.shift(100)
    long_leg = -(panel.close - _ts_min(panel.close, 100))
    return long_leg.where(trend <= 0.05, -_delta(panel.close, 3)).where(_valid(trend))


def alpha025(panel: Panel) -> pd.DataFrame:
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    vwap = _vwap(panel)
    return _cs_rank(
        (-_daily_return(panel.close)) * _adv(panel, 20) * vwap * (panel.high - panel.close)
    )


def alpha027(panel: Panel) -> pd.DataFrame:
    """(0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? -1 : 1"""
    corr = _ts_corr(_cs_rank(panel.volume), _cs_rank(_vwap(panel)), 6)
    rank = _cs_rank(_ts_sum(corr, 2) / 2.0)
    half = pd.DataFrame(0.5, index=rank.index, columns=rank.columns)
    return _if_less(half, rank, -1.0, 1.0)


def alpha028(panel: Panel) -> pd.DataFrame:
    """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    return _scale(
        _ts_corr(_adv(panel, 20), panel.low, 5) + (panel.high + panel.low) / 2 - panel.close
    )


def alpha029(panel: Panel) -> pd.DataFrame:
    """(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 5), 1) + ts_rank(delay((-1 * returns), 6), 5))"""
    step = -_cs_rank(_delta(panel.close - 1, 5))
    inner = _cs_rank(_cs_rank(step))
    inner = _ts_min(inner, 2)
    inner = _cs_rank(_cs_rank(_scale(cast(pd.DataFrame, np.log(inner)))))
    return pd.DataFrame(
        np.minimum(_ts_product(inner, 5), 1.0), index=panel.close.index, columns=panel.close.columns
    ) + _ts_rank((-_daily_return(panel.close)).shift(6), 5)


def alpha030(panel: Panel) -> pd.DataFrame:
    """(((1.0 - rank(((sign(delta(close,1)) + sign(delta(close,1) 的移位)) + ...))) * sum(volume, 5)) / sum(volume, 20)"""
    direction = cast(
        pd.DataFrame,
        np.sign(_delta(panel.close, 1))
        + np.sign(_delta(panel.close, 1).shift(1))
        + np.sign(_delta(panel.close, 1).shift(2)),
    )
    money = _ts_sum(panel.volume, 5) / _ts_sum(panel.volume, 20)
    return (1.0 - _cs_rank(direction)) * money


def alpha031(panel: Panel) -> pd.DataFrame:
    """rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank(-delta(close, 3)) + sign(scale(correlation(adv20, low, 12)))"""
    core = _decay_linear(-_cs_rank(_cs_rank(_delta(panel.close, 10))), 10)
    return (
        _cs_rank(_cs_rank(_cs_rank(core)))
        + _cs_rank(-_delta(panel.close, 3))
        + np.sign(_scale(_ts_corr(_adv(panel, 20), panel.low, 12)))
    )


def alpha032(panel: Panel) -> pd.DataFrame:
    """scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))"""
    return _scale(_ts_sum(panel.close, 7) / 7 - panel.close) + 20 * _scale(
        _ts_corr(_vwap(panel), panel.close.shift(5), 230)
    )


def alpha035(panel: Panel) -> pd.DataFrame:
    """((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))"""
    return (
        _ts_rank(panel.volume, 32)
        * (1 - _ts_rank(panel.close + panel.high - panel.low, 16))
        * (1 - _ts_rank(_daily_return(panel.close), 32))
    )


def alpha036(panel: Panel) -> pd.DataFrame:
    """2.21*rank(corr(close-open, delay(volume,1), 15)) + 0.7*rank(open-close) + 0.73*rank(ts_rank(delay(-returns,6),5)) + rank(abs(corr(vwap,adv20,6))) + 0.6*rank((sum(close,200)/200 - open)*(close-open))"""
    vwap = _vwap(panel)
    returns = _daily_return(panel.close)
    return (
        2.21 * _cs_rank(_ts_corr(panel.close - panel.open, panel.volume.shift(1), 15))
        + 0.7 * _cs_rank(panel.open - panel.close)
        + 0.73 * _cs_rank(_ts_rank((-returns).shift(6), 5))
        + _cs_rank(_ts_corr(vwap, _adv(panel, 20), 6).abs())
        + 0.6
        * _cs_rank((_ts_sum(panel.close, 200) / 200 - panel.open) * (panel.close - panel.open))
    )


def alpha037(panel: Panel) -> pd.DataFrame:
    """(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    return _cs_rank(_ts_corr((panel.open - panel.close).shift(1), panel.close, 200)) + _cs_rank(
        panel.open - panel.close
    )


def alpha039(panel: Panel) -> pd.DataFrame:
    """((-1 * rank(((sum(close, 7) / 7) - close))) * (1 + rank((sum(returns, 250)))))"""
    returns = _daily_return(panel.close)
    return -_cs_rank(_ts_sum(panel.close, 7) / 7 - panel.close) * (
        1 + _cs_rank(_ts_sum(returns, 250))
    )


def alpha041(panel: Panel) -> pd.DataFrame:
    """(((high * low) ^ 0.5) - vwap)"""
    return cast(pd.DataFrame, (panel.high * panel.low) ** 0.5) - _vwap(panel)


def alpha042(panel: Panel) -> pd.DataFrame:
    """(rank((vwap - close)) / rank((vwap + close)))"""
    vwap = _vwap(panel)
    return _cs_rank(vwap - panel.close) / _cs_rank(vwap + panel.close)


def alpha043(panel: Panel) -> pd.DataFrame:
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    return _ts_rank(panel.volume / _adv(panel, 20), 20) * _ts_rank(-_delta(panel.close, 7), 8)


def alpha045(panel: Panel) -> pd.DataFrame:
    """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    return (
        -_cs_rank(_ts_sum(panel.close.shift(5), 20) / 20)
        * _ts_corr(panel.close, panel.volume, 2)
        * _cs_rank(_ts_corr(_ts_sum(panel.close, 5), _ts_sum(panel.close, 20), 2))
    )


def alpha047(panel: Panel) -> pd.DataFrame:
    """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    vwap = _vwap(panel)
    return (
        (_cs_rank(1.0 / panel.close) * panel.volume / _adv(panel, 20))
        * (panel.high * _cs_rank(panel.high - panel.close))
        / (_ts_sum(panel.high, 5) / 5)
    ) - _cs_rank(vwap - vwap.shift(5))


def alpha050(panel: Panel) -> pd.DataFrame:
    """-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)"""
    corr = _ts_corr(_cs_rank(panel.volume), _cs_rank(_vwap(panel)), 5)
    return -_ts_max(_cs_rank(corr), 5)


def alpha052(panel: Panel) -> pd.DataFrame:
    """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    returns = _daily_return(panel.close)
    low_min = _ts_min(panel.low, 5)
    return (
        (-low_min + low_min.shift(5))
        * _cs_rank((_ts_sum(returns, 240) - _ts_sum(returns, 20)) / 220)
        * _ts_rank(panel.volume, 5)
    )


def alpha055(panel: Panel) -> pd.DataFrame:
    """-1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)"""
    low_min = _ts_min(panel.low, 12)
    position = (panel.close - low_min) / (_ts_max(panel.high, 12) - low_min)
    return -_ts_corr(_cs_rank(position), _cs_rank(panel.volume), 6)


def alpha061(panel: Panel) -> pd.DataFrame:
    """(rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18))) ? 1 : -1"""
    vwap = _vwap(panel)
    left = _cs_rank(vwap - _ts_min(vwap, 16))
    right = _cs_rank(_ts_corr(vwap, _adv(panel, 180), 18))
    return _if_less(left, right, 1.0, -1.0)


def alpha062(panel: Panel) -> pd.DataFrame:
    """((rank(correlation(vwap, sum(adv20, 22), 10)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    vwap = _vwap(panel)
    left = _cs_rank(_ts_corr(vwap, _ts_sum(_adv(panel, 20), 22), 10))
    inner = (_cs_rank(panel.open) + _cs_rank(panel.open)) < (
        _cs_rank((panel.high + panel.low) / 2) + _cs_rank(panel.high)
    )
    right = _cs_rank(inner.astype(float).where(_valid(panel.open, panel.high, panel.low)))
    return _if_less(left, right, -1.0, 0.0)


def alpha064(panel: Panel) -> pd.DataFrame:
    """((rank(correlation(sum((open*0.178404 + low*0.821596), 13), sum(adv120, 13), 17)) < rank(delta((((high+low)/2)*0.178404 + vwap*0.821596), 4))) * -1)"""
    vwap = _vwap(panel)
    left = _cs_rank(
        _ts_corr(
            _ts_sum(panel.open * 0.178404 + panel.low * 0.821596, 13),
            _ts_sum(_adv(panel, 120), 13),
            17,
        )
    )
    right = _cs_rank(_delta((panel.high + panel.low) / 2 * 0.178404 + vwap * 0.821596, 4))
    return _if_less(left, right, -1.0, 0.0)


def alpha065(panel: Panel) -> pd.DataFrame:
    """((rank(correlation((open*0.00817205 + vwap*0.99182795), sum(adv60, 9), 6)) < rank((open - ts_min(open, 14)))) * -1)"""
    vwap = _vwap(panel)
    left = _cs_rank(
        _ts_corr(panel.open * 0.00817205 + vwap * 0.99182795, _ts_sum(_adv(panel, 60), 9), 6)
    )
    right = _cs_rank(panel.open - _ts_min(panel.open, 14))
    return _if_less(left, right, -1.0, 0.0)


def alpha066(panel: Panel) -> pd.DataFrame:
    """((rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear((((low*0.96633 + low*0.03367) - vwap) / (open - ((high+low)/2))), 11), 7)) * -1)"""
    vwap = _vwap(panel)
    return -(
        _cs_rank(_decay_linear(_delta(vwap, 4), 7))
        + _ts_rank(
            _decay_linear((panel.low - vwap) / (panel.open - (panel.high + panel.low) / 2), 11), 7
        )
    )


def alpha068(panel: Panel) -> pd.DataFrame:
    """((ts_rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta((close*0.518371 + low*0.481629), 1))) * -1)"""
    left = _ts_rank(_ts_corr(_cs_rank(panel.high), _cs_rank(_adv(panel, 15)), 9), 14)
    right = _cs_rank(_delta(panel.close * 0.518371 + panel.low * 0.481629, 1))
    return _if_less(left, right, -1.0, 0.0)


def alpha071(panel: Panel) -> pd.DataFrame:
    """((ts_rank(correlation(close, sum(adv20, 9), 15), 15) < rank((ts_rank(close, 3) * ts_rank(ts_rank(adv180, 12), 4)))) * -1)"""
    left = _ts_rank(_ts_corr(panel.close, _ts_sum(_adv(panel, 20), 9), 15), 15)
    right = _cs_rank(_ts_rank(panel.close, 3) * _ts_rank(_ts_rank(_adv(panel, 180), 12), 4))
    return _if_less(left, right, -1.0, 0.0)


def alpha072(panel: Panel) -> pd.DataFrame:
    """rank(decay_linear(correlation(((high+low)/2), adv40, 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))"""
    vwap = _vwap(panel)
    numerator = _cs_rank(
        _decay_linear(_ts_corr((panel.high + panel.low) / 2, _adv(panel, 40), 9), 10)
    )
    denominator = _cs_rank(
        _decay_linear(_ts_corr(_ts_rank(vwap, 4), _ts_rank(panel.volume, 19), 7), 3)
    )
    return numerator / denominator


def alpha073(panel: Panel) -> pd.DataFrame:
    """(max(rank(decay_linear(delta(vwap, 5), 3)), ts_rank(decay_linear((((high*0.147155 + low*0.852845) - vwap) / (open - ((high+low)/2))), 3), 12)) * -1)"""
    vwap = _vwap(panel)
    first = _cs_rank(_decay_linear(_delta(vwap, 5), 3))
    second = _ts_rank(
        _decay_linear(
            (panel.high * 0.147155 + panel.low * 0.852845 - vwap)
            / (panel.open - (panel.high + panel.low) / 2),
            3,
        ),
        12,
    )
    return -_max(first, second)


def alpha074(panel: Panel) -> pd.DataFrame:
    """((rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(((high+low)/2)), rank(adv15), 9))) * -1)"""
    left = _cs_rank(_ts_corr(panel.close, _ts_sum(_adv(panel, 30), 37), 15))
    right = _cs_rank(_ts_corr(_cs_rank((panel.high + panel.low) / 2), _cs_rank(_adv(panel, 15)), 9))
    return _if_less(left, right, -1.0, 0.0)


def alpha075(panel: Panel) -> pd.DataFrame:
    """(rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))) ? 1 : -1"""
    vwap = _vwap(panel)
    left = _cs_rank(_ts_corr(vwap, panel.volume, 4))
    right = _cs_rank(_ts_corr(_cs_rank(panel.low), _cs_rank(_adv(panel, 50)), 12))
    return _if_less(left, right, 1.0, -1.0)


def alpha077(panel: Panel) -> pd.DataFrame:
    """min(rank(decay_linear(((((high+low)/2) + high) - (vwap + high)), 20)), rank(decay_linear(correlation(((high+low)/2), adv40, 3), 6)))"""
    vwap = _vwap(panel)
    first = _cs_rank(
        _decay_linear(((panel.high + panel.low) / 2 + panel.high) - (vwap + panel.high), 20)
    )
    second = _cs_rank(_decay_linear(_ts_corr((panel.high + panel.low) / 2, _adv(panel, 40), 3), 6))
    return _min(first, second)


def alpha078(panel: Panel) -> pd.DataFrame:
    """rank(correlation(sum((low*0.352233 + vwap*0.647767), 20), sum(adv40, 20), 7)) ^ rank(correlation(rank(vwap), rank(volume), 6))"""
    vwap = _vwap(panel)
    base = _cs_rank(
        _ts_corr(
            _ts_sum(panel.low * 0.352233 + vwap * 0.647767, 20), _ts_sum(_adv(panel, 40), 20), 7
        )
    )
    exponent = _cs_rank(_ts_corr(_cs_rank(vwap), _cs_rank(panel.volume), 6))
    return base**exponent


def alpha081(panel: Panel) -> pd.DataFrame:
    """((rank(log(product(rank((rank(correlation(vwap, sum(adv10, 50), 8)) ^ 4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5))) * -1)"""
    vwap = _vwap(panel)
    inner = _cs_rank(_ts_corr(vwap, _ts_sum(_adv(panel, 10), 50), 8)) ** 4
    left = _cs_rank(cast(pd.DataFrame, np.log(_ts_product(_cs_rank(inner), 15))))
    right = _cs_rank(_ts_corr(_cs_rank(vwap), _cs_rank(panel.volume), 5))
    return _if_less(left, right, -1.0, 0.0)


def alpha083(panel: Panel) -> pd.DataFrame:
    """((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
    vwap = _vwap(panel)
    span = (panel.high - panel.low) / (_ts_sum(panel.close, 5) / 5)
    return (_cs_rank(span.shift(2)) * _cs_rank(_cs_rank(panel.volume))) / (
        span / (vwap - panel.close)
    )


def alpha084(panel: Panel) -> pd.DataFrame:
    """signedpower(ts_rank((vwap - ts_max(vwap, 15)), 21), 1)"""
    vwap = _vwap(panel)
    return _signed_power(_ts_rank(vwap - _ts_max(vwap, 15), 21), 1)


def alpha085(panel: Panel) -> pd.DataFrame:
    """(rank(correlation((high*0.876703 + close*0.123297), adv30, 10)) < rank(correlation(ts_rank(((high+low)/2), 4), ts_rank(volume, 10), 7))) * -1"""
    left = _cs_rank(_ts_corr(panel.high * 0.876703 + panel.close * 0.123297, _adv(panel, 30), 10))
    right = _cs_rank(
        _ts_corr(_ts_rank((panel.high + panel.low) / 2, 4), _ts_rank(panel.volume, 10), 7)
    )
    return _if_less(left, right, -1.0, 0.0)


def alpha086(panel: Panel) -> pd.DataFrame:
    """((ts_rank(correlation(close, sum(adv20, 15), 6), 20) < rank(((open + close) - (vwap + open)))) * -1)"""
    vwap = _vwap(panel)
    left = _ts_rank(_ts_corr(panel.close, _ts_sum(_adv(panel, 20), 15), 6), 20)
    right = _cs_rank((panel.open + panel.close) - (vwap + panel.open))
    return _if_less(left, right, -1.0, 0.0)


def alpha088(panel: Panel) -> pd.DataFrame:
    """min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3))"""
    first = _cs_rank(
        _decay_linear(
            (_cs_rank(panel.open) + _cs_rank(panel.low))
            - (_cs_rank(panel.high) + _cs_rank(panel.close)),
            8,
        )
    )
    second = _ts_rank(
        _decay_linear(_ts_corr(_ts_rank(panel.close, 8), _ts_rank(_adv(panel, 60), 21), 8), 7), 3
    )
    return _min(first, second)


def alpha092(panel: Panel) -> pd.DataFrame:
    """min(ts_rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 15), 19), ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))"""
    first = _ts_rank(
        _decay_linear(
            ((panel.high + panel.low) / 2 + panel.close < panel.low + panel.open)
            .astype(float)
            .where(_valid(panel.open, panel.high, panel.low, panel.close)),
            15,
        ),
        19,
    )
    second = _ts_rank(
        _decay_linear(_ts_corr(_cs_rank(panel.low), _cs_rank(_adv(panel, 30)), 8), 7), 7
    )
    return _min(first, second)


def alpha094(panel: Panel) -> pd.DataFrame:
    """((rank((vwap - ts_min(vwap, 12))) < rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18))) * -1)"""
    vwap = _vwap(panel)
    left = _cs_rank(vwap - _ts_min(vwap, 12))
    right = _cs_rank(_ts_corr(_ts_rank(vwap, 20), _ts_rank(_adv(panel, 60), 4), 18))
    return _if_less(left, right, -1.0, 0.0)


def alpha095(panel: Panel) -> pd.DataFrame:
    """(rank((open - ts_min(open, 12))) < ts_rank((rank(correlation(sum(((high + low) / 2), 19), sum(adv40, 19), 12)), 12))) ? 1 : -1"""
    left = _cs_rank(panel.open - _ts_min(panel.open, 12))
    right = _ts_rank(
        _cs_rank(
            _ts_corr(_ts_sum((panel.high + panel.low) / 2, 19), _ts_sum(_adv(panel, 40), 19), 12)
        ),
        12,
    )
    return _if_less(left, right, 1.0, -1.0)


def alpha098(panel: Panel) -> pd.DataFrame:
    """rank(decay_linear(correlation(vwap, sum(adv5, 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(open, 31), 3), 3))"""
    vwap = _vwap(panel)
    return _cs_rank(_decay_linear(_ts_corr(vwap, _ts_sum(_adv(panel, 5), 26), 5), 7)) - _cs_rank(
        _decay_linear(_ts_rank(_ts_argmin(panel.open, 31), 3), 3)
    )


def alpha099(panel: Panel) -> pd.DataFrame:
    """((rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)) < rank(correlation(low, volume, 6))) * -1)"""
    left = _cs_rank(
        _ts_corr(_ts_sum((panel.high + panel.low) / 2, 20), _ts_sum(_adv(panel, 60), 20), 9)
    )
    right = _cs_rank(_ts_corr(panel.low, panel.volume, 6))
    return _if_less(left, right, -1.0, 0.0)


def alpha101(panel: Panel) -> pd.DataFrame:
    """(close - open) / ((high - low) + 0.001)"""
    return (panel.close - panel.open) / (panel.high - panel.low + 0.001)


AlphaFunc = Callable[[Panel], pd.DataFrame]

ALPHAS: dict[str, AlphaFunc] = {
    "alpha001": alpha001,
    "alpha002": alpha002,
    "alpha003": alpha003,
    "alpha004": alpha004,
    "alpha005": alpha005,
    "alpha006": alpha006,
    "alpha008": alpha008,
    "alpha009": alpha009,
    "alpha010": alpha010,
    "alpha011": alpha011,
    "alpha012": alpha012,
    "alpha013": alpha013,
    "alpha015": alpha015,
    "alpha016": alpha016,
    "alpha017": alpha017,
    "alpha018": alpha018,
    "alpha019": alpha019,
    "alpha020": alpha020,
    "alpha021": alpha021,
    "alpha022": alpha022,
    "alpha023": alpha023,
    "alpha024": alpha024,
    "alpha025": alpha025,
    "alpha026": alpha026,
    "alpha027": alpha027,
    "alpha028": alpha028,
    "alpha029": alpha029,
    "alpha030": alpha030,
    "alpha031": alpha031,
    "alpha032": alpha032,
    "alpha033": alpha033,
    "alpha034": alpha034,
    "alpha035": alpha035,
    "alpha036": alpha036,
    "alpha037": alpha037,
    "alpha038": alpha038,
    "alpha039": alpha039,
    "alpha040": alpha040,
    "alpha041": alpha041,
    "alpha042": alpha042,
    "alpha043": alpha043,
    "alpha044": alpha044,
    "alpha045": alpha045,
    "alpha046": alpha046,
    "alpha047": alpha047,
    "alpha049": alpha049,
    "alpha050": alpha050,
    "alpha051": alpha051,
    "alpha052": alpha052,
    "alpha053": alpha053,
    "alpha054": alpha054,
    "alpha055": alpha055,
    "alpha060": alpha060,
    "alpha061": alpha061,
    "alpha062": alpha062,
    "alpha064": alpha064,
    "alpha065": alpha065,
    "alpha066": alpha066,
    "alpha068": alpha068,
    "alpha071": alpha071,
    "alpha072": alpha072,
    "alpha073": alpha073,
    "alpha074": alpha074,
    "alpha075": alpha075,
    "alpha077": alpha077,
    "alpha078": alpha078,
    "alpha081": alpha081,
    "alpha083": alpha083,
    "alpha084": alpha084,
    "alpha085": alpha085,
    "alpha086": alpha086,
    "alpha088": alpha088,
    "alpha092": alpha092,
    "alpha094": alpha094,
    "alpha095": alpha095,
    "alpha098": alpha098,
    "alpha099": alpha099,
    "alpha101": alpha101,
}


def compute_alphas(panel: Panel, names: tuple[str, ...] | list[str]) -> dict[str, pd.DataFrame]:
    """按名称批量计算 Alpha101 与 Alpha158，返回 因子名 -> 日期 × 代码 的宽表。"""
    from hkq.alpha158 import ALPHA158_NAMES, compute_alpha158

    wanted = list(names)
    unknown = sorted(set(wanted) - set(ALPHAS) - set(ALPHA158_NAMES))
    if unknown:
        raise KeyError(f"未实现的因子：{unknown}")

    frames = {name: ALPHAS[name](panel) for name in wanted if name in ALPHAS}
    wanted_158 = [name for name in wanted if name in ALPHA158_NAMES]
    if wanted_158:
        frames.update(compute_alpha158(panel, wanted_158))
    return {name: frames[name] for name in wanted}
