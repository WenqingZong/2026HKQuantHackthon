"""Alpha158（Qlib）的标准价量特征。

Qlib 的 Alpha158 由 9 个 K 线形态、4 个价格位置、以及 5/10/20/30/60 五个窗口上的
27 类滚动统计组成。这里按 Qlib 的公式实现，vwap 用 HLC3 代理（日频 OHLCV 没有真实成交额
加权价），这对排序类特征影响很小。

只为 psq 相关性包里出现过的 111 个列名建了注册表——她那份是完整 158 里的一个子集。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from hkq.data import Panel

EPS = 1e-12
WINDOWS = (5, 10, 20, 30, 60)

# psq 相关性包里实际用到的 111 个 Alpha158 列名
ALPHA158_NAMES: tuple[str, ...] = (
    "BETA10",
    "BETA20",
    "BETA30",
    "BETA5",
    "BETA60",
    "CNTN10",
    "CNTP20",
    "CNTN5",
    "CNTP30",
    "CNTP60",
    "CORD10",
    "CORD20",
    "CORD30",
    "CORD5",
    "CORD60",
    "CORR10",
    "CORR20",
    "CORR30",
    "CORR5",
    "CORR60",
    "HIGH0",
    "IMAX10",
    "IMAX20",
    "IMAX30",
    "IMAX5",
    "IMAX60",
    "IMIN10",
    "IMIN20",
    "IMIN30",
    "IMIN5",
    "IMIN60",
    "IMXD10",
    "IMXD20",
    "IMXD30",
    "IMXD5",
    "IMXD60",
    "KLEN",
    "KLOW",
    "KLOW2",
    "KMID",
    "KMID2",
    "KSFT",
    "KSFT2",
    "KUP",
    "KUP2",
    "LOW0",
    "MA10",
    "MA5",
    "QTLD60",
    "MAX10",
    "MAX5",
    "MAX60",
    "MIN10",
    "MIN20",
    "MIN5",
    "MIN60",
    "QTLD10",
    "QTLD30",
    "QTLD5",
    "QTLU10",
    "QTLU20",
    "QTLU5",
    "QTLU60",
    "RANK10",
    "RANK20",
    "RANK5",
    "RANK60",
    "RESI10",
    "RESI20",
    "RESI30",
    "RESI5",
    "RESI60",
    "ROC10",
    "ROC20",
    "ROC30",
    "ROC5",
    "ROC60",
    "RSQR10",
    "RSQR20",
    "RSQR30",
    "RSQR5",
    "RSQR60",
    "RSV10",
    "RSV20",
    "RSV5",
    "STD10",
    "STD20",
    "STD5",
    "STD60",
    "SUMP10",
    "SUMN20",
    "SUMP30",
    "SUMN5",
    "SUMN60",
    "VMA10",
    "VMA30",
    "VMA5",
    "VSTD10",
    "VSTD20",
    "VSTD5",
    "VSTD60",
    "VSUMN10",
    "VSUMD20",
    "VSUMP30",
    "VSUMN5",
    "VSUMN60",
    "WVMA10",
    "WVMA20",
    "WVMA30",
    "WVMA5",
    "WVMA60",
)


def compute_alpha158(panel: Panel, names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """一次算完所有窗口，再按名字取子集（比逐个因子重算快得多）。"""
    wanted = list(names)
    unknown = sorted(set(wanted) - set(ALPHA158_NAMES))
    if unknown:
        raise KeyError(f"未实现的 Alpha158 特征：{unknown}")
    everything = _all_features(panel)
    return {name: everything[name] for name in wanted}


def _all_features(panel: Panel) -> dict[str, pd.DataFrame]:
    close, open_, high, low, volume = panel.close, panel.open, panel.high, panel.low, panel.volume
    vwap = (high + low + close) / 3.0  # HLC3 代理
    eps = EPS

    features: dict[str, pd.DataFrame] = {}

    # --- K 线形态（9）---
    body_high = np.maximum(open_.to_numpy(), close.to_numpy())
    body_low = np.minimum(open_.to_numpy(), close.to_numpy())
    features["KMID"] = (close - open_) / (open_ + eps)
    features["KLEN"] = (high - low) / (open_ + eps)
    features["KMID2"] = (close - open_) / (high - low + eps)
    features["KUP"] = (high - body_high) / (open_ + eps)
    features["KUP2"] = (high - body_high) / (high - low + eps)
    features["KLOW"] = (body_low - low) / (open_ + eps)
    features["KLOW2"] = (body_low - low) / (high - low + eps)
    features["KSFT"] = (2 * close - high - low) / (open_ + eps)
    features["KSFT2"] = (2 * close - high - low) / (high - low + eps)

    # --- 价格位置（4）---
    features["OPEN0"] = open_ / close
    features["HIGH0"] = high / close
    features["LOW0"] = low / close
    features["VWAP0"] = vwap / close

    returns = close / close.shift(1) - 1.0
    change = close.diff()
    volume_change = volume.diff()
    log_volume = cast(pd.DataFrame, np.log(volume + 1.0))

    for window in WINDOWS:
        suffix = str(window)
        rolling_close = close.rolling(window, min_periods=window)

        features["ROC" + suffix] = close / close.shift(window) - 1.0
        features["MA" + suffix] = rolling_close.mean() / (close + eps)
        features["STD" + suffix] = rolling_close.std() / (close + eps)
        features["MAX" + suffix] = high.rolling(window, min_periods=window).max() / (close + eps)
        features["MIN" + suffix] = low.rolling(window, min_periods=window).min() / (close + eps)
        features["QTLU" + suffix] = rolling_close.quantile(0.8) / (close + eps)
        features["QTLD" + suffix] = rolling_close.quantile(0.2) / (close + eps)
        features["RANK" + suffix] = _rolling_rank(close, window)

        low_min = low.rolling(window, min_periods=window).min()
        high_max = high.rolling(window, min_periods=window).max()
        features["RSV" + suffix] = (close - low_min) / (high_max - low_min + eps)

        imax = _rolling_argmax(high, window) / window
        imin = _rolling_argmin(low, window) / window
        features["IMAX" + suffix] = imax
        features["IMIN" + suffix] = imin
        features["IMXD" + suffix] = imax - imin

        beta, rsqr, resi = _rolling_regression(close, window)
        features["BETA" + suffix] = beta / (close + eps)
        features["RSQR" + suffix] = rsqr
        features["RESI" + suffix] = resi / (close + eps)

        features["CORR" + suffix] = close.rolling(window, min_periods=window).corr(log_volume)
        features["CORD" + suffix] = returns.rolling(window, min_periods=window).corr(
            cast(pd.DataFrame, np.log(volume / volume.shift(1) + 1.0))
        )

        # 布尔比较会把缺失变成 False（0），未上市/缺数据的日子必须保持缺失，不能记成「没涨」
        up = (change > 0).astype(float).where(change.notna())
        down = (change < 0).astype(float).where(change.notna())
        up = up.rolling(window, min_periods=window).mean()
        down = down.rolling(window, min_periods=window).mean()
        features["CNTP" + suffix] = up
        features["CNTN" + suffix] = down
        features["CNTD" + suffix] = up - down

        abs_change = change.abs().rolling(window, min_periods=window).sum()
        features["SUMP" + suffix] = change.clip(lower=0).rolling(
            window, min_periods=window
        ).sum() / (abs_change + eps)
        features["SUMN" + suffix] = (-change.clip(upper=0)).rolling(
            window, min_periods=window
        ).sum() / (abs_change + eps)
        features["SUMD" + suffix] = features["SUMP" + suffix] - features["SUMN" + suffix]

        features["VMA" + suffix] = volume.rolling(window, min_periods=window).mean() / (
            volume + eps
        )
        features["VSTD" + suffix] = volume.rolling(window, min_periods=window).std() / (
            volume + eps
        )

        weighted = returns.abs() * volume
        features["WVMA" + suffix] = weighted.rolling(window, min_periods=window).std() / (
            weighted.rolling(window, min_periods=window).mean() + eps
        )

        abs_volume_change = volume_change.abs().rolling(window, min_periods=window).sum()
        features["VSUMP" + suffix] = volume_change.clip(lower=0).rolling(
            window, min_periods=window
        ).sum() / (abs_volume_change + eps)
        features["VSUMN" + suffix] = (-volume_change.clip(upper=0)).rolling(
            window, min_periods=window
        ).sum() / (abs_volume_change + eps)
        features["VSUMD" + suffix] = features["VSUMP" + suffix] - features["VSUMN" + suffix]

    return features


def _windows(frame: pd.DataFrame, window: int) -> np.ndarray:
    """(n - window + 1, 列数, window) 的滑动窗口视图。"""
    values = frame.to_numpy(dtype=float)
    return np.lib.stride_tricks.sliding_window_view(values, window, axis=0)


def _wrap(values: np.ndarray, frame: pd.DataFrame, window: int) -> pd.DataFrame:
    index = frame.index[window - 1 :]
    return pd.DataFrame(values, index=index, columns=frame.columns)


def _rolling_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """窗口内最后一个值的百分位排名。"""
    windows = _windows(frame, window)
    valid = np.isfinite(windows).all(axis=-1)
    ranks = (windows <= windows[..., -1:]).mean(axis=-1)
    return _wrap(np.where(valid, ranks, np.nan), frame, window)


def _rolling_argmax(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    windows = _windows(frame, window)
    valid = np.isfinite(windows).all(axis=-1)
    position = np.argmax(windows, axis=-1) + 1.0
    return _wrap(np.where(valid, position, np.nan), frame, window)


def _rolling_argmin(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    windows = _windows(frame, window)
    valid = np.isfinite(windows).all(axis=-1)
    position = np.argmin(windows, axis=-1) + 1.0
    return _wrap(np.where(valid, position, np.nan), frame, window)


def _rolling_regression(
    frame: pd.DataFrame, window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """滚动线性回归：返回斜率、R²、最后一个点的残差（Qlib 的 BETA / RSQR / RESI）。"""
    windows = _windows(frame, window)
    valid = np.isfinite(windows).all(axis=-1)

    x = np.arange(1.0, window + 1.0)
    x_centered = x - x.mean()
    sxx = float((x_centered**2).sum())

    y_mean = windows.mean(axis=-1)
    slope = (windows * x_centered).sum(axis=-1) / sxx
    residual = windows[..., -1] - (y_mean + slope * x_centered[-1])
    ss_total = ((windows - y_mean[..., None]) ** 2).sum(axis=-1)
    ss_resid = ss_total - slope**2 * sxx
    r2 = np.where(ss_total > 0, 1.0 - ss_resid / np.where(ss_total > 0, ss_total, 1.0), np.nan)

    return (
        _wrap(np.where(valid, slope, np.nan), frame, window),
        _wrap(np.where(valid, r2, np.nan), frame, window),
        _wrap(np.where(valid, residual, np.nan), frame, window),
    )
