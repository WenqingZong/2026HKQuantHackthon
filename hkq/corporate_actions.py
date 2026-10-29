"""分红与拆股的复权：把 Databento 的原始价还原成含分红再投资的总收益序列。

Databento 给的是未经复权的原始成交价：除息日价格会自然下跌，拆股日价格会按比例缩水。
不处理的话，标签会把分红当成亏损（高股息 ETF 一次除息就有 1% 上下），拆股更会变成
一根 -67% 的假 K 线——样本期内 SCHD、SCHP、XLE/XLK/XLB/XLU/XLY、PALL、PPLT、ZCSH、
ETHA 都拆过股。

复权因子 f 的定义是「复权价 = 原始价 × f」，事件日满足::

    f_t / f_{t-1} = 拆股比例_t + 每股分红_t / 原始收盘价_t

于是 adj_t / adj_{t-1} - 1 = (close_t × 比例 + 分红) / close_{t-1} - 1，正好是含分红的
真实收益。因子最后按最新一日归一化，让当前价格保持原始水平（对收益没有影响，只是好读）。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from hkq.data import Panel

logger = logging.getLogger(__name__)

# 复权后还能看到的单日涨跌幅（超过这个数说明有未被识别的事件，值得人看一眼）
SUSPICIOUS_DAILY_MOVE = 0.35


def download_actions(
    *, symbols: list[str], start: str, end: str | None, output: Path
) -> pd.DataFrame:
    """从 yfinance 拉分红与拆股，落成 date/symbol/dividend/split_ratio 长表。

    end 为 None 时取当天；yfinance 的 end 是开区间，和 Databento 的习惯一致。
    """
    finish = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    logger.info(f"[actions] yfinance {len(symbols)} symbols | {start} -> {finish}")

    with warnings.catch_warnings():
        # yfinance 内部还在用 pandas 已弃用的 inplace 参数
        warnings.filterwarnings("ignore", message="The inplace keyword")
        warnings.filterwarnings("ignore", category=FutureWarning)
        raw = yf.download(
            symbols,
            start=start,
            end=finish,
            actions=True,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )

    returned = set(raw.columns.get_level_values(0))
    missing = sorted(set(symbols) - returned)
    if missing:
        logger.warning(f"[actions] yfinance returned no data for {missing}")

    frames = [_actions_of(raw[symbol], symbol) for symbol in symbols if symbol in returned]
    actions = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["date", "symbol", "dividend", "split_ratio"])
    )
    actions = actions.sort_values(["date", "symbol"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    actions.to_parquet(output, index=False)
    logger.info(
        f"[actions] rows={len(actions)} dividends={int((actions['dividend'] > 0).sum())} "
        f"splits={int((actions['split_ratio'] != 1.0).sum())} -> {output}"
    )
    splits = actions.loc[actions["split_ratio"] != 1.0, ["symbol", "date", "split_ratio"]]
    for record in splits.to_dict("records"):
        logger.info(
            f"[actions] split {record['symbol']} {record['date'].date()} "
            f"ratio={float(record['split_ratio']):.4g}"
        )
    return actions


def _actions_of(block: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把单个代码的下载结果压成只有「有事件的那几天」的长表。"""
    zero = pd.Series(0.0, index=block.index)
    dividends = block["Dividends"].fillna(0.0) if "Dividends" in block else zero
    capital_gains = block["Capital Gains"].fillna(0.0) if "Capital Gains" in block else zero
    splits = block["Stock Splits"].fillna(0.0) if "Stock Splits" in block else zero

    frame = pd.DataFrame(
        {
            # 资本利得分配和分红一样是「钱发出去、价格掉下来」，合并进分红项
            "dividend": dividends + capital_gains,
            "split_ratio": splits.where(splits > 0, 1.0),
        }
    )
    frame = frame[(frame["dividend"] > 0) | (frame["split_ratio"] != 1.0)]
    frame = frame.assign(symbol=symbol)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame.reset_index(names="date")


def load_actions(path: Path) -> pd.DataFrame:
    """读取已经下载好的分红/拆股表；缺失时给出可执行的提示。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，先跑一次 main.py download 把分红与拆股记录拉下来")
    actions = pd.read_parquet(path)
    actions["date"] = pd.to_datetime(actions["date"])
    return actions


def adjustment_factors(
    close: pd.DataFrame, actions: pd.DataFrame, *, include_dividends: bool = True
) -> pd.DataFrame:
    """按日期的复权因子，最后一天归一化为 1。"""
    dividends = _event_matrix(actions, "dividend", close, fill=0.0)
    splits = _event_matrix(actions, "split_ratio", close, fill=1.0)

    step = splits if not include_dividends else splits + dividends / close
    factor = step.fillna(1.0).cumprod()
    return factor / factor.iloc[-1]


def _event_matrix(
    actions: pd.DataFrame, column: str, close: pd.DataFrame, *, fill: float
) -> pd.DataFrame:
    """把长表事件摊成 日期 × 代码 的矩阵，没事件的位置填 fill。"""
    events = actions[actions[column].fillna(fill) != fill]
    if events.empty:
        return pd.DataFrame(fill, index=close.index, columns=close.columns)

    matrix = events.pivot_table(index="date", columns="symbol", values=column, aggfunc="sum")
    matrix = matrix.reindex(index=close.index, columns=close.columns)
    dropped = int(events.shape[0] - int(matrix.notna().to_numpy().sum()))
    if dropped:
        logger.warning(f"[actions] {dropped} {column} event(s) fall outside the price calendar")
    return matrix.fillna(fill)


def adjust_panel(panel: Panel, actions: pd.DataFrame) -> Panel:
    """复权：价格乘总收益因子，成交量乘拆股因子（拆股后股数口径变了）。"""
    price_factor = adjustment_factors(panel.close, actions)
    # 拆股后股数口径变了，成交量要按价格因子的倒数调整，这样「量 × 价」保持不变
    volume_factor = 1.0 / adjustment_factors(panel.close, actions, include_dividends=False)

    adjusted = Panel(
        open=panel.open * price_factor,
        high=panel.high * price_factor,
        low=panel.low * price_factor,
        close=panel.close * price_factor,
        volume=panel.volume * volume_factor,
    )
    _warn_suspicious_moves(adjusted)
    return adjusted


def _warn_suspicious_moves(panel: Panel, threshold: float = SUSPICIOUS_DAILY_MOVE) -> None:
    """复权后仍有巨大单日跳空，说明 yfinance 漏了事件或者数据本身有问题。"""
    returns = panel.close.pct_change()
    jumps = returns.where(returns.abs() > threshold).stack().dropna()
    if jumps.empty:
        return
    examples = ", ".join(f"{date.date()}:{symbol}" for date, symbol in jumps.index[:5])
    logger.warning(
        f"[actions] {len(jumps)} daily move(s) still exceed {threshold:.0%} after adjustment "
        f"(possible missing dividend/split): {examples}"
    )
