"""分红与拆股的复权。

Databento 给的是未经复权的原始成交价：除息日价格会自然下跌，拆股日价格按比例缩水。
不处理的话，标签会把分红当成亏损，拆股更会变成一根 -67% 的假 K 线——样本期内 SCHD、
SCHP、XLE/XLK/XLB/XLU/XLY、PALL、PPLT、ZCSH、ETHA 都拆过股。

复权因子 f 的定义是「复权价 = 原始价 × f」，事件日满足::

    f_t / f_{t-1} = 拆股比例_t + 每股分红_t / 原始收盘价_t

于是 adj_t / adj_{t-1} - 1 = (close_t × 比例 + 分红) / close_{t-1} - 1，正好是含分红的
真实收益。

**这里有个容易踩的坑**：把累积因子再除以最后一天（教科书式的「后复权」）会让 t 日的
价格水平依赖 t 之后发生的所有分红和拆股——那是未来函数。样本期内 2024-08-01 就有 24 只
ETF 的「未来缩放」偏离超过 10%，极端如 PPLT（10:1，0.10）、ETHA（1:3 反向，3.00）。
收益率不受影响（常数在比值里抵消），但用价格差构造的因子会被污染。

所以本模块只提供**因果**因子（只用当日及之前的事件），并且不做事后归一化：

- `adjust_panel` / `cumulative_price_factor`：给收益率、标签用，全程只用过去的事件；
- `point_in_time_panels`：给因子用，逐日把最近若干天缩放到「以当日为止」的价格单位，
  和当天的交易所见即所得，不含任何未来信息。
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
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


def cumulative_price_factor(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """因果累积复权因子：f_t 只用 t 及之前的事件，不做任何事后归一化。"""
    dividends = _event_matrix(actions, "dividend", close, fill=0.0)
    splits = _event_matrix(actions, "split_ratio", close, fill=1.0)
    return (splits + dividends / close).fillna(1.0).cumprod()


def cumulative_volume_factor(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """成交量换算成「期初股数」口径：拆股后 300 新股东等价于 100 老股。"""
    splits = _event_matrix(actions, "split_ratio", close, fill=1.0)
    return 1.0 / splits.cumprod()


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
    """因果复权：收益率和标签用这一版。

    不做「除以最后一天」的后复权——那会让历史价格依赖未来的分红/拆股。代价是价格水平
    停在各标的自身的累计口径上，但对收益率没有影响。
    """
    price_factor = cumulative_price_factor(panel.close, actions)
    volume_factor = cumulative_volume_factor(panel.close, actions)

    adjusted = Panel(
        open=panel.open * price_factor,
        high=panel.high * price_factor,
        low=panel.low * price_factor,
        close=panel.close * price_factor,
        volume=panel.volume * volume_factor,
    )
    _warn_suspicious_moves(adjusted)
    return adjusted


def point_in_time_panels(
    panel: Panel, actions: pd.DataFrame, *, window: int
) -> Iterator[tuple[pd.Timestamp, Panel]]:
    """逐日产出「以该日为止」的复权窗口，因子用这一版。

    每一天都把窗口缩放到当日的价格单位：只用当日及之前的事件，未来怎么复权都不会影响
    今天算出来的因子。window 必须不小于因子里最长的回看天数，否则窗口头部的因子算不出来。
    """
    if window < 2:
        raise ValueError("window 至少要有 2 个交易日")

    price_factor = cumulative_price_factor(panel.close, actions)
    volume_factor = cumulative_volume_factor(panel.close, actions)

    for position in range(window - 1, len(panel.dates)):
        rows = slice(position - window + 1, position + 1)
        price_scale = price_factor.iloc[rows].div(price_factor.iloc[position])
        volume_scale = volume_factor.iloc[rows].div(volume_factor.iloc[position])
        yield (
            panel.dates[position],
            Panel(
                open=panel.open.iloc[rows] * price_scale,
                high=panel.high.iloc[rows] * price_scale,
                low=panel.low.iloc[rows] * price_scale,
                close=panel.close.iloc[rows] * price_scale,
                volume=panel.volume.iloc[rows] * volume_scale,
            ),
        )


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
