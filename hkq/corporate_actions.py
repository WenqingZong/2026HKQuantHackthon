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
- 因子和标签都从 `adjust_panel` 的 causal 序列上算，不含未来事件。

代价是价格水平停在「各标的自身的累计口径」上：拆过股的标的，历史价格和没拆过的相比
差一个固定的倍数。对收益率、比率类因子没有影响；对用价格差构造的因子，这是跨标的的
口径差异（不是未来信息），只涉及样本期内拆过股的十来只 ETF。
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from hkq.data import Panel

logger = logging.getLogger(__name__)

# 复权后还能看到的单日涨跌幅（超过这个数说明有未被识别的事件，值得人看一眼）
SUSPICIOUS_DAILY_MOVE = 0.35
# yfinance 会限流：分批请求，失败就退避重试，别一次性把 182 个代码怼上去
CHUNK_SIZE = 40
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 30.0


def download_actions(
    *, symbols: list[str], start: str, end: str | None, output: Path
) -> pd.DataFrame:
    """从 yfinance 拉分红与拆股，落成 date/symbol/dividend/split_ratio 长表。

    end 为 None 时取当天；yfinance 的 end 是开区间，和 Databento 的习惯一致。
    """
    finish = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    logger.info(f"[actions] yfinance {len(symbols)} symbols | {start} -> {finish}")

    blocks: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[offset : offset + CHUNK_SIZE]
        blocks.update(_fetch_actions(chunk, start=start, end=finish))
        time.sleep(2.0)

    missing = sorted(set(symbols) - set(blocks))
    if missing:
        logger.warning(f"[actions] yfinance returned no data for {len(missing)} symbols: {missing}")
    if not blocks:
        raise RuntimeError(
            "yfinance 一个代码都没返回（多半是被限流了）；等几分钟再跑，"
            "不要把空表写盘覆盖掉已有的分红/拆股数据"
        )

    frames = [_actions_of(block, symbol) for symbol, block in blocks.items()]
    actions = pd.concat(frames, ignore_index=True)
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


def _fetch_actions(symbols: list[str], *, start: str, end: str) -> dict[str, pd.DataFrame]:
    """分批拉，遇到限流就退避重试；返回真正拿到数据的那些代码。"""
    pending = list(symbols)
    blocks: dict[str, pd.DataFrame] = {}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        if attempt > 1:
            wait = BACKOFF_SECONDS * (attempt - 1)
            logger.warning(
                f"[actions] retry {attempt}/{MAX_ATTEMPTS} for {len(pending)} symbols "
                f"in {wait:.0f}s"
            )
            time.sleep(wait)

        with warnings.catch_warnings():
            # yfinance 内部还在用 pandas 已弃用的 inplace 参数
            warnings.filterwarnings("ignore", message="The inplace keyword")
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = yf.download(
                pending,
                start=start,
                end=end,
                actions=True,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
            )

        if raw.empty:
            continue
        returned = set(raw.columns.get_level_values(0))
        for symbol in pending:
            if symbol in returned:
                blocks[symbol] = raw[symbol]
        pending = [symbol for symbol in pending if symbol not in blocks]

    return blocks


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


def splits_from_price_jumps(
    panel: Panel, *, threshold: float = 0.20, tolerance: float = 0.06
) -> pd.DataFrame:
    """yfinance 不可用时的兜底：从价格跳空反推拆股，分红补不了（只能记 0）。

    判据是「跳空幅度非常接近某个整数倍」，所以真实的暴涨暴跌（MSOS 的 ±30%）不会被误判成拆股。
    这条路径只保证不出现「3 倍假暴跌」，除息日的价格下跌仍然是未复权的。
    """
    rows: list[dict[str, object]] = []
    returns = panel.close.pct_change()
    for symbol in panel.symbols:
        for date, ret in returns[symbol].items():
            if not np.isfinite(ret) or abs(ret) < threshold:
                continue
            implied = 1.0 / (1.0 + ret)
            ratio = np.nan
            for candidate in range(2, 21):
                if abs(implied / candidate - 1) < tolerance:
                    ratio = float(candidate)
                    break
            if not np.isfinite(ratio):
                grown = 1.0 + ret
                for candidate in range(2, 21):
                    if abs(grown / candidate - 1) < tolerance:
                        ratio = 1.0 / candidate
                        break
            if np.isfinite(ratio):
                rows.append({"date": date, "symbol": symbol, "dividend": 0.0, "split_ratio": ratio})
    frame = pd.DataFrame(rows, columns=["date", "symbol", "dividend", "split_ratio"])
    logger.warning(
        "[actions] yfinance 不可用：从价格跳空反推出 %d 个拆股事件，分红未复权", len(frame)
    )
    return frame


def load_actions(path: Path) -> pd.DataFrame:
    """读取已经下载好的分红/拆股表；缺失时给出可执行的提示。

    文件里存的是 yfinance 原样返回的数值，分红在这里换算回当时每股实际派的金额。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，先跑一次 main.py download 把分红与拆股记录拉下来")
    actions = pd.read_parquet(path)
    actions["date"] = pd.to_datetime(actions["date"])
    return raw_dividends(actions)


def raw_dividends(actions: pd.DataFrame) -> pd.DataFrame:
    """把 yfinance 的分红还原成除息日当时每股的实际金额。

    yfinance 的分红按**之后**发生的拆股追溯调整过：SCHD 2024-10 拆 3:1 之前每季派约 0.75 美元，
    yfinance 给的是 0.25。复权因子用的是「分红 / 当日原始收盘价」，两者口径不一致会把拆股前的
    股息率低估成 1/3。这里乘回除息日之后所有拆股比例的乘积（反向拆股的比例 < 1，同样适用）。

    这一步用到了「未来的拆股」，但只是还原当时已经公开的派息金额，不引入任何未来信息。
    """
    splits = actions.loc[
        actions["split_ratio"].fillna(1.0) != 1.0, ["symbol", "date", "split_ratio"]
    ]
    multiplier = pd.Series(1.0, index=actions.index)
    for symbol, date, ratio in splits.itertuples(index=False, name=None):
        earlier = (actions["symbol"] == symbol) & (actions["date"] < date)
        multiplier = multiplier.where(~earlier, multiplier * float(ratio))
    return actions.assign(dividend=actions["dividend"] * multiplier)


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
