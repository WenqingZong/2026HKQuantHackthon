"""用各交易场所的 Databento 原始数据拼出 2018 年起的长历史日线。

合并全市场的 EQUS.SUMMARY 只从 2024-07 开始，EQUS.MINI 从 2023-03 开始；单个交易所的
数据集（NYSE Arca、Nasdaq、Cboe BZX/EDGX、NYSE）从 2018-05 就有。代价是没有合并行情，
每个字段要自己定口径。下面的规则是对着 EQUS.SUMMARY 在 2024-07 之后的重叠区间逐项验证过的：

- **开盘 / 收盘**：上市交易所的官方开盘、收盘竞价价（statistics 里 stat_type 1 / 11）。
  一只 ETF 可能有好几个交易所发布「收盘价」，取竞价成交量最大的那一家，也就是上市交易所。
  直接取「成交量最大场所」日线的最后一笔会落在连续竞价的某个价格上，
  一半的样本和官方收盘差 5bp 以上。
- **最高 / 最低**：各场所 1 小时 K 线里纽约时间 09:00–15:59 的极值。日线里含盘前盘后，
  零星的盘外成交会把极值拉出 1% 以上（约 8% 的样本）；09:00 那根含了半小时盘前，
  是能拿到的最近似口径。
- **成交量**：各场所日线成交量求和。只覆盖交易所的显性成交（约为合并成交量的 40%，
  比例逐年稳定），比率类因子不受整体缩放影响。
- 竞价价缺失时（约 3%，多在 2018–2021 年）：开盘取 09:00 那根小时线、收盘取 15:00 那根小时线里
  成交量最大场所的开/收价（和官方价差 1% 以上的比例分别约 1.7% / 0.07%）；日线的开收盘含盘前盘后，
  差得多（约 9.8% / 4.4%），只在小时线也缺的时候才用。各环节的条数写进 manifest。

ticker 会被重复使用（IBIT 在 2024-01 现货比特币 ETF 上市前是另一只证券），也有产品中途换了
策略，所以还要按 selected_etfs 里的 history_start_date / inception_date 裁掉之前的行情。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import databento as db
import pandas as pd

from hkq.data import FIELDS, NEW_YORK, api_key

logger = logging.getLogger(__name__)

DAILY = "ohlcv-1d"
HOURLY = "ohlcv-1h"
STATISTICS = "statistics"
# databento StatType：官方开盘价 / 官方收盘价
OPENING_PRICE = 1
CLOSING_PRICE = 11
# 1h K 线的起始时刻（纽约时间）落在这个区间内才算常规交易时段
SESSION_HOURS = range(9, 16)

KEY = ["date", "symbol"]


def venue_file(directory: Path, venue: str, schema: str) -> Path:
    """每个场所每种 schema 一个原始文件，日线不带后缀（和已有的下载目录保持一致）。"""
    return directory / (f"{venue}.parquet" if schema == DAILY else f"{venue}.{schema}.parquet")


def download_venue_files(
    *,
    venues: tuple[str, ...],
    auction_venues: tuple[str, ...],
    symbols: list[str],
    start: str,
    end: str | None,
    directory: Path,
) -> None:
    """把每个场所需要的原始数据下载到 directory；文件已存在就跳过（删掉文件即可重新拉）。"""
    jobs = [(venue, schema) for venue in venues for schema in (DAILY, HOURLY)]
    jobs += [(venue, STATISTICS) for venue in auction_venues]
    missing = [job for job in jobs if not venue_file(directory, *job).exists()]
    logger.info(f"[venues] {len(jobs) - len(missing)} file(s) cached, {len(missing)} to download")
    if not missing:
        return

    client = db.Historical(api_key())
    directory.mkdir(parents=True, exist_ok=True)
    for venue, schema in missing:
        path = venue_file(directory, venue, schema)
        finish = end or str(client.metadata.get_dataset_range(dataset=venue)["end"])[:10]
        logger.info(
            f"[venues] download {venue}/{schema} | {len(symbols)} symbols | {start} -> {finish}"
        )
        records = client.timeseries.get_range(
            dataset=venue,
            schema=schema,
            symbols=symbols,
            start=start,
            end=finish,
            stype_in="raw_symbol",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The inplace keyword")
            raw = records.to_df().reset_index()
        if schema == STATISTICS:
            raw = raw[raw["stat_type"].isin([OPENING_PRICE, CLOSING_PRICE])]
        raw.to_parquet(path, index=False)
        logger.info(f"[venues] wrote {path} ({len(raw):,} rows)")


def build_venue_bars(
    directory: Path,
    *,
    venues: tuple[str, ...],
    auction_venues: tuple[str, ...],
    history_start: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """按模块说明里的规则把各场所原始数据合成一根日线，返回 (bars, 各环节的计数)。"""
    daily = pd.concat(
        [_read_daily(venue_file(directory, venue, DAILY)) for venue in venues], ignore_index=True
    )
    hourly = pd.concat(
        [_read_hourly(venue_file(directory, venue, HOURLY)) for venue in venues], ignore_index=True
    )
    auctions = pd.concat(
        [_read_auctions(venue_file(directory, venue, STATISTICS)) for venue in auction_venues],
        ignore_index=True,
    )
    return combine_venue_bars(daily, hourly, auctions, history_start=history_start)


def combine_venue_bars(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    auctions: pd.DataFrame,
    *,
    history_start: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """合成规则本身（不碰文件，方便单测）。

    daily：date, symbol, open, high, low, close, volume（每个场所一行）
    hourly：date, symbol, hour, open, high, low, close, volume（hour 为纽约时间，每个场所一行）
    auctions：date, symbol, stat_type, price, quantity（每个场所一行）
    """
    # 竞价价缺失时的兜底：开盘 09:00 小时线、收盘 15:00 小时线，再不行才用日线
    fallbacks = {
        "open": (
            _most_active(hourly[hourly["hour"] == SESSION_HOURS[0]])["open"],
            _most_active(daily)["open"],
        ),
        "close": (
            _most_active(hourly[hourly["hour"] == SESSION_HOURS[-1]])["close"],
            _most_active(daily)["close"],
        ),
    }
    bars = daily.groupby(KEY).agg(volume=("volume", "sum"))

    session = hourly[hourly["hour"].isin(SESSION_HOURS)].groupby(KEY)
    bars["high"] = session["high"].max()
    bars["low"] = session["low"].min()

    for stat_type, field in ((OPENING_PRICE, "open"), (CLOSING_PRICE, "close")):
        chosen = (
            auctions[auctions["stat_type"] == stat_type]
            .sort_values([*KEY, "quantity"], ascending=[True, True, False])
            .drop_duplicates(KEY)
            .set_index(KEY)["price"]
        )
        bars[field] = chosen

    counts = {
        "open_fallback": int(bars["open"].isna().sum()),
        "close_fallback": int(bars["close"].isna().sum()),
        "high_low_fallback": int((bars["high"].isna() | bars["low"].isna()).sum()),
    }
    for field, (hour_bar, day_bar) in fallbacks.items():
        bars[field] = bars[field].fillna(hour_bar).fillna(day_bar)
    bars["high"] = bars["high"].fillna(daily.groupby(KEY)["high"].max())
    bars["low"] = bars["low"].fillna(daily.groupby(KEY)["low"].min())
    # 小时线和竞价价来自不同的数据流，极值至少要包住开收盘
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)

    bars = bars.reset_index()
    if history_start is not None:
        starts = bars["symbol"].map(history_start)
        before = starts.notna() & (bars["date"] < starts)
        counts["rows_before_history_start"] = int(before.sum())
        bars = bars[~before]

    counts["rows"] = len(bars)
    return bars[[*KEY, *FIELDS]].sort_values(KEY).reset_index(drop=True), counts


def _most_active(frame: pd.DataFrame) -> pd.DataFrame:
    """同一 (date, symbol) 有多个交易所时，取成交量最大的那一行。"""
    return (
        frame.sort_values([*KEY, "volume"], ascending=[True, True, False])
        .drop_duplicates(KEY)
        .set_index(KEY)
    )


def history_start_dates(training_universe: Path, trading_universe: Path) -> pd.Series:
    """每只 ETF 可用行情的起点：训练池表的 history_start_date 与交易池表的 inception_date 取较晚者。

    history_start_date 已经考虑了换策略（VNM 2023-03-17）和换代码（DTCR），
    inception_date 则拦住了 ticker 被重复使用的情况（IBIT 2024-01 之前是另一只证券）。
    """
    training = pd.read_csv(training_universe, encoding="utf-8-sig").set_index("ticker")
    trading = pd.read_csv(trading_universe, encoding="utf-8-sig").set_index("ticker")
    starts = pd.concat(
        [
            pd.to_datetime(training["history_start_date"], errors="coerce"),
            pd.to_datetime(trading["inception_date"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    return starts.dropna().groupby(level=0).max()


def _read_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["ts_event", "symbol", *FIELDS])
    # 日线时间戳是交易日的 UTC 零点
    frame["date"] = pd.to_datetime(frame["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
    frame["volume"] = frame["volume"].astype("float64")
    return frame[[*KEY, *FIELDS]]


def _read_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["ts_event", "symbol", *FIELDS])
    local = pd.to_datetime(frame["ts_event"], utc=True).dt.tz_convert(NEW_YORK)
    frame["date"] = local.dt.tz_localize(None).dt.normalize()
    frame["hour"] = local.dt.hour
    frame["volume"] = frame["volume"].astype("float64")
    return frame[[*KEY, "hour", *FIELDS]]


def _read_auctions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["ts_event", "symbol", "stat_type", "price", "quantity"])
    # 没成交出竞价价的记录 price 是缺失的，留着会在「取竞价量最大」时被选中
    frame = frame[frame["stat_type"].isin([OPENING_PRICE, CLOSING_PRICE]) & (frame["price"] > 0)]
    local = pd.to_datetime(frame["ts_event"], utc=True).dt.tz_convert(NEW_YORK)
    frame = frame.assign(date=local.dt.tz_localize(None).dt.normalize())
    # 同一场所同一天若有更正，保留最后一条
    frame = frame.sort_values("ts_event").drop_duplicates([*KEY, "stat_type"], keep="last")
    return frame[[*KEY, "stat_type", "price", "quantity"]]
