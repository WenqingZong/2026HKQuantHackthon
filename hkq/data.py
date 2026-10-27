"""Databento 原始行情：下载、落盘、读取成宽表面板。"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd

from hkq.config import PROJECT_ROOT

PRICE_FIELDS = ("open", "high", "low", "close")
FIELDS = (*PRICE_FIELDS, "volume")
NEW_YORK = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


def api_key() -> str:
    """优先读环境变量 DATABENTO_API_KEY，其次读仓库根目录的 .env。"""
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key.strip()

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "DATABENTO_API_KEY" and value.strip():
                return value.strip().strip("\"'")

    raise RuntimeError(
        "缺少 Databento API key：请设置环境变量 DATABENTO_API_KEY，"
        "或在仓库根目录的 .env 中写入 DATABENTO_API_KEY=..."
    )


def read_tickers(path: Path) -> list[str]:
    """读取 selected_etfs 目录下的 ticker 列表，去重并保持原顺序。"""
    tickers = pd.read_csv(path)["ticker"].dropna().astype(str).str.strip()
    return list(dict.fromkeys(tickers))


def download_bars(
    *,
    dataset: str,
    schema: str,
    symbols: list[str],
    start: str,
    end: str | None,
    output: Path,
) -> pd.DataFrame:
    """从 Databento 下载行情，整理成 (date, symbol) 长表后写入 parquet。"""
    client = db.Historical(api_key())
    if end is None:
        end = str(client.metadata.get_dataset_range(dataset=dataset)["end"])[:10]

    logger.info(f"[download] {dataset}/{schema} | {len(symbols)} symbols | {start} -> {end}")
    records = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=symbols,
        start=start,
        end=end,
        stype_in="raw_symbol",
    )
    with warnings.catch_warnings():
        # databento 内部 to_df() 会触发 pandas 的 set_index(inplace=) 弃用告警
        warnings.filterwarnings("ignore", message="The inplace keyword")
        raw = records.to_df()
    bars = _to_daily_frame(raw, schema=schema)
    degraded = _degraded_days(client, dataset=dataset, start=start, end=end)
    if degraded:
        logger.warning(f"[download] {len(degraded)} day(s) with reduced data quality: {degraded}")

    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(output, index=False)
    _write_manifest(
        output,
        {
            "dataset": dataset,
            "schema": schema,
            "start": start,
            "end": end,
            "n_symbols_requested": len(symbols),
            "n_symbols_returned": int(bars["symbol"].nunique()),
            "n_rows": len(bars),
            "date_range": [str(bars["date"].min().date()), str(bars["date"].max().date())],
            "degraded_days": degraded,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    logger.info(f"[download] wrote {output} ({len(bars):,} rows)")
    return bars


def _degraded_days(client: db.Historical, *, dataset: str, start: str, end: str) -> list[str]:
    """列出数据质量降级（degraded）的交易日，回测时要留意。"""
    conditions = client.metadata.get_dataset_condition(
        dataset=dataset, start_date=start, end_date=end
    )
    return [str(item["date"]) for item in conditions if item.get("condition") != "available"]


def _to_daily_frame(raw: pd.DataFrame, *, schema: str) -> pd.DataFrame:
    """把 Databento 的原始记录转成 date/symbol/OHLCV 长表。"""
    frame = raw.reset_index()
    timestamps = pd.to_datetime(frame["ts_event"], utc=True)
    if schema.startswith("ohlcv-1d"):
        # 日线的时间戳是交易日的 UTC 零点，直接取日期即可
        frame["date"] = timestamps.dt.tz_localize(None).dt.normalize()
    else:
        # 分钟级数据要换成纽约时间，才能把盘后成交归到正确的交易日
        frame["date"] = timestamps.dt.tz_convert(NEW_YORK).dt.tz_localize(None).dt.normalize()

    frame["volume"] = frame["volume"].astype("float64")
    return _consolidate(frame[["date", "symbol", *FIELDS]])


def _consolidate(frame: pd.DataFrame) -> pd.DataFrame:
    """同一交易日若有多条记录（多个交易场所），合并成一根日线。

    合并全市场的数据集（EQUS.SUMMARY）每个代码每天只有一条记录，这里直接返回；
    只覆盖部分场所的数据集（DBEQ.BASIC）会出现多条记录，此时按成交量加权的
    思路合并：开/收盘取成交量最大场所的价格，最高/最低取各场所极值，成交量求和。
    """
    frame = frame.sort_values(["date", "symbol"])
    if not frame.duplicated(["date", "symbol"]).any():
        return frame.reset_index(drop=True)

    ordered = frame.sort_values(["date", "symbol", "volume"], ascending=[True, True, False])
    return (
        ordered.groupby(["date", "symbol"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "first"),
            volume=("volume", "sum"),
        )
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def _write_manifest(output: Path, info: dict) -> None:
    """记录下载参数，方便复现和排查数据问题。"""
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Panel:
    """宽表面板：每个字段是 日期 × 代码 的 DataFrame。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> Panel:
        if bars.duplicated(["date", "symbol"]).any():
            raise ValueError("行情里存在重复的 (date, symbol)，请先调用 _consolidate")

        frames = {
            field: bars.pivot(index="date", columns="symbol", values=field).sort_index()
            for field in FIELDS
        }
        # 成交量用浮点，避免无符号整数做差分时溢出
        frames["volume"] = frames["volume"].astype("float64")
        return cls(**frames)

    def field(self, name: str) -> pd.DataFrame:
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise KeyError(f"未知字段 {name!r}，可选：{FIELDS}") from error

    @property
    def dates(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.close.index)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def load_panel(path: Path) -> Panel:
    return Panel.from_bars(pd.read_parquet(path))
