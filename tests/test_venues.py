"""多交易所拼日线的口径：开收盘用官方竞价价，极值只看常规时段，ticker 复用之前的行情要裁掉。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hkq import venues
from hkq.data import NEW_YORK
from hkq.venues import (
    CLOSING_PRICE,
    DAILY,
    HOURLY,
    OPENING_PRICE,
    STATISTICS,
    build_venue_bars,
    combine_venue_bars,
    download_venue_files,
    history_start_dates,
    venue_file,
)

DAY = pd.Timestamp("2024-01-02")


def daily_rows(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    """(symbol, open, high, low, close, volume)，每行代表一个交易所的日线。"""
    frame = pd.DataFrame(rows, columns=["symbol", "open", "high", "low", "close", "volume"])
    return frame.assign(date=DAY)


def hourly_rows(
    rows: list[tuple[int, float, float, float, float]], volume: float = 1.0
) -> pd.DataFrame:
    """(hour, open, high, low, close)，纽约时间整点开始的小时线。"""
    frame = pd.DataFrame(rows, columns=["hour", "open", "high", "low", "close"])
    return frame.assign(date=DAY, symbol="AAA", volume=volume)


def no_auctions() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "symbol", "stat_type", "price", "quantity"])


def test_auction_prices_come_from_the_largest_auction() -> None:
    daily = daily_rows(
        [("AAA", 10.0, 11.0, 9.0, 10.4, 500.0), ("AAA", 10.1, 10.9, 9.1, 10.3, 100.0)]
    )
    hourly = hourly_rows(
        [(9, 10.0, 10.6, 9.8, 10.1), (12, 10.1, 10.8, 9.9, 10.2), (15, 10.2, 10.7, 9.7, 10.4)]
    )
    auctions = pd.DataFrame(
        {
            "date": DAY,
            "symbol": "AAA",
            "stat_type": [OPENING_PRICE, OPENING_PRICE, CLOSING_PRICE, CLOSING_PRICE],
            "price": [10.05, 10.02, 10.50, 10.45],
            "quantity": [9000, 50, 20000, 300],
        }
    )

    bars, counts = combine_venue_bars(daily, hourly, auctions)
    row = bars.iloc[0]

    assert (row["open"], row["close"]) == (10.05, 10.50)
    assert (row["high"], row["low"]) == (10.8, 9.7)
    assert row["volume"] == 600.0  # 各交易所成交量相加
    assert counts["open_fallback"] == counts["close_fallback"] == 0


def test_extended_hours_do_not_set_the_range() -> None:
    daily = daily_rows([("AAA", 10.0, 13.0, 7.0, 10.0, 100.0)])
    hourly = hourly_rows(
        [(4, 10.0, 13.0, 7.0, 10.0), (10, 10.0, 10.2, 9.9, 10.0), (17, 10.0, 12.0, 8.0, 10.0)]
    )
    auctions = pd.DataFrame(
        {
            "date": DAY,
            "symbol": "AAA",
            "stat_type": [OPENING_PRICE, CLOSING_PRICE],
            "price": [10.0, 10.1],
            "quantity": [1, 1],
        }
    )

    bars, _ = combine_venue_bars(daily, hourly, auctions)

    assert (bars.loc[0, "high"], bars.loc[0, "low"]) == (10.2, 9.9)


def test_missing_auction_falls_back_to_session_hour_bars() -> None:
    # 日线的开收盘含盘前盘后（9.0 / 9.1），兜底要用 09:00 的开、15:00 的收
    daily = daily_rows([("AAA", 9.0, 10.5, 8.8, 9.1, 300.0)])
    hourly = hourly_rows([(9, 10.0, 10.2, 9.9, 10.1), (15, 10.1, 10.3, 10.0, 10.25)])

    bars, counts = combine_venue_bars(daily, hourly, no_auctions())
    row = bars.iloc[0]

    assert (row["open"], row["close"]) == (10.0, 10.25)
    assert counts["open_fallback"] == counts["close_fallback"] == 1


def test_daily_bar_is_last_resort_and_range_covers_open_close() -> None:
    daily = daily_rows([("AAA", 10.0, 10.5, 9.5, 10.3, 300.0), ("AAA", 9.0, 9.2, 8.8, 9.1, 10.0)])
    hourly = hourly_rows([(11, 10.15, 10.2, 10.1, 10.15)])

    bars, _ = combine_venue_bars(daily, hourly, no_auctions())
    row = bars.iloc[0]

    # 没有竞价价、也没有 09:00 / 15:00 小时线：用成交量最大那家交易所的日线开收盘
    assert (row["open"], row["close"]) == (10.0, 10.3)
    assert (row["high"], row["low"]) == (10.3, 10.0)


def test_rows_before_history_start_are_dropped() -> None:
    days = pd.to_datetime(["2023-06-01", "2024-01-11"])
    daily = pd.DataFrame(
        {
            "date": days,
            "symbol": "IBIT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    hourly = pd.DataFrame(
        columns=["date", "symbol", "hour", "open", "high", "low", "close", "volume"]
    )

    bars, counts = combine_venue_bars(
        daily, hourly, no_auctions(), history_start=pd.Series({"IBIT": pd.Timestamp("2024-01-11")})
    )

    assert bars["date"].tolist() == [pd.Timestamp("2024-01-11")]
    assert counts["rows_before_history_start"] == 1


def test_history_start_takes_the_later_of_both_tables(tmp_path: Path) -> None:
    training = tmp_path / "training.csv"
    trading = tmp_path / "trading.csv"
    training.write_text(
        "﻿ticker,history_start_date\nVNM,2023-03-17\nSPY,1993-01-22\n", encoding="utf-8"
    )
    trading.write_text("﻿ticker,inception_date\nVNM,2009-08-14\nIBIT,2024-01-05\n", encoding="utf-8")

    starts = history_start_dates(training, trading)

    assert starts.to_dict() == {
        "IBIT": pd.Timestamp("2024-01-05"),
        "SPY": pd.Timestamp("1993-01-22"),
        "VNM": pd.Timestamp("2023-03-17"),
    }


def test_default_config_points_at_venue_history() -> None:
    from hkq.config import Config

    cfg = Config.load()

    assert set(cfg.data.auction_venues) <= set(cfg.data.venues)
    assert cfg.universe.training_universe is not None


def write_raw_venue(directory: Path, venue: str, *, close: float, auction: bool) -> None:
    """按 databento to_df() 的列名写一份最小的原始文件（日线、小时线、statistics）。"""
    utc_day = pd.Timestamp("2024-01-02", tz="UTC")
    bar = {"symbol": "AAA", "open": close, "high": close + 1, "low": close - 1, "close": close}
    pd.DataFrame([{"ts_event": utc_day, **bar, "volume": 100}]).to_parquet(
        venue_file(directory, venue, DAILY)
    )
    hours = pd.date_range("2024-01-02 09:00", periods=7, freq="h", tz=NEW_YORK)
    pd.DataFrame(
        [{"ts_event": t.tz_convert("UTC"), **bar, "volume": 10} for t in hours]
    ).to_parquet(venue_file(directory, venue, HOURLY))
    if auction:
        stamps = [
            pd.Timestamp("2024-01-02 09:30", tz=NEW_YORK),
            pd.Timestamp("2024-01-02 16:00", tz=NEW_YORK),
        ]
        pd.DataFrame(
            {
                "ts_event": [t.tz_convert("UTC") for t in stamps] * 2,
                "symbol": "AAA",
                "stat_type": [OPENING_PRICE, CLOSING_PRICE] * 2,
                # 第二组是没有成交出竞价价的记录：量更大但价格缺失，不能被选中
                "price": [close, close + 0.5, float("nan"), float("nan")],
                "quantity": [10, 10, 999, 999],
            }
        ).to_parquet(venue_file(directory, venue, STATISTICS))


def test_build_venue_bars_reads_raw_files(tmp_path: Path) -> None:
    write_raw_venue(tmp_path, "ARCX.PILLAR", close=10.0, auction=True)
    write_raw_venue(tmp_path, "EDGX.PITCH", close=11.0, auction=False)

    bars, counts = build_venue_bars(
        tmp_path, venues=("ARCX.PILLAR", "EDGX.PITCH"), auction_venues=("ARCX.PILLAR",)
    )

    row = bars.iloc[0]
    assert row["date"] == DAY
    assert (row["open"], row["close"], row["volume"]) == (10.0, 10.5, 200.0)
    assert counts["close_fallback"] == 0


def test_cached_venue_files_are_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_raw_venue(tmp_path, "ARCX.PILLAR", close=10.0, auction=True)

    def no_network(*_: object, **__: object) -> None:
        raise AssertionError("文件都在，不应该再连 Databento")

    monkeypatch.setattr(venues.db, "Historical", no_network)
    download_venue_files(
        venues=("ARCX.PILLAR",),
        auction_venues=("ARCX.PILLAR",),
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-03",
        directory=tmp_path,
    )


def test_missing_venue_files_are_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[tuple[str, str]] = []

    class FakeStore:
        def __init__(self, schema: str) -> None:
            self.schema = schema

        def to_df(self) -> pd.DataFrame:
            frame = pd.DataFrame({"symbol": ["AAA", "AAA"], "stat_type": [OPENING_PRICE, 2]})
            return frame.rename_axis("ts_recv")

    class FakeClient:
        def __init__(self, key: str) -> None:
            assert key == "test-key"
            self.timeseries = self

        def get_range(self, *, dataset: str, schema: str, **_: object) -> FakeStore:
            requested.append((dataset, schema))
            return FakeStore(schema)

    monkeypatch.setattr(venues.db, "Historical", FakeClient)
    monkeypatch.setattr(venues, "api_key", lambda: "test-key")
    download_venue_files(
        venues=("XNAS.ITCH",),
        auction_venues=("XNAS.ITCH",),
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-03",
        directory=tmp_path,
    )

    assert requested == [("XNAS.ITCH", DAILY), ("XNAS.ITCH", HOURLY), ("XNAS.ITCH", STATISTICS)]
    # statistics 只留开盘 / 收盘价两类
    stats = pd.read_parquet(venue_file(tmp_path, "XNAS.ITCH", STATISTICS))
    assert stats["stat_type"].tolist() == [OPENING_PRICE]
