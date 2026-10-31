"""滚动重训（expanding window）：每年用之前的全部数据重训一次，拼出多年的样本外预测。

单次切分的测试集只覆盖一段行情，偶然性很大。这里对每个测试年 Y：

    训练 = 数据起点 ~ Y-2 年底，验证（早停）= Y-1 年，测试 = Y 年

训练集、验证集末尾各去掉 label_lag + horizon 个交易日：这些样本的标签会跨进下一段的行情
（purge），不去掉的话验证集早停会间接看到测试段的价格。各年的测试预测拼起来跑一次连续回测。
模型参数直接用 config 里的，不做任何调参。

用法::

    uv run python scripts/walk_forward.py --models xgboost,lightgbm --first-year 2022
    uv run python scripts/walk_forward.py --models lightgbm --factor-set top30 \\
        --params '{"num_leaves": 63}' --first-year 2024 --tag lgb_top30

--factor-set 取自 sweep_models.py 写出的 artifacts/sweep/factor_sets.json。那些子集是用主切分的
训练集（2019-05 ~ 2022-12）选出来的，配置又是在 2023 年的验证集上挑的，所以只有 2024 年及之后的
测试年对这类配置是干净的样本外。

结果写到 artifacts/walk_forward/：每个模型的 predictions.parquet / backtest_daily.csv /
holdings.csv，以及按年和全区间汇总的 summary.csv。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkq.config import Config
from hkq.corporate_actions import adjust_panel, load_actions
from hkq.data import load_panel
from hkq.dataset import LABEL, feature_columns
from hkq.ic import daily_rank_ic, ic_summary
from hkq.models import build_model
from hkq.portfolio import TRADING_DAYS_PER_YEAR, backtest_topk, performance

logger = logging.getLogger("walk_forward")


def purge_tail(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """去掉最后 days 个交易日：它们的标签用到了下一段的价格。"""
    dates = frame.index.get_level_values("date").unique().sort_values()
    if len(dates) <= days:
        raise ValueError(f"只有 {len(dates)} 个交易日，不够 purge {days} 天")
    return frame[frame.index.get_level_values("date") <= dates[-days - 1]]


def year_slice(data: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    years = data.index.get_level_values("date").year
    return data[(years >= first) & (years <= last)]


def fit_predict(
    cfg: Config,
    name: str,
    params: dict[str, object],
    data: pd.DataFrame,
    features: list[str],
    year: int,
    purge: int,
) -> pd.DataFrame:
    """训练到 year-2 年底、year-1 年早停，返回 year 年的预测。"""
    first_year = int(data.index.get_level_values("date").year.min())
    train = purge_tail(year_slice(data, first_year, year - 2), purge)
    valid = purge_tail(year_slice(data, year - 1, year - 1), purge)
    test = year_slice(data, year, year)

    model = build_model(
        name,
        params,
        seed=cfg.train.seed,
        early_stopping_rounds=cfg.train.early_stopping_rounds,
    )
    model.fit(
        train[features].to_numpy("float64"),
        train[LABEL].to_numpy("float64"),
        valid[features].to_numpy("float64"),
        valid[LABEL].to_numpy("float64"),
    )
    frame = pd.DataFrame({LABEL: test[LABEL]}, index=test.index)
    frame["prediction"] = model.predict(test[features].to_numpy("float64"))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=None, help="逗号分隔，默认读 config 的 train.models")
    parser.add_argument("--first-year", type=int, default=2022, help="第一个样本外测试年")
    parser.add_argument("--factor-set", default="all", help="factor_sets.json 里的名字")
    parser.add_argument("--params", default="{}", help="JSON，覆盖 config 里该模型的参数")
    parser.add_argument("--tag", default="", help="结果子目录名，默认 walk_forward")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = Config.load()
    data = pd.read_parquet(cfg.dataset.output)
    features = feature_columns(data)
    if args.factor_set != "all":
        sets = json.loads((cfg.train.artifacts / "sweep" / "factor_sets.json").read_text("utf-8"))
        features = sets[args.factor_set]
    overrides = json.loads(args.params)
    purge = cfg.dataset.label_lag + cfg.dataset.horizon
    last_year = int(data.index.get_level_values("date").year.max())
    years = list(range(args.first_year, last_year + 1))
    models = tuple(args.models.split(",")) if args.models else cfg.train.models
    logger.info(f"features={len(features)} years={years} models={models} purge={purge}d")

    returns = adjust_panel(
        load_panel(cfg.data.output), load_actions(cfg.data.corporate_actions)
    ).close.pct_change(fill_method=None)
    out = cfg.train.artifacts / "walk_forward" / (args.tag or "default")
    rows: list[dict[str, object]] = []

    for name in models:
        started = time.time()
        frames = []
        for year in years:
            params = {**cfg.models[name], **overrides}
            frames.append(fit_predict(cfg, name, params, data, features, year, purge))
            ic = ic_summary(daily_rank_ic(frames[-1]["prediction"], frames[-1][LABEL]))
            logger.info(
                f"[{name}] {year}: IC={ic['ic_mean']:+.4f} ICIR={ic['icir']:+.3f} "
                f"days={ic['n_days']} ({time.time() - started:.0f}s)"
            )
        predictions = pd.concat(frames).sort_index()
        backtest = backtest_topk(
            predictions["prediction"].unstack("symbol"),
            returns,
            top_k=cfg.backtest.top_k,
            rebalance_days=cfg.backtest.rebalance_days,
            cost_bps=cfg.backtest.cost_bps,
            execution_lag=cfg.dataset.label_lag,
            seed=cfg.train.seed,
        )

        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(directory / "predictions.parquet")
        backtest.daily.to_csv(directory / "backtest_daily.csv")
        backtest.holdings.to_csv(directory / "holdings.csv", index=False)

        daily = backtest.daily
        periods = [("all", daily)] + [
            (str(year), daily[daily.index.year == year]) for year in years
        ]
        for label, part in periods:
            part_predictions = (
                predictions
                if label == "all"
                else predictions[predictions.index.get_level_values("date").year == int(label)]
            )
            ic = ic_summary(daily_rank_ic(part_predictions["prediction"], part_predictions[LABEL]))
            strategy = performance(part["net_return"], TRADING_DAYS_PER_YEAR)
            benchmark = performance(part["benchmark_return"], TRADING_DAYS_PER_YEAR)
            rows.append(
                {
                    "model": name,
                    "period": label,
                    "ic": ic["ic_mean"],
                    "icir": ic["icir"],
                    "cagr": strategy["cagr"],
                    "sharpe": strategy["sharpe"],
                    "max_drawdown": strategy["max_drawdown"],
                    "bench_cagr": benchmark["cagr"],
                    "bench_sharpe": benchmark["sharpe"],
                    "days": len(part),
                }
            )
        logger.info(f"[{name}] done in {time.time() - started:.0f}s -> {directory}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    logger.info("\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
