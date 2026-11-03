"""模拟盘信号：用 sweep 选出的最优配置、截至最新交易日的数据重训，输出每个模型对全部 ETF 的排名。

做法和 walk_forward.py 一致，只是窗口按最新日期滚动：

    验证（早停）= 最后一个有标签的交易日往前 1 年
    训练 = 数据起点 ~ 验证开始，末尾 purge label_lag + horizon 天

超参和因子子集按 sweep_models.py 的规则取（各模型验证集 ICIR 最高的那组），信号日取数据的最后一个
交易日，按回测口径在信号日的下一个交易日收盘成交。只出选股建议，不下单。

用法::

    uv run python scripts/paper_signal.py --sweep-tag h5 --rebalance-days 10

输出到 artifacts/paper/<信号日>/：rankings.csv（每个模型 × 每只 ETF 一行）、top_picks.csv
（每个模型前 top_k 只及按 --capital 折算的建仓金额）、manifest.json（训练窗口、配置、验证集 IC），
以及 models/ 下训练好的模型（pickle，复现用）。
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkq.config import DEFAULT_CONFIG, PROJECT_ROOT, Config
from hkq.corporate_actions import adjust_panel, load_actions
from hkq.data import load_panel
from hkq.dataset import LABEL, build_dataset
from hkq.ic import daily_rank_ic, ic_summary
from hkq.models import build_model

logger = logging.getLogger("paper_signal")

FAMILIES = ("xgboost", "lightgbm", "mlp")
THREAD_PARAM = {"xgboost": "n_jobs", "lightgbm": "n_jobs", "mlp": "threads"}


def best_configs(sweep_dir: Path) -> dict[str, dict[str, Any]]:
    """各模型验证集 ICIR（带符号）最高的那组超参和因子集，和 sweep_models.py 的选法一致。"""
    table = pd.concat(
        [pd.read_csv(path) for path in sorted(sweep_dir.glob("sweep*.csv"))], ignore_index=True
    )
    sets = json.loads((sweep_dir / "factor_sets.json").read_text("utf-8"))
    best: dict[str, dict[str, Any]] = {}
    for family, subset in table.groupby("family"):
        row = subset.loc[subset["valid_icir"].idxmax()]
        best[str(family)] = {
            "run": row["run"],
            "config": row["config"],
            "factor_set": row["factor_set"],
            "features": sets[row["factor_set"]],
            "params": json.loads(row["params"]),
            "sweep_valid_icir": float(row["valid_icir"]),
        }
    return best


def rolling_windows(labeled: pd.DataFrame, purge: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """验证 = 最后一个有标签日往前 1 年；训练 = 之前的全部，末尾去掉 purge 天。"""
    dates = labeled.index.get_level_values("date")
    valid_start = dates.max() - pd.DateOffset(years=1)
    train = labeled[dates <= valid_start]
    train_dates = train.index.get_level_values("date").unique().sort_values()
    train = train[train.index.get_level_values("date") <= train_dates[-purge - 1]]
    return train, labeled[dates > valid_start]


def date_range(frame: pd.DataFrame) -> list[str]:
    dates = frame.index.get_level_values("date")
    return [str(dates.min().date()), str(dates.max().date())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--sweep-tag", default="", help="artifacts/sweep/ 下取最优配置的子目录")
    parser.add_argument("--models", default=",".join(FAMILIES))
    parser.add_argument("--rebalance-days", type=int, default=None, help="默认读 config")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="折算建仓金额用")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = Config.load(args.config)
    rebalance = args.rebalance_days or cfg.backtest.rebalance_days
    raw = load_panel(cfg.data.output)
    actions = load_actions(cfg.data.corporate_actions)
    data = build_dataset(
        raw,
        actions,
        alphas=cfg.dataset.alphas,
        horizon=cfg.dataset.horizon,
        normalize=cfg.dataset.normalize,
        min_history=cfg.dataset.min_history,
        label_lag=cfg.dataset.label_lag,
        label_norm=cfg.dataset.label_norm,
        drop_unlabeled=False,
    )
    dates = data.index.get_level_values("date")
    signal_date = dates.max()
    # 和回测的 tradable 口径一致：信号日有收盘价、能算出当日收益
    returns = adjust_panel(raw, actions).close.pct_change(fill_method=None)
    tradable = returns.loc[signal_date].dropna().index
    today = data[(dates == signal_date) & data.index.get_level_values("symbol").isin(tradable)]

    labeled = data.dropna(subset=[LABEL])
    purge = cfg.dataset.label_lag + cfg.dataset.horizon
    train, valid = rolling_windows(labeled, purge)
    logger.info(
        f"[signal] date={signal_date.date()} symbols={len(today)} | train {date_range(train)} "
        f"({len(train):,} rows) | valid {date_range(valid)} ({len(valid):,} rows) | purge={purge}d"
    )

    universe = pd.read_csv(cfg.universe.training_universe, encoding="utf-8-sig").set_index("ticker")
    trading = pd.read_csv(cfg.universe.trading_universe, encoding="utf-8-sig").set_index("ticker")
    asset_class = universe["asset_class"].combine_first(trading["asset_class"])
    names = universe["name"].combine_first(trading["name"])

    best = best_configs(cfg.train.artifacts / "sweep" / args.sweep_tag)
    out = cfg.train.artifacts / "paper" / str(signal_date.date())
    (out / "models").mkdir(parents=True, exist_ok=True)

    rankings, manifest_models = [], {}
    for family in args.models.split(","):
        spec = best[family]
        started = time.time()
        params = {**spec["params"], THREAD_PARAM[family]: args.threads}
        model = build_model(
            family,
            params,
            seed=cfg.train.seed,
            early_stopping_rounds=cfg.train.early_stopping_rounds,
        )
        columns = spec["features"]
        model.fit(
            train[columns].to_numpy("float64"),
            train[LABEL].to_numpy("float64"),
            valid[columns].to_numpy("float64"),
            valid[LABEL].to_numpy("float64"),
        )
        valid_ic = ic_summary(
            daily_rank_ic(
                pd.Series(model.predict(valid[columns].to_numpy("float64")), index=valid.index),
                valid[LABEL],
            )
        )
        with (out / "models" / f"{family}.pkl").open("wb") as handle:
            pickle.dump(model, handle)

        symbols = today.index.get_level_values("symbol")
        score = pd.Series(model.predict(today[columns].to_numpy("float64")), index=symbols)
        rank = score.rank(ascending=False, method="first").astype(int)
        rankings.append(
            pd.DataFrame(
                {
                    "model": family,
                    "model_config": spec["config"],
                    "factor_set": spec["factor_set"],
                    "symbol": symbols,
                    "name": symbols.map(names),
                    "asset_class": symbols.map(asset_class),
                    "score": score.to_numpy(),
                    "rank": rank.to_numpy(),
                    "percentile": 1.0 - (rank.to_numpy() - 1) / len(score),
                    "close_price_raw": raw.close.loc[signal_date, symbols].to_numpy(),
                }
            )
        )
        manifest_models[family] = {
            "run": spec["run"],
            "factor_set": spec["factor_set"],
            "n_factors": len(columns),
            "params": spec["params"],
            "sweep_valid_icir_2023": spec["sweep_valid_icir"],
            "refit_valid_ic": valid_ic["ic_mean"],
            "refit_valid_icir": valid_ic["icir"],
        }
        logger.info(
            f"[{family}] {spec['run']} | refit valid IC={valid_ic['ic_mean']:+.4f} "
            f"ICIR={valid_ic['icir']:+.3f} | top: {', '.join(score.nlargest(5).index)} "
            f"({time.time() - started:.0f}s)"
        )

    table = pd.concat(rankings, ignore_index=True).sort_values(["model", "rank"])
    table.insert(0, "signal_date", signal_date.date())
    table.to_csv(out / "rankings.csv", index=False, float_format="%.6g")

    top_k = cfg.backtest.top_k
    picks = table[table["rank"] <= top_k].copy()
    picks["target_weight"] = 1.0 / top_k
    picks["target_usd"] = args.capital / top_k
    picks["shares_at_signal_close"] = (picks["target_usd"] / picks["close_price_raw"]).round(2)
    picks.to_csv(out / "top_picks.csv", index=False, float_format="%.6g")

    calendar = returns.index
    trade_offset = cfg.dataset.label_lag
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    ).stdout.strip()
    manifest = {
        "signal_date": str(signal_date.date()),
        "execution": f"close, {trade_offset} trading day(s) after signal_date (backtest rule)",
        "rebalance_days": rebalance,
        "next_signal_date": f"signal_date + {rebalance} trading days",
        "label_horizon_days": cfg.dataset.horizon,
        "top_k": top_k,
        "capital_for_sizing": args.capital,
        "train_range": date_range(train),
        "valid_range": date_range(valid),
        "purge_days": purge,
        "n_symbols_ranked": len(today),
        "data_last_date": str(calendar.max().date()),
        "sweep": str(cfg.train.artifacts / "sweep" / args.sweep_tag),
        "git_commit": commit,
        "models": manifest_models,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), "utf-8")
    logger.info(f"[signal] wrote {out}")


if __name__ == "__main__":
    main()
