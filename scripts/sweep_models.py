"""超参搜索：因子集 × 模型超参的网格，按验证集 ICIR 选优（测试集只记录、不参与选择）。

因子集也当成超参：全部 186 个、只用 Alpha101、只用 Alpha158，以及按**训练集** |ICIR| 排序、
相关性去重后的前 N 个。因子子集只看训练集，这样验证集只被「选配置」用一次；
测试集的指标照样算出来写进表里，但选最优时不看。

用法::

    uv run python scripts/sweep_models.py --families xgboost,lightgbm --threads 4 \\
        --out sweep_trees.csv
    uv run python scripts/sweep_models.py --families mlp --threads 8 --out sweep_mlp.csv

结果逐行追加到 artifacts/sweep/<out>，中断后重跑会跳过已经跑完的组合；
每个组合的验证集 / 测试集预测存在 artifacts/sweep/predictions/。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkq.alpha158 import ALPHA158_NAMES
from hkq.config import Config
from hkq.corporate_actions import adjust_panel, load_actions
from hkq.data import load_panel
from hkq.dataset import LABEL, feature_columns, split_by_date
from hkq.ic import daily_rank_ic, ic_summary
from hkq.models import build_model
from hkq.portfolio import backtest_topk

logger = logging.getLogger("sweep")

# 两只因子日均 |Spearman| 超过这个数就算同一个信号
MAX_ABS_RHO = 0.8
TOP_N = (15, 30, 60, 100)

XGBOOST_GRID: dict[str, dict[str, Any]] = {
    "xgb_d3": {
        "n_estimators": 900,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 100,
        "subsample": 0.9,
        "colsample_bytree": 0.6,
        "reg_lambda": 1.0,
    },
    "xgb_d5": {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_lambda": 2.0,
    },
    "xgb_d7": {
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 300,
        "subsample": 0.7,
        "colsample_bytree": 0.4,
        "reg_lambda": 5.0,
    },
    "xgb_d9": {
        "n_estimators": 3000,
        "learning_rate": 0.015,
        "max_depth": 9,
        "min_child_weight": 500,
        "subsample": 0.6,
        "colsample_bytree": 0.3,
        "reg_lambda": 10.0,
    },
}

LIGHTGBM_GRID: dict[str, dict[str, Any]] = {
    "lgb_l31": {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 200,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_lambda": 3.0,
    },
    "lgb_l63": {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 300,
        "subsample": 0.7,
        "colsample_bytree": 0.4,
        "reg_lambda": 5.0,
    },
    "lgb_l127": {
        "n_estimators": 3000,
        "learning_rate": 0.015,
        "num_leaves": 127,
        "min_child_samples": 500,
        "subsample": 0.6,
        "colsample_bytree": 0.3,
        "reg_lambda": 10.0,
    },
    "lgb_l255": {
        "n_estimators": 4000,
        "learning_rate": 0.01,
        "num_leaves": 255,
        "min_child_samples": 800,
        "subsample": 0.6,
        "colsample_bytree": 0.25,
        "reg_lambda": 20.0,
    },
}

MLP_GRID: dict[str, dict[str, Any]] = {
    "mlp_256_128": {
        "hidden": [256, 128],
        "dropout": 0.2,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 2048,
        "max_epochs": 400,
        "patience": 40,
    },
    "mlp_512_256": {
        "hidden": [512, 256],
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 2048,
        "max_epochs": 400,
        "patience": 40,
    },
    "mlp_1024_512": {
        "hidden": [1024, 512],
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 4096,
        "max_epochs": 500,
        "patience": 50,
    },
    "mlp_2048_1024_512": {
        "hidden": [2048, 1024, 512],
        "dropout": 0.4,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "batch_size": 4096,
        "max_epochs": 500,
        "patience": 50,
    },
    # 上面几组学习率下，验证集 MSE 在第 1 个 epoch 就到了最低点（之后一直过拟合），
    # 下面把学习率降 10–50 倍、正则加重，让早停点落到后面的 epoch
    "mlp_64_slow": {
        "hidden": [64],
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 4096,
        "max_epochs": 300,
        "patience": 30,
    },
    "mlp_256_128_slow": {
        "hidden": [256, 128],
        "dropout": 0.5,
        "lr": 3e-5,
        "weight_decay": 1e-2,
        "batch_size": 2048,
        "max_epochs": 300,
        "patience": 30,
    },
    "mlp_512_256_slow": {
        "hidden": [512, 256],
        "dropout": 0.5,
        "lr": 1e-5,
        "weight_decay": 1e-2,
        "batch_size": 2048,
        "max_epochs": 300,
        "patience": 30,
    },
    # 验证集 MSE 几乎不动（信噪比太低），下面改成按验证集相关系数早停（stop_on: corr）
    "mlp_64_corr": {
        "hidden": [64],
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 4096,
        "max_epochs": 300,
        "patience": 20,
        "stop_on": "corr",
    },
    "mlp_256_128_corr": {
        "hidden": [256, 128],
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 2048,
        "max_epochs": 300,
        "patience": 20,
        "stop_on": "corr",
    },
    "mlp_512_256_corr": {
        "hidden": [512, 256],
        "dropout": 0.4,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 2048,
        "max_epochs": 300,
        "patience": 20,
        "stop_on": "corr",
    },
    "mlp_1024_512_corr": {
        "hidden": [1024, 512],
        "dropout": 0.5,
        "lr": 5e-5,
        "weight_decay": 1e-2,
        "batch_size": 4096,
        "max_epochs": 300,
        "patience": 20,
        "stop_on": "corr",
    },
}

GRIDS = {"xgboost": XGBOOST_GRID, "lightgbm": LIGHTGBM_GRID, "mlp": MLP_GRID}
THREAD_PARAM = {"xgboost": "n_jobs", "lightgbm": "n_jobs", "mlp": "threads"}


def daily_ranks(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """按日横截面 rank 后去均值：之后的点积就是逐日 Spearman 的分子。"""
    ranks = data[columns].groupby(level="date").rank()
    return ranks - ranks.groupby(level="date").transform("mean")


def train_ic(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """训练集逐因子的日 Rank IC 统计（向量化，和 hkq.ic.daily_rank_ic 的口径一致）。"""
    centred = daily_ranks(train, [*features, LABEL])
    numerator = centred[features].mul(centred[LABEL], axis=0).groupby(level="date").sum()
    denominator = np.sqrt(
        (centred[features] ** 2)
        .groupby(level="date")
        .sum()
        .mul((centred[LABEL] ** 2).groupby(level="date").sum(), axis=0)
    )
    ic = numerator / denominator.replace(0.0, np.nan)
    return pd.DataFrame({"ic": ic.mean(), "icir": ic.mean() / ic.std()})


def train_correlation(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """训练集上因子两两的日均 Spearman（逐日算完再平均，常数列当天不计入）。"""
    centred = daily_ranks(train, features)
    total = np.zeros((len(features), len(features)))
    count = np.zeros_like(total)
    for _, day in centred.groupby(level="date"):
        values = day.to_numpy()
        norms = np.sqrt((values**2).sum(axis=0))
        valid = norms > 0
        z = np.where(valid, values / np.where(valid, norms, 1.0), 0.0)
        pair = np.outer(valid, valid)
        total += np.where(pair, z.T @ z, 0.0)
        count += pair
    return pd.DataFrame(total / np.maximum(count, 1), index=features, columns=features)


def factor_sets(train: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    """候选因子集。子集只用训练集的信息构造，不碰验证集和测试集。"""
    ic = train_ic(train, features)
    corr = train_correlation(train, features)
    order = ic["icir"].abs().sort_values(ascending=False).index

    deduped: list[str] = []
    for name in order:
        if all(abs(corr.loc[name, kept]) < MAX_ABS_RHO for kept in deduped):
            deduped.append(str(name))
    logger.info(
        f"train-only dedupe |rho|<{MAX_ABS_RHO}: {len(deduped)}/{len(features)} factors kept; "
        f"top by |ICIR|: {', '.join(f'{n}={ic.loc[n, "icir"]:+.3f}' for n in deduped[:8])}"
    )

    sets = {
        "all": list(features),
        "alpha101": [f for f in features if f not in ALPHA158_NAMES],
        "alpha158": [f for f in features if f in ALPHA158_NAMES],
        "dedup": deduped,
    }
    for n in TOP_N:
        if n < len(deduped):
            sets[f"top{n}"] = deduped[:n]
    return sets


def evaluate(
    predictions: pd.Series, labels: pd.Series, returns: pd.DataFrame, cfg: Config
) -> dict[str, float]:
    ic = ic_summary(daily_rank_ic(predictions, labels))
    backtest = backtest_topk(
        predictions.unstack("symbol"),
        returns,
        top_k=cfg.backtest.top_k,
        rebalance_days=cfg.backtest.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        execution_lag=cfg.dataset.label_lag,
        seed=cfg.train.seed,
    )
    return {
        "ic": ic["ic_mean"],
        "icir": ic["icir"],
        "cagr": backtest.strategy["cagr"],
        "sharpe": backtest.strategy["sharpe"],
        "maxdd": backtest.strategy["max_drawdown"],
        "bench_cagr": backtest.benchmark["cagr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="因子集 × 超参网格搜索")
    parser.add_argument("--families", default="xgboost,lightgbm,mlp")
    parser.add_argument("--sets", default=None, help="逗号分隔的因子集名，默认全部")
    parser.add_argument("--configs", default=None, help="逗号分隔的网格名，默认该模型的全部")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="sweep.csv")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = Config.load()
    data = pd.read_parquet(cfg.dataset.output)
    splits = split_by_date(data, cfg.split.ranges())
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    features = feature_columns(data)
    returns = adjust_panel(
        load_panel(cfg.data.output), load_actions(cfg.data.corporate_actions)
    ).close.pct_change(fill_method=None)

    out_dir = cfg.train.artifacts / "sweep"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    sets = factor_sets(train, features)
    (out_dir / "factor_sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if args.sets:
        sets = {name: sets[name] for name in args.sets.split(",")}
    logger.info("factor sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    out = out_dir / args.out
    done = set(pd.read_csv(out)["run"]) if out.exists() else set()
    jobs = [
        (family, config, set_name)
        for set_name in sets
        for family in args.families.split(",")
        for config in GRIDS[family]
        if args.configs is None or config in args.configs.split(",")
    ]
    logger.info(f"{len(jobs)} runs, {len(done)} already done -> {out}")

    for index, (family, config, set_name) in enumerate(jobs, start=1):
        run = f"{family}|{config}|{set_name}"
        if run in done:
            continue
        columns = sets[set_name]
        params = {**GRIDS[family][config], THREAD_PARAM[family]: args.threads}
        started = time.time()
        model = build_model(
            family,
            params,
            seed=cfg.train.seed,
            early_stopping_rounds=cfg.train.early_stopping_rounds,
        )
        model.fit(
            train[columns].to_numpy("float64"),
            train[LABEL].to_numpy("float64"),
            valid[columns].to_numpy("float64"),
            valid[LABEL].to_numpy("float64"),
        )
        row: dict[str, Any] = {
            "run": run,
            "family": family,
            "config": config,
            "factor_set": set_name,
            "n_factors": len(columns),
        }
        frames = []
        for split, frame in (("valid", valid), ("test", test)):
            prediction = pd.Series(
                model.predict(frame[columns].to_numpy("float64")), index=frame.index
            )
            metrics = evaluate(prediction, frame[LABEL], returns, cfg)
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
            frames.append(
                pd.DataFrame({"prediction": prediction, LABEL: frame[LABEL], "split": split})
            )
        row["seconds"] = round(time.time() - started, 1)
        row["params"] = json.dumps(GRIDS[family][config])

        pd.concat(frames).to_parquet(out_dir / "predictions" / f"{run.replace('|', '__')}.parquet")
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        logger.info(
            f"[{index}/{len(jobs)}] {run:40s} n={len(columns):3d} | "
            f"valid IC={row['valid_ic']:+.4f} ICIR={row['valid_icir']:+.3f} "
            f"Sharpe={row['valid_sharpe']:+.2f} | test IC={row['test_ic']:+.4f} "
            f"CAGR={row['test_cagr']:+.2%} ({row['seconds']:.0f}s)"
        )

    table = pd.read_csv(out)
    for family, subset in table.groupby("family"):
        # 按带符号的 ICIR 取最大：负 ICIR 说明排序方向反了，不能因为绝对值大就当成最好
        best = subset.loc[subset["valid_icir"].idxmax()]
        logger.info(
            f"BEST {family} by valid ICIR -> {best['run']} | valid ICIR={best['valid_icir']:+.3f} "
            f"| test IC={best['test_ic']:+.4f} CAGR={best['test_cagr']:+.2%} "
            f"Sharpe={best['test_sharpe']:+.2f}"
        )


if __name__ == "__main__":
    main()
