"""三个模型各跑一组超参，按验证集 ICIR 选优（测试集只看不选）。

用法::

    uv run python scripts/sweep_models.py

结果写到 artifacts/sweep.csv，最优参数会打印出来，手动填回 config/default.yaml。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkq.config import Config
from hkq.corporate_actions import adjust_panel, load_actions
from hkq.data import load_panel
from hkq.dataset import LABEL, feature_columns, split_by_date
from hkq.ic import daily_rank_ic, ic_summary
from hkq.models import build_model
from hkq.portfolio import backtest_topk

logger = logging.getLogger("sweep")

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
    "mlp_1024_512_256": {
        "hidden": [1024, 512, 256],
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 4096,
        "max_epochs": 500,
        "patience": 50,
    },
    "mlp_2048_1024": {
        "hidden": [2048, 1024],
        "dropout": 0.4,
        "lr": 3e-4,
        "weight_decay": 1e-2,
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
}

GRIDS = {"xgboost": XGBOOST_GRID, "lightgbm": LIGHTGBM_GRID, "mlp": MLP_GRID}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = Config.load()
    data = pd.read_parquet(cfg.dataset.output)
    splits = split_by_date(data, cfg.split.ranges())
    features = feature_columns(data)
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    logger.info(
        "features=%d train=%d valid=%d test=%d", len(features), len(train), len(valid), len(test)
    )

    panel = adjust_panel(load_panel(cfg.data.output), load_actions(cfg.data.corporate_actions))
    returns = panel.close.pct_change()

    rows = []
    for family, grid in GRIDS.items():
        for name, params in grid.items():
            model = build_model(
                family,
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
            valid_ic = ic_summary(
                daily_rank_ic(
                    pd.Series(
                        model.predict(valid[features].to_numpy("float64")), index=valid.index
                    ),
                    valid[LABEL],
                )
            )
            test_pred = pd.Series(
                model.predict(test[features].to_numpy("float64")), index=test.index
            )
            test_ic = ic_summary(daily_rank_ic(test_pred, test[LABEL]))
            backtest = backtest_topk(
                test_pred.unstack("symbol"),
                returns,
                top_k=cfg.backtest.top_k,
                rebalance_days=cfg.backtest.rebalance_days,
                cost_bps=cfg.backtest.cost_bps,
                execution_lag=cfg.dataset.label_lag,
                seed=cfg.train.seed,
            )
            row = {
                "config": name,
                "family": family,
                "valid_ic": valid_ic["ic_mean"],
                "valid_icir": valid_ic["icir"],
                "test_ic": test_ic["ic_mean"],
                "test_icir": test_ic["icir"],
                "test_cagr": backtest.strategy["cagr"],
                "test_sharpe": backtest.strategy["sharpe"],
                "test_maxdd": backtest.strategy["max_drawdown"],
                "params": json.dumps(params, ensure_ascii=False),
            }
            rows.append(row)
            logger.info(
                "%-18s validIC=%+.4f ICIR=%+.2f | testIC=%+.4f CAGR=%+.2f%% Sharpe=%+.2f",
                name,
                row["valid_ic"],
                row["valid_icir"],
                row["test_ic"],
                100 * row["test_cagr"],
                row["test_sharpe"],
            )

    table = pd.DataFrame(rows)
    out = cfg.train.artifacts / "sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info("wrote %s", out)

    for family in GRIDS:
        subset = table[table["family"] == family].copy()
        best = subset.loc[subset["valid_icir"].abs().idxmax()]
        logger.info(
            "BEST %s -> %s | validIC=%+.4f testIC=%+.4f testCAGR=%+.2f%% params=%s",
            family,
            best["config"],
            best["valid_ic"],
            best["test_ic"],
            100 * best["test_cagr"],
            best["params"],
        )


if __name__ == "__main__":
    main()
