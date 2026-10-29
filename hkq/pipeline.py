"""流水线：下载行情 -> 生成因子数据集 -> 训练与评估。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from hkq.config import Config
from hkq.corporate_actions import adjust_panel, download_actions, load_actions
from hkq.data import Panel, download_bars, load_panel, read_tickers
from hkq.dataset import LABEL, build_dataset, feature_columns, split_by_date
from hkq.ic import alpha_ic_table, daily_rank_ic, ic_summary
from hkq.models import Regressor, build_model
from hkq.portfolio import BacktestResult, backtest_topk, cost_sensitivity
from hkq.report import split_summary, write_json, write_model_artifacts, write_report

logger = logging.getLogger(__name__)


def run_download(cfg: Config) -> pd.DataFrame:
    """下载训练池与交易池的日频行情，以及分红/拆股记录。"""
    training = read_tickers(cfg.universe.training_tickers)
    trading = read_tickers(cfg.universe.trading_tickers)
    symbols = sorted(set(training) | set(trading))
    logger.info(f"[download] training={len(training)} trading={len(trading)} union={len(symbols)}")

    bars = download_bars(
        dataset=cfg.data.dataset,
        schema=cfg.data.schema,
        symbols=symbols,
        start=cfg.data.start,
        end=cfg.data.end,
        output=cfg.data.output,
    )
    logger.info(
        f"[download] symbols={bars['symbol'].nunique()} rows={len(bars):,} "
        f"range={bars['date'].min().date()} -> {bars['date'].max().date()}"
    )

    download_actions(
        symbols=symbols,
        start=cfg.data.start,
        end=str((bars["date"].max() + pd.Timedelta(days=1)).date()),
        output=cfg.data.corporate_actions,
    )
    return bars


def run_features(cfg: Config) -> pd.DataFrame:
    """计算因子与前瞻收益，生成 (date, symbol) 长表数据集。"""
    panel = load_panel(cfg.data.output)
    data = build_dataset(
        panel,
        load_actions(cfg.data.corporate_actions),
        alphas=cfg.dataset.alphas,
        horizon=cfg.dataset.horizon,
        normalize=cfg.dataset.normalize,
        min_history=cfg.dataset.min_history,
        label_lag=cfg.dataset.label_lag,
        label_norm=cfg.dataset.label_norm,
    )

    cfg.dataset.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cfg.dataset.output)
    write_json(
        cfg.dataset.output.with_suffix(".meta.json"),
        {
            "alphas": list(cfg.dataset.alphas),
            "horizon": cfg.dataset.horizon,
            "label_lag": cfg.dataset.label_lag,
            "label_norm": cfg.dataset.label_norm,
            "normalize": cfg.dataset.normalize,
            "min_history": cfg.dataset.min_history,
            "n_rows": len(data),
            "n_dates": int(data.index.get_level_values("date").nunique()),
            "n_symbols": int(data.index.get_level_values("symbol").nunique()),
            "date_range": [
                str(data.index.get_level_values("date").min().date()),
                str(data.index.get_level_values("date").max().date()),
            ],
        },
    )

    logger.info(
        f"[features] rows={len(data):,} dates={data.index.get_level_values('date').nunique()} "
        f"symbols={data.index.get_level_values('symbol').nunique()} "
        f"features={len(feature_columns(data))} -> {cfg.dataset.output}"
    )
    return data


def run_train(cfg: Config, models: tuple[str, ...] | None = None) -> pd.DataFrame:
    """训练多个模型，输出 IC、回测绩效与报告。"""
    data = pd.read_parquet(cfg.dataset.output)
    splits = split_by_date(data, cfg.split.ranges())
    _require_non_empty(splits)
    features = feature_columns(data)

    split_table = split_summary(splits)
    logger.info("[data] split summary:\n" + split_table.to_string())

    if cfg.backtest.rebalance_days != cfg.dataset.horizon:
        logger.warning(
            f"rebalance_days={cfg.backtest.rebalance_days} != horizon="
            f"{cfg.dataset.horizon}; the backtest holds a different period than the label"
        )

    alpha_ic = alpha_ic_table(splits["train"], features)
    cfg.train.artifacts.mkdir(parents=True, exist_ok=True)
    alpha_ic.to_csv(cfg.train.artifacts / "alpha_ic.csv")
    logger.info(
        "[factors] rank IC on the train split (sorted by ic_mean):\n"
        + alpha_ic[["ic_mean", "icir", "ic_win_rate", "n_days"]].round(4).to_string()
    )

    returns = _adjusted_panel(cfg).close.pct_change()

    rows: list[dict[str, Any]] = []
    sensitivities: dict[str, pd.Series] = {}
    for name in models or cfg.train.models:
        logger.info(f"=== {name} ===")
        row, sensitivity = _train_one(cfg, name, splits, features, returns)
        rows.append(row)
        sensitivities[name] = sensitivity

    summary_frame = pd.DataFrame(rows)
    logger.info("[summary]\n" + summary_frame.round(4).to_string(index=False))
    report_path = cfg.train.artifacts / "report.md"
    write_report(report_path, summary_frame, alpha_ic, split_table, sensitivities, cfg)
    logger.info(f"[report] wrote {report_path}")
    return summary_frame


def _train_one(
    cfg: Config,
    name: str,
    splits: dict[str, pd.DataFrame],
    features: list[str],
    returns: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series]:
    """训练单个模型：拟合 -> 预测 -> 回测 -> 落盘 -> 打日志，返回汇总行与成本敏感度。"""
    model = _fit(cfg, name, splits, features)
    predictions = {
        split: _predict_frame(model, splits[split], features) for split in ("valid", "test")
    }
    ic = {
        split: daily_rank_ic(frame["prediction"], frame[LABEL])
        for split, frame in predictions.items()
    }
    backtest = _backtest(predictions["test"], returns, cfg)
    importance = _feature_importance(model, features)
    sensitivity = cost_sensitivity(backtest.daily, cfg.backtest.cost_bps)
    valid_ic, test_ic = ic_summary(ic["valid"]), ic_summary(ic["test"])

    metrics = {
        "model": name,
        "valid": valid_ic,
        "test": test_ic,
        "strategy": backtest.strategy,
        "benchmark": backtest.benchmark,
        "avg_turnover": backtest.avg_turnover,
    }
    write_model_artifacts(
        cfg.train.artifacts / name,
        predictions=predictions,
        backtest=backtest,
        feature_importance=importance,
        sensitivity=sensitivity,
        metrics=metrics,
    )
    _log_model(cfg, name, valid_ic, test_ic, backtest, sensitivity, importance)

    row: dict[str, Any] = {
        "model": name,
        "valid_ic": valid_ic["ic_mean"],
        "valid_icir": valid_ic["icir"],
        "test_ic": test_ic["ic_mean"],
        "test_icir": test_ic["icir"],
        "test_cagr": backtest.strategy["cagr"],
        "test_sharpe": backtest.strategy["sharpe"],
        "test_maxdd": backtest.strategy["max_drawdown"],
        "bench_cagr": backtest.benchmark["cagr"],
        "turnover": backtest.avg_turnover,
    }
    return row, sensitivity


def _fit(cfg: Config, name: str, splits: dict[str, pd.DataFrame], features: list[str]) -> Regressor:
    """只用训练集拟合、用验证集早停；测试集在这里完全不出现。"""
    train, valid = splits["train"], splits["valid"]
    model = build_model(
        name,
        cfg.models[name],
        seed=cfg.train.seed,
        early_stopping_rounds=cfg.train.early_stopping_rounds,
    )
    model.fit(
        _matrix(train, features),
        train[LABEL].to_numpy(dtype="float64"),
        _matrix(valid, features),
        valid[LABEL].to_numpy(dtype="float64"),
    )
    return model


def _feature_importance(model: Regressor, features: list[str]) -> pd.Series | None:
    importance = getattr(model, "feature_importance_", None)
    if importance is None:
        return None
    return pd.Series(importance.to_numpy(), index=features).sort_values(ascending=False)


def _log_model(
    cfg: Config,
    name: str,
    valid_ic: dict[str, float],
    test_ic: dict[str, float],
    backtest: BacktestResult,
    sensitivity: pd.Series,
    importance: pd.Series | None,
) -> None:
    """一个模型跑完，把交易员真正关心的几行打出来。"""
    logger.info(
        f"  valid IC={valid_ic['ic_mean']:+.4f} ICIR={valid_ic['icir']:+.2f} | "
        f"test IC={test_ic['ic_mean']:+.4f} ICIR={test_ic['icir']:+.2f}"
    )
    logger.info(
        f"  test top{cfg.backtest.top_k}: CAGR={backtest.strategy['cagr']:+.2%} "
        f"Sharpe={backtest.strategy['sharpe']:+.2f} MDD={backtest.strategy['max_drawdown']:.2%} | "
        f"benchmark CAGR={backtest.benchmark['cagr']:+.2%}"
    )
    logger.info(
        f"  turnover={backtest.avg_turnover:.2f} per rebalance | "
        f"rebalances={int((backtest.daily['traded_notional'] > 0).sum())} "
        f"over {len(backtest.daily)} trading days"
    )
    if importance is not None:
        logger.info(
            "  top factors: " + ", ".join(f"{k}={v:.4g}" for k, v in importance.head(5).items())
        )
    logger.info(
        "  CAGR under cost "
        + ", ".join(f"{bps:g}bp={value:+.1%}" for bps, value in sensitivity.items())
    )


def _matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    return data[features].to_numpy(dtype="float64")


def _predict_frame(model: Regressor, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({LABEL: data[LABEL]}, index=data.index)
    frame["prediction"] = model.predict(_matrix(data, features))
    return frame


def _backtest(predictions: pd.DataFrame, returns: pd.DataFrame, cfg: Config) -> BacktestResult:
    wide = predictions["prediction"].unstack("symbol")
    return backtest_topk(
        wide,
        returns,
        top_k=cfg.backtest.top_k,
        rebalance_days=cfg.backtest.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        execution_lag=cfg.dataset.label_lag,
        seed=cfg.train.seed,
    )


def _adjusted_panel(cfg: Config) -> Panel:
    """因果复权后的行情，回测的日收益从这里取（收益率与归一化方式无关）。

    特征的复权走 dataset 里的 point-in-time 窗口，两者口径不同但收益率完全一致。
    """
    return adjust_panel(load_panel(cfg.data.output), load_actions(cfg.data.corporate_actions))


def _require_non_empty(splits: dict[str, pd.DataFrame]) -> None:
    for name, split in splits.items():
        if split.empty:
            raise ValueError(
                f"{name} 切分没有样本，检查 config 里的 split 日期与 data.start 是否重叠"
            )
