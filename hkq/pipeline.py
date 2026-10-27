"""流水线：下载行情 -> 生成因子数据集 -> 训练与评估。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hkq.config import Config
from hkq.data import download_bars, load_panel, read_tickers
from hkq.dataset import LABEL, build_dataset, feature_columns, forward_return, split_by_date
from hkq.evaluate import (
    BacktestResult,
    alpha_ic_table,
    backtest_topk,
    cost_sensitivity,
    daily_rank_ic,
    ic_summary,
)
from hkq.models import Regressor, build_model
from hkq.report import write_json, write_report

logger = logging.getLogger(__name__)


def run_download(cfg: Config) -> pd.DataFrame:
    """下载训练池与交易池的日频行情。"""
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
    return bars


def run_features(cfg: Config) -> pd.DataFrame:
    """计算因子与前瞻收益，生成 (date, symbol) 长表数据集。"""
    panel = load_panel(cfg.data.output)
    data = build_dataset(
        panel,
        alphas=cfg.dataset.alphas,
        horizon=cfg.dataset.horizon,
        normalize=cfg.dataset.normalize,
        min_history=cfg.dataset.min_history,
        label_lag=cfg.dataset.label_lag,
        label_norm=cfg.dataset.label_norm,
    )

    cfg.dataset.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cfg.dataset.output)
    meta = {
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
    }
    write_json(cfg.dataset.output.with_suffix(".meta.json"), meta)

    logger.info(
        f"[features] rows={meta['n_rows']:,} dates={meta['n_dates']} "
        f"symbols={meta['n_symbols']} features={len(feature_columns(data))}"
    )
    logger.info(f"[features] wrote {cfg.dataset.output}")
    return data


def run_train(cfg: Config, models: tuple[str, ...] | None = None) -> pd.DataFrame:
    """训练多个模型，输出 IC、回测绩效与报告。"""
    data = pd.read_parquet(cfg.dataset.output)
    splits = split_by_date(data, cfg.split.ranges())
    features = feature_columns(data)
    for name, split in splits.items():
        if split.empty:
            raise ValueError(
                f"{name} 切分没有样本，检查 config 里的 split 日期与 data.start 是否重叠"
            )
        n_dates = split.index.get_level_values("date").nunique()
        logger.info(f"[data] {name:5s} rows={len(split):>7,} dates={n_dates:>4}")

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

    close = load_panel(cfg.data.output).close
    forward_returns = forward_return(
        close, horizon=cfg.backtest.rebalance_days, lag=cfg.dataset.label_lag
    )

    summary: list[dict[str, float | str]] = []
    sensitivities: dict[str, pd.Series] = {}
    for name in models or cfg.train.models:
        logger.info(f"=== {name} ===")
        row, sensitivity = _train_one(cfg, name, splits, features, forward_returns)
        summary.append(row)
        sensitivities[name] = sensitivity

    summary_frame = pd.DataFrame(summary)
    logger.info("[summary]\n" + summary_frame.round(4).to_string(index=False))
    report_path = cfg.train.artifacts / "report.md"
    write_report(report_path, summary_frame, alpha_ic, splits, sensitivities, cfg)
    logger.info(f"[report] wrote {report_path}")
    return summary_frame


def _train_one(
    cfg: Config,
    name: str,
    splits: dict[str, pd.DataFrame],
    features: list[str],
    forward_returns: pd.DataFrame,
) -> tuple[dict[str, float | str], pd.Series]:
    """训练单个模型并落盘预测、绩效与因子重要性，返回汇总行与成本敏感度。"""
    train, valid, test = splits["train"], splits["valid"], splits["test"]
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

    predictions = {
        "valid": _predict_frame(model, valid, features),
        "test": _predict_frame(model, test, features),
    }
    ic = {
        split: daily_rank_ic(frame["prediction"], frame[LABEL])
        for split, frame in predictions.items()
    }
    backtest = _backtest(predictions["test"], forward_returns, cfg)

    output_dir = cfg.train.artifacts / name
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, names=["split"]).to_parquet(output_dir / "predictions.parquet")
    backtest.period.to_csv(output_dir / "backtest_periods.csv")
    backtest.holdings.to_csv(output_dir / "holdings.csv")

    valid_ic, test_ic = ic_summary(ic["valid"]), ic_summary(ic["test"])
    avg_turnover = float(backtest.period["turnover"].mean())
    write_json(
        output_dir / "metrics.json",
        {
            "model": name,
            "valid": valid_ic,
            "test": test_ic,
            "strategy": backtest.strategy,
            "benchmark": backtest.benchmark,
            "avg_turnover": avg_turnover,
        },
    )

    logger.info(
        f"  valid IC={valid_ic['ic_mean']:+.4f} ICIR={valid_ic['icir']:+.2f} | "
        f"test IC={test_ic['ic_mean']:+.4f} ICIR={test_ic['icir']:+.2f}"
    )
    logger.info(
        f"  test top{cfg.backtest.top_k}: CAGR={backtest.strategy['cagr']:+.2%} "
        f"Sharpe={backtest.strategy['sharpe']:+.2f} MDD={backtest.strategy['max_drawdown']:.2%} | "
        f"benchmark CAGR={backtest.benchmark['cagr']:+.2%}"
    )

    importance = getattr(model, "feature_importance_", None)
    if importance is not None:
        ranking = pd.Series(importance.to_numpy(), index=features).sort_values(ascending=False)
        ranking.to_csv(output_dir / "feature_importance.csv")
        logger.info(
            "  top factors: " + ", ".join(f"{k}={v:.4g}" for k, v in ranking.head(5).items())
        )

    sensitivity = cost_sensitivity(backtest.period, cfg.backtest.rebalance_days)
    sensitivity.to_csv(output_dir / "cost_sensitivity.csv", header=["cagr"])
    logger.info(
        "  CAGR under cost "
        + ", ".join(f"{bps}bp={value:+.1%}" for bps, value in sensitivity.items())
    )

    row: dict[str, float | str] = {
        "model": name,
        "valid_ic": valid_ic["ic_mean"],
        "valid_icir": valid_ic["icir"],
        "test_ic": test_ic["ic_mean"],
        "test_icir": test_ic["icir"],
        "test_cagr": backtest.strategy["cagr"],
        "test_sharpe": backtest.strategy["sharpe"],
        "test_maxdd": backtest.strategy["max_drawdown"],
        "bench_cagr": backtest.benchmark["cagr"],
        "turnover": avg_turnover,
    }
    return row, sensitivity


def _matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    return data[features].to_numpy(dtype="float64")


def _predict_frame(model: Regressor, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({LABEL: data[LABEL]}, index=data.index)
    frame["prediction"] = model.predict(_matrix(data, features))
    return frame


def _backtest(
    predictions: pd.DataFrame, forward_returns: pd.DataFrame, cfg: Config
) -> BacktestResult:
    wide = predictions["prediction"].unstack("symbol")
    return backtest_topk(
        wide,
        forward_returns,
        top_k=cfg.backtest.top_k,
        rebalance_days=cfg.backtest.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        seed=cfg.train.seed,
    )
