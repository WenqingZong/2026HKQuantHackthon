"""流水线：下载行情 -> 生成因子数据集 -> 训练与评估。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hkq.config import Config
from hkq.data import download_bars, load_panel, read_tickers
from hkq.dataset import LABEL, build_dataset, feature_columns, forward_return, split_by_date
from hkq.evaluate import BacktestResult, backtest_topk, daily_rank_ic, ic_summary, performance
from hkq.models import build_model


def run_download(cfg: Config) -> pd.DataFrame:
    """下载训练池与交易池的日频行情。"""
    training = read_tickers(cfg.universe.training_tickers)
    trading = read_tickers(cfg.universe.trading_tickers)
    symbols = sorted(set(training) | set(trading))
    print(f"[download] training={len(training)} trading={len(trading)} union={len(symbols)}")

    bars = download_bars(
        dataset=cfg.data.dataset,
        schema=cfg.data.schema,
        symbols=symbols,
        start=cfg.data.start,
        end=cfg.data.end,
        output=cfg.data.output,
    )
    print(
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
    _write_json(cfg.dataset.output.with_suffix(".meta.json"), meta)

    print(
        f"[features] rows={meta['n_rows']:,} dates={meta['n_dates']} "
        f"symbols={meta['n_symbols']} features={len(feature_columns(data))}"
    )
    print(f"[features] wrote {cfg.dataset.output}")
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
        print(f"[data] {name:5s} rows={len(split):>7,} dates={n_dates:>4}")

    if cfg.backtest.rebalance_days != cfg.dataset.horizon:
        print(
            f"[warn] rebalance_days={cfg.backtest.rebalance_days} != horizon="
            f"{cfg.dataset.horizon}; the backtest holds a different period than the label"
        )

    alpha_ic = _alpha_ic_table(splits["train"], features)
    cfg.train.artifacts.mkdir(parents=True, exist_ok=True)
    alpha_ic.to_csv(cfg.train.artifacts / "alpha_ic.csv")
    print("\n[factors] rank IC on the train split (sorted by ic_mean):")
    print(alpha_ic[["ic_mean", "icir", "ic_win_rate", "n_days"]].round(4).to_string())

    close = load_panel(cfg.data.output).close
    forward_returns = forward_return(
        close, horizon=cfg.backtest.rebalance_days, lag=cfg.dataset.label_lag
    )

    summary, sensitivities = [], {}
    for name in models or cfg.train.models:
        print(f"\n=== {name} ===")
        row, sensitivity = _train_one(cfg, name, splits, features, forward_returns)
        summary.append(row)
        sensitivities[name] = sensitivity

    summary_frame = pd.DataFrame(summary)
    print("\n[summary]")
    print(summary_frame.round(4).to_string(index=False))
    _write_report(
        cfg.train.artifacts / "report.md", summary_frame, alpha_ic, splits, sensitivities, cfg
    )
    print(f"\n[report] wrote {cfg.train.artifacts / 'report.md'}")
    return summary_frame


def _train_one(
    cfg: Config,
    name: str,
    splits: dict[str, pd.DataFrame],
    features: list[str],
    forward_returns: pd.DataFrame,
) -> dict:
    """训练单个模型并落盘预测、绩效与因子重要性。"""
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

    metrics = {
        "model": name,
        "valid": ic_summary(ic["valid"]),
        "test": ic_summary(ic["test"]),
        "strategy": backtest.strategy,
        "benchmark": backtest.benchmark,
        "avg_turnover": float(backtest.period["turnover"].mean()),
    }
    _write_json(output_dir / "metrics.json", metrics)

    print(
        f"  valid IC={metrics['valid']['ic_mean']:+.4f} ICIR={metrics['valid']['icir']:+.2f} | "
        f"test IC={metrics['test']['ic_mean']:+.4f} ICIR={metrics['test']['icir']:+.2f}"
    )
    print(
        f"  test top{cfg.backtest.top_k}: CAGR={backtest.strategy['cagr']:+.2%} "
        f"Sharpe={backtest.strategy['sharpe']:+.2f} MDD={backtest.strategy['max_drawdown']:.2%} | "
        f"benchmark CAGR={backtest.benchmark['cagr']:+.2%}"
    )

    importance = getattr(model, "feature_importance_", None)
    if importance is not None:
        ranking = pd.Series(importance.to_numpy(), index=features).sort_values(ascending=False)
        ranking.to_csv(output_dir / "feature_importance.csv")
        print("  top factors: " + ", ".join(f"{k}={v:.4g}" for k, v in ranking.head(5).items()))

    sensitivity = _cost_sensitivity(backtest.period, cfg.backtest.rebalance_days)
    sensitivity.to_csv(output_dir / "cost_sensitivity.csv", header=["cagr"])
    print(
        "  CAGR under cost "
        + ", ".join(f"{bps}bp={value:+.1%}" for bps, value in sensitivity.items())
    )

    row = {
        "model": name,
        "valid_ic": metrics["valid"]["ic_mean"],
        "valid_icir": metrics["valid"]["icir"],
        "test_ic": metrics["test"]["ic_mean"],
        "test_icir": metrics["test"]["icir"],
        "test_cagr": backtest.strategy["cagr"],
        "test_sharpe": backtest.strategy["sharpe"],
        "test_maxdd": backtest.strategy["max_drawdown"],
        "bench_cagr": backtest.benchmark["cagr"],
        "turnover": metrics["avg_turnover"],
    }
    return row, sensitivity


def _matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    return data[features].to_numpy(dtype="float64")


def _predict_frame(model, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
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


_COST_LEVELS_BPS = (0, 5, 10, 20, 30)


def _cost_sensitivity(period: pd.DataFrame, rebalance_days: int) -> pd.Series:
    """不同单边成本假设下的年化收益，用来看策略吃不吃交易成本。

    回测已经把每期的毛收益和换手存下来了，这里只是换个数重算一遍，不用重新训练。
    """
    periods_per_year = 252.0 / rebalance_days
    return pd.Series(
        {
            bps: performance(
                period["gross_return"] - period["turnover"] * bps / 1e4, periods_per_year
            )["cagr"]
            for bps in _COST_LEVELS_BPS
        },
        name="cagr",
    )


def _alpha_ic_table(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """逐因子计算 Rank IC 统计，作为后续因子筛选的基线。"""
    rows = {name: ic_summary(daily_rank_ic(train[name], train[LABEL])) for name in features}
    table = pd.DataFrame(rows).T
    table["n_days"] = table["n_days"].astype(int)
    table.index.name = "factor"
    return table.sort_values("ic_mean", ascending=False)


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")


def _json_default(value):
    """把 numpy 标量转成原生类型，方便写 JSON。"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"无法序列化 {type(value)}")


def _write_report(
    path,
    summary: pd.DataFrame,
    alpha_ic: pd.DataFrame,
    splits,
    sensitivities: dict[str, pd.Series],
    cfg: Config,
) -> None:
    """生成一份给团队看的训练报告。"""
    sizes = pd.DataFrame(
        {
            "交易日": [split.index.get_level_values("date").nunique() for split in splits.values()],
            "样本数": [len(split) for split in splits.values()],
            "起": [
                str(split.index.get_level_values("date").min().date()) for split in splits.values()
            ],
            "止": [
                str(split.index.get_level_values("date").max().date()) for split in splits.values()
            ],
        },
        index=list(splits),
    )
    sizes.index.name = "split"
    lines = [
        "# 训练报告",
        "",
        f"- 数据源：{cfg.data.dataset} / {cfg.data.schema}",
        f"- 因子：{', '.join(cfg.dataset.alphas)}",
        (
            f"- 标签：{cfg.dataset.label_lag} 个交易日后进场、"
            f"持有 {cfg.dataset.horizon} 个交易日的收益，"
            f"按横截面 {cfg.dataset.label_norm} 标准化"
        ),
        f"- 因子标准化：横截面 {cfg.dataset.normalize}",
        (
            f"- 回测：每 {cfg.backtest.rebalance_days} 个交易日调仓，"
            f"等权持有前 {cfg.backtest.top_k} 只，单边成本 {cfg.backtest.cost_bps} bp"
        ),
        "",
        "## 数据切分",
        "",
        _markdown_table(sizes),
        "",
        "## 因子 Rank IC（训练集）",
        "",
        _markdown_table(_with_integer_days(alpha_ic)),
        "",
        "## 模型表现",
        "",
        _markdown_table(summary.set_index("model")),
        "",
        "## 成本敏感度（年化收益）",
        "",
        _markdown_table(_cost_table(sensitivities)),
        "",
        (
            f"回测默认单边成本 {cfg.backtest.cost_bps} bp（手续费 + 滑点），按每期换手计提；"
            "基准为全池等权、不计成本。"
        ),
        (
            "test_cagr 为测试集上的年化收益，bench_cagr 为同期全池等权基准；"
            "turnover 为每次调仓的单边换手率。"
        ),
        "样本期较短，结论仅供流水线验证。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _with_integer_days(table: pd.DataFrame) -> pd.DataFrame:
    """n_days 是整数，转成字符串避免在表格里显示成 228.0000。"""
    view = table[["ic_mean", "ic_std", "icir", "ic_win_rate", "n_days"]].copy()
    view["n_days"] = view["n_days"].astype(int).astype(str)
    return view


def _cost_table(sensitivities: dict[str, pd.Series]) -> pd.DataFrame:
    table = pd.DataFrame(sensitivities)
    table.index.name = "单边成本(bp)"
    return table.map(lambda value: f"{value:.2%}")


def _format_cell(value) -> str:
    """整数不加小数位，浮点保留 4 位，其余按字符串输出。"""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    """把 DataFrame 渲染成 markdown 表格（不额外依赖 tabulate）。"""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join([frame.index.name or "", *columns]) + " |",
        "|" + "|".join(["---"] * (len(columns) + 1)) + "|",
    ]
    for index, row in frame.iterrows():
        cells = [_format_cell(value) for value in row]
        lines.append("| " + " | ".join([str(index), *cells]) + " |")
    return "\n".join(lines)
