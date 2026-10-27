"""训练产物落盘：JSON 元数据与给团队看的 markdown 报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hkq.config import Config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")


def _json_default(value: object) -> object:
    """把 numpy 标量转成原生类型，方便写 JSON。"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"无法序列化 {type(value)}")


def write_report(
    path: Path,
    summary: pd.DataFrame,
    alpha_ic: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
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
        markdown_table(sizes),
        "",
        "## 因子 Rank IC（训练集）",
        "",
        markdown_table(_with_integer_days(alpha_ic)),
        "",
        "## 模型表现",
        "",
        markdown_table(summary.set_index("model")),
        "",
        "## 成本敏感度（年化收益）",
        "",
        markdown_table(_cost_table(sensitivities)),
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


def _format_cell(value: object) -> str:
    """整数不加小数位，浮点保留 4 位，其余按字符串输出。"""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """把 DataFrame 渲染成 markdown 表格（不额外依赖 tabulate）。"""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join([str(frame.index.name or ""), *columns]) + " |",
        "|" + "|".join(["---"] * (len(columns) + 1)) + "|",
    ]
    for index, row in frame.iterrows():
        cells = [_format_cell(value) for value in row]
        lines.append("| " + " | ".join([str(index), *cells]) + " |")
    return "\n".join(lines)
