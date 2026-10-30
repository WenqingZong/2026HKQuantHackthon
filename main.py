"""命令行入口。

用法::

    uv run python main.py download   # 从 Databento 下载日频行情
    uv run python main.py features   # 计算 Alpha101 因子并生成数据集
    uv run python main.py train      # 训练 MLP / XGBoost / LightGBM 并评估
    uv run python main.py all        # 一次跑完全流程
"""

from __future__ import annotations

import argparse
import logging

from hkq import pipeline
from hkq.config import DEFAULT_CONFIG, Config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ETF 日频因子 + 机器学习流水线")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="从 Databento 下载日频 OHLCV")
    subparsers.add_parser("features", help="计算 Alpha101 因子并生成数据集")

    subparsers.add_parser("select", help="按 IC 与相关性筛选因子")
    train_parser = subparsers.add_parser("train", help="训练并评估模型")
    train_parser.add_argument("--models", default=None, help="逗号分隔的模型名，默认读配置")
    subparsers.add_parser("all", help="依次执行 download -> features -> train")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.load(args.config)

    if args.command == "download":
        pipeline.run_download(cfg)
    elif args.command == "features":
        pipeline.run_features(cfg)
    elif args.command == "select":
        pipeline.run_select(cfg)
    elif args.command == "train":
        models = tuple(args.models.split(",")) if args.models else None
        pipeline.run_train(cfg, models)
    else:
        pipeline.run_download(cfg)
        pipeline.run_features(cfg)
        pipeline.run_train(cfg)


if __name__ == "__main__":
    main()
