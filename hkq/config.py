"""项目配置：从 YAML 读取，相对路径统一相对仓库根目录解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _path(value: str | Path) -> Path:
    """相对路径按仓库根目录解析，绝对路径原样返回。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class UniverseConfig:
    training_tickers: Path
    trading_tickers: Path


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    schema: str
    start: str
    end: str | None
    output: Path


@dataclass(frozen=True)
class DatasetConfig:
    alphas: tuple[str, ...]
    horizon: int
    label_lag: int
    label_norm: str
    normalize: str
    min_history: int
    output: Path


@dataclass(frozen=True)
class SplitConfig:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    def ranges(self) -> dict[str, tuple[str, str]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    models: tuple[str, ...]
    early_stopping_rounds: int
    artifacts: Path


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int
    rebalance_days: int
    cost_bps: float


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig
    data: DataConfig
    dataset: DatasetConfig
    split: SplitConfig
    train: TrainConfig
    models: dict[str, dict]
    backtest: BacktestConfig

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            universe=UniverseConfig(
                training_tickers=_path(raw["universe"]["training_tickers"]),
                trading_tickers=_path(raw["universe"]["trading_tickers"]),
            ),
            data=DataConfig(
                dataset=raw["data"]["dataset"],
                schema=raw["data"]["schema"],
                start=str(raw["data"]["start"]),
                end=None if raw["data"]["end"] is None else str(raw["data"]["end"]),
                output=_path(raw["data"]["output"]),
            ),
            dataset=DatasetConfig(
                alphas=tuple(raw["dataset"]["alphas"]),
                horizon=int(raw["dataset"]["horizon"]),
                label_lag=int(raw["dataset"]["label_lag"]),
                label_norm=raw["dataset"]["label_norm"],
                normalize=raw["dataset"]["normalize"],
                min_history=int(raw["dataset"]["min_history"]),
                output=_path(raw["dataset"]["output"]),
            ),
            split=SplitConfig(
                train=tuple(raw["split"]["train"]),
                valid=tuple(raw["split"]["valid"]),
                test=tuple(raw["split"]["test"]),
            ),
            train=TrainConfig(
                seed=int(raw["train"]["seed"]),
                models=tuple(raw["train"]["models"]),
                early_stopping_rounds=int(raw["train"]["early_stopping_rounds"]),
                artifacts=_path(raw["train"]["artifacts"]),
            ),
            models=dict(raw["models"]),
            backtest=BacktestConfig(
                top_k=int(raw["backtest"]["top_k"]),
                rebalance_days=int(raw["backtest"]["rebalance_days"]),
                cost_bps=float(raw["backtest"]["cost_bps"]),
            ),
        )
