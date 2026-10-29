"""项目配置：从 YAML 读取，相对路径统一相对仓库根目录解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"

# models.<name> 下允许出现的超参。写错键名（比如 n_estimator）会在这里直接报错，
# 而不是等到 fit 的时候被第三方库静默忽略。要加新参数，先补到这个表里。
KNOWN_MODEL_PARAMS: dict[str, frozenset[str]] = {
    "xgboost": frozenset(
        {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "reg_lambda",
            "reg_alpha",
            "gamma",
        }
    ),
    "lightgbm": frozenset(
        {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "min_split_gain",
        }
    ),
    "mlp": frozenset(
        {
            "hidden",
            "dropout",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "device",
        }
    ),
}


def _validate_models(train_models: tuple[str, ...], models: dict[str, dict[str, Any]]) -> None:
    """训练前就拦住配置错误：模型没配参数、参数名拼错、MLP 缺 hidden。"""
    for name in train_models:
        if name not in KNOWN_MODEL_PARAMS:
            raise ValueError(f"未知模型 {name}，可选：{sorted(KNOWN_MODEL_PARAMS)}")
        if name not in models:
            raise ValueError(f"train.models 里有 {name}，但 models 段没有对应的参数配置")

        unknown = sorted(set(models[name]) - KNOWN_MODEL_PARAMS[name])
        if unknown:
            raise ValueError(
                f"models.{name} 里有不认识的参数 {unknown}；"
                f"确实要用的话先在 config.py 的 KNOWN_MODEL_PARAMS 里登记"
            )
    if "mlp" in train_models and "hidden" not in models.get("mlp", {}):
        raise ValueError("models.mlp 必须指定 hidden，例如 [128, 64]")


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
    corporate_actions: Path


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
    models: dict[str, dict[str, Any]]
    backtest: BacktestConfig

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        train_models = tuple(raw["train"]["models"])
        models = dict(raw["models"])
        _validate_models(train_models, models)
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
                corporate_actions=_path(raw["data"]["corporate_actions"]),
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
                models=train_models,
                early_stopping_rounds=int(raw["train"]["early_stopping_rounds"]),
                artifacts=_path(raw["train"]["artifacts"]),
            ),
            models=models,
            backtest=BacktestConfig(
                top_k=int(raw["backtest"]["top_k"]),
                rebalance_days=int(raw["backtest"]["rebalance_days"]),
                cost_bps=float(raw["backtest"]["cost_bps"]),
            ),
        )
