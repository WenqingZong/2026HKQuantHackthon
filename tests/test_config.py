"""配置校验：模型参数写错名字要在训练之前就报错。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from hkq.config import DEFAULT_CONFIG, Config

Mutate = Callable[[dict[str, Any]], None]


def load_with(tmp_path: Path, mutate: Mutate) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return Config.load(path)


def test_default_config_loads() -> None:
    cfg = Config.load()

    assert cfg.backtest.cost_bps == 1.5
    assert set(cfg.train.models) <= set(cfg.models)


def test_misspelled_model_param_is_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["models"]["xgboost"]["n_estimator"] = 100  # 少一个 s

    with pytest.raises(ValueError, match="不认识的参数"):
        load_with(tmp_path, mutate)


def test_model_without_params_block_is_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict) -> None:
        del raw["models"]["lightgbm"]

    with pytest.raises(ValueError, match="没有对应的参数配置"):
        load_with(tmp_path, mutate)


def test_mlp_requires_hidden(tmp_path: Path) -> None:
    def mutate(raw: dict) -> None:
        del raw["models"]["mlp"]["hidden"]

    with pytest.raises(ValueError, match="hidden"):
        load_with(tmp_path, mutate)
