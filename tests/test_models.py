"""模型：未 fit 时报错，以及不给验证集时也能训练。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from hkq.models import MLPModel, build_model

PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {"n_estimators": 5},
    "lightgbm": {"n_estimators": 5},
    "mlp": {"hidden": [4], "max_epochs": 2},
}


@pytest.mark.parametrize("name", list(PARAMS))
def test_predict_before_fit_raises(name: str) -> None:
    model = build_model(name, PARAMS[name], seed=0)

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 3)))


@pytest.mark.parametrize("name", list(PARAMS))
def test_fit_without_validation_set(name: str) -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), rng.normal(size=200)
    model = build_model(name, PARAMS[name], seed=0)

    model.fit(X, y)

    assert model.predict(X).shape == (200,)


def test_mlp_logs_early_stop(caplog: pytest.LogCaptureFixture) -> None:
    # 训练目标是 +1、验证目标是 -1：越训练验证损失越大，第一轮之后就该早停
    X = np.random.default_rng(0).normal(size=(64, 3))
    model = MLPModel(hidden=(4,), lr=1e-2, max_epochs=50, patience=1, device="cpu")

    with caplog.at_level(logging.INFO, logger="hkq.models"):
        model.fit(X, np.ones(64), X, -np.ones(64))

    assert any("MLP early stop" in record.message for record in caplog.records)


def test_mlp_corr_early_stop_keeps_the_best_ranking_epoch() -> None:
    # 训练标签和 x0 正相关、验证标签和 x0 负相关：越训练验证集相关性越差，应该保留最早那一轮
    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 3))
    model = MLPModel(
        hidden=(8,), lr=5e-2, max_epochs=40, patience=3, device="cpu", stop_on="corr", seed=0
    )

    model.fit(X, X[:, 0], X, -X[:, 0])

    late = MLPModel(hidden=(8,), lr=5e-2, max_epochs=40, patience=40, device="cpu", seed=0)
    late.fit(X, X[:, 0])
    corr_kept = np.corrcoef(model.predict(X), -X[:, 0])[0, 1]
    corr_late = np.corrcoef(late.predict(X), -X[:, 0])[0, 1]
    assert corr_kept > corr_late


def test_mlp_rejects_unknown_stop_metric() -> None:
    with pytest.raises(ValueError, match="stop_on"):
        MLPModel(hidden=(4,), stop_on="ic", device="cpu").fit(np.zeros((8, 2)), np.zeros(8))
