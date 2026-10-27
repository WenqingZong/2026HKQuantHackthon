"""三个基学习器：XGBoost、LightGBM、MLP。

统一接口（训练脚本只依赖这三个方法）::

    model.fit(X_train, y_train, X_valid, y_valid)
    model.predict(X)

X / y 都是 numpy 数组；X_valid 为空时关闭早停。
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor, early_stopping
from torch import nn
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class Regressor(Protocol):
    """训练脚本用到的最小接口。"""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class XGBoostModel:
    """XGBoost 回归器：把未来收益当连续值回归，按预测值排序选股。"""

    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 50
    seed: int = 42
    feature_importance_: pd.Series | None = field(default=None, init=False, repr=False)
    _model: XGBRegressor | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> None:
        params = {
            "objective": "reg:squarederror",
            "random_state": self.seed,
            "n_jobs": -1,
            **self.params,
        }
        if X_valid is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = XGBRegressor(**params)
        eval_set = [(X_valid, y_valid)] if X_valid is not None else None
        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        self.feature_importance_ = pd.Series(self._model.feature_importances_, name="importance")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型还没有 fit")
        return np.asarray(self._model.predict(X))


@dataclass
class LightGBMModel:
    """LightGBM 回归器，参数风格与 XGBoost 保持一致。"""

    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 50
    seed: int = 42
    feature_importance_: pd.Series | None = field(default=None, init=False, repr=False)
    _model: LGBMRegressor | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> None:
        self._model = LGBMRegressor(
            objective="regression",
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
            **self.params,
        )
        if X_valid is None or y_valid is None:
            self._model.fit(X, y)
        else:
            self._model.fit(
                X,
                y,
                eval_X=(X_valid,),
                eval_y=(y_valid,),
                callbacks=[early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        self.feature_importance_ = pd.Series(self._model.feature_importances_, name="importance")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型还没有 fit")
        return np.asarray(self._model.predict(X))


@dataclass
class MLPModel:
    """多层感知机回归器（PyTorch），按验证集损失做早停。"""

    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 300
    patience: int = 30
    seed: int = 42
    device: str = "auto"
    _net: nn.Module | None = field(default=None, init=False, repr=False)

    def _resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_net(self, n_features: int) -> nn.Module:
        layers: list[nn.Module] = []
        width = n_features
        for hidden in self.hidden:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(self.seed)
        device = self._resolve_device()

        self._net = self._build_net(X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        x_train = torch.tensor(X, dtype=torch.float32, device=device)
        y_train = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        has_valid = X_valid is not None and len(X_valid) > 0
        if has_valid:
            x_valid = torch.tensor(X_valid, dtype=torch.float32, device=device)
            y_valid_t = torch.tensor(y_valid, dtype=torch.float32, device=device).view(-1, 1)

        best_loss, best_state, bad_epochs = math.inf, None, 0
        n_samples = len(x_train)
        for epoch in range(self.max_epochs):
            self._net.train()
            order = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._net(x_train[batch]), y_train[batch])
                loss.backward()
                optimizer.step()

            if not has_valid:
                continue

            self._net.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(self._net(x_valid), y_valid_t))
            if valid_loss < best_loss - 1e-9:
                best_loss, bad_epochs = valid_loss, 0
                best_state = copy.deepcopy(self._net.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    logger.info(
                        f"  MLP early stop at epoch {epoch + 1} (valid MSE {best_loss:.6f})"
                    )
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("模型还没有 fit")
        device = next(self._net.parameters()).device
        self._net.eval()
        with torch.no_grad():
            output = self._net(torch.tensor(X, dtype=torch.float32, device=device))
        return output.cpu().numpy().ravel()


def build_model(
    name: str,
    params: dict[str, Any],
    *,
    seed: int,
    early_stopping_rounds: int = 50,
) -> Regressor:
    """按名字创建模型，params 来自配置文件。"""
    if name == "xgboost":
        return XGBoostModel(params=params, early_stopping_rounds=early_stopping_rounds, seed=seed)
    if name == "lightgbm":
        return LightGBMModel(params=params, early_stopping_rounds=early_stopping_rounds, seed=seed)
    if name == "mlp":
        return MLPModel(seed=seed, **{**params, "hidden": tuple(params["hidden"])})
    raise ValueError(f"未知模型：{name}，可选：xgboost / lightgbm / mlp")
