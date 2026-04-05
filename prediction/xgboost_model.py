"""XGBoost 예측 모델 — XGBoost 기반 모델 구현.

Requirements: 5.2, 5.3, 5.4
"""

import logging
import os
import pickle

import numpy as np
import xgboost as xgb

from prediction.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost 기반 예측 모델.

    Parameters
    ----------
    feature_names : list[str] | None
        피처 이름 리스트.
    n_estimators : int
        트리 수 (기본 100).
    max_depth : int
        최대 깊이 (기본 6).
    learning_rate : float
        학습률 (기본 0.1).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        super().__init__(feature_names)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: xgb.XGBRegressor | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """XGBoost 모델 학습."""
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._is_trained = True
        logger.info("XGBoost 모델 학습 완료: %d 샘플, %d 피처", len(X), X.shape[1])

    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        """XGBoost 모델로 예측 수행."""
        if timeframe not in ("short", "mid", "long"):
            raise ValueError(f"유효하지 않은 timeframe: {timeframe}")
        if not self._is_trained or self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        # 마지막 행으로 예측
        X_pred = X[-1:] if X.ndim == 2 else X.reshape(1, -1)
        predicted_price = float(self._model.predict(X_pred)[0])

        # 방향 확률 계산: 현재 종가(close) vs 예측 가격
        close_idx = self.feature_names.index("close") if "close" in self.feature_names else 0
        last_price = float(X[-1, close_idx])
        price_change_ratio = (predicted_price - last_price) / max(abs(last_price), 1e-8)

        up_prob = 1.0 / (1.0 + np.exp(-price_change_ratio * 100))
        up_probability = round(up_prob * 100, 2)
        down_probability = round(100.0 - up_probability, 2)

        confidence = min(100.0, max(0.0, abs(price_change_ratio) * 500))
        confidence = round(confidence, 2)

        return PredictionResult(
            predicted_price=round(predicted_price, 6),
            up_probability=up_probability,
            down_probability=down_probability,
            confidence=confidence,
            timeframe=timeframe,
            feature_importance=self.get_feature_importance(),
        )

    def save(self, path: str) -> None:
        """모델을 파일로 저장."""
        if self._model is None:
            raise RuntimeError("저장할 모델이 없습니다.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "model": self._model,
            "feature_names": self.feature_names,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("XGBoost 모델 저장: %s", path)

    def load(self, path: str) -> None:
        """저장된 모델 로드."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._model = data["model"]
        self.feature_names = data.get("feature_names", [])
        config = data.get("config", {})
        self.n_estimators = config.get("n_estimators", self.n_estimators)
        self.max_depth = config.get("max_depth", self.max_depth)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self._is_trained = True
        logger.info("XGBoost 모델 로드: %s", path)

    def get_feature_importance(self) -> dict[str, float]:
        """XGBoost 내장 피처 기여도 반환. 합 ≈ 1.0."""
        if self._model is None or not self._is_trained:
            return super().get_feature_importance()

        raw_importance = self._model.feature_importances_
        total = raw_importance.sum()
        if total == 0:
            return super().get_feature_importance()

        normalized = raw_importance / total
        result = {}
        for i, name in enumerate(self.feature_names):
            if i < len(normalized):
                result[name] = float(normalized[i])
        return result
