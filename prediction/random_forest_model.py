"""Random Forest 예측 모델 — scikit-learn 기반 Random Forest 모델 구현.

Requirements: 5.2, 5.3, 5.4
"""

import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from prediction.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """scikit-learn 기반 Random Forest 예측 모델.

    Parameters
    ----------
    feature_names : list[str] | None
        피처 이름 리스트.
    n_estimators : int
        트리 수 (기본 100).
    max_depth : int | None
        최대 깊이 (기본 None, 제한 없음).
    random_state : int
        랜덤 시드 (기본 42).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(feature_names)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: RandomForestRegressor | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Random Forest 모델 학습."""
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._is_trained = True
        logger.info("Random Forest 모델 학습 완료: %d 샘플, %d 피처", len(X), X.shape[1])

    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        """Random Forest 모델로 예측 수행."""
        if timeframe not in ("short", "mid", "long"):
            raise ValueError(f"유효하지 않은 timeframe: {timeframe}")
        if not self._is_trained or self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        X_pred = X[-1:] if X.ndim == 2 else X.reshape(1, -1)
        predicted_price = float(self._model.predict(X_pred)[0])

        # 방향 확률: 개별 트리 예측의 분포로 계산 — 현재 종가(close) 기준
        close_idx = self.feature_names.index("close") if "close" in self.feature_names else 0
        last_price = float(X[-1, close_idx])
        tree_predictions = np.array([
            tree.predict(X_pred)[0] for tree in self._model.estimators_
        ])
        up_count = (tree_predictions > last_price).sum()
        total_trees = len(tree_predictions)

        up_probability = round((up_count / total_trees) * 100, 2)
        down_probability = round(100.0 - up_probability, 2)

        # 신뢰도: 트리 예측의 일관성 (표준편차가 작을수록 높음)
        pred_std = tree_predictions.std()
        pred_mean = abs(tree_predictions.mean()) if tree_predictions.mean() != 0 else 1.0
        cv = pred_std / max(pred_mean, 1e-8)  # 변동계수
        confidence = min(100.0, max(0.0, (1.0 - cv) * 100))
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
                "random_state": self.random_state,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Random Forest 모델 저장: %s", path)

    def load(self, path: str) -> None:
        """저장된 모델 로드."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._model = data["model"]
        self.feature_names = data.get("feature_names", [])
        config = data.get("config", {})
        self.n_estimators = config.get("n_estimators", self.n_estimators)
        self.max_depth = config.get("max_depth", self.max_depth)
        self.random_state = config.get("random_state", self.random_state)
        self._is_trained = True
        logger.info("Random Forest 모델 로드: %s", path)

    def get_feature_importance(self) -> dict[str, float]:
        """Random Forest 내장 피처 기여도 반환. 합 ≈ 1.0."""
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
