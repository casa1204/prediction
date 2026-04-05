"""예측 모델 기본 인터페이스 — 모든 Individual_Model이 구현해야 하는 공통 인터페이스.

PredictionResult 데이터클래스와 BaseModel 추상 클래스를 정의한다.

Requirements: 5.2, 5.3, 5.4, 5.10
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictionResult:
    """개별 모델의 예측 결과를 담는 데이터클래스.

    Attributes
    ----------
    predicted_price : float
        예측 가격.
    up_probability : float
        상승 확률 (0~100). down_probability와 합이 100.
    down_probability : float
        하락 확률 (0~100). up_probability와 합이 100.
    confidence : float
        신뢰도 (0~100).
    timeframe : str
        예측 타임프레임 ("short", "mid", "long").
    feature_importance : dict[str, float]
        피처별 기여도 (합 ≈ 1.0).
    """

    predicted_price: float
    up_probability: float
    down_probability: float
    confidence: float
    timeframe: str
    feature_importance: dict[str, float] = field(default_factory=dict)


class BaseModel(ABC):
    """모든 Individual_Model이 구현해야 하는 추상 기본 클래스.

    각 모델은 train, predict, save, load를 구현하고,
    get_feature_importance로 피처 기여도를 반환한다.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names: list[str] = feature_names or []
        self._is_trained: bool = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """모델 학습.

        Parameters
        ----------
        X : np.ndarray
            피처 매트릭스 (n_samples, n_features).
        y : np.ndarray
            타겟 벡터 (다음 날 종가).
        """

    @abstractmethod
    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        """예측 수행.

        Parameters
        ----------
        X : np.ndarray
            피처 매트릭스.
        timeframe : str
            "short", "mid", "long" 중 하나.

        Returns
        -------
        PredictionResult
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """모델을 파일로 저장."""

    @abstractmethod
    def load(self, path: str) -> None:
        """저장된 모델 로드."""

    def get_feature_importance(self) -> dict[str, float]:
        """피처 기여도 반환. 합 ≈ 1.0.

        기본 구현은 균등 분배. 서브클래스에서 오버라이드 가능.
        """
        if not self.feature_names:
            return {}
        n = len(self.feature_names)
        equal_weight = 1.0 / n
        return {name: equal_weight for name in self.feature_names}
