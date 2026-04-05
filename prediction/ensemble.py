"""앙상블 엔진 — 개별 모델의 예측을 가중 평균/투표로 통합한다.

Requirements: 5.5, 5.6, 5.7, 5.8, 5.11
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from prediction.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """앙상블 통합 예측 결과.

    Attributes
    ----------
    final_price : float
        가중 평균 최종 예측 가격.
    final_direction : str
        가중 투표 최종 방향 ("up" or "down").
    integrated_up_probability : float
        통합 상승 확률 (0~100).
    integrated_down_probability : float
        통합 하락 확률 (0~100).
    individual_results : dict[str, PredictionResult]
        개별 모델 예측 결과.
    weights : dict[str, float]
        사용된 모델별 가중치.
    """

    final_price: float
    final_direction: str
    integrated_up_probability: float
    integrated_down_probability: float
    individual_results: dict[str, PredictionResult] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)


class EnsembleEngine:
    """개별 모델의 예측을 가중 평균/투표로 통합하는 앙상블 엔진.

    Parameters
    ----------
    models : dict[str, BaseModel]
        모델명 → BaseModel 인스턴스 매핑.
    """

    def __init__(self, models: dict[str, BaseModel]):
        self.models = models
        # 초기 가중치: 균등 분배
        n = len(models)
        if n > 0:
            equal_weight = 1.0 / n
            self.weights: dict[str, float] = {
                name: equal_weight for name in models
            }
        else:
            self.weights = {}

    def predict(self, X: np.ndarray, timeframe: str) -> EnsembleResult:
        """모든 Individual_Model 예측 후 가중 평균/투표로 통합.

        개별 모델 실패 시 해당 모델을 제외하고 나머지로 통합한다.
        모든 모델이 실패하면 RuntimeError를 발생시킨다.
        """
        if timeframe not in ("short", "mid", "long"):
            raise ValueError(f"유효하지 않은 timeframe: {timeframe}")

        individual_results: dict[str, PredictionResult] = {}

        # 개별 모델 예측 수행 (실패 시 제외)
        for name, model in self.models.items():
            try:
                result = model.predict(X, timeframe)
                individual_results[name] = result
            except Exception as e:
                logger.warning("모델 '%s' 예측 실패, 앙상블에서 제외: %s", name, e)

        if not individual_results:
            raise RuntimeError("모든 모델이 예측에 실패했습니다.")

        # 성공한 모델의 가중치만 추출하고 정규화
        active_weights = {
            name: self.weights.get(name, 0.0)
            for name in individual_results
        }
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            # 가중치가 모두 0이면 균등 분배
            equal_w = 1.0 / len(active_weights)
            active_weights = {name: equal_w for name in active_weights}
            total_weight = 1.0

        # 가중 평균 최종 예측 가격
        final_price = sum(
            active_weights[name] * res.predicted_price
            for name, res in individual_results.items()
        ) / total_weight

        # 가중 투표로 최종 방향 결정
        final_direction = self.weighted_vote(individual_results)

        # 통합 상승/하락 확률 (가중 평균)
        integrated_up = sum(
            active_weights[name] * res.up_probability
            for name, res in individual_results.items()
        ) / total_weight
        integrated_down = 100.0 - integrated_up

        return EnsembleResult(
            final_price=round(final_price, 6),
            final_direction=final_direction,
            integrated_up_probability=round(integrated_up, 2),
            integrated_down_probability=round(integrated_down, 2),
            individual_results=individual_results,
            weights=dict(active_weights),
        )

    def weighted_vote(self, results: dict[str, PredictionResult]) -> str:
        """가중 투표로 최종 방향(up/down) 결정.

        각 모델의 예측 방향에 해당 모델의 가중치를 곱하여 합산하고,
        가중치 합이 더 큰 방향을 최종 방향으로 결정한다.
        """
        up_weight = 0.0
        down_weight = 0.0

        for name, result in results.items():
            w = self.weights.get(name, 0.0)
            if result.up_probability > result.down_probability:
                up_weight += w
            elif result.down_probability > result.up_probability:
                down_weight += w
            else:
                # 동률이면 양쪽에 절반씩
                up_weight += w / 2
                down_weight += w / 2

        return "up" if up_weight >= down_weight else "down"

    def update_weights(self, accuracy_history: dict[str, float]) -> None:
        """과거 정확도 기반으로 Ensemble_Weight 동적 조정.

        정확도가 높은 모델에 더 높은 가중치를 부여한다.
        정확도가 50% 미만인 모델은 가중치를 하향 조정한다.
        모든 가중치의 합은 1.0이 된다.

        Parameters
        ----------
        accuracy_history : dict[str, float]
            모델명 → 정확도(0~100) 매핑.
        """
        if not accuracy_history:
            return

        # 이전 가중치 저장 (저성능 모델 하향 조정 비교용)
        old_weights = dict(self.weights)

        # 정확도를 가중치로 변환 (정확도가 0이면 최소값 부여)
        raw_weights: dict[str, float] = {}
        for name in self.models:
            acc = accuracy_history.get(name, 50.0)
            # 50% 미만 모델은 페널티 적용: 정확도의 제곱 / 100
            if acc < 50.0:
                raw_weights[name] = (acc * acc) / 10000.0  # 0~0.25 범위
            else:
                raw_weights[name] = acc / 100.0  # 0.5~1.0 범위

        total = sum(raw_weights.values())
        if total == 0:
            # 모든 정확도가 0이면 균등 분배
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models}
            return

        # 정규화하여 합 = 1.0
        self.weights = {
            name: raw_weights[name] / total
            for name in self.models
        }

        # 50% 미만 모델의 가중치가 이전보다 감소했는지 확인 (로깅)
        for name, acc in accuracy_history.items():
            if acc < 50.0 and name in old_weights and name in self.weights:
                if self.weights[name] < old_weights[name]:
                    logger.info(
                        "모델 '%s' 정확도 %.1f%% < 50%%: 가중치 %.4f → %.4f (하향 조정)",
                        name, acc, old_weights[name], self.weights[name],
                    )
