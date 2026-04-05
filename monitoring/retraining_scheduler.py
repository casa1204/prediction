"""재학습 스케줄러 — Champion/Challenger 패턴으로 모델 재학습을 관리한다.

Requirements: 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from db.models import RetrainingHistory
from prediction.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """모델 성능 지표."""

    mae: float
    mape: float
    direction_accuracy: float  # 0~100


@dataclass
class RetrainingRecord:
    """재학습 결과 레코드."""

    timestamp: datetime
    model_name: str
    method: str  # "incremental" or "full"
    data_period_start: datetime | None
    data_period_end: datetime | None
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics | None
    replaced: bool
    error_message: str | None = None


class RetrainingScheduler:
    """Champion/Challenger 패턴으로 모델 재학습을 관리한다.

    Parameters
    ----------
    session : Session | None
        SQLAlchemy DB 세션.
    champion_dir : str
        Champion 모델 저장 디렉토리.
    challenger_dir : str
        Challenger 모델 저장 디렉토리.
    """

    def __init__(
        self,
        session: Session | None = None,
        champion_dir: str = "models/champion",
        challenger_dir: str = "models/challenger",
    ):
        self.session = session
        self.champion_dir = champion_dir
        self.challenger_dir = challenger_dir

    # ── 재학습 ────────────────────────────────────────────────

    def retrain(
        self,
        model: BaseModel,
        method: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> BaseModel:
        """Incremental 또는 Full 방식으로 재학습하여 Challenger 모델 반환.

        Parameters
        ----------
        model : BaseModel
            기존 모델 (복사하여 재학습).
        method : str
            "incremental" or "full".
        X : np.ndarray
            학습 피처 데이터.
        y : np.ndarray
            학습 타겟 데이터.

        Returns
        -------
        BaseModel
            재학습된 Challenger 모델.

        Raises
        ------
        ValueError
            유효하지 않은 method.
        """
        if method not in ("incremental", "full"):
            raise ValueError(f"유효하지 않은 재학습 방식: {method}")

        model.train(X, y)
        return model

    # ── 성능 비교 ─────────────────────────────────────────────

    @staticmethod
    def compare_models(
        champion_metrics: ModelMetrics,
        challenger_metrics: ModelMetrics,
    ) -> bool:
        """Champion vs Challenger 성능 비교.

        Challenger의 모든 지표가 Champion보다 우수하면 True.
        - MAE: 낮을수록 좋음
        - MAPE: 낮을수록 좋음
        - direction_accuracy: 높을수록 좋음

        Parameters
        ----------
        champion_metrics : ModelMetrics
            Champion 모델 성능 지표.
        challenger_metrics : ModelMetrics
            Challenger 모델 성능 지표.

        Returns
        -------
        bool
            True면 Challenger가 우수하여 교체 대상.
        """
        mae_better = challenger_metrics.mae <= champion_metrics.mae
        mape_better = challenger_metrics.mape <= champion_metrics.mape
        dir_better = (
            challenger_metrics.direction_accuracy
            >= champion_metrics.direction_accuracy
        )
        return mae_better and mape_better and dir_better

    # ── 교체 로직 ─────────────────────────────────────────────

    def swap_if_better(
        self,
        model_name: str,
        champion_metrics: ModelMetrics,
        challenger_metrics: ModelMetrics,
    ) -> bool:
        """Challenger가 우수하면 교체, 아니면 Champion 유지.

        Parameters
        ----------
        model_name : str
            모델명.
        champion_metrics : ModelMetrics
            Champion 성능 지표.
        challenger_metrics : ModelMetrics
            Challenger 성능 지표.

        Returns
        -------
        bool
            True면 교체 수행됨.
        """
        should_replace = self.compare_models(champion_metrics, challenger_metrics)
        if should_replace:
            logger.info(
                "모델 '%s': Challenger가 Champion보다 우수 — 교체 수행",
                model_name,
            )
        else:
            logger.info(
                "모델 '%s': Champion 유지 (Challenger 성능 미달)",
                model_name,
            )
        return should_replace

    # ── 재학습 실행 + 이력 저장 ───────────────────────────────

    def run_retraining(
        self,
        model_name: str,
        champion: BaseModel,
        method: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        data_period_start: datetime | None = None,
        data_period_end: datetime | None = None,
        evaluate_fn=None,
    ) -> RetrainingRecord:
        """재학습 전체 파이프라인 실행.

        오류 발생 시 Champion 유지, 오류 로그 기록.

        Parameters
        ----------
        model_name : str
            모델명.
        champion : BaseModel
            현재 Champion 모델.
        method : str
            "incremental" or "full".
        X_train, y_train : np.ndarray
            학습 데이터.
        X_test, y_test : np.ndarray
            테스트 데이터.
        data_period_start, data_period_end : datetime | None
            학습 데이터 기간.
        evaluate_fn : callable | None
            모델 평가 함수. (model, X_test, y_test) -> ModelMetrics.

        Returns
        -------
        RetrainingRecord
            재학습 결과 레코드.
        """
        timestamp = datetime.now(timezone.utc)

        try:
            # Champion 성능 평가
            if evaluate_fn is None:
                raise ValueError("evaluate_fn이 필요합니다.")
            champion_metrics = evaluate_fn(champion, X_test, y_test)

            # Challenger 재학습
            challenger = self.retrain(champion, method, X_train, y_train)
            challenger_metrics = evaluate_fn(challenger, X_test, y_test)

            # 비교 및 교체
            replaced = self.swap_if_better(
                model_name, champion_metrics, challenger_metrics
            )

            record = RetrainingRecord(
                timestamp=timestamp,
                model_name=model_name,
                method=method,
                data_period_start=data_period_start,
                data_period_end=data_period_end,
                champion_metrics=champion_metrics,
                challenger_metrics=challenger_metrics,
                replaced=replaced,
                error_message=None,
            )

        except Exception as e:
            logger.error(
                "모델 '%s' 재학습 오류 — Champion 유지: %s", model_name, e
            )
            record = RetrainingRecord(
                timestamp=timestamp,
                model_name=model_name,
                method=method,
                data_period_start=data_period_start,
                data_period_end=data_period_end,
                champion_metrics=ModelMetrics(mae=0, mape=0, direction_accuracy=0),
                challenger_metrics=None,
                replaced=False,
                error_message=str(e),
            )

        # DB 저장
        self._save_history(record)
        return record

    # ── DB 저장 ───────────────────────────────────────────────

    def _save_history(self, record: RetrainingRecord) -> RetrainingHistory | None:
        """재학습 이력을 RetrainingHistory 테이블에 저장."""
        if self.session is None:
            return None

        challenger_mae = (
            record.challenger_metrics.mae if record.challenger_metrics else None
        )
        challenger_mape = (
            record.challenger_metrics.mape if record.challenger_metrics else None
        )
        challenger_dir_acc = (
            record.challenger_metrics.direction_accuracy
            if record.challenger_metrics
            else None
        )

        history = RetrainingHistory(
            timestamp=record.timestamp,
            model_name=record.model_name,
            method=record.method,
            data_period_start=record.data_period_start,
            data_period_end=record.data_period_end,
            champion_mae=record.champion_metrics.mae,
            champion_mape=record.champion_metrics.mape,
            champion_direction_accuracy=record.champion_metrics.direction_accuracy,
            challenger_mae=challenger_mae,
            challenger_mape=challenger_mape,
            challenger_direction_accuracy=challenger_dir_acc,
            replaced=record.replaced,
            error_message=record.error_message,
        )
        self.session.add(history)
        self.session.commit()
        return history

    def get_history(self, model_name: str | None = None) -> list[RetrainingHistory]:
        """재학습 이력 조회."""
        if self.session is None:
            raise RuntimeError("DB 세션이 설정되지 않았습니다.")

        query = self.session.query(RetrainingHistory)
        if model_name:
            query = query.filter(RetrainingHistory.model_name == model_name)
        return query.order_by(RetrainingHistory.timestamp.desc()).all()
