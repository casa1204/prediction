"""적중률 추적 모듈 — 예측 결과와 실제 가격을 비교하여 적중률을 산출한다.

Requirements: 8.1, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from db.models import HitRateRecord

logger = logging.getLogger(__name__)

# 타임프레임별 허용 오차 (단기 ±3%, 중기 ±5%, 장기 ±10%)
TOLERANCE = {"short": 0.03, "mid": 0.05, "long": 0.10}


@dataclass
class HitRateResult:
    """적중률 산출 결과."""

    model_name: str
    timeframe: str
    direction_hit_rate: float  # 0~100
    range_hit_rate: float  # 0~100
    date: datetime


class HitRateTracker:
    """각 모델과 앙상블의 적중률을 추적하고 관리한다."""

    def __init__(self, session: Session | None = None):
        self.session = session

    # ── MAE / MAPE ────────────────────────────────────────────

    @staticmethod
    def calculate_mae(predictions: list[float], actuals: list[float]) -> float:
        """MAE = mean(|predicted - actual|).

        Parameters
        ----------
        predictions : list[float]
            예측 가격 리스트.
        actuals : list[float]
            실제 가격 리스트.

        Returns
        -------
        float
            MAE 값.
        """
        if not predictions or not actuals or len(predictions) != len(actuals):
            return 0.0
        n = len(predictions)
        return sum(abs(p - a) for p, a in zip(predictions, actuals)) / n

    @staticmethod
    def calculate_mape(predictions: list[float], actuals: list[float]) -> float:
        """MAPE = mean(|predicted - actual| / actual) × 100.

        actual이 0인 항목은 제외한다.

        Parameters
        ----------
        predictions : list[float]
            예측 가격 리스트.
        actuals : list[float]
            실제 가격 리스트.

        Returns
        -------
        float
            MAPE 값 (%).
        """
        if not predictions or not actuals or len(predictions) != len(actuals):
            return 0.0
        valid = [(p, a) for p, a in zip(predictions, actuals) if a != 0.0]
        if not valid:
            return 0.0
        return (sum(abs(p - a) / abs(a) for p, a in valid) / len(valid)) * 100.0

    # ── Direction Hit Rate ────────────────────────────────────

    @staticmethod
    def calculate_direction_hit_rate(
        predicted_directions: list[str],
        actual_directions: list[str],
    ) -> float:
        """방향 적중률 = (일치 건수 / 전체 건수) × 100.

        Parameters
        ----------
        predicted_directions : list[str]
            예측 방향 리스트 ("up" or "down").
        actual_directions : list[str]
            실제 방향 리스트 ("up" or "down").

        Returns
        -------
        float
            방향 적중률 (0~100).
        """
        if (
            not predicted_directions
            or not actual_directions
            or len(predicted_directions) != len(actual_directions)
        ):
            return 0.0
        n = len(predicted_directions)
        matches = sum(
            1 for p, a in zip(predicted_directions, actual_directions) if p == a
        )
        return (matches / n) * 100.0

    # ── Range Hit Rate ────────────────────────────────────────

    @staticmethod
    def calculate_range_hit_rate(
        predictions: list[float],
        actuals: list[float],
        timeframe: str,
    ) -> float:
        """범위 적중률 계산.

        허용 오차: 단기 ±3%, 중기 ±5%, 장기 ±10%.
        적중 = |predicted - actual| / actual <= tolerance.

        Parameters
        ----------
        predictions : list[float]
            예측 가격 리스트.
        actuals : list[float]
            실제 가격 리스트.
        timeframe : str
            "short", "mid", "long".

        Returns
        -------
        float
            범위 적중률 (0~100).
        """
        if (
            not predictions
            or not actuals
            or len(predictions) != len(actuals)
        ):
            return 0.0

        tolerance = TOLERANCE.get(timeframe, 0.05)
        n = len(predictions)
        hits = 0
        for p, a in zip(predictions, actuals):
            if a == 0.0:
                continue
            if abs(p - a) / abs(a) <= tolerance:
                hits += 1
        return (hits / n) * 100.0

    # ── 저성능 감지 ───────────────────────────────────────────

    def check_underperformance(
        self,
        model_name: str,
        model_hit_rates: list[float],
        ensemble_hit_rates: list[float],
        consecutive_days: int = 30,
        threshold_pp: float = 10.0,
    ) -> bool:
        """30일 연속 앙상블 대비 10%p 이상 낮은 모델 감지.

        Parameters
        ----------
        model_name : str
            모델명.
        model_hit_rates : list[float]
            모델의 일별 Direction Hit Rate 리스트 (최근 순).
        ensemble_hit_rates : list[float]
            앙상블의 일별 Direction Hit Rate 리스트 (최근 순).
        consecutive_days : int
            연속 일수 기준 (기본 30).
        threshold_pp : float
            퍼센트포인트 기준 (기본 10.0).

        Returns
        -------
        bool
            True면 "성능 저하 지속" 경고 대상.
        """
        if (
            len(model_hit_rates) < consecutive_days
            or len(ensemble_hit_rates) < consecutive_days
        ):
            return False

        for i in range(consecutive_days):
            diff = ensemble_hit_rates[i] - model_hit_rates[i]
            if diff < threshold_pp:
                return False

        logger.warning(
            "모델 '%s': %d일 연속 앙상블 대비 %.1f%%p 이상 저성능 — 성능 저하 지속 경고",
            model_name,
            consecutive_days,
            threshold_pp,
        )
        return True

    # ── DB 저장 / 조회 ────────────────────────────────────────

    def save_hit_rate(self, result: HitRateResult) -> HitRateRecord:
        """적중률을 HitRateRecord 테이블에 저장.

        Parameters
        ----------
        result : HitRateResult
            저장할 적중률 결과.

        Returns
        -------
        HitRateRecord
            저장된 ORM 레코드.
        """
        if self.session is None:
            raise RuntimeError("DB 세션이 설정되지 않았습니다.")

        record = HitRateRecord(
            date=result.date,
            model_name=result.model_name,
            timeframe=result.timeframe,
            direction_hit_rate=result.direction_hit_rate,
            range_hit_rate=result.range_hit_rate,
        )
        self.session.add(record)
        self.session.commit()
        return record

    def get_hit_rate(
        self, model_name: str, timeframe: str, date: datetime
    ) -> HitRateRecord | None:
        """날짜와 모델명으로 적중률 조회.

        Parameters
        ----------
        model_name : str
            모델명.
        timeframe : str
            타임프레임.
        date : datetime
            조회 날짜.

        Returns
        -------
        HitRateRecord | None
        """
        if self.session is None:
            raise RuntimeError("DB 세션이 설정되지 않았습니다.")

        return (
            self.session.query(HitRateRecord)
            .filter(
                HitRateRecord.model_name == model_name,
                HitRateRecord.timeframe == timeframe,
                HitRateRecord.date == date,
            )
            .first()
        )
