"""재학습 스케줄러 속성 기반 테스트.

Property 21: Champion/Challenger 교체 정확성
Property 22: 재학습 이력 필수 필드
Property 23: 재학습 오류 시 모델 보존
"""

from datetime import datetime

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, RetrainingHistory
from monitoring.retraining_scheduler import (
    ModelMetrics,
    RetrainingRecord,
    RetrainingScheduler,
)
from prediction.base_model import BaseModel, PredictionResult


# ── 테스트용 더미 모델 ────────────────────────────────────────

class DummyModel(BaseModel):
    """테스트용 더미 모델."""

    def __init__(self, predict_price: float = 1.0):
        super().__init__()
        self._predict_price = predict_price
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._trained = True
        self._is_trained = True

    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        return PredictionResult(
            predicted_price=self._predict_price,
            up_probability=60.0,
            down_probability=40.0,
            confidence=70.0,
            timeframe=timeframe,
        )

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


class ErrorModel(BaseModel):
    """학습 시 오류를 발생시키는 모델."""

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise RuntimeError("학습 중 오류 발생")

    def predict(self, X: np.ndarray, timeframe: str) -> PredictionResult:
        raise RuntimeError("예측 불가")

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# ── 헬퍼 ──────────────────────────────────────────────────────

@pytest.fixture
def db_session():
    """인메모리 SQLite 세션."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def make_evaluate_fn(metrics: ModelMetrics):
    """고정 메트릭을 반환하는 평가 함수 생성."""
    def evaluate(model, X_test, y_test):
        return metrics
    return evaluate


# ── Property 21: Champion/Challenger 교체 정확성 ─────────────
# Feature: xrp-price-prediction-dashboard, Property 21: Champion/Challenger 교체 정확성

@settings(max_examples=100)
@given(
    champ_mae=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_mape=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_dir=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    improvement=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_swap_when_challenger_better(champ_mae, champ_mape, champ_dir, improvement):
    """Property 21: Challenger의 모든 지표가 우수하면 교체가 수행되어야 한다.

    **Validates: Requirements 7.8, 7.9, 7.10**
    """
    champion_metrics = ModelMetrics(
        mae=champ_mae, mape=champ_mape, direction_accuracy=champ_dir
    )
    challenger_metrics = ModelMetrics(
        mae=champ_mae - improvement,  # 더 낮은 MAE (더 좋음)
        mape=champ_mape - improvement,  # 더 낮은 MAPE (더 좋음)
        direction_accuracy=champ_dir + improvement,  # 더 높은 정확도 (더 좋음)
    )

    result = RetrainingScheduler.compare_models(champion_metrics, challenger_metrics)
    assert result is True, "Challenger가 모든 지표에서 우수하면 교체되어야 한다"


@settings(max_examples=100)
@given(
    champ_mae=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_mape=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_dir=st.floats(min_value=10.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    worse_amount=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    worse_metric=st.sampled_from(["mae", "mape", "direction"]),
)
def test_no_swap_when_challenger_worse_in_any(
    champ_mae, champ_mape, champ_dir, worse_amount, worse_metric
):
    """Property 21: Challenger가 하나라도 저하되면 Champion이 유지되어야 한다.

    **Validates: Requirements 7.8, 7.9, 7.10**
    """
    # 기본적으로 Challenger가 동일
    chal_mae = champ_mae
    chal_mape = champ_mape
    chal_dir = champ_dir

    # 하나의 지표를 악화시킴
    if worse_metric == "mae":
        chal_mae = champ_mae + worse_amount  # 더 높은 MAE (더 나쁨)
    elif worse_metric == "mape":
        chal_mape = champ_mape + worse_amount  # 더 높은 MAPE (더 나쁨)
    else:
        chal_dir = champ_dir - worse_amount  # 더 낮은 정확도 (더 나쁨)

    champion_metrics = ModelMetrics(
        mae=champ_mae, mape=champ_mape, direction_accuracy=champ_dir
    )
    challenger_metrics = ModelMetrics(
        mae=chal_mae, mape=chal_mape, direction_accuracy=chal_dir
    )

    result = RetrainingScheduler.compare_models(champion_metrics, challenger_metrics)
    assert result is False, "Challenger가 하나라도 저하되면 Champion이 유지되어야 한다"


# ── Property 22: 재학습 이력 필수 필드 ────────────────────────
# Feature: xrp-price-prediction-dashboard, Property 22: 재학습 이력 필수 필드

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    model_name=st.sampled_from(["lstm", "xgboost", "rf", "transformer"]),
    method=st.sampled_from(["incremental", "full"]),
    champ_mae=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_mape=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    champ_dir=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    day=st.integers(min_value=1, max_value=28),
    month=st.integers(min_value=1, max_value=12),
)
def test_retraining_history_required_fields(
    db_session, model_name, method, champ_mae, champ_mape, champ_dir, day, month
):
    """Property 22: 재학습 이력에 필수 필드가 모두 포함되어야 한다.

    **Validates: Requirements 7.11**
    """
    scheduler = RetrainingScheduler(session=db_session)

    champion_metrics = ModelMetrics(mae=champ_mae, mape=champ_mape, direction_accuracy=champ_dir)
    challenger_metrics = ModelMetrics(mae=champ_mae, mape=champ_mape, direction_accuracy=champ_dir)

    data_start = datetime(2024, month, day)
    data_end = datetime(2024, month, day, 23, 59)

    dummy = DummyModel()
    eval_fn = make_evaluate_fn(challenger_metrics)

    record = scheduler.run_retraining(
        model_name=model_name,
        champion=dummy,
        method=method,
        X_train=np.array([[1.0]]),
        y_train=np.array([1.0]),
        X_test=np.array([[1.0]]),
        y_test=np.array([1.0]),
        data_period_start=data_start,
        data_period_end=data_end,
        evaluate_fn=eval_fn,
    )

    # 필수 필드 검증
    assert record.timestamp is not None, "실행 일시 필수"
    assert record.model_name == model_name, "모델명 필수"
    assert record.method in ("incremental", "full"), "재학습 방식 필수"
    assert record.data_period_start is not None, "학습 데이터 시작 기간 필수"
    assert record.data_period_end is not None, "학습 데이터 종료 기간 필수"
    assert record.champion_metrics is not None, "Champion 성능 필수"
    assert record.challenger_metrics is not None, "Challenger 성능 필수"
    assert isinstance(record.replaced, bool), "교체 여부 필수"

    # DB에도 저장되었는지 확인
    history = db_session.query(RetrainingHistory).order_by(
        RetrainingHistory.id.desc()
    ).first()
    assert history is not None
    assert history.timestamp is not None
    assert history.model_name == model_name
    assert history.method == method
    assert history.champion_mae is not None
    assert history.champion_mape is not None
    assert history.champion_direction_accuracy is not None
    assert isinstance(history.replaced, bool)

    # 테스트 간 격리
    db_session.query(RetrainingHistory).delete()
    db_session.commit()


# ── Property 23: 재학습 오류 시 모델 보존 ────────────────────
# Feature: xrp-price-prediction-dashboard, Property 23: 재학습 오류 시 모델 보존

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    model_name=st.sampled_from(["lstm", "xgboost", "rf", "transformer"]),
    method=st.sampled_from(["incremental", "full"]),
)
def test_champion_preserved_on_error(db_session, model_name, method):
    """Property 23: 재학습 오류 시 Champion 모델이 변경되지 않고 오류가 로그에 기록되어야 한다.

    **Validates: Requirements 7.12**
    """
    scheduler = RetrainingScheduler(session=db_session)

    error_model = ErrorModel()

    record = scheduler.run_retraining(
        model_name=model_name,
        champion=error_model,
        method=method,
        X_train=np.array([[1.0]]),
        y_train=np.array([1.0]),
        X_test=np.array([[1.0]]),
        y_test=np.array([1.0]),
        evaluate_fn=lambda m, x, y: (_ for _ in ()).throw(RuntimeError("평가 오류")),
    )

    # Champion 유지 (교체 안 됨)
    assert record.replaced is False, "오류 시 Champion이 유지되어야 한다"
    # 오류 메시지 기록
    assert record.error_message is not None, "오류 메시지가 기록되어야 한다"
    assert len(record.error_message) > 0

    # DB에도 오류가 기록되었는지 확인
    history = db_session.query(RetrainingHistory).order_by(
        RetrainingHistory.id.desc()
    ).first()
    assert history is not None
    assert history.replaced is False
    assert history.error_message is not None

    # 테스트 간 격리
    db_session.query(RetrainingHistory).delete()
    db_session.commit()
