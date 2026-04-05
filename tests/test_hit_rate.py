"""적중률 추적 모듈 속성 기반 테스트.

Property 24: MAE/MAPE 수학적 정확성
Property 26: Direction Hit Rate 계산 정확성
Property 27: Range Hit Rate 계산 정확성
Property 28: 적중률 시계열 저장 라운드트립
Property 29: 장기 저성능 모델 경고 정확성
"""

from datetime import datetime

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, HitRateRecord
from monitoring.hit_rate_tracker import HitRateTracker, HitRateResult


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


# ── Property 24: MAE/MAPE 수학적 정확성 ──────────────────────
# Feature: xrp-price-prediction-dashboard, Property 24: MAE/MAPE 수학적 정확성

@settings(max_examples=100)
@given(
    data=st.lists(
        st.tuples(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_mae_mape_mathematical_accuracy(data):
    """Property 24: MAE = mean(|predicted - actual|), MAPE = mean(|predicted - actual| / actual) × 100.

    **Validates: Requirements 8.1**
    """
    predictions = [p for p, _ in data]
    actuals = [a for _, a in data]
    n = len(data)

    # MAE 검증
    expected_mae = sum(abs(p - a) for p, a in data) / n
    actual_mae = HitRateTracker.calculate_mae(predictions, actuals)
    assert abs(actual_mae - expected_mae) < 1e-6, (
        f"MAE mismatch: {actual_mae} != {expected_mae}"
    )

    # MAPE 검증 (actual > 0 보장됨 — min_value=0.01)
    expected_mape = (sum(abs(p - a) / a for p, a in data) / n) * 100.0
    actual_mape = HitRateTracker.calculate_mape(predictions, actuals)
    assert abs(actual_mape - expected_mape) < 1e-4, (
        f"MAPE mismatch: {actual_mape} != {expected_mape}"
    )


# ── Property 26: Direction Hit Rate 계산 정확성 ──────────────
# Feature: xrp-price-prediction-dashboard, Property 26: Direction Hit Rate 계산 정확성

@settings(max_examples=100)
@given(
    directions=st.lists(
        st.tuples(
            st.sampled_from(["up", "down"]),
            st.sampled_from(["up", "down"]),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_direction_hit_rate_accuracy(directions):
    """Property 26: Direction Hit Rate = (일치 건수 / 전체 건수) × 100.

    **Validates: Requirements 8.6**
    """
    predicted = [p for p, _ in directions]
    actual = [a for _, a in directions]
    n = len(directions)

    expected = (sum(1 for p, a in directions if p == a) / n) * 100.0
    result = HitRateTracker.calculate_direction_hit_rate(predicted, actual)
    assert abs(result - expected) < 1e-6, (
        f"Direction hit rate mismatch: {result} != {expected}"
    )


# ── Property 27: Range Hit Rate 계산 정확성 ──────────────────
# Feature: xrp-price-prediction-dashboard, Property 27: Range Hit Rate 계산 정확성

@settings(max_examples=100)
@given(
    data=st.lists(
        st.tuples(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=50,
    ),
    timeframe=st.sampled_from(["short", "mid", "long"]),
)
def test_range_hit_rate_accuracy(data, timeframe):
    """Property 27: Range Hit Rate는 허용 오차 내 비율을 올바르게 계산해야 한다.

    **Validates: Requirements 8.7**
    """
    predictions = [p for p, _ in data]
    actuals = [a for _, a in data]
    n = len(data)

    tolerance_map = {"short": 0.03, "mid": 0.05, "long": 0.10}
    tol = tolerance_map[timeframe]

    expected_hits = sum(
        1 for p, a in data if abs(p - a) / a <= tol
    )
    expected_rate = (expected_hits / n) * 100.0

    result = HitRateTracker.calculate_range_hit_rate(predictions, actuals, timeframe)
    assert abs(result - expected_rate) < 1e-6, (
        f"Range hit rate mismatch: {result} != {expected_rate}"
    )


# ── Property 28: 적중률 시계열 저장 라운드트립 ────────────────
# Feature: xrp-price-prediction-dashboard, Property 28: 적중률 시계열 저장 라운드트립

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    model_name=st.sampled_from(["lstm", "xgboost", "rf", "transformer", "ensemble"]),
    timeframe=st.sampled_from(["short", "mid", "long"]),
    direction_hr=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    range_hr=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    day=st.integers(min_value=1, max_value=28),
    month=st.integers(min_value=1, max_value=12),
)
def test_hit_rate_roundtrip(db_session, model_name, timeframe, direction_hr, range_hr, day, month):
    """Property 28: DB에 저장한 후 조회하면 원본과 동일한 적중률 데이터가 반환되어야 한다.

    **Validates: Requirements 8.10**
    """
    date = datetime(2024, month, day)
    result = HitRateResult(
        model_name=model_name,
        timeframe=timeframe,
        direction_hit_rate=direction_hr,
        range_hit_rate=range_hr,
        date=date,
    )

    tracker = HitRateTracker(session=db_session)
    tracker.save_hit_rate(result)

    loaded = tracker.get_hit_rate(model_name, timeframe, date)
    assert loaded is not None
    assert loaded.model_name == model_name
    assert loaded.timeframe == timeframe
    assert abs(loaded.direction_hit_rate - direction_hr) < 1e-6
    assert abs(loaded.range_hit_rate - range_hr) < 1e-6
    assert loaded.date == date

    # 테스트 간 격리: 저장된 레코드 삭제
    db_session.query(HitRateRecord).delete()
    db_session.commit()


# ── Property 29: 장기 저성능 모델 경고 정확성 ────────────────
# Feature: xrp-price-prediction-dashboard, Property 29: 장기 저성능 모델 경고 정확성

@settings(max_examples=100)
@given(
    gap=st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    ensemble_base=st.floats(min_value=50.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    extra_days=st.integers(min_value=0, max_value=10),
)
def test_underperformance_warning_when_all_below(gap, ensemble_base, extra_days):
    """Property 29: 30일 연속 10%p 이상 낮으면 경고가 생성되어야 한다.

    **Validates: Requirements 8.11**
    """
    days = 30 + extra_days
    ensemble_rates = [ensemble_base] * days
    model_rates = [ensemble_base - gap] * days  # gap >= 10.0

    tracker = HitRateTracker()
    assert tracker.check_underperformance("test_model", model_rates, ensemble_rates) is True


@settings(max_examples=100)
@given(
    gap=st.floats(min_value=0.0, max_value=9.99, allow_nan=False, allow_infinity=False),
    ensemble_base=st.floats(min_value=50.0, max_value=90.0, allow_nan=False, allow_infinity=False),
)
def test_no_underperformance_warning_when_gap_small(gap, ensemble_base):
    """Property 29 (보완): gap < 10%p이면 경고가 생성되지 않아야 한다.

    **Validates: Requirements 8.11**
    """
    days = 30
    ensemble_rates = [ensemble_base] * days
    model_rates = [ensemble_base - gap] * days

    tracker = HitRateTracker()
    assert tracker.check_underperformance("test_model", model_rates, ensemble_rates) is False


@settings(max_examples=100)
@given(
    gap=st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    ensemble_base=st.floats(min_value=50.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    break_day=st.integers(min_value=0, max_value=29),
)
def test_no_underperformance_warning_when_not_consecutive(gap, ensemble_base, break_day):
    """Property 29 (보완): 30일 중 하루라도 gap < 10%p이면 경고가 생성되지 않아야 한다.

    **Validates: Requirements 8.11**
    """
    days = 30
    ensemble_rates = [ensemble_base] * days
    model_rates = [ensemble_base - gap] * days
    # break_day에서 gap을 0으로 만듦
    model_rates[break_day] = ensemble_base

    tracker = HitRateTracker()
    assert tracker.check_underperformance("test_model", model_rates, ensemble_rates) is False
