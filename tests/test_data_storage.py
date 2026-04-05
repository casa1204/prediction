"""Property 3: 데이터 저장 라운드트립

구조화된 데이터 레코드(기술 지표, 와이코프 이벤트, 온체인 데이터)를 DB에 저장한 후
타임스탬프로 조회하면 원본과 동일한 데이터가 반환되어야 한다.

**Validates: Requirements 1.4, 1.16, 4.5**

Tags: Feature: xrp-price-prediction-dashboard, Property 3: 데이터 저장 라운드트립
"""

import math
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, CollectionLog, OnchainData, TechnicalIndicator, WyckoffData


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """인메모리 SQLite 세션을 생성하고 테스트 후 정리한다."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def make_session():
    """hypothesis 테스트 내부에서 사용할 세션 팩토리."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


# ── Strategies ────────────────────────────────────────────────

# 유한한 float만 생성 (NaN, Inf 제외)
finite_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_float = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
score_float = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)

reasonable_datetime = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

positive_int = st.integers(min_value=0, max_value=10**9)


# ── Helper ────────────────────────────────────────────────────


def floats_equal(a, b):
    """SQLite float 저장 시 미세한 부동소수점 차이를 허용하여 비교."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# ── Property Tests ────────────────────────────────────────────


class TestDataStorageRoundTrip:
    """Property 3: 데이터 저장 라운드트립

    **Validates: Requirements 1.4, 1.16, 4.5**
    """

    @given(
        timestamp=reasonable_datetime,
        rsi_14=st.one_of(st.none(), score_float),
        macd=st.one_of(st.none(), finite_float),
        macd_signal=st.one_of(st.none(), finite_float),
        macd_histogram=st.one_of(st.none(), finite_float),
        bb_upper=st.one_of(st.none(), positive_float),
        bb_middle=st.one_of(st.none(), positive_float),
        bb_lower=st.one_of(st.none(), positive_float),
        sma_5=st.one_of(st.none(), positive_float),
        sma_10=st.one_of(st.none(), positive_float),
        sma_20=st.one_of(st.none(), positive_float),
        sma_50=st.one_of(st.none(), positive_float),
        sma_200=st.one_of(st.none(), positive_float),
        ema_12=st.one_of(st.none(), positive_float),
        ema_26=st.one_of(st.none(), positive_float),
    )
    @settings(max_examples=100)
    def test_technical_indicator_round_trip(
        self,
        timestamp,
        rsi_14,
        macd,
        macd_signal,
        macd_histogram,
        bb_upper,
        bb_middle,
        bb_lower,
        sma_5,
        sma_10,
        sma_20,
        sma_50,
        sma_200,
        ema_12,
        ema_26,
    ):
        """기술 지표 레코드를 저장 후 타임스탬프로 조회하면 원본과 동일해야 한다.

        **Validates: Requirements 1.4**
        """
        session, engine = make_session()
        try:
            record = TechnicalIndicator(
                timestamp=timestamp,
                rsi_14=rsi_14,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                sma_5=sma_5,
                sma_10=sma_10,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_12=ema_12,
                ema_26=ema_26,
            )
            session.add(record)
            session.commit()

            loaded = (
                session.query(TechnicalIndicator)
                .filter(TechnicalIndicator.timestamp == timestamp)
                .first()
            )

            assert loaded is not None, "저장된 레코드를 타임스탬프로 조회할 수 없음"
            assert loaded.timestamp == timestamp
            assert floats_equal(loaded.rsi_14, rsi_14)
            assert floats_equal(loaded.macd, macd)
            assert floats_equal(loaded.macd_signal, macd_signal)
            assert floats_equal(loaded.macd_histogram, macd_histogram)
            assert floats_equal(loaded.bb_upper, bb_upper)
            assert floats_equal(loaded.bb_middle, bb_middle)
            assert floats_equal(loaded.bb_lower, bb_lower)
            assert floats_equal(loaded.sma_5, sma_5)
            assert floats_equal(loaded.sma_10, sma_10)
            assert floats_equal(loaded.sma_20, sma_20)
            assert floats_equal(loaded.sma_50, sma_50)
            assert floats_equal(loaded.sma_200, sma_200)
            assert floats_equal(loaded.ema_12, ema_12)
            assert floats_equal(loaded.ema_26, ema_26)
        finally:
            session.close()
            engine.dispose()

    @given(
        timestamp=reasonable_datetime,
        event_type=st.sampled_from(
            ["PS", "SC", "AR", "ST", "Spring", "SOS", "LPS", "PSY", "BC", "UTAD", "LPSY", "SOW"]
        ),
        pattern_type=st.sampled_from(["accumulation", "distribution"]),
        price=positive_float,
        volume=positive_float,
        market_phase=st.one_of(
            st.none(),
            st.sampled_from(["Accumulation", "Markup", "Distribution", "Markdown"]),
        ),
        wyckoff_phase=st.one_of(st.none(), st.sampled_from(["A", "B", "C", "D", "E"])),
        confidence_score=st.one_of(st.none(), score_float),
        is_trend_reversal=st.booleans(),
    )
    @settings(max_examples=100)
    def test_wyckoff_data_round_trip(
        self,
        timestamp,
        event_type,
        pattern_type,
        price,
        volume,
        market_phase,
        wyckoff_phase,
        confidence_score,
        is_trend_reversal,
    ):
        """와이코프 이벤트 레코드를 저장 후 타임스탬프로 조회하면 원본과 동일해야 한다.

        **Validates: Requirements 1.16**
        """
        session, engine = make_session()
        try:
            record = WyckoffData(
                timestamp=timestamp,
                event_type=event_type,
                pattern_type=pattern_type,
                price=price,
                volume=volume,
                market_phase=market_phase,
                wyckoff_phase=wyckoff_phase,
                confidence_score=confidence_score,
                is_trend_reversal=is_trend_reversal,
            )
            session.add(record)
            session.commit()

            loaded = (
                session.query(WyckoffData)
                .filter(WyckoffData.timestamp == timestamp)
                .first()
            )

            assert loaded is not None, "저장된 레코드를 타임스탬프로 조회할 수 없음"
            assert loaded.timestamp == timestamp
            assert loaded.event_type == event_type
            assert loaded.pattern_type == pattern_type
            assert floats_equal(loaded.price, price)
            assert floats_equal(loaded.volume, volume)
            assert loaded.market_phase == market_phase
            assert loaded.wyckoff_phase == wyckoff_phase
            assert floats_equal(loaded.confidence_score, confidence_score)
            assert loaded.is_trend_reversal == is_trend_reversal
        finally:
            session.close()
            engine.dispose()

    @given(
        timestamp=reasonable_datetime,
        active_wallets=st.one_of(st.none(), positive_int),
        new_wallets=st.one_of(st.none(), positive_int),
        transaction_count=st.one_of(st.none(), positive_int),
        total_volume_xrp=st.one_of(st.none(), positive_float),
        whale_tx_count=st.one_of(st.none(), positive_int),
        whale_tx_volume=st.one_of(st.none(), positive_float),
    )
    @settings(max_examples=100)
    def test_onchain_data_round_trip(
        self,
        timestamp,
        active_wallets,
        new_wallets,
        transaction_count,
        total_volume_xrp,
        whale_tx_count,
        whale_tx_volume,
    ):
        """온체인 데이터 레코드를 저장 후 타임스탬프로 조회하면 원본과 동일해야 한다.

        **Validates: Requirements 4.5**
        """
        session, engine = make_session()
        try:
            record = OnchainData(
                timestamp=timestamp,
                active_wallets=active_wallets,
                new_wallets=new_wallets,
                transaction_count=transaction_count,
                total_volume_xrp=total_volume_xrp,
                whale_tx_count=whale_tx_count,
                whale_tx_volume=whale_tx_volume,
            )
            session.add(record)
            session.commit()

            loaded = (
                session.query(OnchainData)
                .filter(OnchainData.timestamp == timestamp)
                .first()
            )

            assert loaded is not None, "저장된 레코드를 타임스탬프로 조회할 수 없음"
            assert loaded.timestamp == timestamp
            assert loaded.active_wallets == active_wallets
            assert loaded.new_wallets == new_wallets
            assert loaded.transaction_count == transaction_count
            assert floats_equal(loaded.total_volume_xrp, total_volume_xrp)
            assert loaded.whale_tx_count == whale_tx_count
            assert floats_equal(loaded.whale_tx_volume, whale_tx_volume)
        finally:
            session.close()
            engine.dispose()


class TestCollectionLogRequiredFields:
    """Property 20: 수집 로그 필수 필드

    데이터 수집 작업에 대해, 로그 레코드에는 소스명, 시작 시간, 종료 시간,
    성공/실패 상태가 반드시 포함되어야 하며, 종료 시간은 시작 시간 이후여야 한다.

    **Validates: Requirements 7.4**

    Tags: Feature: xrp-price-prediction-dashboard, Property 20: 수집 로그 필수 필드
    """

    @given(
        source=st.sampled_from([
            "price_collector",
            "pair_collector",
            "sentiment_collector",
            "onchain_collector",
        ]),
        start_time=reasonable_datetime,
        duration_seconds=st.integers(min_value=1, max_value=86400),
        status=st.sampled_from(["success", "failure"]),
        error_message=st.one_of(st.none(), st.text(min_size=1, max_size=200)),
        consecutive_failures=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_collection_log_required_fields(
        self,
        source,
        start_time,
        duration_seconds,
        status,
        error_message,
        consecutive_failures,
    ):
        """수집 로그 레코드에 필수 필드가 모두 존재하고, 종료 시간이 시작 시간 이후여야 한다.

        **Validates: Requirements 7.4**
        """
        end_time = start_time + timedelta(seconds=duration_seconds)

        session, engine = make_session()
        try:
            record = CollectionLog(
                source=source,
                start_time=start_time,
                end_time=end_time,
                status=status,
                error_message=error_message,
                consecutive_failures=consecutive_failures,
            )
            session.add(record)
            session.commit()

            loaded = (
                session.query(CollectionLog)
                .filter(CollectionLog.id == record.id)
                .first()
            )

            # 필수 필드 존재 확인
            assert loaded is not None, "저장된 수집 로그를 조회할 수 없음"
            assert loaded.source is not None and loaded.source != "", \
                "소스명이 비어 있으면 안 됨"
            assert loaded.start_time is not None, "시작 시간이 None이면 안 됨"
            assert loaded.end_time is not None, "종료 시간이 None이면 안 됨"
            assert loaded.status is not None and loaded.status != "", \
                "상태가 비어 있으면 안 됨"

            # 상태 값 유효성
            assert loaded.status in ("success", "failure"), \
                f"상태는 'success' 또는 'failure'여야 함, 실제: {loaded.status}"

            # 종료 시간 >= 시작 시간
            assert loaded.end_time >= loaded.start_time, \
                f"종료 시간({loaded.end_time})이 시작 시간({loaded.start_time}) 이전임"

            # 라운드트립 정확성
            assert loaded.source == source
            assert loaded.start_time == start_time
            assert loaded.end_time == end_time
            assert loaded.status == status
        finally:
            session.close()
            engine.dispose()
