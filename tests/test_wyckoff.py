"""와이코프 모듈 단위 테스트."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analysis.wyckoff import (
    WyckoffModule,
    WyckoffEvent,
    MarketPhase,
    WyckoffPhaseResult,
    VolumeAnalysis,
)
from db.models import Base, WyckoffData


# ── 테스트 헬퍼 ──────────────────────────────────────────────


def _make_ohlcv(
    n: int = 250,
    base_price: float = 0.5,
    trend: str = "flat",
    seed: int = 42,
) -> pd.DataFrame:
    """테스트용 OHLCV 데이터프레임 생성.

    Parameters
    ----------
    n : int
        캔들 수.
    base_price : float
        시작 가격.
    trend : str
        "flat", "up", "down", "accumulation", "distribution"
    seed : int
        난수 시드.
    """
    rng = np.random.RandomState(seed)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    closes = np.zeros(n)
    closes[0] = base_price

    if trend == "flat":
        for i in range(1, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.005))
    elif trend == "up":
        for i in range(1, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(0.002, 0.005))
    elif trend == "down":
        for i in range(1, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(-0.002, 0.005))
    elif trend == "accumulation":
        # 하락 → 횡보 → 상승 패턴
        phase_len = n // 3
        for i in range(1, phase_len):
            closes[i] = closes[i - 1] * (1 + rng.normal(-0.003, 0.005))
        for i in range(phase_len, 2 * phase_len):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.003))
        for i in range(2 * phase_len, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(0.003, 0.005))
    elif trend == "distribution":
        # 상승 → 횡보 → 하락 패턴
        phase_len = n // 3
        for i in range(1, phase_len):
            closes[i] = closes[i - 1] * (1 + rng.normal(0.003, 0.005))
        for i in range(phase_len, 2 * phase_len):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.003))
        for i in range(2 * phase_len, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(-0.003, 0.005))

    closes = np.maximum(closes, 0.01)  # 가격은 양수
    highs = closes * (1 + rng.uniform(0.001, 0.02, n))
    lows = closes * (1 - rng.uniform(0.001, 0.02, n))
    opens = closes * (1 + rng.normal(0, 0.005, n))
    volumes = rng.uniform(1000, 10000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

def _make_accumulation_df(n: int = 300, seed: int = 100) -> pd.DataFrame:
    """축적 패턴이 포함된 OHLCV 데이터 생성.

    하락 → SC(급락+고거래량) → AR(반등) → ST(재하락) → Spring(지지선 이탈 후 복귀) → SOS → LPS
    """
    rng = np.random.RandomState(seed)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    opens = np.zeros(n)
    volumes = np.zeros(n)

    base = 1.0
    closes[0] = base

    # Phase 1: 하락 추세 (0~60) — PS 발생 조건
    for i in range(1, 60):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.002, 0.008))
        volumes[i] = rng.uniform(5000, 8000)

    # PS 이벤트: 높은 거래량으로 지지 시도
    volumes[58] = 15000  # 높은 거래량

    # Phase 2: SC (급락 + 매우 높은 거래량) (60~80)
    for i in range(60, 80):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.01, 0.02))
        volumes[i] = rng.uniform(15000, 25000)  # 매우 높은 거래량

    support_level = closes[79]

    # Phase 3: AR (반등) (80~110)
    for i in range(80, 110):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.005, 0.015))
        volumes[i] = rng.uniform(5000, 10000)

    resistance_level = closes[109]

    # Phase 4: ST (SC 근처까지 재하락, 낮은 거래량) (110~150)
    for i in range(110, 150):
        target = support_level + (resistance_level - support_level) * 0.1
        diff = target - closes[i - 1]
        closes[i] = closes[i - 1] + diff * 0.05 + rng.normal(0, 0.002)
        volumes[i] = rng.uniform(2000, 4000)  # 낮은 거래량

    # Phase 5: Spring (지지선 아래로 이탈 후 복귀) (150~170)
    for i in range(150, 160):
        closes[i] = support_level * (1 - rng.uniform(0.005, 0.02))
        volumes[i] = rng.uniform(3000, 6000)

    # Spring 복귀
    for i in range(160, 170):
        closes[i] = support_level * (1 + rng.uniform(0.005, 0.02))
        volumes[i] = rng.uniform(5000, 10000)

    # 지지선 아래로 low, close는 위로 — Spring 조건
    lows[155] = support_level * 0.97
    closes[155] = support_level * 1.01

    # Phase 6: SOS (강한 상승 + 높은 거래량) (170~220)
    for i in range(170, 220):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.005, 0.012))
        volumes[i] = rng.uniform(10000, 18000)

    # Phase 7: LPS (되돌림 + 낮은 거래량) (220~260)
    for i in range(220, min(260, n)):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.002, 0.006))
        volumes[i] = rng.uniform(2000, 4000)

    # 나머지 채우기
    for i in range(260, n):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.001, 0.005))
        volumes[i] = rng.uniform(5000, 10000)

    # highs, lows, opens 생성
    for i in range(n):
        if closes[i] == 0:
            closes[i] = closes[max(0, i - 1)] if i > 0 else base
        highs[i] = closes[i] * (1 + rng.uniform(0.005, 0.02))
        lows[i] = closes[i] * (1 - rng.uniform(0.005, 0.02))
        opens[i] = closes[i] * (1 + rng.normal(0, 0.005))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _make_distribution_df(n: int = 300, seed: int = 200) -> pd.DataFrame:
    """분배 패턴이 포함된 OHLCV 데이터 생성."""
    rng = np.random.RandomState(seed)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    opens = np.zeros(n)
    volumes = np.zeros(n)

    base = 1.0
    closes[0] = base

    # Phase 1: 상승 추세 (0~60) — PSY 발생 조건
    for i in range(1, 60):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.002, 0.008))
        volumes[i] = rng.uniform(5000, 8000)

    volumes[58] = 15000  # PSY: 높은 거래량

    # Phase 2: BC (급등 + 매우 높은 거래량) (60~80)
    for i in range(60, 80):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.01, 0.02))
        volumes[i] = rng.uniform(15000, 25000)

    resistance_level = closes[79]

    # Phase 3: AR (하락) (80~110)
    for i in range(80, 110):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.005, 0.015))
        volumes[i] = rng.uniform(5000, 10000)

    support_level = closes[109]

    # Phase 4: ST (저항선 근처까지 재상승, 낮은 거래량) (110~150)
    for i in range(110, 150):
        target = resistance_level - (resistance_level - support_level) * 0.1
        diff = target - closes[i - 1]
        closes[i] = closes[i - 1] + diff * 0.05 + rng.normal(0, 0.002)
        volumes[i] = rng.uniform(2000, 4000)

    # Phase 5: UTAD (저항선 위로 이탈 후 하락) (150~170)
    for i in range(150, 160):
        closes[i] = resistance_level * (1 + rng.uniform(0.005, 0.02))
        volumes[i] = rng.uniform(5000, 10000)

    for i in range(160, 170):
        closes[i] = resistance_level * (1 - rng.uniform(0.005, 0.02))
        volumes[i] = rng.uniform(5000, 10000)

    highs[155] = resistance_level * 1.03
    closes[155] = resistance_level * 0.99

    # Phase 6: LPSY (약한 반등 + 낮은 거래량) (170~220)
    for i in range(170, 190):
        closes[i] = closes[i - 1] * (1 + rng.uniform(0.001, 0.004))
        volumes[i] = rng.uniform(2000, 4000)

    for i in range(190, 220):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.001, 0.003))
        volumes[i] = rng.uniform(3000, 6000)

    # Phase 7: SOW (강한 하락 + 높은 거래량) (220~260)
    for i in range(220, min(260, n)):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.008, 0.015))
        volumes[i] = rng.uniform(12000, 20000)

    for i in range(260, n):
        closes[i] = closes[i - 1] * (1 - rng.uniform(0.001, 0.005))
        volumes[i] = rng.uniform(5000, 10000)

    for i in range(n):
        if closes[i] == 0:
            closes[i] = closes[max(0, i - 1)] if i > 0 else base
        highs[i] = closes[i] * (1 + rng.uniform(0.005, 0.02))
        lows[i] = closes[i] * (1 - rng.uniform(0.005, 0.02))
        opens[i] = closes[i] * (1 + rng.normal(0, 0.005))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ── 테스트 픽스처 ────────────────────────────────────────────


@pytest.fixture
def module():
    return WyckoffModule()


@pytest.fixture
def flat_df():
    return _make_ohlcv(250, trend="flat")


@pytest.fixture
def up_df():
    return _make_ohlcv(250, trend="up")


@pytest.fixture
def down_df():
    return _make_ohlcv(250, trend="down")


@pytest.fixture
def small_df():
    return _make_ohlcv(50, trend="flat")


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ── detect_accumulation 테스트 ────────────────────────────────


class TestDetectAccumulation:
    def test_returns_empty_for_insufficient_data(self, module, small_df):
        result = module.detect_accumulation(small_df)
        assert result == []

    def test_returns_list_of_wyckoff_events(self, module, flat_df):
        result = module.detect_accumulation(flat_df)
        assert isinstance(result, list)
        for evt in result:
            assert isinstance(evt, WyckoffEvent)

    def test_all_events_are_accumulation_type(self, module, flat_df):
        result = module.detect_accumulation(flat_df)
        for evt in result:
            assert evt.pattern_type == "accumulation"

    def test_event_types_are_valid(self, module, flat_df):
        valid_types = {"PS", "SC", "AR", "ST", "Spring", "SOS", "LPS"}
        result = module.detect_accumulation(flat_df)
        for evt in result:
            assert evt.event_type in valid_types

    def test_spring_has_trend_reversal_true(self, module):
        df = _make_accumulation_df()
        result = module.detect_accumulation(df)
        springs = [e for e in result if e.event_type == "Spring"]
        for s in springs:
            assert s.is_trend_reversal is True

    def test_non_spring_events_default_false(self, module):
        df = _make_accumulation_df()
        result = module.detect_accumulation(df)
        non_springs = [e for e in result if e.event_type not in ("Spring",)]
        for evt in non_springs:
            assert evt.is_trend_reversal is False

    def test_events_have_required_fields(self, module, flat_df):
        result = module.detect_accumulation(flat_df)
        for evt in result:
            assert evt.timestamp is not None
            assert evt.price > 0
            assert evt.volume >= 0


# ── detect_distribution 테스트 ────────────────────────────────


class TestDetectDistribution:
    def test_returns_empty_for_insufficient_data(self, module, small_df):
        result = module.detect_distribution(small_df)
        assert result == []

    def test_returns_list_of_wyckoff_events(self, module, flat_df):
        result = module.detect_distribution(flat_df)
        assert isinstance(result, list)
        for evt in result:
            assert isinstance(evt, WyckoffEvent)

    def test_all_events_are_distribution_type(self, module, flat_df):
        result = module.detect_distribution(flat_df)
        for evt in result:
            assert evt.pattern_type == "distribution"

    def test_event_types_are_valid(self, module, flat_df):
        valid_types = {"PSY", "BC", "AR", "ST", "UTAD", "LPSY", "SOW"}
        result = module.detect_distribution(flat_df)
        for evt in result:
            assert evt.event_type in valid_types

    def test_utad_has_trend_reversal_true(self, module):
        df = _make_distribution_df()
        result = module.detect_distribution(df)
        utads = [e for e in result if e.event_type == "UTAD"]
        for u in utads:
            assert u.is_trend_reversal is True

    def test_non_utad_events_default_false(self, module):
        df = _make_distribution_df()
        result = module.detect_distribution(df)
        non_utads = [e for e in result if e.event_type not in ("UTAD",)]
        for evt in non_utads:
            assert evt.is_trend_reversal is False


# ── determine_market_phase 테스트 ─────────────────────────────


class TestDetermineMarketPhase:
    def test_returns_market_phase(self, module, flat_df):
        result = module.determine_market_phase(flat_df)
        assert isinstance(result, MarketPhase)

    def test_phase_is_valid(self, module, flat_df):
        valid_phases = {"Accumulation", "Markup", "Distribution", "Markdown"}
        result = module.determine_market_phase(flat_df)
        assert result.phase in valid_phases

    def test_confidence_in_range(self, module, flat_df):
        result = module.determine_market_phase(flat_df)
        assert 0 <= result.confidence_score <= 100

    def test_uptrend_suggests_markup(self, module, up_df):
        result = module.determine_market_phase(up_df)
        assert result.phase in {"Markup", "Distribution"}  # 상승 추세

    def test_downtrend_suggests_markdown(self, module, down_df):
        result = module.determine_market_phase(down_df)
        assert result.phase in {"Markdown", "Accumulation"}  # 하락 추세

    def test_small_data_returns_default(self, module):
        df = _make_ohlcv(10, trend="flat")
        result = module.determine_market_phase(df)
        assert isinstance(result, MarketPhase)
        assert result.confidence_score == 0.0


# ── determine_wyckoff_phase 테스트 ────────────────────────────


class TestDetermineWyckoffPhase:
    def test_returns_wyckoff_phase_result(self, module, flat_df):
        result = module.determine_wyckoff_phase(flat_df)
        assert isinstance(result, WyckoffPhaseResult)

    def test_phase_is_valid(self, module, flat_df):
        valid_phases = {"A", "B", "C", "D", "E"}
        result = module.determine_wyckoff_phase(flat_df)
        assert result.phase in valid_phases

    def test_progress_in_range(self, module, flat_df):
        result = module.determine_wyckoff_phase(flat_df)
        assert 0 <= result.progress <= 100

    def test_no_events_returns_phase_a_or_e(self, module, flat_df):
        result = module.determine_wyckoff_phase(flat_df)
        assert result.phase in {"A", "E"}

    def test_with_accumulation_events(self, module):
        df = _make_accumulation_df()
        module.detect_accumulation(df)
        result = module.determine_wyckoff_phase(df)
        assert result.phase in {"A", "B", "C", "D", "E"}


# ── analyze_volume 테스트 ─────────────────────────────────────


class TestAnalyzeVolume:
    def test_returns_volume_analysis(self, module, flat_df):
        result = module.analyze_volume(flat_df)
        assert isinstance(result, VolumeAnalysis)

    def test_relationship_is_valid(self, module, flat_df):
        result = module.analyze_volume(flat_df)
        assert result.relationship in {"divergence", "convergence"}

    def test_details_is_string(self, module, flat_df):
        result = module.analyze_volume(flat_df)
        assert isinstance(result.details, str)
        assert len(result.details) > 0

    def test_small_data_returns_convergence(self, module):
        df = _make_ohlcv(5, trend="flat")
        result = module.analyze_volume(df)
        assert result.relationship == "convergence"


# ── save_to_db 테스트 ─────────────────────────────────────────


class TestSaveToDb:
    def test_saves_events_to_db(self, module, db_session):
        events = [
            WyckoffEvent(
                event_type="PS",
                pattern_type="accumulation",
                timestamp=datetime(2024, 1, 1),
                price=0.5,
                volume=10000.0,
            ),
            WyckoffEvent(
                event_type="Spring",
                pattern_type="accumulation",
                timestamp=datetime(2024, 1, 2),
                price=0.48,
                volume=8000.0,
                is_trend_reversal=True,
            ),
        ]
        mp = MarketPhase(phase="Accumulation", confidence_score=75.0)
        wp = WyckoffPhaseResult(phase="C", progress=55.0)

        saved = module.save_to_db(events, db_session, mp, wp)
        assert saved == 2

        records = db_session.query(WyckoffData).all()
        assert len(records) == 2

        # 첫 번째 레코드 검증
        r0 = records[0]
        assert r0.event_type == "PS"
        assert r0.pattern_type == "accumulation"
        assert r0.price == 0.5
        assert r0.market_phase == "Accumulation"
        assert r0.wyckoff_phase == "C"
        assert r0.confidence_score == 75.0
        assert r0.is_trend_reversal is False

        # Spring 레코드 검증
        r1 = records[1]
        assert r1.event_type == "Spring"
        assert r1.is_trend_reversal is True

    def test_saves_empty_list(self, module, db_session):
        saved = module.save_to_db([], db_session)
        assert saved == 0

    def test_saves_without_phase_info(self, module, db_session):
        events = [
            WyckoffEvent(
                event_type="SC",
                pattern_type="accumulation",
                timestamp=datetime(2024, 1, 1),
                price=0.45,
                volume=20000.0,
            ),
        ]
        saved = module.save_to_db(events, db_session)
        assert saved == 1

        record = db_session.query(WyckoffData).first()
        assert record.market_phase is None
        assert record.wyckoff_phase is None


# ── 통합 테스트 ───────────────────────────────────────────────


class TestIntegration:
    def test_full_accumulation_workflow(self, module, db_session):
        df = _make_accumulation_df()
        events = module.detect_accumulation(df)
        phase = module.determine_market_phase(df)
        wp = module.determine_wyckoff_phase(df)
        vol = module.analyze_volume(df)

        assert isinstance(phase, MarketPhase)
        assert isinstance(wp, WyckoffPhaseResult)
        assert isinstance(vol, VolumeAnalysis)

        if events:
            saved = module.save_to_db(events, db_session, phase, wp)
            assert saved == len(events)

    def test_full_distribution_workflow(self, module, db_session):
        df = _make_distribution_df()
        events = module.detect_distribution(df)
        phase = module.determine_market_phase(df)
        wp = module.determine_wyckoff_phase(df)
        vol = module.analyze_volume(df)

        assert isinstance(phase, MarketPhase)
        assert isinstance(wp, WyckoffPhaseResult)
        assert isinstance(vol, VolumeAnalysis)

        if events:
            saved = module.save_to_db(events, db_session, phase, wp)
            assert saved == len(events)


# ── 속성 기반 테스트 (Property-Based Tests) ────────────────────


from hypothesis import given, settings, assume
from hypothesis import strategies as st


def _ohlcv_strategy(min_candles: int = 200):
    """hypothesis 전략: 양수 가격/거래량을 가진 OHLCV 데이터프레임 생성."""
    return st.integers(min_value=min_candles, max_value=400).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.sampled_from(["flat", "up", "down", "accumulation", "distribution"]),
            st.integers(min_value=0, max_value=10000),
        )
    )


class TestProperty8WyckoffAnalysisValidity:
    """Property 8: 와이코프 분석 결과 유효성

    Feature: xrp-price-prediction-dashboard
    Property 8: 가격 데이터에 대해, determine_market_phase가 반환하는 시장 단계는
    Accumulation, Markup, Distribution, Markdown 중 하나이고 신뢰도 점수는 0~100 범위이며,
    determine_wyckoff_phase가 반환하는 Phase는 A~E 중 하나여야 한다.

    **Validates: Requirements 1.12, 1.13**
    """

    @given(data=_ohlcv_strategy())
    @settings(max_examples=100)
    def test_market_phase_and_wyckoff_phase_validity(self, data):
        n, base_price, trend, seed = data
        df = _make_ohlcv(n=n, base_price=base_price, trend=trend, seed=seed)

        module = WyckoffModule()

        # 축적/분배 이벤트 감지를 먼저 수행하여 내부 상태 구축
        module.detect_accumulation(df)
        module.detect_distribution(df)

        # determine_market_phase 검증
        market_phase = module.determine_market_phase(df)
        assert isinstance(market_phase, MarketPhase)
        assert market_phase.phase in {
            "Accumulation", "Markup", "Distribution", "Markdown"
        }, f"Invalid market phase: {market_phase.phase}"
        assert 0 <= market_phase.confidence_score <= 100, (
            f"Confidence score out of range: {market_phase.confidence_score}"
        )

        # determine_wyckoff_phase 검증
        wyckoff_phase = module.determine_wyckoff_phase(df)
        assert isinstance(wyckoff_phase, WyckoffPhaseResult)
        assert wyckoff_phase.phase in {
            "A", "B", "C", "D", "E"
        }, f"Invalid Wyckoff phase: {wyckoff_phase.phase}"


class TestProperty9SpringUpthrustRequiredFields:
    """Property 9: Spring/Upthrust 이벤트 필수 필드

    Feature: xrp-price-prediction-dashboard
    Property 9: 감지된 Spring 또는 Upthrust WyckoffEvent에 대해,
    발생 시간, 가격, 거래량이 존재하고 is_trend_reversal이 True여야 한다.

    **Validates: Requirements 1.15**
    """

    @given(data=_ohlcv_strategy())
    @settings(max_examples=100)
    def test_spring_upthrust_events_have_required_fields(self, data):
        n, base_price, trend, seed = data
        df = _make_ohlcv(n=n, base_price=base_price, trend=trend, seed=seed)

        module = WyckoffModule()
        accum_events = module.detect_accumulation(df)
        dist_events = module.detect_distribution(df)

        all_events = accum_events + dist_events
        # Spring 이벤트 (축적 패턴) 및 UTAD 이벤트 (분배 패턴 = Upthrust)
        spring_upthrust = [
            e for e in all_events if e.event_type in ("Spring", "UTAD")
        ]

        for event in spring_upthrust:
            # 발생 시간이 존재
            assert event.timestamp is not None, (
                f"{event.event_type} event missing timestamp"
            )
            # 가격이 존재하고 양수
            assert event.price is not None and event.price > 0, (
                f"{event.event_type} event has invalid price: {event.price}"
            )
            # 거래량이 존재하고 양수
            assert event.volume is not None and event.volume > 0, (
                f"{event.event_type} event has invalid volume: {event.volume}"
            )
            # 추세 전환 신호
            assert event.is_trend_reversal is True, (
                f"{event.event_type} event should have is_trend_reversal=True, "
                f"got {event.is_trend_reversal}"
            )
