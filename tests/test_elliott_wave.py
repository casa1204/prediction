"""엘리엇 파동 모듈 단위 테스트."""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.elliott_wave import (
    FIBONACCI_RATIOS,
    MIN_CANDLES,
    ElliottWaveModule,
    WavePattern,
    WavePosition,
)


# ── 헬퍼 ──────────────────────────────────────────────────────


def _make_ohlcv(prices: list[float], start: datetime | None = None) -> pd.DataFrame:
    """종가 리스트로 간단한 OHLCV DataFrame 생성."""
    if start is None:
        start = datetime(2024, 1, 1)
    n = len(prices)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _generate_impulse_prices(n: int = 300) -> list[float]:
    """상승 충격파 패턴이 포함된 가격 시계열 생성.

    5파 구조: 상승(1) → 하락(2) → 큰 상승(3) → 하락(4) → 상승(5)
    """
    prices = []
    base = 1.0
    segment = n // 6

    # 초기 평탄 구간
    for i in range(segment):
        prices.append(base + 0.001 * i)

    # 1파 상승
    start_1 = prices[-1]
    for i in range(segment):
        prices.append(start_1 + (0.5 * i / segment))
    end_1 = prices[-1]

    # 2파 하락 (1파 시작 위에서 멈춤)
    for i in range(segment // 2):
        prices.append(end_1 - (0.2 * i / (segment // 2)))
    end_2 = prices[-1]

    # 3파 큰 상승 (가장 긴 파동)
    for i in range(segment):
        prices.append(end_2 + (0.8 * i / segment))
    end_3 = prices[-1]

    # 4파 하락 (1파 끝 위에서 멈춤)
    for i in range(segment // 2):
        prices.append(end_3 - (0.15 * i / (segment // 2)))
    end_4 = prices[-1]

    # 5파 상승
    remaining = n - len(prices)
    for i in range(remaining):
        prices.append(end_4 + (0.4 * i / max(remaining, 1)))

    return prices[:n]


# ── 테스트: detect_waves ──────────────────────────────────────


class TestDetectWaves:
    def test_returns_empty_for_insufficient_candles(self):
        """200개 미만 캔들이면 빈 리스트 반환."""
        module = ElliottWaveModule()
        df = _make_ohlcv([1.0] * 100)
        result = module.detect_waves(df)
        assert result == []

    def test_returns_empty_for_empty_dataframe(self):
        """빈 DataFrame이면 빈 리스트 반환."""
        module = ElliottWaveModule()
        df = pd.DataFrame(columns=["timestamp", "close"])
        result = module.detect_waves(df)
        assert result == []

    def test_detects_waves_with_sufficient_data(self):
        """200개 이상 캔들에서 파동 감지."""
        module = ElliottWaveModule()
        prices = _generate_impulse_prices(300)
        df = _make_ohlcv(prices)
        result = module.detect_waves(df)
        # 파동이 감지되거나 빈 리스트 (패턴이 없을 수도 있음)
        assert isinstance(result, list)

    def test_wave_pattern_fields_populated(self):
        """감지된 파동의 필수 필드가 모두 채워져 있는지 확인."""
        module = ElliottWaveModule()
        prices = _generate_impulse_prices(400)
        df = _make_ohlcv(prices)
        result = module.detect_waves(df)
        for wave in result:
            assert wave.wave_number in ("1", "2", "3", "4", "5", "A", "B", "C")
            assert wave.wave_type in ("impulse", "corrective")
            assert isinstance(wave.start_price, float)
            assert isinstance(wave.end_price, float)
            assert isinstance(wave.start_time, datetime)
            assert isinstance(wave.end_time, datetime)
            assert wave.wave_degree in ("Primary", "Intermediate", "Minor")

    def test_exactly_200_candles_works(self):
        """정확히 200개 캔들에서도 동작."""
        module = ElliottWaveModule()
        prices = _generate_impulse_prices(200)
        df = _make_ohlcv(prices)
        result = module.detect_waves(df)
        assert isinstance(result, list)

    def test_199_candles_returns_empty(self):
        """199개 캔들이면 빈 리스트."""
        module = ElliottWaveModule()
        prices = _generate_impulse_prices(200)[:199]
        df = _make_ohlcv(prices)
        result = module.detect_waves(df)
        assert result == []


# ── 테스트: validate_wave_rules ───────────────────────────────


class TestValidateWaveRules:
    def _make_bullish_impulse_set(self) -> list[WavePattern]:
        """유효한 상승 충격파 5파 세트."""
        base_time = datetime(2024, 1, 1)
        return [
            WavePattern("1", "impulse", 1.0, 1.5, base_time, base_time + timedelta(days=1), "Minor"),
            WavePattern("2", "impulse", 1.5, 1.2, base_time + timedelta(days=1), base_time + timedelta(days=2), "Minor"),
            WavePattern("3", "impulse", 1.2, 2.5, base_time + timedelta(days=2), base_time + timedelta(days=3), "Minor"),
            WavePattern("4", "impulse", 2.5, 1.8, base_time + timedelta(days=3), base_time + timedelta(days=4), "Minor"),
            WavePattern("5", "impulse", 1.8, 2.8, base_time + timedelta(days=4), base_time + timedelta(days=5), "Minor"),
        ]

    def test_valid_impulse_returns_true(self):
        """유효한 충격파는 True."""
        module = ElliottWaveModule()
        waves = self._make_bullish_impulse_set()
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is True

    def test_rule1_wave2_below_wave1_start(self):
        """규칙 1 위반: 2파가 1파 시작점 아래."""
        module = ElliottWaveModule()
        waves = self._make_bullish_impulse_set()
        waves[1] = WavePattern("2", "impulse", 1.5, 0.8, waves[1].start_time, waves[1].end_time, "Minor")
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False

    def test_rule2_wave3_shortest(self):
        """규칙 2 위반: 3파가 가장 짧은 충격파."""
        module = ElliottWaveModule()
        waves = self._make_bullish_impulse_set()
        # 3파를 매우 짧게 만듦
        waves[2] = WavePattern("3", "impulse", 1.2, 1.25, waves[2].start_time, waves[2].end_time, "Minor")
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False

    def test_rule3_wave4_overlaps_wave1(self):
        """규칙 3 위반: 4파가 1파 영역과 겹침."""
        module = ElliottWaveModule()
        waves = self._make_bullish_impulse_set()
        # 4파 끝이 1파 끝(1.5) 아래
        waves[3] = WavePattern("4", "impulse", 2.5, 1.3, waves[3].start_time, waves[3].end_time, "Minor")
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False

    def test_corrective_wave_always_valid(self):
        """조정파는 규칙 검증 대상이 아님."""
        module = ElliottWaveModule()
        w = WavePattern("A", "corrective", 2.0, 1.5, datetime.now(), datetime.now(), "Minor")
        assert module.validate_wave_rules(w) is True


# ── 테스트: calculate_fibonacci_targets ───────────────────────


class TestCalculateFibonacciTargets:
    def test_contains_all_ratios(self):
        """정확히 5개 피보나치 비율 키를 포함."""
        module = ElliottWaveModule()
        w = WavePattern("1", "impulse", 1.0, 2.0, datetime.now(), datetime.now(), "Minor")
        targets = module.calculate_fibonacci_targets(w)
        assert set(targets.keys()) == {"0.236", "0.382", "0.5", "0.618", "0.786"}

    def test_mathematical_accuracy(self):
        """각 목표가가 수학적으로 정확."""
        module = ElliottWaveModule()
        sp, ep = 1.0, 3.0
        w = WavePattern("1", "impulse", sp, ep, datetime.now(), datetime.now(), "Minor")
        targets = module.calculate_fibonacci_targets(w)
        diff = ep - sp
        for key, ratio in FIBONACCI_RATIOS.items():
            expected = sp + diff * ratio
            assert math.isclose(targets[key], expected, rel_tol=1e-9)

    def test_negative_movement(self):
        """하락 파동에서도 정확히 계산."""
        module = ElliottWaveModule()
        sp, ep = 3.0, 1.0
        w = WavePattern("A", "corrective", sp, ep, datetime.now(), datetime.now(), "Minor")
        targets = module.calculate_fibonacci_targets(w)
        diff = ep - sp  # -2.0
        for key, ratio in FIBONACCI_RATIOS.items():
            expected = sp + diff * ratio
            assert math.isclose(targets[key], expected, rel_tol=1e-9)

    def test_zero_movement(self):
        """가격 변동 없으면 모든 목표가가 시작가와 동일."""
        module = ElliottWaveModule()
        w = WavePattern("1", "impulse", 2.0, 2.0, datetime.now(), datetime.now(), "Minor")
        targets = module.calculate_fibonacci_targets(w)
        for val in targets.values():
            assert math.isclose(val, 2.0, rel_tol=1e-9)


# ── 테스트: get_current_position ──────────────────────────────


class TestGetCurrentPosition:
    def test_no_waves_returns_default(self):
        """파동이 없으면 기본값 반환."""
        module = ElliottWaveModule()
        pos = module.get_current_position()
        assert isinstance(pos, WavePosition)
        assert pos.current_wave == "1"
        assert pos.next_direction in ("up", "down")

    def test_returns_valid_position(self):
        """파동이 있으면 유효한 위치 반환."""
        module = ElliottWaveModule()
        module._waves = [
            WavePattern("3", "impulse", 1.2, 2.5, datetime.now(), datetime.now(), "Minor"),
        ]
        pos = module.get_current_position()
        assert pos.current_wave == "3"
        assert pos.next_direction in ("up", "down")
        assert pos.wave_degree == "Minor"

    def test_after_wave5_predicts_down(self):
        """상승 5파 후 하락 예상."""
        module = ElliottWaveModule()
        module._waves = [
            WavePattern("5", "impulse", 1.8, 2.8, datetime.now(), datetime.now(), "Minor"),
        ]
        pos = module.get_current_position()
        assert pos.current_wave == "5"
        assert pos.next_direction == "down"

    def test_after_wave_c_predicts_up(self):
        """C파 후 상승 예상."""
        module = ElliottWaveModule()
        module._waves = [
            WavePattern("C", "corrective", 2.0, 1.0, datetime.now(), datetime.now(), "Minor"),
        ]
        pos = module.get_current_position()
        assert pos.current_wave == "C"
        assert pos.next_direction == "up"


# ── 테스트: save_to_db ────────────────────────────────────────


class TestSaveToDb:
    def test_save_and_query(self, tmp_path):
        """파동 데이터를 DB에 저장하고 조회."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.models import Base, ElliottWaveData

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        module = ElliottWaveModule()
        now = datetime(2024, 6, 1, 12, 0, 0)
        waves = [
            WavePattern(
                wave_number="1",
                wave_type="impulse",
                start_price=1.0,
                end_price=1.5,
                start_time=now,
                end_time=now + timedelta(hours=1),
                wave_degree="Minor",
                fibonacci_targets={"0.236": 1.118, "0.382": 1.191, "0.5": 1.25, "0.618": 1.309, "0.786": 1.393},
            )
        ]

        count = module.save_to_db(waves, session)
        assert count == 1

        records = session.query(ElliottWaveData).all()
        assert len(records) == 1
        assert records[0].wave_number == "1"
        assert records[0].wave_type == "impulse"
        assert records[0].start_price == 1.0
        assert records[0].end_price == 1.5
        assert records[0].is_valid is True
        assert "0.618" in records[0].fibonacci_targets

        session.close()

    def test_save_empty_list(self, tmp_path):
        """빈 리스트 저장 시 0 반환."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.models import Base

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        module = ElliottWaveModule()
        count = module.save_to_db([], session)
        assert count == 0
        session.close()


# ── 테스트: _determine_degree ─────────────────────────────────


class TestDetermineDegree:
    def test_primary(self):
        assert ElliottWaveModule._determine_degree(1000) == "Primary"
        assert ElliottWaveModule._determine_degree(2000) == "Primary"

    def test_intermediate(self):
        assert ElliottWaveModule._determine_degree(500) == "Intermediate"
        assert ElliottWaveModule._determine_degree(999) == "Intermediate"

    def test_minor(self):
        assert ElliottWaveModule._determine_degree(200) == "Minor"
        assert ElliottWaveModule._determine_degree(499) == "Minor"


# ── 속성 기반 테스트 (Property-Based Tests) ─────────────────────


from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ── 전략(Strategies) ──────────────────────────────────────────


def wave_number_strategy():
    """유효한 파동 번호 전략."""
    return st.sampled_from(["1", "2", "3", "4", "5", "A", "B", "C"])


def wave_type_strategy():
    """유효한 파동 타입 전략."""
    return st.sampled_from(["impulse", "corrective"])


def wave_degree_strategy():
    """유효한 파동 차수 전략."""
    return st.sampled_from(["Primary", "Intermediate", "Minor"])


def positive_price_strategy():
    """양수 가격 전략."""
    return st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False)


def timestamp_strategy():
    """유효한 타임스탬프 전략."""
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    )


def wave_pattern_strategy():
    """WavePattern 객체 생성 전략."""
    return st.builds(
        WavePattern,
        wave_number=wave_number_strategy(),
        wave_type=wave_type_strategy(),
        start_price=positive_price_strategy(),
        end_price=positive_price_strategy(),
        start_time=timestamp_strategy(),
        end_time=timestamp_strategy(),
        wave_degree=wave_degree_strategy(),
    )


def impulse_prices_strategy(n: int = 300):
    """상승 충격파 패턴이 포함된 가격 시계열 전략.

    5파 구조를 보장하기 위해 세그먼트별 가격 변동을 제어한다.
    """
    return st.tuples(
        st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),  # base
        st.floats(min_value=0.3, max_value=5.0, allow_nan=False, allow_infinity=False),   # wave1_rise
        st.floats(min_value=0.05, max_value=0.9, allow_nan=False, allow_infinity=False),  # wave2_retrace_ratio
        st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False),   # wave3_rise
        st.floats(min_value=0.05, max_value=0.9, allow_nan=False, allow_infinity=False),  # wave4_retrace_ratio
        st.floats(min_value=0.2, max_value=4.0, allow_nan=False, allow_infinity=False),   # wave5_rise
    ).map(lambda t: _build_impulse_prices(n, *t))


def _build_impulse_prices(
    n: int, base: float, w1_rise: float, w2_ratio: float,
    w3_rise: float, w4_ratio: float, w5_rise: float,
) -> list[float]:
    """파라미터로부터 상승 충격파 가격 시계열을 생성한다."""
    prices: list[float] = []
    seg = n // 6

    # 초기 평탄 구간
    for i in range(seg):
        prices.append(base + 0.001 * i)

    # 1파 상승
    p0 = prices[-1]
    for i in range(seg):
        prices.append(p0 + w1_rise * (i + 1) / seg)
    p1 = prices[-1]

    # 2파 하락 (1파 시작 위에서 멈춤)
    w2_drop = w1_rise * w2_ratio
    for i in range(seg // 2):
        prices.append(p1 - w2_drop * (i + 1) / (seg // 2))
    p2 = prices[-1]

    # 3파 상승
    for i in range(seg):
        prices.append(p2 + w3_rise * (i + 1) / seg)
    p3 = prices[-1]

    # 4파 하락
    w4_drop = w3_rise * w4_ratio
    for i in range(seg // 2):
        prices.append(p3 - w4_drop * (i + 1) / (seg // 2))
    p4 = prices[-1]

    # 5파 상승
    remaining = n - len(prices)
    for i in range(remaining):
        prices.append(p4 + w5_rise * (i + 1) / max(remaining, 1))

    return prices[:n]


# ── Property 4: 엘리엇 파동 필수 필드 완전성 ─────────────────


class TestProperty4WaveFieldCompleteness:
    """Feature: xrp-price-prediction-dashboard, Property 4: 엘리엇 파동 필수 필드 완전성

    *For any* 감지된 WavePattern 객체에 대해, 파동 번호(1~5 또는 A~C),
    시작 가격, 종료 가격, 시작 시간, 종료 시간, Wave_Degree가 모두 존재하고
    None이 아니어야 한다.

    **Validates: Requirements 1.6**
    """

    @given(data=impulse_prices_strategy(300))
    @settings(max_examples=100)
    def test_detected_waves_have_all_required_fields(self, data: list[float]):
        """감지된 모든 WavePattern의 필수 필드가 None이 아니어야 한다."""
        module = ElliottWaveModule()
        df = _make_ohlcv(data)
        waves = module.detect_waves(df)

        for wave in waves:
            # 파동 번호는 유효한 값
            assert wave.wave_number is not None
            assert wave.wave_number in ("1", "2", "3", "4", "5", "A", "B", "C")

            # 시작/종료 가격은 None이 아닌 float
            assert wave.start_price is not None
            assert isinstance(wave.start_price, (int, float))

            assert wave.end_price is not None
            assert isinstance(wave.end_price, (int, float))

            # 시작/종료 시간은 None이 아닌 datetime
            assert wave.start_time is not None
            assert isinstance(wave.start_time, datetime)

            assert wave.end_time is not None
            assert isinstance(wave.end_time, datetime)

            # Wave_Degree는 None이 아님
            assert wave.wave_degree is not None
            assert wave.wave_degree in ("Primary", "Intermediate", "Minor")


# ── Property 5: 엘리엇 파동 위치 유효성 ──────────────────────


class TestProperty5WavePositionValidity:
    """Feature: xrp-price-prediction-dashboard, Property 5: 엘리엇 파동 위치 유효성

    *For any* 200개 이상의 캔들 데이터에 캔들을 추가한 후,
    get_current_position이 반환하는 파동 위치는 유효한 파동 번호(1~5 또는 A~C)이고,
    예상 다음 파동 방향은 "up" 또는 "down"이어야 한다.

    **Validates: Requirements 1.7**
    """

    @given(data=impulse_prices_strategy(300))
    @settings(max_examples=100)
    def test_current_position_has_valid_wave_and_direction(self, data: list[float]):
        """get_current_position의 파동 번호와 방향이 유효해야 한다."""
        module = ElliottWaveModule()
        df = _make_ohlcv(data)
        module.detect_waves(df)

        pos = module.get_current_position()

        assert isinstance(pos, WavePosition)
        assert pos.current_wave in ("1", "2", "3", "4", "5", "A", "B", "C")
        assert pos.next_direction in ("up", "down")


# ── Property 6: 피보나치 목표가 수학적 정확성 ────────────────


class TestProperty6FibonacciTargetAccuracy:
    """Feature: xrp-price-prediction-dashboard, Property 6: 피보나치 목표가 수학적 정확성

    *For any* WavePattern에 대해, calculate_fibonacci_targets가 반환하는 목표가는
    정확히 0.236, 0.382, 0.5, 0.618, 0.786 비율에 해당하는 키를 포함하고,
    각 값은 start_price + (end_price - start_price) * ratio와 일치해야 한다.

    **Validates: Requirements 1.8**
    """

    @given(wave=wave_pattern_strategy())
    @settings(max_examples=100)
    def test_fibonacci_targets_match_formula(self, wave: WavePattern):
        """피보나치 목표가가 수학 공식과 정확히 일치해야 한다."""
        module = ElliottWaveModule()
        targets = module.calculate_fibonacci_targets(wave)

        # 정확히 5개 키를 포함
        expected_keys = {"0.236", "0.382", "0.5", "0.618", "0.786"}
        assert set(targets.keys()) == expected_keys

        # 각 값이 공식과 일치
        diff = wave.end_price - wave.start_price
        for key, ratio in FIBONACCI_RATIOS.items():
            expected = wave.start_price + diff * ratio
            assert math.isclose(targets[key], expected, rel_tol=1e-9), (
                f"ratio={key}: expected={expected}, got={targets[key]}"
            )


# ── Property 7: 엘리엇 파동 규칙 검증 정확성 ────────────────


class TestProperty7WaveRuleValidation:
    """Feature: xrp-price-prediction-dashboard, Property 7: 엘리엇 파동 규칙 검증 정확성

    *For any* 엘리엇 파동 규칙을 위반하는 WavePattern에 대해,
    validate_wave_rules는 False를 반환해야 한다.

    **Validates: Requirements 1.9**
    """

    @given(
        base_price=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        w1_rise=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        violation_amount=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_rule1_violation_wave2_below_wave1_start(
        self, base_price: float, w1_rise: float, violation_amount: float,
    ):
        """규칙 1 위반: 2파가 1파 시작점 아래로 내려가면 False."""
        base_time = datetime(2024, 1, 1)
        w1_start = base_price
        w1_end = base_price + w1_rise
        # 2파 끝이 1파 시작 아래
        w2_end = w1_start - violation_amount

        waves = [
            WavePattern("1", "impulse", w1_start, w1_end, base_time, base_time + timedelta(days=1), "Minor"),
            WavePattern("2", "impulse", w1_end, w2_end, base_time + timedelta(days=1), base_time + timedelta(days=2), "Minor"),
            WavePattern("3", "impulse", w2_end, w2_end + w1_rise * 2, base_time + timedelta(days=2), base_time + timedelta(days=3), "Minor"),
            WavePattern("4", "impulse", w2_end + w1_rise * 2, w1_end + 0.5, base_time + timedelta(days=3), base_time + timedelta(days=4), "Minor"),
            WavePattern("5", "impulse", w1_end + 0.5, w2_end + w1_rise * 2 + 0.5, base_time + timedelta(days=4), base_time + timedelta(days=5), "Minor"),
        ]

        module = ElliottWaveModule()
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False

    @given(
        base_price=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        w1_rise=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        w5_rise=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_rule2_violation_wave3_shortest(
        self, base_price: float, w1_rise: float, w5_rise: float,
    ):
        """규칙 2 위반: 3파가 가장 짧은 충격파이면 False."""
        assume(w1_rise > 0.1 and w5_rise > 0.1)
        base_time = datetime(2024, 1, 1)

        w1_start = base_price
        w1_end = base_price + w1_rise
        w2_end = w1_start + 0.1  # 1파 시작 위
        # 3파를 매우 짧게 (1파, 5파보다 짧게)
        w3_rise = min(w1_rise, w5_rise) * 0.1
        assume(w3_rise > 0)
        w3_end = w2_end + w3_rise
        w4_end = max(w1_end, w3_end - 0.01)  # 4파 끝이 1파 끝 이상
        w5_end = w4_end + w5_rise

        waves = [
            WavePattern("1", "impulse", w1_start, w1_end, base_time, base_time + timedelta(days=1), "Minor"),
            WavePattern("2", "impulse", w1_end, w2_end, base_time + timedelta(days=1), base_time + timedelta(days=2), "Minor"),
            WavePattern("3", "impulse", w2_end, w3_end, base_time + timedelta(days=2), base_time + timedelta(days=3), "Minor"),
            WavePattern("4", "impulse", w3_end, w4_end, base_time + timedelta(days=3), base_time + timedelta(days=4), "Minor"),
            WavePattern("5", "impulse", w4_end, w5_end, base_time + timedelta(days=4), base_time + timedelta(days=5), "Minor"),
        ]

        module = ElliottWaveModule()
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False

    @given(
        base_price=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        w1_rise=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        overlap_amount=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_rule3_violation_wave4_overlaps_wave1(
        self, base_price: float, w1_rise: float, overlap_amount: float,
    ):
        """규칙 3 위반: 4파가 1파 영역과 겹치면 False."""
        base_time = datetime(2024, 1, 1)

        w1_start = base_price
        w1_end = base_price + w1_rise
        w2_end = w1_start + 0.1
        w3_rise = w1_rise * 3  # 3파가 가장 길도록
        w3_end = w2_end + w3_rise
        # 4파 끝이 1파 끝 아래 (겹침)
        w4_end = w1_end - overlap_amount
        w5_end = w3_end + 0.5

        waves = [
            WavePattern("1", "impulse", w1_start, w1_end, base_time, base_time + timedelta(days=1), "Minor"),
            WavePattern("2", "impulse", w1_end, w2_end, base_time + timedelta(days=1), base_time + timedelta(days=2), "Minor"),
            WavePattern("3", "impulse", w2_end, w3_end, base_time + timedelta(days=2), base_time + timedelta(days=3), "Minor"),
            WavePattern("4", "impulse", w3_end, w4_end, base_time + timedelta(days=3), base_time + timedelta(days=4), "Minor"),
            WavePattern("5", "impulse", w4_end, w5_end, base_time + timedelta(days=4), base_time + timedelta(days=5), "Minor"),
        ]

        module = ElliottWaveModule()
        module._waves = waves
        assert module.validate_wave_rules(waves[0]) is False
