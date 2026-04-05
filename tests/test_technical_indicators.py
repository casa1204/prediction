"""TechnicalIndicatorModule 단위 테스트."""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analysis.technical_indicators import TechnicalIndicatorModule
from db.models import Base, TechnicalIndicator


@pytest.fixture
def module():
    return TechnicalIndicatorModule()


@pytest.fixture
def ohlcv_df():
    """200개 이상의 캔들을 가진 테스트용 OHLCV DataFrame."""
    n = 250
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    # 간단한 사인파 기반 가격 데이터 생성
    import numpy as np

    np.random.seed(42)
    close_prices = 0.5 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n)) + 0.01 * np.random.randn(n)
    close_prices = np.abs(close_prices)  # 양수 보장

    data = {
        "timestamp": timestamps,
        "open": close_prices * (1 + 0.005 * np.random.randn(n)),
        "high": close_prices * (1 + abs(0.01 * np.random.randn(n))),
        "low": close_prices * (1 - abs(0.01 * np.random.randn(n))),
        "close": close_prices,
        "volume": np.random.uniform(1e6, 1e8, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def db_session():
    """인메모리 SQLite 세션."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestInterpolateMissing:
    def test_no_missing_values(self, module):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = module.interpolate_missing(df)
        assert result.isna().sum().sum() == 0

    def test_fills_nan_values(self, module):
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, float("nan")]})
        result = module.interpolate_missing(df)
        assert result.isna().sum().sum() == 0
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_empty_dataframe(self, module):
        df = pd.DataFrame()
        result = module.interpolate_missing(df)
        assert result.empty

    def test_leading_nan(self, module):
        df = pd.DataFrame({"a": [float("nan"), float("nan"), 3.0, 4.0]})
        result = module.interpolate_missing(df)
        assert result.isna().sum().sum() == 0


class TestCalculateAll:
    def test_returns_all_indicator_columns(self, module, ohlcv_df):
        result = module.calculate_all(ohlcv_df)
        expected_cols = [
            "timestamp", "rsi_14", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_range(self, module, ohlcv_df):
        result = module.calculate_all(ohlcv_df)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_band_ordering(self, module, ohlcv_df):
        result = module.calculate_all(ohlcv_df)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_sma_ema_positive(self, module, ohlcv_df):
        result = module.calculate_all(ohlcv_df)
        for col in ["sma_5", "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26"]:
            valid = result[col].dropna()
            assert (valid > 0).all(), f"{col} has non-positive values"

    def test_interpolation_applied(self, module, ohlcv_df):
        """보간 후 지표 컬럼에 NaN이 없어야 한다."""
        result = module.calculate_all(ohlcv_df)
        indicator_cols = [c for c in result.columns if c != "timestamp"]
        assert result[indicator_cols].isna().sum().sum() == 0

    def test_empty_dataframe(self, module):
        df = pd.DataFrame()
        result = module.calculate_all(df)
        assert result.empty

    def test_row_count_preserved(self, module, ohlcv_df):
        result = module.calculate_all(ohlcv_df)
        assert len(result) == len(ohlcv_df)


class TestSaveToDb:
    def test_saves_records(self, module, ohlcv_df, db_session):
        result = module.calculate_all(ohlcv_df)
        count = module.save_to_db(result, db_session)
        assert count == len(result)
        db_count = db_session.query(TechnicalIndicator).count()
        assert db_count == len(result)

    def test_skips_duplicate_timestamps(self, module, ohlcv_df, db_session):
        result = module.calculate_all(ohlcv_df)
        first_count = module.save_to_db(result, db_session)
        second_count = module.save_to_db(result, db_session)
        assert first_count == len(result)
        assert second_count == 0

    def test_roundtrip_values(self, module, ohlcv_df, db_session):
        """저장 후 조회하면 원본 값과 일치해야 한다."""
        result = module.calculate_all(ohlcv_df)
        module.save_to_db(result, db_session)

        first_row = result.iloc[0]
        ts = first_row["timestamp"]
        record = db_session.query(TechnicalIndicator).filter_by(timestamp=ts).first()
        assert record is not None
        assert record.rsi_14 == pytest.approx(first_row["rsi_14"], abs=1e-6)

    def test_empty_dataframe(self, module, db_session):
        df = pd.DataFrame()
        count = module.save_to_db(df, db_session)
        assert count == 0


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def ohlcv_strategy(draw):
    """200개 이상의 양수 OHLCV 캔들 데이터를 생성하는 hypothesis 전략.

    - close: 0.01 ~ 10000 양수 float
    - high >= close, low <= close, open > 0
    - volume > 0
    """
    n = draw(st.integers(min_value=200, max_value=300))

    close = draw(
        arrays(np.float64, n, elements=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False))
    )
    high_add = draw(
        arrays(np.float64, n, elements=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    )
    low_frac = draw(
        arrays(np.float64, n, elements=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False))
    )
    open_fac = draw(
        arrays(np.float64, n, elements=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False))
    )
    volume = draw(
        arrays(np.float64, n, elements=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False))
    )

    high = close + high_add
    low = np.maximum(close * (1.0 - low_frac), 0.001)
    opn = np.maximum(close * open_fac, 0.001)

    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestInterpolateMissingCompletenessProperty:
    """Property 2: 결측값 보간 완전성

    **Validates: Requirements 1.3**

    결측값이 포함된 OHLCV 데이터프레임에 대해, interpolate_missing 적용 후
    결과 데이터프레임에는 NaN 값이 존재하지 않아야 한다.

    Feature: xrp-price-prediction-dashboard
    Property 2: 결측값 보간 완전성
    """

    @given(data=st.data())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow, HealthCheck.data_too_large],
        deadline=None,
    )
    def test_interpolate_missing_removes_all_nans(self, data):
        """속성 테스트: 결측값이 포함된 DataFrame에 interpolate_missing 적용 후 NaN이 없어야 한다."""
        module = TechnicalIndicatorModule()

        # 컬럼 수: 2~6개
        n_cols = data.draw(st.integers(min_value=2, max_value=6), label="n_cols")
        # 행 수: 5~100개
        n_rows = data.draw(st.integers(min_value=5, max_value=100), label="n_rows")

        col_names = [f"col_{i}" for i in range(n_cols)]
        df_dict = {}

        for col in col_names:
            # 각 컬럼에 대해 유효한 float 값 배열 생성
            values = data.draw(
                arrays(
                    np.float64,
                    n_rows,
                    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                ),
                label=f"values_{col}",
            )

            # 랜덤하게 NaN 삽입 (최소 1개의 유효값은 보존)
            nan_mask = data.draw(
                arrays(
                    np.bool_,
                    n_rows,
                    elements=st.booleans(),
                ),
                label=f"nan_mask_{col}",
            )
            # 최소 1개의 유효값 보장: 모두 True(NaN)이면 첫 번째를 False로
            if nan_mask.all():
                nan_mask[0] = False

            values_with_nans = values.copy()
            values_with_nans[nan_mask] = np.nan
            df_dict[col] = values_with_nans

        df = pd.DataFrame(df_dict)

        result = module.interpolate_missing(df)

        # 보간 후 NaN이 없어야 한다
        nan_count = result.isna().sum().sum()
        assert nan_count == 0, (
            f"interpolate_missing 후에도 {nan_count}개의 NaN이 남아 있습니다.\n"
            f"입력 NaN 개수: {df.isna().sum().sum()}\n"
            f"결과 NaN 위치:\n{result.isna()}"
        )

        # 결과 shape이 입력과 동일해야 한다
        assert result.shape == df.shape, (
            f"결과 shape {result.shape}이 입력 shape {df.shape}과 다릅니다."
        )


class TestTechnicalIndicatorRangeProperty:
    """Property 1: 기술 지표 범위 유효성

    **Validates: Requirements 1.1**

    모든 OHLCV 데이터프레임에 대해, calculate_all로 계산된:
    - RSI는 0~100 범위
    - MACD 히스토그램은 실수
    - 볼린저밴드 상단 >= 중단, 하단 <= 중단
    - SMA/EMA는 양수

    Feature: xrp-price-prediction-dashboard
    Property 1: 기술 지표 범위 유효성
    """

    @given(df=ohlcv_strategy())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow, HealthCheck.data_too_large],
        deadline=None,
    )
    def test_indicator_ranges_property(self, df):
        """속성 테스트: 임의의 OHLCV 데이터에 대해 기술 지표 범위가 유효해야 한다."""
        module = TechnicalIndicatorModule()
        result = module.calculate_all(df)

        # RSI: 0 ~ 100
        rsi = result["rsi_14"]
        assert (rsi >= 0).all(), "RSI has values below 0"
        assert (rsi <= 100).all(), "RSI has values above 100"

        # MACD histogram: 실수 (NaN 없음, 보간 후)
        macd_hist = result["macd_histogram"]
        assert macd_hist.apply(lambda x: isinstance(x, (int, float)) and not math.isnan(x)).all(), (
            "MACD histogram contains non-finite values"
        )

        # 볼린저밴드: upper >= middle >= lower
        assert (result["bb_upper"] >= result["bb_middle"]).all(), (
            "Bollinger upper band is less than middle band"
        )
        assert (result["bb_lower"] <= result["bb_middle"]).all(), (
            "Bollinger lower band is greater than middle band"
        )

        # SMA: 양수
        for period in [5, 10, 20, 50, 200]:
            col = f"sma_{period}"
            assert (result[col] > 0).all(), f"{col} has non-positive values"

        # EMA: 양수
        for period in [12, 26]:
            col = f"ema_{period}"
            assert (result[col] > 0).all(), f"{col} has non-positive values"
