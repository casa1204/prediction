"""FeatureEngineering 모듈 테스트.

기술 지표, 엘리엇 파동, 와이코프, 페어, 심리, 온체인 데이터를 통합하여
학습 데이터셋을 올바르게 구성하는지 검증한다.

Requirements: 5.1, 5.9
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import (
    Base,
    ElliottWaveData,
    OnchainData,
    PairData,
    PriceData,
    SentimentData,
    TechnicalIndicator,
    WyckoffData,
)
from prediction.feature_engineering import FeatureEngineering


@pytest.fixture
def session():
    """인메모리 SQLite DB 세션을 생성한다."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()


def _seed_price_data(session, n_days: int, start_date=None):
    """n_days일 분량의 가격 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        price = 0.5 + i * 0.01
        session.add(PriceData(
            timestamp=ts,
            open=price - 0.01,
            high=price + 0.02,
            low=price - 0.02,
            close=price,
            volume=1000.0 + i,
        ))
    session.commit()


def _seed_technical_indicators(session, n_days: int, start_date=None):
    """n_days일 분량의 기술 지표 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        session.add(TechnicalIndicator(
            timestamp=ts,
            rsi_14=50.0 + i * 0.1,
            macd=0.01 * i,
            macd_signal=0.005 * i,
            macd_histogram=0.005 * i,
            bb_upper=0.6 + i * 0.01,
            bb_middle=0.5 + i * 0.01,
            bb_lower=0.4 + i * 0.01,
            sma_5=0.5 + i * 0.01,
            sma_10=0.5 + i * 0.01,
            sma_20=0.5 + i * 0.01,
            sma_50=0.5 + i * 0.01,
            sma_200=0.5 + i * 0.01,
            ema_12=0.5 + i * 0.01,
            ema_26=0.5 + i * 0.01,
        ))
    session.commit()


def _seed_elliott_wave(session, n_days: int, start_date=None):
    """n_days일 분량의 엘리엇 파동 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    wave_numbers = ["1", "2", "3", "4", "5", "A", "B", "C"]
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        wn = wave_numbers[i % len(wave_numbers)]
        wt = "impulse" if wn in ("1", "2", "3", "4", "5") else "corrective"
        session.add(ElliottWaveData(
            timestamp=ts,
            wave_number=wn,
            wave_type=wt,
            start_price=0.5,
            end_price=0.6,
            start_time=ts,
            end_time=ts + timedelta(hours=12),
        ))
    session.commit()


def _seed_wyckoff(session, n_days: int, start_date=None):
    """n_days일 분량의 와이코프 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    phases = ["Accumulation", "Markup", "Distribution", "Markdown"]
    w_phases = ["A", "B", "C", "D", "E"]
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        session.add(WyckoffData(
            timestamp=ts,
            event_type="PS",
            pattern_type="accumulation",
            price=0.5 + i * 0.01,
            volume=1000.0,
            market_phase=phases[i % len(phases)],
            wyckoff_phase=w_phases[i % len(w_phases)],
            confidence_score=70.0 + i * 0.1,
        ))
    session.commit()


def _seed_pair_data(session, n_days: int, start_date=None):
    """n_days일 분량의 페어 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    assets = ["BTC/USD", "ETH/USD"]
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        for asset in assets:
            session.add(PairData(
                timestamp=ts,
                asset_name=asset,
                price=100.0 + i,
                correlation_with_xrp=0.5 + i * 0.001,
            ))
    session.commit()


def _seed_sentiment(session, n_days: int, start_date=None):
    """n_days일 분량의 심리 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        session.add(SentimentData(
            timestamp=ts,
            google_trend_score=50.0 + i * 0.1,
            sns_mention_score=40.0 + i * 0.1,
            sns_sentiment_score=60.0 + i * 0.1,
            fear_greed_index=45.0 + i * 0.1,
        ))
    session.commit()


def _seed_onchain(session, n_days: int, start_date=None):
    """n_days일 분량의 온체인 데이터를 시드한다."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    for i in range(n_days):
        ts = start_date + timedelta(days=i)
        session.add(OnchainData(
            timestamp=ts,
            active_wallets=1000 + i,
            new_wallets=50 + i,
            transaction_count=5000 + i * 10,
            total_volume_xrp=1e6 + i * 1000,
            whale_tx_count=5 + i % 3,
            whale_tx_volume=5e6 + i * 10000,
        ))
    session.commit()


def _seed_all(session, n_days: int):
    """모든 데이터 소스를 시드한다."""
    _seed_price_data(session, n_days)
    _seed_technical_indicators(session, n_days)
    _seed_elliott_wave(session, n_days)
    _seed_wyckoff(session, n_days)
    _seed_pair_data(session, n_days)
    _seed_sentiment(session, n_days)
    _seed_onchain(session, n_days)


# ── 테스트 ───────────────────────────────────────────────────


class TestFeatureEngineeringBuildDataset:
    """build_dataset 메서드 테스트."""

    def test_empty_db_returns_empty(self, session):
        """빈 DB에서는 빈 배열과 insufficient_data=True를 반환한다."""
        fe = FeatureEngineering()
        X, y, names, insufficient = fe.build_dataset(session)
        assert X.size == 0
        assert y.size == 0
        assert insufficient is True

    def test_full_dataset_shape(self, session):
        """모든 데이터 소스가 있을 때 올바른 shape을 반환한다."""
        n_days = 100
        _seed_all(session, n_days)

        fe = FeatureEngineering()
        X, y, names, insufficient = fe.build_dataset(session)

        # 타겟이 shift(-1)이므로 마지막 행 제외 → n_days - 1
        assert X.shape[0] == n_days - 1
        assert X.shape[1] == len(names)
        assert y.shape[0] == n_days - 1
        assert insufficient is False

    def test_feature_names_include_all_sources(self, session):
        """피처 이름에 모든 데이터 소스의 피처가 포함된다."""
        _seed_all(session, 100)

        fe = FeatureEngineering()
        _, _, names, _ = fe.build_dataset(session)

        # 기술 지표
        assert "rsi_14" in names
        assert "macd" in names
        assert "sma_200" in names

        # 엘리엇 파동
        assert "current_wave_number" in names
        assert "next_direction" in names

        # 와이코프
        assert "market_phase" in names
        assert "wyckoff_phase" in names
        assert "confidence_score" in names

        # 심리
        assert "google_trend_score" in names
        assert "fear_greed_index" in names

        # 온체인
        assert "active_wallets" in names
        assert "whale_tx_count" in names

        # 페어 (BTC/USD, ETH/USD)
        assert "BTC_USD_price" in names
        assert "ETH_USD_corr" in names

    def test_insufficient_data_warning(self, session):
        """90일 미만 데이터 시 insufficient_data=True를 반환한다."""
        _seed_all(session, 50)

        fe = FeatureEngineering()
        X, y, names, insufficient = fe.build_dataset(session)

        assert insufficient is True
        assert X.shape[0] == 49  # 50 - 1 (shift)

    def test_exactly_90_days_no_warning(self, session):
        """정확히 90일 데이터 시 insufficient_data=False를 반환한다."""
        # 90일 가격 + shift(-1) → 89 샘플, 89 < 90이므로 insufficient
        # 91일이면 90 샘플 → insufficient=False
        _seed_all(session, 91)

        fe = FeatureEngineering()
        _, _, _, insufficient = fe.build_dataset(session)

        assert insufficient is False

    def test_no_nan_in_output(self, session):
        """출력 X, y에 NaN이 없어야 한다."""
        _seed_all(session, 100)

        fe = FeatureEngineering()
        X, y, _, _ = fe.build_dataset(session)

        assert not np.isnan(X).any(), "X에 NaN이 존재합니다"
        assert not np.isnan(y).any(), "y에 NaN이 존재합니다"

    def test_target_is_next_day_close(self, session):
        """타겟(y)이 다음 날 종가인지 확인한다."""
        _seed_price_data(session, 10)

        fe = FeatureEngineering()
        X, y, _, _ = fe.build_dataset(session)

        # 가격: 0.5, 0.51, 0.52, ..., 0.59
        # y[0] = 다음 날 종가 = 0.51
        assert y.shape[0] == 9
        np.testing.assert_almost_equal(y[0], 0.51, decimal=5)
        np.testing.assert_almost_equal(y[1], 0.52, decimal=5)

    def test_missing_source_fills_zero(self, session):
        """일부 데이터 소스가 없어도 0으로 채워져 동작한다."""
        _seed_price_data(session, 100)
        # 기술 지표만 추가, 나머지는 없음
        _seed_technical_indicators(session, 100)

        fe = FeatureEngineering()
        X, y, names, _ = fe.build_dataset(session)

        assert X.shape[0] == 99
        # 엘리엇 파동 피처는 0으로 채워져야 함
        wave_idx = names.index("current_wave_number")
        assert (X[:, wave_idx] == 0.0).all()

    def test_partial_date_coverage(self, session):
        """데이터 소스 간 날짜 범위가 다를 때 누락 피처가 ffill/0으로 처리된다."""
        start = datetime(2024, 1, 1)
        _seed_price_data(session, 100, start)
        # 기술 지표는 50일부터만 존재
        _seed_technical_indicators(session, 50, start + timedelta(days=50))

        fe = FeatureEngineering()
        X, y, names, _ = fe.build_dataset(session)

        # NaN이 없어야 함
        assert not np.isnan(X).any()


class TestFeatureEngineeringEncoding:
    """인코딩 로직 테스트."""

    def test_elliott_wave_encoding(self, session):
        """엘리엇 파동 번호가 올바르게 인코딩된다."""
        _seed_price_data(session, 10)
        start = datetime(2024, 1, 1)
        # 파동 번호 "3" → 인코딩 3
        session.add(ElliottWaveData(
            timestamp=start,
            wave_number="3",
            wave_type="impulse",
            start_price=0.5,
            end_price=0.6,
            start_time=start,
        ))
        session.commit()

        fe = FeatureEngineering()
        X, _, names, _ = fe.build_dataset(session)

        wave_idx = names.index("current_wave_number")
        # 첫 번째 행의 파동 번호가 3이어야 함
        assert X[0, wave_idx] == 3.0

    def test_wyckoff_phase_encoding(self, session):
        """와이코프 시장 단계가 올바르게 인코딩된다."""
        _seed_price_data(session, 10)
        start = datetime(2024, 1, 1)
        session.add(WyckoffData(
            timestamp=start,
            event_type="PS",
            pattern_type="accumulation",
            price=0.5,
            volume=1000.0,
            market_phase="Distribution",
            wyckoff_phase="C",
            confidence_score=85.0,
        ))
        session.commit()

        fe = FeatureEngineering()
        X, _, names, _ = fe.build_dataset(session)

        mp_idx = names.index("market_phase")
        wp_idx = names.index("wyckoff_phase")
        cs_idx = names.index("confidence_score")

        assert X[0, mp_idx] == 3.0  # Distribution → 3
        assert X[0, wp_idx] == 3.0  # C → 3
        assert X[0, cs_idx] == 85.0
