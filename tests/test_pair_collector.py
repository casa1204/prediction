"""PairCollector 단위 테스트.

CoinGecko API 및 Yahoo Finance 응답을 모킹하여 PairCollector의
데이터 수집, 파싱, DB 저장, 타임존 변환, 중복 방지 로직을 검증한다.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.pair_collector import (
    PairCollector,
    _ms_to_et_datetime,
    _pandas_ts_to_et_datetime,
    _save_records,
)
from config import TIMEZONE
from db.models import Base, PairData

_ET = ZoneInfo(TIMEZONE)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def in_memory_session():
    """인메모리 SQLite 세션을 생성한다."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_coingecko_response():
    """CoinGecko market_chart 응답 샘플."""
    return {
        "prices": [
            [1705320000000, 0.000015],  # 2024-01-15 12:00 UTC
            [1705323600000, 0.000016],  # 2024-01-15 13:00 UTC
        ],
        "total_volumes": [
            [1705320000000, 500.0],
            [1705323600000, 600.0],
        ],
    }


@pytest.fixture
def sample_yfinance_hist():
    """Yahoo Finance history 샘플 DataFrame."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-12 16:00:00", tz="America/New_York"),
            pd.Timestamp("2024-01-15 16:00:00", tz="America/New_York"),
        ],
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [4780.0, 4800.0],
            "High": [4790.0, 4820.0],
            "Low": [4770.0, 4790.0],
            "Close": [4785.0, 4810.0],
            "Volume": [3000000, 3500000],
        },
        index=idx,
    )


# ── 헬퍼 함수 테스트 ─────────────────────────────────────────


class TestMsToEtDatetime:
    def test_converts_utc_to_eastern(self):
        # 2024-01-15 12:00:00 UTC → EST(UTC-5) = 07:00:00
        ts_ms = 1705320000000
        result = _ms_to_et_datetime(ts_ms)
        utc_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        expected = utc_dt.astimezone(_ET).replace(tzinfo=None)
        assert result == expected

    def test_result_has_no_tzinfo(self):
        result = _ms_to_et_datetime(1705320000000)
        assert result.tzinfo is None


class TestPandasTsToEtDatetime:
    def test_converts_tz_aware_timestamp(self):
        ts = pd.Timestamp("2024-01-15 16:00:00", tz="America/New_York")
        result = _pandas_ts_to_et_datetime(ts)
        assert result.tzinfo is None
        assert result.hour == 16

    def test_converts_tz_naive_timestamp(self):
        ts = pd.Timestamp("2024-01-15 12:00:00")
        result = _pandas_ts_to_et_datetime(ts)
        assert result.tzinfo is None
        # UTC 12:00 → ET 07:00 (EST)
        assert result.hour == 7


# ── _save_records 테스트 ──────────────────────────────────────


class TestSaveRecords:
    def test_saves_records_to_db(self, in_memory_session):
        records = [
            {
                "timestamp": datetime(2024, 1, 15, 7, 0, 0),
                "asset_name": "XRP/BTC",
                "price": 0.000015,
            },
            {
                "timestamp": datetime(2024, 1, 15, 8, 0, 0),
                "asset_name": "XRP/BTC",
                "price": 0.000016,
            },
        ]
        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            _save_records(records)

        saved = in_memory_session.query(PairData).all()
        assert len(saved) == 2

    def test_skips_duplicate_timestamp_asset(self, in_memory_session):
        records = [
            {
                "timestamp": datetime(2024, 1, 15, 7, 0, 0),
                "asset_name": "XRP/BTC",
                "price": 0.000015,
            },
        ]
        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            _save_records(records)
            _save_records(records)  # 중복 저장 시도

        saved = in_memory_session.query(PairData).all()
        assert len(saved) == 1

    def test_allows_same_timestamp_different_asset(self, in_memory_session):
        records = [
            {
                "timestamp": datetime(2024, 1, 15, 7, 0, 0),
                "asset_name": "XRP/BTC",
                "price": 0.000015,
            },
            {
                "timestamp": datetime(2024, 1, 15, 7, 0, 0),
                "asset_name": "XRP/ETH",
                "price": 0.00025,
            },
        ]
        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            _save_records(records)

        saved = in_memory_session.query(PairData).all()
        assert len(saved) == 2


# ── PairCollector 테스트 ──────────────────────────────────────


class TestPairCollectorSourceName:
    def test_source_name_is_pair(self):
        collector = PairCollector()
        assert collector.source_name == "pair"


class TestPairCollectorCollectCrypto:
    @pytest.mark.asyncio
    async def test_collect_crypto_pairs(self, in_memory_session, sample_coingecko_response):
        """CoinGecko API에서 암호화폐 페어 데이터를 수집한다."""
        response = httpx.Response(
            200,
            json=sample_coingecko_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        collector = PairCollector()

        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                with patch.object(collector, "_collect_indices", return_value=[]):
                    result = await collector.collect()

        assert result["source"] == "pair"
        assert result["count"] > 0
        # 5 pairs × 2 data points = 10
        assert result["count"] == 10

    @pytest.mark.asyncio
    async def test_crypto_timestamps_are_eastern(self, sample_coingecko_response):
        """수집된 암호화폐 타임스탬프가 동부시간으로 변환되었는지 확인."""
        response = httpx.Response(
            200,
            json=sample_coingecko_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        collector = PairCollector()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            records = await collector._collect_crypto_pairs()

        # 첫 번째 레코드: 2024-01-15 12:00 UTC → EST 07:00
        assert records[0]["timestamp"].tzinfo is None
        utc_dt = datetime.fromtimestamp(1705320000, tz=timezone.utc)
        expected = utc_dt.astimezone(_ET).replace(tzinfo=None)
        assert records[0]["timestamp"] == expected


class TestPairCollectorCollectIndices:
    def test_collect_indices(self, in_memory_session, sample_yfinance_hist):
        """Yahoo Finance에서 지수 데이터를 수집한다."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yfinance_hist

        collector = PairCollector()

        with patch("collectors.pair_collector.yf.Ticker", return_value=mock_ticker):
            records = collector._collect_indices()

        # 2 indices × 2 data points = 4
        assert len(records) == 4
        sp500_records = [r for r in records if r["asset_name"] == "S&P500"]
        nasdaq_records = [r for r in records if r["asset_name"] == "NASDAQ"]
        assert len(sp500_records) == 2
        assert len(nasdaq_records) == 2

    def test_index_timestamps_are_eastern(self, sample_yfinance_hist):
        """지수 타임스탬프가 동부시간으로 변환되었는지 확인."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yfinance_hist

        collector = PairCollector()

        with patch("collectors.pair_collector.yf.Ticker", return_value=mock_ticker):
            records = collector._collect_indices()

        for rec in records:
            assert rec["timestamp"].tzinfo is None

    def test_empty_history_skipped(self):
        """빈 히스토리는 건너뛴다."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        collector = PairCollector()

        with patch("collectors.pair_collector.yf.Ticker", return_value=mock_ticker):
            records = collector._collect_indices()

        assert len(records) == 0


class TestPairCollectorFullCollect:
    @pytest.mark.asyncio
    async def test_full_collect_saves_to_db(
        self, in_memory_session, sample_coingecko_response, sample_yfinance_hist
    ):
        """전체 collect()가 암호화폐 + 지수 데이터를 DB에 저장한다."""
        response = httpx.Response(
            200,
            json=sample_coingecko_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yfinance_hist

        collector = PairCollector()

        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                with patch("collectors.pair_collector.yf.Ticker", return_value=mock_ticker):
                    result = await collector.collect()

        assert result["source"] == "pair"
        # 5 crypto pairs × 2 + 2 indices × 2 = 14
        assert result["count"] == 14

        saved = in_memory_session.query(PairData).all()
        assert len(saved) == 14

    @pytest.mark.asyncio
    async def test_collect_deduplicates(
        self, in_memory_session, sample_coingecko_response, sample_yfinance_hist
    ):
        """동일 timestamp+asset_name 조합은 중복 저장하지 않는다."""
        response = httpx.Response(
            200,
            json=sample_coingecko_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yfinance_hist

        collector = PairCollector()

        with patch("collectors.pair_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                with patch("collectors.pair_collector.yf.Ticker", return_value=mock_ticker):
                    await collector.collect()
                    await collector.collect()  # 두 번째 호출

        saved = in_memory_session.query(PairData).all()
        assert len(saved) == 14  # 중복 없이 14건

    @pytest.mark.asyncio
    async def test_collect_raises_on_api_error(self):
        """API 오류 시 예외를 발생시킨다."""
        collector = PairCollector()

        async def mock_get(url, **kwargs):
            raise httpx.HTTPStatusError(
                "500 Server Error",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await collector.collect()
