"""PriceCollector 단위 테스트.

CoinGecko API 응답을 모킹하여 PriceCollector의 데이터 수집, 파싱,
DB 저장, 타임존 변환 로직을 검증한다.

**Validates: Requirements 1.1, 1.2**
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.price_collector import (
    PriceCollector,
    _build_volume_map,
    _find_nearest_volume,
    _ms_to_et_datetime,
    _parse_ohlc,
)
from config import TIMEZONE
from db.models import Base, PriceData

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
def sample_ohlc_data():
    """CoinGecko OHLC 응답 샘플 데이터."""
    # 2024-01-15 12:00:00 UTC = 1705320000000 ms
    return [
        [1705320000000, 0.62, 0.63, 0.61, 0.625],
        [1705323600000, 0.625, 0.64, 0.62, 0.635],
        [1705327200000, 0.635, 0.65, 0.63, 0.645],
    ]


@pytest.fixture
def sample_market_data():
    """CoinGecko market_chart 응답 샘플 데이터."""
    return {
        "prices": [
            [1705320000000, 0.625],
            [1705323600000, 0.635],
        ],
        "total_volumes": [
            [1705320000000, 1500000.0],
            [1705323600000, 2000000.0],
            [1705327200000, 1800000.0],
        ],
    }


# ── 헬퍼 함수 테스트 ─────────────────────────────────────────


class TestBuildVolumeMap:
    def test_builds_map_from_valid_data(self):
        volumes = [[1705320000000, 1500000.0], [1705323600000, 2000000.0]]
        result = _build_volume_map(volumes)
        assert result == {1705320000000: 1500000.0, 1705323600000: 2000000.0}

    def test_empty_input(self):
        assert _build_volume_map([]) == {}

    def test_skips_invalid_entries(self):
        volumes = [[1705320000000, 1500000.0], [1705323600000]]
        result = _build_volume_map(volumes)
        assert len(result) == 1


class TestFindNearestVolume:
    def test_exact_match(self):
        volume_map = {1705320000000: 1500000.0, 1705323600000: 2000000.0}
        assert _find_nearest_volume(volume_map, 1705320000000) == 1500000.0

    def test_nearest_within_threshold(self):
        volume_map = {1705320000000: 1500000.0}
        # 30분 차이 (1800000ms) — 1시간 이내
        assert _find_nearest_volume(volume_map, 1705321800000) == 1500000.0

    def test_returns_zero_when_too_far(self):
        volume_map = {1705320000000: 1500000.0}
        # 2시간 차이 (7200000ms) — 1시간 초과
        assert _find_nearest_volume(volume_map, 1705327200000) == 0.0

    def test_empty_map(self):
        assert _find_nearest_volume({}, 1705320000000) == 0.0


class TestMsToEtDatetime:
    def test_converts_utc_to_eastern(self):
        # 2024-01-15 12:00:00 UTC
        ts_ms = 1705320000000
        result = _ms_to_et_datetime(ts_ms)
        # UTC-5 (EST) → 07:00:00 ET
        utc_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        expected = utc_dt.astimezone(_ET).replace(tzinfo=None)
        assert result == expected

    def test_result_has_no_tzinfo(self):
        result = _ms_to_et_datetime(1705320000000)
        assert result.tzinfo is None


class TestParseOhlc:
    def test_parses_valid_data(self, sample_ohlc_data):
        volume_map = {
            1705320000000: 1500000.0,
            1705323600000: 2000000.0,
            1705327200000: 1800000.0,
        }
        records = _parse_ohlc(sample_ohlc_data, volume_map)
        assert len(records) == 3
        assert records[0]["open"] == 0.62
        assert records[0]["high"] == 0.63
        assert records[0]["low"] == 0.61
        assert records[0]["close"] == 0.625
        assert records[0]["volume"] == 1500000.0

    def test_skips_invalid_entries(self):
        data = [
            [1705320000000, 0.62, 0.63, 0.61, 0.625],
            "invalid",
            [1705323600000, 0.625],  # too short
        ]
        records = _parse_ohlc(data, {})
        assert len(records) == 1

    def test_empty_data(self):
        assert _parse_ohlc([], {}) == []


# ── PriceCollector 통합 테스트 ────────────────────────────────


class TestPriceCollectorCollect:
    """PriceCollector.collect() 메서드 테스트."""

    @pytest.mark.asyncio
    async def test_collect_success(
        self, in_memory_session, sample_ohlc_data, sample_market_data
    ):
        """정상적인 API 응답으로 데이터를 수집하고 DB에 저장한다."""
        ohlc_response = httpx.Response(
            200, json=sample_ohlc_data, request=httpx.Request("GET", "http://test")
        )
        market_response = httpx.Response(
            200, json=sample_market_data, request=httpx.Request("GET", "http://test")
        )

        async def mock_get(url, **kwargs):
            if "ohlc" in url:
                return ohlc_response
            return market_response

        collector = PriceCollector()

        with patch("collectors.price_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await collector.collect()

        assert result["source"] == "price"
        assert result["count"] == 3
        assert len(result["records"]) == 3

        # DB에 저장되었는지 확인
        saved = in_memory_session.query(PriceData).all()
        assert len(saved) == 3

    @pytest.mark.asyncio
    async def test_collect_deduplicates(
        self, in_memory_session, sample_ohlc_data, sample_market_data
    ):
        """동일 타임스탬프의 데이터는 중복 저장하지 않는다."""
        ohlc_response = httpx.Response(
            200, json=sample_ohlc_data, request=httpx.Request("GET", "http://test")
        )
        market_response = httpx.Response(
            200, json=sample_market_data, request=httpx.Request("GET", "http://test")
        )

        async def mock_get(url, **kwargs):
            if "ohlc" in url:
                return ohlc_response
            return market_response

        collector = PriceCollector()

        with patch("collectors.price_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                await collector.collect()
                await collector.collect()  # 두 번째 호출

        saved = in_memory_session.query(PriceData).all()
        assert len(saved) == 3  # 중복 없이 3건

    @pytest.mark.asyncio
    async def test_collect_raises_on_api_error(self):
        """API 오류 시 예외를 발생시킨다 (BaseCollector의 재시도 로직이 처리)."""
        collector = PriceCollector()

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

    @pytest.mark.asyncio
    async def test_source_name(self):
        """source_name이 'price'인지 확인한다."""
        collector = PriceCollector()
        assert collector.source_name == "price"

    @pytest.mark.asyncio
    async def test_timestamps_are_eastern_time(
        self, in_memory_session, sample_ohlc_data, sample_market_data
    ):
        """저장된 타임스탬프가 미국 동부시간으로 변환되었는지 확인한다."""
        ohlc_response = httpx.Response(
            200, json=sample_ohlc_data, request=httpx.Request("GET", "http://test")
        )
        market_response = httpx.Response(
            200, json=sample_market_data, request=httpx.Request("GET", "http://test")
        )

        async def mock_get(url, **kwargs):
            if "ohlc" in url:
                return ohlc_response
            return market_response

        collector = PriceCollector()

        with patch("collectors.price_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await collector.collect()

        # 첫 번째 레코드: 2024-01-15 12:00:00 UTC → EST(UTC-5) = 07:00:00
        saved = in_memory_session.query(PriceData).order_by(PriceData.timestamp).first()
        utc_dt = datetime.fromtimestamp(1705320000, tz=timezone.utc)
        expected_et = utc_dt.astimezone(_ET).replace(tzinfo=None)
        assert saved.timestamp == expected_et
