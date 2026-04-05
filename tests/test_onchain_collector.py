"""OnchainCollector 단위 테스트.

XRPL Data API / Bithomp API 응답을 모킹하여
OnchainCollector의 데이터 수집, 고래 거래 필터링, DB 저장, 타임존 변환 로직을 검증한다.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.onchain_collector import (
    OnchainCollector,
    _save_record,
    filter_whale_transactions,
)
from config import WHALE_THRESHOLD_XRP
from db.models import Base, OnchainData


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
def sample_network_metrics_response():
    """XRPL server_info JSON-RPC 응답 샘플."""
    return {
        "result": {
            "info": {
                "validated_ledger": {
                    "accounts": 45000,
                    "reserve_base_xrp": 10,
                },
                "load": {
                    "transaction_count": 1500000,
                },
            },
            "status": "success",
        }
    }


@pytest.fixture
def sample_transactions_response():
    """XRPL ledger JSON-RPC 응답 샘플 (고래 거래 포함)."""
    return {
        "result": {
            "ledger": {
                "transactions": [
                    {"TransactionType": "Payment", "Amount": "500000000000"},      # 500K XRP
                    {"TransactionType": "Payment", "Amount": "2000000000000"},     # 2M XRP
                    {"TransactionType": "Payment", "Amount": "100000000"},          # 100 XRP
                    {"TransactionType": "Payment", "Amount": "1000000000000"},     # 1M XRP
                    {"TransactionType": "Payment", "Amount": "5000000000000"},     # 5M XRP
                    {"TransactionType": "Payment", "Amount": "999999000000"},      # 999999 XRP
                    {"TransactionType": "OfferCreate", "Amount": "9000000000000"}, # not Payment
                ]
            },
            "status": "success",
        }
    }


# ── OnchainCollector 기본 테스트 ──────────────────────────────


class TestOnchainCollectorBasic:
    def test_source_name(self):
        collector = OnchainCollector()
        assert collector.source_name == "onchain"

    def test_default_rpc_url(self):
        collector = OnchainCollector()
        assert "ripple.com" in collector._rpc_url

    def test_custom_rpc_url(self):
        collector = OnchainCollector(rpc_url="http://test-xrpl")
        assert collector._rpc_url == "http://test-xrpl"


# ── 고래 거래 필터링 테스트 ───────────────────────────────────


class TestFilterWhaleTransactions:
    def test_filters_above_threshold(self):
        """threshold 이상인 거래만 반환한다."""
        transactions = [
            {"amount": 500000},
            {"amount": 2000000},
            {"amount": 1000000},
            {"amount": 999999},
        ]
        result = filter_whale_transactions(transactions)
        assert len(result) == 2
        assert result[0]["amount"] == 2000000
        assert result[1]["amount"] == 1000000

    def test_exact_threshold_included(self):
        """정확히 threshold와 같은 거래도 포함한다."""
        transactions = [{"amount": WHALE_THRESHOLD_XRP}]
        result = filter_whale_transactions(transactions)
        assert len(result) == 1

    def test_below_threshold_excluded(self):
        """threshold 미만인 거래는 제외한다."""
        transactions = [{"amount": WHALE_THRESHOLD_XRP - 1}]
        result = filter_whale_transactions(transactions)
        assert len(result) == 0

    def test_empty_list(self):
        """빈 리스트를 입력하면 빈 리스트를 반환한다."""
        result = filter_whale_transactions([])
        assert result == []

    def test_custom_threshold(self):
        """커스텀 threshold를 사용할 수 있다."""
        transactions = [
            {"amount": 50},
            {"amount": 100},
            {"amount": 150},
        ]
        result = filter_whale_transactions(transactions, threshold=100)
        assert len(result) == 2

    def test_missing_amount_defaults_to_zero(self):
        """amount 키가 없으면 0으로 처리하여 제외한다."""
        transactions = [{"hash": "tx1"}, {"amount": 2000000}]
        result = filter_whale_transactions(transactions)
        assert len(result) == 1
        assert result[0]["amount"] == 2000000

    def test_string_amount_converted(self):
        """amount가 문자열이어도 float 변환하여 필터링한다."""
        transactions = [
            {"amount": "2000000"},
            {"amount": "500000"},
        ]
        result = filter_whale_transactions(transactions)
        assert len(result) == 1

    def test_no_whale_transactions(self):
        """고래 거래가 없으면 빈 리스트를 반환한다."""
        transactions = [
            {"amount": 100},
            {"amount": 200},
            {"amount": 500},
        ]
        result = filter_whale_transactions(transactions)
        assert len(result) == 0

    def test_all_whale_transactions(self):
        """모든 거래가 고래 거래이면 전부 반환한다."""
        transactions = [
            {"amount": 1000000},
            {"amount": 2000000},
            {"amount": 5000000},
        ]
        result = filter_whale_transactions(transactions)
        assert len(result) == 3


# ── DB 저장 테스트 ────────────────────────────────────────────


class TestSaveRecord:
    def test_saves_record(self, in_memory_session):
        """레코드를 DB에 저장한다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "active_wallets": 45000,
            "new_wallets": 1200,
            "transaction_count": 1500000,
            "total_volume_xrp": 850000000.5,
            "whale_tx_count": 3,
            "whale_tx_volume": 8000000.0,
        }

        with patch(
            "collectors.onchain_collector.get_session",
            return_value=in_memory_session,
        ):
            _save_record(record)

        saved = in_memory_session.query(OnchainData).all()
        assert len(saved) == 1
        assert saved[0].active_wallets == 45000
        assert saved[0].new_wallets == 1200
        assert saved[0].transaction_count == 1500000
        assert saved[0].total_volume_xrp == 850000000.5
        assert saved[0].whale_tx_count == 3
        assert saved[0].whale_tx_volume == 8000000.0

    def test_skips_duplicate_timestamp(self, in_memory_session):
        """중복 타임스탬프는 건너뛴다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "active_wallets": 45000,
            "new_wallets": 1200,
            "transaction_count": 1500000,
            "total_volume_xrp": 850000000.5,
            "whale_tx_count": 3,
            "whale_tx_volume": 8000000.0,
        }

        with patch(
            "collectors.onchain_collector.get_session",
            return_value=in_memory_session,
        ):
            _save_record(record)
            _save_record(record)  # 두 번째 호출

        saved = in_memory_session.query(OnchainData).all()
        assert len(saved) == 1

    def test_saves_with_none_values(self, in_memory_session):
        """일부 값이 None이어도 저장한다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "active_wallets": None,
            "new_wallets": None,
            "transaction_count": None,
            "total_volume_xrp": None,
            "whale_tx_count": 0,
            "whale_tx_volume": 0.0,
        }

        with patch(
            "collectors.onchain_collector.get_session",
            return_value=in_memory_session,
        ):
            _save_record(record)

        saved = in_memory_session.query(OnchainData).first()
        assert saved.active_wallets is None
        assert saved.whale_tx_count == 0


# ── OnchainCollector.collect() 통합 테스트 ────────────────────


class TestOnchainCollectorCollect:
    @pytest.mark.asyncio
    async def test_collect_success(
        self,
        in_memory_session,
        sample_network_metrics_response,
        sample_transactions_response,
    ):
        """네트워크 메트릭과 고래 거래를 수집하고 DB에 저장한다."""
        collector = OnchainCollector()

        async def mock_post(url, **kwargs):
            body = kwargs.get("json", {})
            method = body.get("method", "")
            if method == "server_info":
                return httpx.Response(200, json=sample_network_metrics_response, request=httpx.Request("POST", url))
            elif method == "ledger":
                return httpx.Response(200, json=sample_transactions_response, request=httpx.Request("POST", url))
            return httpx.Response(404, request=httpx.Request("POST", url))

        with patch("collectors.onchain_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = mock_post
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await collector.collect()

        assert result["source"] == "onchain"
        assert result["active_wallets"] == 45000
        # 고래 거래: 2M, 1M, 5M XRP → 3건
        assert result["whale_tx_count"] == 3

        saved = in_memory_session.query(OnchainData).all()
        assert len(saved) == 1

    @pytest.mark.asyncio
    async def test_collect_deduplicates(
        self,
        in_memory_session,
        sample_network_metrics_response,
        sample_transactions_response,
    ):
        """동일 날짜에 두 번 수집해도 중복 저장하지 않는다."""
        collector = OnchainCollector()

        async def mock_post(url, **kwargs):
            body = kwargs.get("json", {})
            method = body.get("method", "")
            if method == "server_info":
                return httpx.Response(200, json=sample_network_metrics_response, request=httpx.Request("POST", url))
            return httpx.Response(200, json=sample_transactions_response, request=httpx.Request("POST", url))

        with patch("collectors.onchain_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = mock_post
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                await collector.collect()
                await collector.collect()

        saved = in_memory_session.query(OnchainData).all()
        assert len(saved) == 1

    @pytest.mark.asyncio
    async def test_timestamp_is_eastern_time_midnight(
        self,
        in_memory_session,
        sample_network_metrics_response,
        sample_transactions_response,
    ):
        """저장된 타임스탬프가 동부시간 자정으로 정규화되었는지 확인한다."""
        collector = OnchainCollector()

        async def mock_post(url, **kwargs):
            body = kwargs.get("json", {})
            method = body.get("method", "")
            if method == "server_info":
                return httpx.Response(200, json=sample_network_metrics_response, request=httpx.Request("POST", url))
            return httpx.Response(200, json=sample_transactions_response, request=httpx.Request("POST", url))

        with patch("collectors.onchain_collector.get_session", return_value=in_memory_session):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = mock_post
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                await collector.collect()

        saved = in_memory_session.query(OnchainData).first()
        assert saved.timestamp.hour == 0
        assert saved.timestamp.minute == 0

    @pytest.mark.asyncio
    async def test_collect_raises_on_rpc_failure(self):
        """XRPL RPC 실패 시 예외를 발생시킨다."""
        collector = OnchainCollector()

        async def mock_post(url, **kwargs):
            raise httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", url), response=httpx.Response(500),
            )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception):
                await collector.collect()
