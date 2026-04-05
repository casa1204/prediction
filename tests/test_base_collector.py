"""BaseCollector 단위 테스트.

collect_with_retry 재시도 로직, 지수 백오프, 연속 실패 알림,
log_collection 메서드를 검증한다.

**Validates: Requirements 2.4, 3.4, 4.4, 7.3, 7.4**
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.base_collector import BaseCollector
from db.models import Base, CollectionLog


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """인메모리 SQLite 세션을 생성한다."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, engine
    session.close()
    engine.dispose()


# ── 테스트용 구현체 ──────────────────────────────────────────


class SuccessCollector(BaseCollector):
    """항상 성공하는 수집기."""

    source_name = "test_success"

    async def collect(self) -> dict:
        return {"price": 0.5}


class FailCollector(BaseCollector):
    """항상 실패하는 수집기."""

    source_name = "test_fail"

    async def collect(self) -> dict:
        raise ConnectionError("API 연결 실패")


class FailThenSucceedCollector(BaseCollector):
    """처음 N회 실패 후 성공하는 수집기."""

    source_name = "test_fail_then_succeed"

    def __init__(self, fail_count: int):
        super().__init__()
        self._fail_count = fail_count
        self._attempt = 0

    async def collect(self) -> dict:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            raise ConnectionError(f"실패 #{self._attempt}")
        return {"price": 1.0}


# ── Tests ─────────────────────────────────────────────────────


class TestCollectWithRetry:
    """collect_with_retry 재시도 및 폴백 동작 검증."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self, db_session):
        """첫 시도에 성공하면 데이터를 즉시 반환한다."""
        session, engine = db_session
        collector = SuccessCollector()

        with patch("collectors.base_collector.get_session", return_value=session):
            result = await collector.collect_with_retry()

        assert result == {"price": 0.5}
        assert collector._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_retry_then_success(self, db_session):
        """1회 실패 후 2번째 시도에 성공한다."""
        session, engine = db_session
        collector = FailThenSucceedCollector(fail_count=1)

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                result = await collector.collect_with_retry()

        assert result == {"price": 1.0}
        assert collector._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_all_retries_fail_returns_last_success(self, db_session):
        """모든 재시도 실패 시 마지막 성공 데이터를 반환한다."""
        session, engine = db_session
        collector = FailCollector()
        collector._last_success_data = {"price": 0.3}

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                result = await collector.collect_with_retry()

        assert result == {"price": 0.3}

    @pytest.mark.asyncio
    async def test_all_retries_fail_no_previous_data(self, db_session):
        """이전 성공 데이터 없이 모든 재시도 실패 시 빈 dict 반환."""
        session, engine = db_session
        collector = FailCollector()

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                result = await collector.collect_with_retry()

        assert result == {}

    @pytest.mark.asyncio
    async def test_max_retries_count(self, db_session):
        """정확히 MAX_RETRIES(3)회 시도한다."""
        session, engine = db_session
        collector = FailCollector()

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                await collector.collect_with_retry()

        assert collector._consecutive_failures == 3


class TestLogCollection:
    """log_collection 메서드 검증."""

    def test_log_saved_to_db(self, db_session):
        """수집 로그가 DB에 정상 저장된다."""
        session, engine = db_session
        collector = SuccessCollector()
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 5)

        with patch("collectors.base_collector.get_session", return_value=session):
            collector.log_collection("success", start, end)

        log = session.query(CollectionLog).first()
        assert log is not None
        assert log.source == "test_success"
        assert log.status == "success"
        assert log.start_time == start
        assert log.end_time == end
        assert log.error_message is None

    def test_failure_log_with_error_message(self, db_session):
        """실패 로그에 에러 메시지가 포함된다."""
        session, engine = db_session
        collector = FailCollector()
        collector._consecutive_failures = 1
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 1)

        with patch("collectors.base_collector.get_session", return_value=session):
            collector.log_collection("failure", start, end, error_message="timeout")

        log = session.query(CollectionLog).first()
        assert log is not None
        assert log.status == "failure"
        assert log.error_message == "timeout"
        assert log.consecutive_failures == 1


class TestConsecutiveFailureAlert:
    """연속 실패 카운터 및 알림 로직 검증."""

    @pytest.mark.asyncio
    async def test_alert_at_threshold(self, db_session):
        """연속 실패가 정확히 3회에 도달하면 알림이 발생한다."""
        session, engine = db_session
        collector = FailCollector()

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                with patch.object(
                    collector, "_alert_consecutive_failures"
                ) as mock_alert:
                    await collector.collect_with_retry()

        mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_counter_resets_on_success(self, db_session):
        """성공 시 연속 실패 카운터가 0으로 리셋된다."""
        session, engine = db_session
        collector = FailThenSucceedCollector(fail_count=2)

        with patch("collectors.base_collector.get_session", return_value=session):
            with patch("asyncio.sleep", return_value=None):
                await collector.collect_with_retry()

        assert collector._consecutive_failures == 0
