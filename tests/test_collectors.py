"""수집기 재시도 및 폴백 속성 기반 테스트.

Property 10: 수집기 재시도 및 폴백
BaseCollector 구현체에서 API 호출이 실패할 때, collect_with_retry는
최대 3회 재시도하고, 모든 재시도 실패 시 마지막 성공 데이터를 반환하며
오류를 로그에 기록해야 한다.

**Validates: Requirements 2.4, 3.4, 4.4**

Feature: xrp-price-prediction-dashboard
Property 10: 수집기 재시도 및 폴백
"""

import asyncio
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.base_collector import BaseCollector
from db.models import Base, CollectionLog


# ── Strategies ────────────────────────────────────────────────

# 수집 데이터로 사용할 dict 전략: 키는 짧은 문자열, 값은 유한한 float
_collected_data_st = st.dictionaries(
    keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
    values=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    min_size=1,
    max_size=5,
)

# 실패 횟수 전략: 0 ~ 5 (MAX_RETRIES=3 전후를 포함)
_fail_count_st = st.integers(min_value=0, max_value=5)

# 마지막 성공 데이터 전략: None 또는 유효한 dict
_last_success_st = st.one_of(
    st.none(),
    _collected_data_st,
)


# ── 테스트용 수집기 구현체 ────────────────────────────────────


class ConfigurableCollector(BaseCollector):
    """테스트용 수집기: fail_count회 실패 후 성공 데이터를 반환한다."""

    source_name = "test_configurable"

    def __init__(self, fail_count: int, success_data: dict):
        super().__init__()
        self._fail_count = fail_count
        self._success_data = success_data
        self._attempt = 0

    async def collect(self) -> dict:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            raise ConnectionError(f"API 연결 실패 #{self._attempt}")
        return self._success_data


class AlwaysFailCollector(BaseCollector):
    """항상 실패하는 수집기."""

    source_name = "test_always_fail"

    async def collect(self) -> dict:
        raise ConnectionError("API 연결 실패")


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


# ── Property Tests ────────────────────────────────────────────


class TestCollectorRetryAndFallbackProperty:
    """Property 10: 수집기 재시도 및 폴백 속성 테스트.

    **Validates: Requirements 2.4, 3.4, 4.4**
    """

    @settings(max_examples=100)
    @given(
        fail_count=st.integers(min_value=0, max_value=2),
        success_data=_collected_data_st,
    )
    @pytest.mark.asyncio
    async def test_retry_eventually_succeeds_within_max_retries(
        self, fail_count, success_data
    ):
        """재시도 횟수가 MAX_RETRIES 이내이면 최종적으로 성공 데이터를 반환한다.

        **Validates: Requirements 2.4, 3.4, 4.4**
        """
        collector = ConfigurableCollector(
            fail_count=fail_count, success_data=success_data
        )

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    result = await collector.collect_with_retry()

            # 재시도 내에 성공하면 성공 데이터를 반환해야 한다
            assert result == success_data
            # 성공 후 연속 실패 카운터는 0이어야 한다
            assert collector._consecutive_failures == 0
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(last_success_data=_last_success_st)
    @pytest.mark.asyncio
    async def test_all_retries_fail_returns_fallback(self, last_success_data):
        """모든 재시도 실패 시 마지막 성공 데이터를 반환한다.

        마지막 성공 데이터가 없으면 빈 dict를 반환한다.

        **Validates: Requirements 2.4, 3.4, 4.4**
        """
        collector = AlwaysFailCollector()
        collector._last_success_data = last_success_data

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    result = await collector.collect_with_retry()

            if last_success_data is not None:
                assert result == last_success_data
            else:
                assert result == {}
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(
        fail_count=st.integers(min_value=0, max_value=5),
        success_data=_collected_data_st,
    )
    @pytest.mark.asyncio
    async def test_max_retry_count_is_three(self, fail_count, success_data):
        """collect_with_retry는 최대 3회(MAX_RETRIES)까지만 시도한다.

        **Validates: Requirements 2.4, 3.4, 4.4**
        """
        collector = ConfigurableCollector(
            fail_count=fail_count, success_data=success_data
        )

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    await collector.collect_with_retry()

            # 시도 횟수는 MAX_RETRIES(3) 이하여야 한다
            assert collector._attempt <= 3
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(
        fail_count=st.integers(min_value=1, max_value=5),
        success_data=_collected_data_st,
    )
    @pytest.mark.asyncio
    async def test_failures_are_logged(self, fail_count, success_data):
        """실패한 시도는 모두 CollectionLog에 'failure' 상태로 기록된다.

        **Validates: Requirements 2.4, 3.4, 4.4**
        """
        collector = ConfigurableCollector(
            fail_count=fail_count, success_data=success_data
        )

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    await collector.collect_with_retry()

            logs = session.query(CollectionLog).all()
            failure_logs = [log for log in logs if log.status == "failure"]
            success_logs = [log for log in logs if log.status == "success"]

            actual_failures = min(fail_count, 3)  # MAX_RETRIES 제한
            succeeded = fail_count < 3

            # 실패 로그 수는 실제 실패 횟수와 일치해야 한다
            assert len(failure_logs) == actual_failures

            # 모든 실패 로그에 에러 메시지가 있어야 한다
            for flog in failure_logs:
                assert flog.error_message is not None
                assert len(flog.error_message) > 0

            # 성공했으면 성공 로그가 1개 있어야 한다
            if succeeded:
                assert len(success_logs) == 1
        finally:
            session.close()
            engine.dispose()


# ── Property 19: 연속 실패 알림 정확성 ────────────────────────


class SequenceCollector(BaseCollector):
    """테스트용 수집기: 호출 시퀀스에 따라 성공/실패를 결정한다.

    outcomes 리스트의 각 원소가 True이면 성공, False이면 실패.
    outcomes를 모두 소진하면 이후 호출은 항상 실패한다.
    """

    source_name = "test_sequence"

    def __init__(self, outcomes: list[bool], success_data: dict):
        super().__init__()
        self._outcomes = list(outcomes)
        self._success_data = success_data
        self._call_index = 0

    async def collect(self) -> dict:
        idx = self._call_index
        self._call_index += 1
        if idx < len(self._outcomes) and self._outcomes[idx]:
            return self._success_data
        raise ConnectionError(f"API 연결 실패 #{idx}")


class TestConsecutiveFailureAlertProperty:
    """Property 19: 연속 실패 알림 정확성 속성 테스트.

    수집 실패 시퀀스에 대해, 연속 실패 횟수가 정확히 3회에 도달했을 때만
    관리자 알림(_alert_consecutive_failures)이 발송되어야 한다.

    **Validates: Requirements 7.3**

    Feature: xrp-price-prediction-dashboard
    Property 19: 연속 실패 알림 정확성
    """

    @settings(max_examples=100)
    @given(
        initial_failures=st.integers(min_value=0, max_value=2),
    )
    @pytest.mark.asyncio
    async def test_alert_fires_exactly_when_threshold_reached(
        self, initial_failures
    ):
        """연속 실패가 정확히 CONSECUTIVE_FAILURE_ALERT_THRESHOLD(3)에 도달하면
        _alert_consecutive_failures가 정확히 1회 호출된다.

        initial_failures(0~2)만큼 이미 실패한 상태에서 collect_with_retry를
        호출하여 모든 재시도가 실패하면, 연속 실패가 3에 도달하는 시점에서
        알림이 정확히 1회 발송되어야 한다.

        **Validates: Requirements 7.3**
        """
        collector = AlwaysFailCollector()
        collector._consecutive_failures = initial_failures

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    with patch.object(
                        collector, "_alert_consecutive_failures"
                    ) as mock_alert:
                        await collector.collect_with_retry()

            # initial_failures + MAX_RETRIES(3) 범위 내에서
            # 연속 실패가 정확히 3에 도달하면 알림 1회, 아니면 0회
            threshold = 3  # CONSECUTIVE_FAILURE_ALERT_THRESHOLD
            final_failures = initial_failures + 3  # MAX_RETRIES=3, 모두 실패

            # 임계값을 이번 호출에서 정확히 통과했는지 확인
            crosses_threshold = (
                initial_failures < threshold <= final_failures
            )

            if crosses_threshold:
                assert mock_alert.call_count == 1, (
                    f"initial={initial_failures}, final={final_failures}: "
                    f"알림이 정확히 1회 호출되어야 하지만 {mock_alert.call_count}회 호출됨"
                )
            else:
                assert mock_alert.call_count == 0, (
                    f"initial={initial_failures}, final={final_failures}: "
                    f"알림이 호출되지 않아야 하지만 {mock_alert.call_count}회 호출됨"
                )
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(
        initial_failures=st.integers(min_value=3, max_value=10),
    )
    @pytest.mark.asyncio
    async def test_alert_not_fired_when_already_past_threshold(
        self, initial_failures
    ):
        """이미 연속 실패가 임계값(3) 이상인 상태에서 추가 실패가 발생해도
        알림이 다시 발송되지 않아야 한다.

        **Validates: Requirements 7.3**
        """
        collector = AlwaysFailCollector()
        collector._consecutive_failures = initial_failures

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    with patch.object(
                        collector, "_alert_consecutive_failures"
                    ) as mock_alert:
                        await collector.collect_with_retry()

            # 이미 임계값을 넘었으므로 알림이 발송되지 않아야 한다
            assert mock_alert.call_count == 0, (
                f"initial={initial_failures}: "
                f"이미 임계값 초과 상태에서 알림이 {mock_alert.call_count}회 호출됨"
            )
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(
        num_rounds=st.integers(min_value=1, max_value=5),
    )
    @pytest.mark.asyncio
    async def test_alert_fires_once_across_multiple_retry_rounds(
        self, num_rounds
    ):
        """여러 번의 collect_with_retry 호출에 걸쳐 연속 실패가 누적될 때,
        임계값(3)에 정확히 도달하는 시점에서만 알림이 1회 발송된다.

        각 라운드에서 모든 재시도가 실패하는 시나리오.

        **Validates: Requirements 7.3**
        """
        collector = AlwaysFailCollector()

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        total_alert_count = 0

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    for _ in range(num_rounds):
                        with patch.object(
                            collector, "_alert_consecutive_failures"
                        ) as mock_alert:
                            await collector.collect_with_retry()
                        total_alert_count += mock_alert.call_count

            # 여러 라운드에 걸쳐 알림은 정확히 1회만 발송되어야 한다
            # (첫 번째 라운드에서 3회 실패 → 임계값 도달 → 알림 1회)
            assert total_alert_count == 1, (
                f"num_rounds={num_rounds}: "
                f"알림이 정확히 1회 발송되어야 하지만 {total_alert_count}회 발송됨"
            )
        finally:
            session.close()
            engine.dispose()

    @settings(max_examples=100)
    @given(
        success_data=_collected_data_st,
    )
    @pytest.mark.asyncio
    async def test_alert_not_fired_when_success_resets_counter(
        self, success_data
    ):
        """수집 성공 시 연속 실패 카운터가 0으로 리셋되므로,
        성공 후 다시 실패해도 임계값에 도달하지 않으면 알림이 발송되지 않는다.

        시나리오: 2회 실패 → 1회 성공(카운터 리셋) → 2회 실패 → 성공
        총 연속 실패가 3에 도달하지 않으므로 알림 0회.

        **Validates: Requirements 7.3**
        """
        # 2회 실패 후 성공, 다시 2회 실패 후 성공 = 총 6번 collect 호출
        # collect_with_retry는 MAX_RETRIES=3이므로 한 번의 호출에서 최대 3번 collect
        # 첫 번째 호출: 2회 실패 + 1회 성공 → 연속 실패 최대 2 → 리셋
        collector = ConfigurableCollector(fail_count=2, success_data=success_data)

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            with patch("collectors.base_collector.get_session", return_value=session):
                with patch("asyncio.sleep", return_value=None):
                    with patch.object(
                        collector, "_alert_consecutive_failures"
                    ) as mock_alert:
                        await collector.collect_with_retry()

            # 2회 실패 후 성공 → 연속 실패 최대 2, 임계값(3) 미도달
            assert mock_alert.call_count == 0, (
                f"2회 실패 후 성공 시 알림이 발송되지 않아야 하지만 "
                f"{mock_alert.call_count}회 호출됨"
            )
            assert collector._consecutive_failures == 0
        finally:
            session.close()
            engine.dispose()


# ── Property 13: 고래 거래 필터링 정확성 ──────────────────────


from collectors.onchain_collector import filter_whale_transactions
from config import WHALE_THRESHOLD_XRP


# 거래 금액 전략: 0 ~ 5,000,000 범위의 양수 float (고래 기준 전후를 포함)
_tx_amount_st = st.floats(min_value=0, max_value=5_000_000, allow_nan=False, allow_infinity=False)

# 단일 거래 dict 전략
_transaction_st = st.fixed_dictionaries(
    {"amount": _tx_amount_st.map(str)},
    optional={
        "hash": st.text(min_size=8, max_size=16, alphabet=st.characters(whitelist_categories=("L", "N"))),
        "destination": st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    },
)

# 거래 목록 전략
_transactions_list_st = st.lists(_transaction_st, min_size=0, max_size=50)


class TestWhaleTransactionFilteringProperty:
    """Property 13: 고래 거래 필터링 정확성 속성 테스트.

    거래 목록에 대해, 고래 거래로 필터링된 결과의 모든 거래는 100만 XRP 이상이어야
    하며, 원본 목록에서 100만 XRP 이상인 거래가 누락되지 않아야 한다.

    **Validates: Requirements 4.3**

    Feature: xrp-price-prediction-dashboard
    Property 13: 고래 거래 필터링 정확성
    """

    @settings(max_examples=100)
    @given(transactions=_transactions_list_st)
    def test_all_filtered_transactions_are_above_threshold(self, transactions):
        """필터링된 모든 거래의 금액은 WHALE_THRESHOLD_XRP(100만 XRP) 이상이어야 한다.

        **Validates: Requirements 4.3**
        """
        result = filter_whale_transactions(transactions, WHALE_THRESHOLD_XRP)

        for tx in result:
            assert float(tx.get("amount", 0)) >= WHALE_THRESHOLD_XRP, (
                f"필터링된 거래의 금액 {tx.get('amount')}이 "
                f"임계값 {WHALE_THRESHOLD_XRP}보다 작다"
            )

    @settings(max_examples=100)
    @given(transactions=_transactions_list_st)
    def test_no_whale_transactions_are_missed(self, transactions):
        """원본 목록에서 WHALE_THRESHOLD_XRP 이상인 거래가 필터링 결과에서 누락되지 않아야 한다.

        **Validates: Requirements 4.3**
        """
        result = filter_whale_transactions(transactions, WHALE_THRESHOLD_XRP)

        # 원본에서 고래 거래에 해당하는 것들을 직접 계산
        expected_whale_txs = [
            tx for tx in transactions
            if float(tx.get("amount", 0)) >= WHALE_THRESHOLD_XRP
        ]

        assert len(result) == len(expected_whale_txs), (
            f"필터링 결과 {len(result)}건이 예상 {len(expected_whale_txs)}건과 다르다"
        )

        # 각 예상 고래 거래가 결과에 포함되어 있는지 확인
        for tx in expected_whale_txs:
            assert tx in result, (
                f"금액 {tx.get('amount')}인 고래 거래가 필터링 결과에서 누락되었다"
            )

    @settings(max_examples=100)
    @given(transactions=_transactions_list_st)
    def test_filtered_result_is_subset_of_original(self, transactions):
        """필터링 결과는 원본 거래 목록의 부분집합이어야 한다.

        **Validates: Requirements 4.3**
        """
        result = filter_whale_transactions(transactions, WHALE_THRESHOLD_XRP)

        for tx in result:
            assert tx in transactions, (
                f"필터링 결과에 원본에 없는 거래가 포함되어 있다: {tx}"
            )
